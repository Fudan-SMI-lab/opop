"""Does the contention diagnostic resolve the case that actually blocks us?

Verified in probe_throttle_v2: with correct controls, co-running a bandwidth hog degrades a
bandwidth-saturated kernel by +19.5% and a compute-saturated one by +5.5% -- a 3.5x
separation, both clearing the ~10% noise floor, in the predicted direction.

Now the real question. Our blocking case is a kernel that is SIMULTANEOUSLY at DRAM 93.7%
and occupancy 17% (register-limited). Utilization alone cannot say which to attack. Can a
pair of cheap perturbation experiments say it?

Two perturbations, each needing no privileges:
  P1  bandwidth pressure   -- co-run a memory hog. Sensitivity => bytes are the constraint.
  P2  concurrency pressure -- launch with fewer blocks than the GPU can hold (grid throttle).
      Sensitivity => the kernel needed concurrency, i.e. latency exposure, not bandwidth.

The two together are informative in a way neither is alone:
  P1 high, P2 low  -> bandwidth-saturated: cut bytes, do NOT chase occupancy
  P1 low,  P2 high -> latency-exposed: raise concurrency, cutting bytes will not help
  P1 low,  P2 low  -> neither; look elsewhere (arithmetic, launch overhead, dependencies)
  P1 high, P2 high -> genuinely coupled; report BOTH and say so

Cost: each perturbation is a re-timing of an ALREADY COMPILED kernel, so ~2 extra launch
batches, not extra trials. That is the cost model that makes this affordable.
"""
import torch
import triton
import triton.language as tl

dev = torch.device("cuda")
torch.manual_seed(0)
DRAM_CEIL_GBS = 924.1
FP32_CEIL_TFLOPS = 54.8

SMS = torch.cuda.get_device_properties(0).multi_processor_count
print("device: %s, %d SMs" % (torch.cuda.get_device_name(0), SMS))


@triton.jit
def mm_k(a_ptr, b_ptr, c_ptr, M, N, K, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pid_m, pid_n = tl.program_id(0), tl.program_id(1)
    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k0 in range(0, K, BK):
        offs_k = k0 + tl.arange(0, BK)
        a = tl.load(a_ptr + offs_m[:, None] * K + offs_k[None, :],
                    mask=(offs_m[:, None] < M) & (offs_k[None, :] < K), other=0.0)
        b = tl.load(b_ptr + offs_k[:, None] * N + offs_n[None, :],
                    mask=(offs_k[:, None] < K) & (offs_n[None, :] < N), other=0.0)
        acc += tl.dot(a, b)
    tl.store(c_ptr + offs_m[:, None] * N + offs_n[None, :], acc,
             mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


@triton.jit
def elem_k(x_ptr, y_ptr, o_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    a = tl.load(x_ptr + offs, mask=m)
    b = tl.load(y_ptr + offs, mask=m)
    tl.store(o_ptr + offs, a * b + a, mask=m)


def timeit(fn, n=40, warmup=12):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(n):
        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    ts.sort()
    med = ts[len(ts) // 2]
    mean = sum(ts) / len(ts)
    cv = (sum((t - mean) ** 2 for t in ts) / len(ts)) ** 0.5 / mean * 100
    return med, cv


hog_a = torch.randn(1 << 24, device=dev)
hog_b = torch.empty_like(hog_a)
hog_stream = torch.cuda.Stream()


def p1_bandwidth_pressure(fn, rounds=25):
    """Median runtime while a memory hog competes for DRAM."""
    for _ in range(8):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(rounds):
        with torch.cuda.stream(hog_stream):
            for _ in range(8):
                hog_b.copy_(hog_a)
        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    ts.sort()
    return ts[len(ts) // 2]


def meta_of(jitfn):
    out = {}
    for dc in getattr(jitfn, "device_caches", {}).values():
        cache = dc[0] if isinstance(dc, tuple) else dc
        for ck in (cache or {}).values():
            md = getattr(ck, "metadata", None)
            if md is None:
                continue
            out = {"shared": getattr(md, "shared", None), "n_regs": getattr(ck, "n_regs", None),
                   "n_spills": getattr(ck, "n_spills", None),
                   "warps": getattr(md, "num_warps", None)}
    return out


print()
print("=" * 84)
print("Two perturbations on three workloads with KNOWN answers (positive + negative controls)")
print("=" * 84)
print("%-22s %10s %8s | %7s %7s | %s" %
      ("workload", "median ms", "CV", "P1 bw", "P2 conc", "verdict"))

cases = []

# --- Case 1: elementwise, huge -> must be bandwidth-bound
N = 1 << 25
ex, ey, eo = (torch.randn(N, device=dev) for _ in range(3))
full_grid_1 = triton.cdiv(N, 1024)
cases.append(("elementwise 32M (BW)", lambda g=full_grid_1: elem_k[(g,)](ex, ey, eo, N, BLOCK=1024),
              lambda g: elem_k[(g,)](ex, ey, eo, N, BLOCK=1024), full_grid_1,
              3 * N * 4, None))

# --- Case 2: big fp32 GEMM -> must be compute-bound
M = 2048
ga, gb = torch.randn(M, M, device=dev), torch.randn(M, M, device=dev)
gc = torch.empty(M, M, device=dev)
grid2 = (triton.cdiv(M, 64), triton.cdiv(M, 64))
cases.append(("gemm 2048 fp32 (CMP)",
              lambda: mm_k[grid2](ga, gb, gc, M, M, M, BM=64, BN=64, BK=32),
              None, grid2[0] * grid2[1], 3 * M * M * 4, 2 * M ** 3))

# --- Case 3: small GEMM, tiny grid -> under-filled, should be concurrency-sensitive
Ms = 256
sa, sb = torch.randn(Ms, Ms, device=dev), torch.randn(Ms, Ms, device=dev)
sc = torch.empty(Ms, Ms, device=dev)
grid3 = (triton.cdiv(Ms, 64), triton.cdiv(Ms, 64))
cases.append(("gemm 256 fp32 (small)",
              lambda: mm_k[grid3](sa, sb, sc, Ms, Ms, Ms, BM=64, BN=64, BK=32),
              None, grid3[0] * grid3[1], 3 * Ms * Ms * 4, 2 * Ms ** 3))

for name, fn, _regrid, nblocks, nbytes, nflops in cases:
    base, cv = timeit(fn)
    p1 = p1_bandwidth_pressure(fn)
    p1_pct = 100 * (p1 - base) / base
    gbs = nbytes / (base / 1e3) / 1e9
    util_mem = 100 * gbs / DRAM_CEIL_GBS
    util_cmp = (100 * (nflops / (base / 1e3) / 1e12) / FP32_CEIL_TFLOPS) if nflops else 0.0
    blocks_per_sm = nblocks / SMS
    print("%-22s %10.4f %7.1f%% | %+6.1f%% %7s | mem %.0f%% cmp %.0f%% blocks/SM %.1f"
          % (name, base, cv, p1_pct, "-", util_mem, util_cmp, blocks_per_sm))

print()
print("=" * 84)
print("P2: concurrency pressure -- same kernel, fewer blocks than the GPU can hold")
print("=" * 84)
print("Grid throttling changes the WORK, so it cannot be a like-for-like re-timing.")
print("Instead: time a FIXED slice of work at several grid widths and compare throughput.")
slice_n = 1 << 22
sx, sy, so = (torch.randn(slice_n, device=dev) for _ in range(3))
full = triton.cdiv(slice_n, 1024)
base_t = None
for frac, label in ((1.0, "full grid"), (0.5, "half"), (0.25, "quarter"), (0.10, "10%")):
    g = max(1, int(full * frac))
    n_eff = min(slice_n, g * 1024)

    def run(g=g, n_eff=n_eff):
        elem_k[(g,)](sx, sy, so, n_eff, BLOCK=1024)

    med, cv = timeit(run, n=40, warmup=12)
    thr = (3 * n_eff * 4) / (med / 1e3) / 1e9
    if base_t is None:
        base_t = thr
    print("  %-10s blocks=%-6d blocks/SM=%5.1f  %8.4f ms  throughput %6.1f GB/s (%5.1f%% of full)"
          % (label, g, g / SMS, med, thr, 100 * thr / base_t))
print("  reading: throughput that HOLDS as blocks shrink => not concurrency-limited;")
print("           throughput that FALLS => the kernel needed the extra concurrency.")

print()
print("=" * 84)
print("compile-only metadata (decides predict-vs-compile), re-checked properly")
print("=" * 84)
for bm, bn, bk in ((64, 64, 32), (128, 128, 32), (128, 128, 64)):
    try:
        kw = mm_k.warmup(ga, gb, gc, M, M, M, BM=bm, BN=bn, BK=bk,
                         grid=(triton.cdiv(M, bm), triton.cdiv(M, bn)))
        md = getattr(kw, "metadata", None)
        print("  tile %-12s shared=%-8s n_regs=%-6s n_spills=%-4s  (NO launch happened)"
              % ("%dx%dx%d" % (bm, bn, bk), getattr(md, "shared", "?"),
                 getattr(kw, "n_regs", "?"), getattr(kw, "n_spills", "?")))
    except Exception as exc:
        print("  tile %-12s compile REFUSED: %s" % ("%dx%dx%d" % (bm, bn, bk),
                                                    str(exc)[:80]))
print("\n=== DONE")
