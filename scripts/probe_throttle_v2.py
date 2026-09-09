"""Second attempt at counter-free bottleneck diagnostics, with VERIFIED controls.

The first attempt failed twice and both failures are instructive:
  - my "occupancy throttle" used tl.zeros(), which lives in registers, so it allocated NO
    shared memory (metadata showed shared=None) and the measured effect was ~0%. That is a
    broken probe, not a negative result.
  - my "bandwidth contention" gave the OPPOSITE of my prediction (bandwidth-bound copy
    degraded 0.0%, "compute-bound" GEMM degraded 27.4%). Checking the arithmetic: 8.6 GFLOP
    in 0.3768 ms = 22.8 TFLOP/s against our measured fp32 ceiling of 54.8 -- only 42%, so my
    "compute-bound control" was never verified to be compute-bound.

Lesson applied here: every control is VERIFIED against a measured ceiling before it is used
to judge a technique.
"""
import torch
import triton
import triton.language as tl

dev = torch.device("cuda")
torch.manual_seed(0)

# Measured ceilings for this box (from our own calibration, not spec sheets).
FP32_CEIL_TFLOPS = 54.8
TF32_CEIL_TFLOPS = 88.1
DRAM_CEIL_GBS = 924.1


def timeit(fn, n=80, warmup=20):
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


print("=" * 78)
print("STEP 1: build controls and VERIFY each is what it claims to be")
print("=" * 78)

# Control 1: bandwidth-bound. Large copy, no reuse possible.
N = 1 << 25
xb = torch.randn(N, device=dev)
ob = torch.empty_like(xb)
copy_bytes = 2 * N * 4
med_bw, cv_bw = timeit(lambda: ob.copy_(xb), n=50)
achieved_gbs = copy_bytes / (med_bw / 1e3) / 1e9
print("control BW  : %.4f ms  CV %.1f%%  -> %.1f GB/s = %.1f%% of measured DRAM ceiling"
      % (med_bw, cv_bw, achieved_gbs, 100 * achieved_gbs / DRAM_CEIL_GBS))

# Control 2: compute-bound. Force tf32 OFF so the fp32 ceiling applies, and use a big
# enough matrix that arithmetic dominates.
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
M = 4096
ma = torch.randn(M, M, device=dev)
mb = torch.randn(M, M, device=dev)
gemm_flops = 2 * M ** 3
med_cp, cv_cp = timeit(lambda: ma @ mb, n=40)
achieved_tflops = gemm_flops / (med_cp / 1e3) / 1e12
gemm_bytes = 3 * M * M * 4
print("control CMP : %.4f ms  CV %.1f%%  -> %.1f TFLOP/s = %.1f%% of measured fp32 ceiling"
      % (med_cp, cv_cp, achieved_tflops, 100 * achieved_tflops / FP32_CEIL_TFLOPS))
print("              (its intensity is %.1f FLOP/byte, so it is compute-side by roofline)"
      % (gemm_flops / gemm_bytes))

ok_bw = achieved_gbs / DRAM_CEIL_GBS > 0.75
ok_cp = achieved_tflops / FP32_CEIL_TFLOPS > 0.60
print("\ncontrols valid? bandwidth %s   compute %s"
      % ("YES" if ok_bw else "NO -- not saturated",
         "YES" if ok_cp else "NO -- not compute-saturated"))
if not (ok_bw and ok_cp):
    print("!! At least one control is not what it claims. Any verdict below is UNSAFE.")

print()
print("=" * 78)
print("STEP 2: occupancy throttle -- with shared memory ACTUALLY allocated")
print("=" * 78)


@triton.jit
def mm_shared(a_ptr, b_ptr, c_ptr, M, N, K,
              BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
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


def shared_and_regs(jitfn):
    out = {}
    for dc in getattr(jitfn, "device_caches", {}).values():
        cache = dc[0] if isinstance(dc, tuple) else dc
        for ck in (cache or {}).values():
            md = getattr(ck, "metadata", None)
            if md is None:
                continue
            out = {"shared": getattr(md, "shared", None),
                   "n_regs": getattr(ck, "n_regs", None),
                   "n_spills": getattr(ck, "n_spills", None),
                   "num_warps": getattr(md, "num_warps", None)}
    return out


S = 2048
sa = torch.randn(S, S, device=dev)
sb = torch.randn(S, S, device=dev)
sc = torch.empty(S, S, device=dev)
print("varying the TILE (which really does move shared+regs) and reading the compiler back:")
print("  %-18s %-9s %-8s %-7s %-8s %-10s %s"
      % ("tile", "shared", "n_regs", "spill", "warps", "median ms", "note"))
rows = []
for (bm, bn, bk) in ((32, 32, 32), (64, 64, 32), (128, 128, 32), (128, 128, 64)):
    def run(bm=bm, bn=bn, bk=bk):
        mm_shared[(triton.cdiv(S, bm), triton.cdiv(S, bn))](
            sa, sb, sc, S, S, S, BM=bm, BN=bn, BK=bk)
    try:
        med, cv = timeit(run, n=30, warmup=10)
        meta = shared_and_regs(mm_shared)
        rows.append((f"{bm}x{bn}x{bk}", meta, med, cv))
        print("  %-18s %-9s %-8s %-7s %-8s %-10.4f CV %.1f%%"
              % (f"{bm}x{bn}x{bk}", meta.get("shared"), meta.get("n_regs"),
                 meta.get("n_spills"), meta.get("num_warps"), med, cv))
    except Exception as exc:
        print("  %-18s FAILED %s: %s" % (f"{bm}x{bn}x{bk}", type(exc).__name__,
                                         str(exc)[:70]))
if len(rows) >= 2:
    best = min(rows, key=lambda r: r[2])
    worst = max(rows, key=lambda r: r[2])
    spread = 100 * (worst[2] - best[2]) / best[2]
    print("\n  spread across tiles: %+.1f%% (%s fastest, %s slowest)"
          % (spread, best[0], worst[0]))
    print("  clears our ~10%% noise floor? %s" % ("YES" if spread > 10 else "NO"))
    print("  --> this is the signal we ALREADY exploit via tuning; the question is whether")
    print("      the shared/regs numbers PREDICT the ordering, which is testable offline.")

print()
print("=" * 78)
print("STEP 3: contention diagnostic, re-run against VERIFIED controls")
print("=" * 78)
hog_a = torch.randn(1 << 24, device=dev)
hog_b = torch.empty_like(hog_a)
hog_stream = torch.cuda.Stream()


def with_hog(fn, rounds=30):
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


for name, fn, n in (("BW-bound copy", lambda: ob.copy_(xb), 30),
                    ("CMP-bound gemm", lambda: ma @ mb, 25)):
    alone, _ = timeit(fn, n=n, warmup=8)
    cont = with_hog(fn)
    print("  %-16s alone %8.4f  contended %8.4f  %+7.1f%%"
          % (name, alone, cont, 100 * (cont - alone) / alone))
print("  a USABLE diagnostic must separate these two by >10% IN THE PREDICTED DIRECTION")
print("  (bandwidth-bound should degrade MORE than compute-bound)")

print()
print("=" * 78)
print("STEP 4: does compile-only (no launch) give us regs? -- decides predict-vs-compile")
print("=" * 78)
try:
    kw = mm_shared.warmup(sa, sb, sc, S, S, S, BM=64, BN=64, BK=32,
                          grid=(triton.cdiv(S, 64), triton.cdiv(S, 64)))
    md = getattr(kw, "metadata", None)
    print("  warmup(): shared = %s" % getattr(md, "shared", "ABSENT"))
    print("  warmup(): n_regs = %s" % getattr(kw, "n_regs", "ABSENT"))
    print("  warmup(): maxnreg = %s" % getattr(md, "maxnreg", "ABSENT"))
    print("  --> shared IS available without launching; n_regs is NOT.")
    print("      Consequence: shared-memory feasibility is decidable pre-launch (we do this),")
    print("      but register pressure / occupancy needs an actual launch or a cubin parse.")
except Exception as exc:
    print("  warmup failed: %s: %s" % (type(exc).__name__, str(exc)[:120]))
print("\n=== DONE")
