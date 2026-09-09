"""Can we diagnose "which resource binds" by DELIBERATELY restricting one, using only timing?

Clock locking is refused in our container (verified: even as root), so the frequency-scaling
diagnostic is unavailable. This tests the two alternatives that need no privileges:

  A. OCCUPANCY THROTTLE -- allocate dummy shared memory to force fewer blocks per SM.
     If runtime is insensitive, the kernel was not latency-bound (already saturated).
     If runtime rises proportionally, concurrency was doing real work.

  B. BANDWIDTH CONTENTION -- co-run a memory-hog stream and see if runtime degrades.
     A bandwidth-saturated kernel should degrade much more than a compute-bound one.

The decisive question for BOTH is whether the effect size clears our ~10% noise floor.
Positive control matters: we run one kernel that is definitely bandwidth-bound (a big copy)
and one that is definitely compute-bound (small-matrix repeated GEMM), so a technique that
cannot separate THOSE two is not worth deploying.
"""
import torch
import triton
import triton.language as tl

torch.manual_seed(0)
dev = torch.device("cuda")


@triton.jit
def copy_k(x_ptr, o_ptr, n, BLOCK: tl.constexpr, DUMMY: tl.constexpr):
    # DUMMY shared memory throttles occupancy without changing the work done.
    pid = tl.program_id(0)
    if DUMMY > 0:
        pad = tl.zeros((DUMMY,), dtype=tl.float32)
        tl.store(o_ptr + tl.arange(0, DUMMY), pad, mask=tl.arange(0, DUMMY) < 0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    tl.store(o_ptr + offs, tl.load(x_ptr + offs, mask=m), mask=m)


def timeit(fn, n=100, warmup=20):
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
    var = sum((t - mean) ** 2 for t in ts) / len(ts)
    return med, (var ** 0.5) / mean * 100


print("=== workloads (positive controls)")
N = 1 << 24                                    # 16M floats = 64 MB each way
xb = torch.randn(N, device=dev)
ob = torch.empty_like(xb)
bw_bytes = 2 * N * 4
mc = torch.randn(1024, 1024, device=dev)       # compute-heavy: repeated small GEMM


def bw_work():
    ob.copy_(xb)


def compute_work():
    r = mc
    for _ in range(8):
        r = r @ mc
    return r


for name, fn, note in (("bandwidth-bound copy", bw_work, "%.1f MB" % (bw_bytes / 1e6)),
                       ("compute-bound GEMM x8", compute_work, "1024^3 x8")):
    med, cv = timeit(fn)
    extra = ""
    if name.startswith("bandwidth"):
        extra = "  -> %.1f GB/s" % (bw_bytes / (med / 1e3) / 1e9)
    print("  %-24s %8.4f ms  CV %4.1f%%  (%s)%s" % (name, med, cv, note, extra))

print("\n=== A. OCCUPANCY THROTTLE via dummy shared memory")
print("    (does restricting concurrency change runtime? effect must clear ~10%)")
base = None
for dummy in (0, 1024, 4096, 8192):
    o2 = torch.empty(max(N, dummy), device=dev)

    def run(d=dummy, out=o2):
        copy_k[(triton.cdiv(N, 1024),)](xb, out, N, BLOCK=1024, DUMMY=d)

    try:
        med, cv = timeit(run, n=60, warmup=15)
        if base is None:
            base = med
        # read back the shared bytes the compiler actually allocated
        shared = None
        for dc in copy_k.device_caches.values():
            cache = dc[0] if isinstance(dc, tuple) else dc
            for ck in (cache or {}).values():
                md = getattr(ck, "metadata", None)
                if md is not None and getattr(md, "shared", None):
                    shared = max(shared or 0, md.shared)
        print("  DUMMY=%-5d shared=%-7s %8.4f ms  CV %4.1f%%  vs base %+6.1f%%"
              % (dummy, shared, med, cv, 100 * (med - base) / base))
    except Exception as e:
        print("  DUMMY=%-5d FAILED: %s: %s" % (dummy, type(e).__name__, str(e)[:90]))

print("\n=== B. BANDWIDTH CONTENTION via a co-running memory hog")
print("    (a bandwidth-bound kernel should degrade far more than a compute-bound one)")
hog_a = torch.randn(1 << 23, device=dev)
hog_b = torch.empty_like(hog_a)
hog_stream = torch.cuda.Stream()


def with_hog(fn, rounds=40):
    # keep the hog stream busy while timing fn on the default stream
    for _ in range(10):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(rounds):
        with torch.cuda.stream(hog_stream):
            for _ in range(6):
                hog_b.copy_(hog_a)
        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    ts.sort()
    return ts[len(ts) // 2]


for name, fn in (("bandwidth-bound copy", bw_work), ("compute-bound GEMM x8", compute_work)):
    alone, _ = timeit(fn, n=40, warmup=10)
    contended = with_hog(fn)
    print("  %-24s alone %8.4f  contended %8.4f  degradation %+6.1f%%"
          % (name, alone, contended, 100 * (contended - alone) / alone))

print("\n=== interpretation guide")
print("  A technique is USABLE for us only if it separates the two positive controls")
print("  by more than the ~10% noise floor. Anything smaller is unmeasurable.")
