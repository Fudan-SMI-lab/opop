"""What does a compiled Triton kernel expose WITHOUT launching it, and can we get grid/launch
info + peak memory? Decides whether "compile and inspect" can replace "predict".

Must be a real file on disk: @triton.jit refuses to work from stdin.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def add_k(x_ptr, y_ptr, o_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    a = tl.load(x_ptr + offs, mask=m)
    b = tl.load(y_ptr + offs, mask=m)
    tl.store(o_ptr + offs, a + b, mask=m)


@triton.jit
def mm_k(a_ptr, b_ptr, c_ptr, M, N, K, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
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


print("torch", torch.__version__, "triton", triton.__version__)

# --- 1. AOT compile WITHOUT launching: what metadata comes back?
print("\n=== 1. compile-only via warmup (no launch)")
try:
    x = torch.randn(4096, device="cuda")
    o = torch.empty_like(x)
    kw = add_k.warmup(x, x, o, x.numel(), BLOCK=1024, grid=(4,))
    print("  warmup returned:", type(kw).__name__)
    for attr in ("n_regs", "n_spills", "shared", "num_warps", "metadata", "name"):
        v = getattr(kw, attr, "ABSENT")
        if attr == "metadata" and v != "ABSENT":
            keys = sorted(v._asdict().keys()) if hasattr(v, "_asdict") else sorted(vars(v))
            print("   metadata keys:", keys)
        else:
            print("  ", attr, "=", v)
except Exception as e:
    print("  warmup failed:", type(e).__name__, e)

# --- 2. after a real launch, what is in the JIT cache?
print("\n=== 2. post-launch cache metadata (the path our harness already uses)")
a = torch.randn(256, 256, device="cuda")
b = torch.randn(256, 256, device="cuda")
c = torch.empty(256, 256, device="cuda")
torch.cuda.reset_peak_memory_stats()
mm_k[(1, 1)](a, b, c, 256, 256, 256, BM=64, BN=64, BK=32)
torch.cuda.synchronize()
for name, jitfn in (("add_k", add_k), ("mm_k", mm_k)):
    for dc in getattr(jitfn, "device_caches", {}).values():
        cache = dc[0] if isinstance(dc, tuple) else dc
        for key, ck in (cache or {}).items():
            md = getattr(ck, "metadata", None)
            fields = {}
            for f in ("n_regs", "n_spills", "shared", "num_warps", "num_stages", "num_ctas",
                      "cluster_dims", "name", "global_scratch_size"):
                val = getattr(md, f, getattr(ck, f, None))
                if val is not None:
                    fields[f] = val
            print(" ", name, "->", fields)

# --- 3. peak memory: the dimension we do not currently record at all
print("\n=== 3. peak memory instrumentation")
torch.cuda.reset_peak_memory_stats()
big = torch.randn(2048, 2048, device="cuda")
tmp = big @ big
torch.cuda.synchronize()
print("  max_memory_allocated MB:", torch.cuda.max_memory_allocated() / 1e6)
print("  max_memory_reserved  MB:", torch.cuda.max_memory_reserved() / 1e6)
del big, tmp

# --- 4. is the grid recoverable? (needed to turn static SASS counts into dynamic ones)
print("\n=== 4. grid / launch count recoverability")
print("  triton stores grid at launch time, not in metadata; a wrapper must record it.")
print("  metadata has global_scratch_size / cluster_dims but no grid -- see fields above.")

# --- 5. L2-flush vs warm timing: is the DIFFERENCE a usable traffic signal?
print("\n=== 5. L2-flushed vs warm timing delta (candidate traffic proxy)")


def timeit(fn, flush_bytes=0, n=50):
    flush = torch.empty(int(flush_bytes // 4), device="cuda") if flush_bytes else None
    for _ in range(10):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(n):
        if flush is not None:
            flush.zero_()
        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e))
    times.sort()
    return times[len(times) // 2]


for size in (512, 2048):
    a = torch.randn(size, size, device="cuda")
    b = torch.randn(size, size, device="cuda")
    warm = timeit(lambda: a @ b)
    flushed = timeit(lambda: a @ b, flush_bytes=128 * 1024 * 1024)
    bytes_touched = 3 * size * size * 4
    print("  N=%-5d warm %.4f ms  flushed %.4f ms  delta %+.1f%%  (min bytes %.1f MB)"
          % (size, warm, flushed, 100 * (flushed - warm) / warm, bytes_touched / 1e6))

print("\n=== DONE")
