"""Hard readiness gate for a new box: a REAL Triton kernel must compile, launch and be correct.

`torch.cuda.is_available()` returning True is not the gate. It was True on a box where the first
Triton launch failed, and "imports fine" is exactly the kind of positive that hides an arch
mismatch. So this compiles a kernel on the actual device, launches it, and checks the numbers --
plus reads back the compile metadata the harness depends on (n_regs / shared / spills), because a
metadata accessor that silently returns nothing disables the whole Tier-1 signal set.
"""
import sys

import torch
import triton
import triton.language as tl


@triton.jit
def _axpy(X, Y, OUT, n, ALPHA: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    off = pid * BLOCK + tl.arange(0, BLOCK)
    m = off < n
    tl.store(OUT + off, ALPHA * tl.load(X + off, mask=m) + tl.load(Y + off, mask=m), mask=m)


def main() -> int:
    if not torch.cuda.is_available():
        print("FAIL: no CUDA device")
        return 1
    dev = torch.device("cuda:0")
    p = torch.cuda.get_device_properties(0)
    print("card        : %s" % p.name)
    print("capability  : sm_%d%d" % (p.major, p.minor))
    print("SMs         : %d" % p.multi_processor_count)
    print("VRAM        : %.1f GiB" % (p.total_memory / 2**30))
    print("shared/block: %d B" % getattr(p, "shared_memory_per_block_optin",
                                        p.shared_memory_per_block))
    print("L2          : %.1f MiB" % (p.L2_cache_size / 2**20))
    print("torch/triton: %s / %s" % (torch.__version__, triton.__version__))
    print()

    n = 1 << 20
    x = torch.randn(n, device=dev)
    y = torch.randn(n, device=dev)
    out = torch.empty_like(x)
    BLOCK = 1024
    grid = (triton.cdiv(n, BLOCK),)
    compiled = _axpy[grid](x, y, out, n, ALPHA=2.5, BLOCK=BLOCK)
    torch.cuda.synchronize(dev)

    err = float(torch.linalg.norm(out - (2.5 * x + y)) / torch.linalg.norm(2.5 * x + y))
    print("launch      : OK, grid=%s" % (grid,))
    print("correctness : rel err %.2e" % err)
    if not (err == err) or err > 1e-6:
        print("FAIL: the kernel launched but computed the wrong answer")
        return 1

    # The metadata the harness's Tier-1 signals are built on. Absent metadata looks identical to
    # "this kernel is unremarkable", so its presence is checked explicitly.
    #
    # READ THE SAME OBJECTS THE HARNESS READS: n_regs/n_spills hang off the compiled kernel, while
    # shared/num_warps/num_stages live on `compiled.metadata`. Reading all four off the kernel made
    # this probe warn that shared and num_warps were blind on a box where the harness's own
    # accessor returns them fine -- a false alarm about the box caused by the probe.
    meta_obj = getattr(compiled, "metadata", None)
    meta = {k: getattr(compiled, k, None) for k in ("n_regs", "n_spills")}
    meta.update({k: (getattr(meta_obj, k, None) if meta_obj else None)
                 for k in ("shared", "num_warps", "num_stages")})
    print("metadata    : %s" % meta)
    # `shared` is legitimately 0 for a kernel that allocates none, so only None means blind.
    missing = [k for k, v in meta.items() if v is None]
    if missing:
        print("WARN: no %s from the compiled kernel -- Tier-1 resource signals would be blind"
              % ", ".join(missing))

    # Measured ceilings, so this box's numbers are on record from the start rather than guessed.
    import statistics
    def timed(fn, k=20, w=8):
        for _ in range(w):
            fn()
        torch.cuda.synchronize(dev)
        s = []
        for _ in range(k):
            a, b = torch.cuda.Event(True), torch.cuda.Event(True)
            a.record(); fn(); b.record(); torch.cuda.synchronize(dev)
            s.append(a.elapsed_time(b))
        return statistics.median(s)

    print()
    big = torch.empty(1 << 27, device=dev)          # 512 MiB
    o = torch.empty_like(big)
    ms = timed(lambda: torch.add(big, 1.0, out=o))
    print("DRAM        : %.1f GB/s (512 MiB streaming add)" % (2 * big.numel() * 4 / (ms * 1e-3) / 1e9))
    del big, o
    torch.cuda.empty_cache()

    m = 8192
    a = torch.randn(m, m, device=dev)
    b = torch.randn(m, m, device=dev)
    f = 2 * m ** 3
    for label, tf32 in (("fp32 ieee", False), ("tf32", True)):
        prev = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = tf32
        try:
            print("%-12s: %.1f TFLOP/s" % (label, f / (timed(lambda: a @ b) * 1e-3) / 1e12))
        finally:
            torch.backends.cuda.matmul.allow_tf32 = prev
    del a, b
    torch.cuda.empty_cache()
    for label, dt in (("fp16", torch.float16), ("bf16", torch.bfloat16)):
        ah = torch.randn(m, m, device=dev, dtype=dt)
        bh = torch.randn(m, m, device=dev, dtype=dt)
        print("%-12s: %.1f TFLOP/s" % (label, f / (timed(lambda: ah @ bh) * 1e-3) / 1e12))
        del ah, bh
        torch.cuda.empty_cache()

    print()
    print("VERDICT: READY -- Triton compiles, launches and is correct on this card.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
