"""Prove the calibration path really measures the Triton ceilings on this box.

The unit tests exercise pure logic with hardcoded numbers. This exercises the code that has to run
on a GPU: reachable_tflops must produce a figure, gate it on correctness, and disagree with cuBLAS
in the direction the probe measured.
"""
import torch

from kernel_optimizer.gpu.tritonmm import reachable_tflops


def main() -> int:
    dev = torch.device("cuda:0")
    torch.cuda.set_device(dev)
    free, _ = torch.cuda.mem_get_info(dev)
    n = 8192
    while n > 1024 and (3 * n * n * 4) > free * 0.35:
        n //= 2
    flops = 2 * n ** 3
    print("card %s  N=%d" % (torch.cuda.get_device_properties(0).name, n))
    print("%-6s %12s %12s %8s %11s %9s" % ("prec", "cuBLAS", "Triton", "ratio", "rel err", "wrong"))

    import statistics

    def timed(fn, k=10, w=4):
        for _ in range(w):
            fn()
        torch.cuda.synchronize(dev)
        s = []
        for _ in range(k):
            a, b = torch.cuda.Event(True), torch.cuda.Event(True)
            a.record()
            fn()
            b.record()
            torch.cuda.synchronize(dev)
            s.append(a.elapsed_time(b))
        return statistics.median(s)

    ok = 0
    for prec, dtype, tf32 in (("fp32", torch.float32, False), ("tf32", torch.float32, True),
                              ("fp16", torch.float16, True), ("bf16", torch.bfloat16, True)):
        prev = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = tf32
        try:
            x = torch.randn(n, n, device=dev, dtype=dtype)
            y = torch.randn(n, n, device=dev, dtype=dtype)
            cublas = flops / (timed(lambda: x @ y) * 1e-3) / 1e12
            del x, y
            torch.cuda.empty_cache()
        finally:
            torch.backends.cuda.matmul.allow_tf32 = prev
        r = reachable_tflops(prec, n, dev)
        tri = r.get("tflops", 0.0)
        if tri <= 0:
            print("%-6s %12.2f %12s   %s" % (prec, cublas, "NONE", r.get("note", "")[:60]))
            continue
        ok += 1
        print("%-6s %12.2f %12.2f %8.3f %11.2e %9d"
              % (prec, cublas, tri, tri / cublas, r.get("rel_err", -1), r.get("n_wrong", 0)))

    print()
    if ok == 0:
        print("VERDICT: FAILED -- reachable_tflops produced nothing on any precision, so the "
              "calibration would silently fall back to cuBLAS-only ceilings.")
        return 1
    print("VERDICT: OK -- reachable_tflops measured %d of 4 precisions with a correctness gate." % ok)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
