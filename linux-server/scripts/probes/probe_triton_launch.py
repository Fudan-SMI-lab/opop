"""Prove a real Triton kernel compiles AND launches on this GPU, and that the check can fail.

`torch.cuda.is_available()` passing tells you nothing about Triton: the compiler runs on the
host, emits PTX for the card's compute capability, and only fails at load time. A version
string is not evidence either. So this launches a kernel, checks the numbers, and then makes
the same check fail on purpose -- a probe that cannot report a failure is worth nothing.

Also reports the compiler metadata the harness harvests (n_regs / n_spills / shared), because
`ProfileRecord` comes from exactly this path and a silent None there degrades every bottleneck
report without erroring.

Run with the WORKER venv's python.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _axpy(x_ptr, y_ptr, out_ptr, a, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    y = tl.load(y_ptr + offs, mask=mask, other=0.0)
    tl.store(out_ptr + offs, a * x + y, mask=mask)


def main() -> int:
    p = torch.cuda.get_device_properties(0)
    print(f"device: {p.name}  sm_{p.major}{p.minor}  torch {torch.__version__}  "
          f"triton {triton.__version__}")

    n = 4096
    x = torch.randn(n, device="cuda")
    y = torch.randn(n, device="cuda")
    out = torch.empty_like(x)
    grid = (triton.cdiv(n, 256),)
    compiled = _axpy[grid](x, y, out, 2.5, n, BLOCK=256)
    torch.cuda.synchronize()

    expected = 2.5 * x + y
    max_err = (out - expected).abs().max().item()
    print(f"launch OK, max abs error vs torch: {max_err:.3e}")
    ok = max_err < 1e-5

    # The metadata the harness turns into ProfileRecord. If these are None the profiler
    # silently degrades, so surface them here rather than discovering it mid-run.
    meta = {k: getattr(compiled, k, None)
            for k in ("n_regs", "n_spills", "shared", "num_warps", "num_stages")}
    print("compiler metadata:", meta)
    if meta["n_regs"] is None:
        print("  WARNING: n_regs is None -- ProfileRecord will be empty and every")
        print("  bottleneck report loses its register-pressure evidence.")

    # tf32 / fp16 tensor-core path: the tasks depend on it, and it is a separate code path
    # in Triton from the elementwise kernel above.
    a = torch.randn(256, 256, device="cuda")
    b = torch.randn(256, 256, device="cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    ref = (a.double() @ b.double()).float()
    got = a @ b
    rel = ((got - ref).abs().max() / ref.abs().max()).item()
    print(f"tf32 matmul relative error: {rel:.3e}  (expected ~1e-3, NOT ~1e-7)")

    # NEGATIVE CONTROL: the correctness check must be able to fail.
    wrong = 3.5 * x + y
    control_err = (wrong - expected).abs().max().item()
    if control_err < 1e-5:
        print("CONTROL BROKEN: a deliberately wrong result passed the tolerance")
        return 1
    print(f"control OK: a wrong coefficient gives max err {control_err:.3f}, "
          f"correctly outside tolerance")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
