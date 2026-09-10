"""The arithmetic ceiling REACHABLE from the backend our candidates are written in (G10).

WHY THIS EXISTS. Calibration measured each precision's compute ceiling with `torch.matmul`, i.e.
cuBLAS. Every candidate this harness generates is Triton. Measured on box 1 (RTX 4090), with every
number gated on correctness against a fp64 reference, the two backends do NOT agree, and they
disagree in BOTH directions:

    prec   cuBLAS   Triton   ratio    consequence of using the cuBLAS figure as the ceiling
    fp32    54.20    45.61   0.841    a Triton kernel at its own structural limit is reported as
                                      having 16% of its headroom unused -- it is measured against
                                      a roof its backend cannot reach
    tf32    88.14    86.58   0.982    agrees
    fp16   159.47   174.78   1.096    the ceiling is BELOW what a candidate achieves, so the
    bf16   162.29   175.18   1.079    fraction exceeds 100% and reads downstream as "saturated,
                                      stop optimizing" for a kernel that is not at any limit

Both failures are the same mistake: treating ONE LIBRARY'S ACHIEVEMENT as the physical roof. So the
rule here is that a ceiling is an UPPER BOUND OVER EVERY PATH MEASURED ON THE BOX -- max(cuBLAS,
Triton) -- and the per-backend figure is recorded next to it so an unreachable roof stays visible
instead of silently demanding the impossible. No ratio is hardcoded; both sides are measured here.

This also refutes the external report that motivated the check (V100: 96.5% of theoretical via
cuBLAS vs 54% via hand-written WMMA). On this card Triton reaches 98.2% of cuBLAS at tf32 and
EXCEEDS it at fp16/bf16. The 54% figure describes a different card, generation and code generator,
and importing it would have produced a ceiling that is wrong on this hardware.

EVERY NUMBER IS GATED ON CORRECTNESS. A kernel that computes the wrong thing is faster, and that
always looks like good news. The first version of the probe behind this module had no correctness
check and reported Triton above cuBLAS on three of four precisions -- a conclusion that happened to
survive the gate, but only because it was checked.
"""
from __future__ import annotations

import statistics

import torch
import triton
import triton.language as tl


@triton.jit
def _mm_kernel(A, B, C, M, N, K,
               sa0, sa1, sb0, sb1, sc0, sc1,
               BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
               GROUP: tl.constexpr, ACC_DTYPE: tl.constexpr, IPREC: tl.constexpr):
    """A plain grouped-ordering tiled matmul: the shape an agent writes, not a tuned library.

    Deliberately unremarkable. The question this answers is what a candidate OF THE KIND OUR AGENTS
    PRODUCE can reach, so a hand-optimised kernel would measure the wrong thing and set a ceiling
    no candidate could approach.
    """
    pid = tl.program_id(0)
    n_m = tl.cdiv(M, BM)
    n_n = tl.cdiv(N, BN)
    per_group = GROUP * n_n
    gid = pid // per_group
    first_m = gid * GROUP
    # tl.minimum, not the Python builtin: `min(tensor, constexpr)` evaluates a comparison and calls
    # bool() on the result, which is not an elementwise minimum. It compiles and produces a
    # plausible throughput number, which is why it is called out here.
    gsize = tl.minimum(n_m - first_m, GROUP)
    pid_m = first_m + ((pid % per_group) % gsize)
    pid_n = (pid % per_group) // gsize

    rm = pid_m * BM + tl.arange(0, BM)
    rn = pid_n * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    acc = tl.zeros((BM, BN), dtype=ACC_DTYPE)
    for k in range(0, tl.cdiv(K, BK)):
        kk = k * BK + rk
        a = tl.load(A + rm[:, None] * sa0 + kk[None, :] * sa1,
                    mask=(rm[:, None] < M) & (kk[None, :] < K), other=0.0)
        b = tl.load(B + kk[:, None] * sb0 + rn[None, :] * sb1,
                    mask=(kk[:, None] < K) & (rn[None, :] < N), other=0.0)
        # input_precision comes from the caller and is PINNED. Without it `tl.dot` on fp32 inputs
        # silently uses tf32 tensor cores, and the "fp32" ceiling is then measured in a different
        # precision than it claims -- which read as Triton being 58% faster than cuBLAS at fp32.
        acc += tl.dot(a, b, input_precision=IPREC)
    tl.store(C + rm[:, None] * sc0 + rn[None, :] * sc1, acc.to(C.dtype.element_ty),
             mask=(rm[:, None] < M) & (rn[None, :] < N))


# Relative Frobenius error a correct GEMM may show, per precision. These are input-rounding
# budgets, not tuning knobs: fp16 inputs with fp32 accumulation over K=8192 give ~2e-4, and the
# measured figure was 2.07e-04. A value far below its tolerance is the evidence that the timed
# kernel computed the right thing.
_TOL = {"fp32": 2e-5, "tf32": 5e-3, "fp16": 5e-2, "bf16": 8e-2}

# A small sweep, not an exhaustive one. Calibration runs at the start of every run, so this must
# cost seconds. These six bracket the configurations that actually won across all four precisions
# in the 72-config probe (BM=128, BN in {128,256}, BK in {32,64}, warps in {4,8}, stages in {2,3}),
# so the best of them is not an artefact of one unlucky tile choice.
_CONFIGS = (
    (128, 128, 32, 4, 2),
    (128, 128, 32, 4, 3),
    (128, 128, 64, 4, 2),
    (128, 256, 32, 8, 3),
    (128, 256, 64, 8, 3),
    (64, 128, 32, 4, 3),
)

# Which `tl.dot` input_precision each calibrated precision must use. fp32 MUST be "ieee": the
# default would put fp32 inputs on tf32 tensor cores and the fp32 ceiling would not be fp32.
_IPREC = {"fp32": "ieee", "tf32": "tf32", "fp16": "tf32", "bf16": "tf32"}


def _timed(fn, device, n: int = 10, warmup: int = 4) -> float:
    """Median of n CUDA-event timings. Median because the compile/cache tail is long."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize(device)
    samples = []
    for _ in range(n):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize(device)
        samples.append(s.elapsed_time(e))
    return statistics.median(samples)


def reachable_tflops(precision: str, n: int, device) -> dict:
    """Best CORRECT Triton throughput at `precision` on an n x n matmul, in TFLOP/s.

    Returns a dict with `tflops` (0.0 when nothing correct compiled), `rel_err` of the winning
    configuration, `config`, and `n_wrong` -- the number of configurations discarded for a wrong
    result. `n_wrong` is reported rather than swallowed: if it is ever large, the fastest of those
    discarded kernels is what a probe without a correctness gate would have called the ceiling.

    Never raises. A box where this cannot run must still calibrate, so every failure degrades to
    `tflops: 0.0` plus a note, and the caller then keeps the cuBLAS figure alone.
    """
    out: dict = {"tflops": 0.0, "n_wrong": 0, "n_failed": 0}
    dtype = {"fp32": torch.float32, "tf32": torch.float32,
             "fp16": torch.float16, "bf16": torch.bfloat16}.get(precision)
    if dtype is None:
        out["note"] = "unknown precision %r" % (precision,)
        return out
    tol = _TOL[precision]
    flops = 2 * n ** 3
    a = b = c = ref64 = None
    try:
        a = torch.randn(n, n, device=device, dtype=dtype)
        b = torch.randn(n, n, device=device, dtype=dtype)
        # fp64 reference from the same operands the timed calls use, so a wrong-answer kernel
        # cannot hide behind different data.
        ref64 = a.to(torch.float64) @ b.to(torch.float64)
        ref_norm = torch.linalg.norm(ref64)
        c = torch.empty((n, n), device=device, dtype=dtype)
        best_tf, best_cfg, best_err = 0.0, None, None
        for BM, BN, BK, warps, stages in _CONFIGS:
            try:
                grid = (triton.cdiv(n, BM) * triton.cdiv(n, BN),)

                def fn(BM=BM, BN=BN, BK=BK, warps=warps, stages=stages):
                    _mm_kernel[grid](
                        a, b, c, n, n, n,
                        a.stride(0), a.stride(1), b.stride(0), b.stride(1),
                        c.stride(0), c.stride(1),
                        BM=BM, BN=BN, BK=BK, GROUP=8,
                        ACC_DTYPE=tl.float32, IPREC=_IPREC[precision],
                        num_warps=warps, num_stages=stages)

                c.zero_()
                fn()
                torch.cuda.synchronize(device)
                err = float(torch.linalg.norm(c.to(torch.float64) - ref64) / ref_norm)
                # THE POSITIVE CONTROL. A configuration whose output is wrong is discarded however
                # fast it was: its throughput measures skipped work, not a reachable ceiling.
                if not (err == err) or err > tol:      # NaN-safe
                    out["n_wrong"] += 1
                    continue
                tfl = flops / (_timed(fn, device) * 1e-3) / 1e12
                if tfl > best_tf:
                    best_tf, best_cfg, best_err = tfl, (BM, BN, BK, warps, stages), err
            except Exception:      # noqa: BLE001 -- infeasible configs are expected; skip them
                out["n_failed"] += 1
                continue
        if best_cfg is None:
            out["note"] = ("no configuration of %d produced a correct result within %.0e, so no "
                           "Triton-reachable figure exists for %s on this box"
                           % (len(_CONFIGS), tol, precision))
            return out
        out.update({"tflops": best_tf, "rel_err": best_err, "config": list(best_cfg)})
        return out
    except Exception as exc:      # noqa: BLE001 -- calibration must never die here
        out["note"] = "%s: %s" % (type(exc).__name__, str(exc)[:160])
        return out
    finally:
        del a, b, c, ref64
        torch.cuda.empty_cache()
