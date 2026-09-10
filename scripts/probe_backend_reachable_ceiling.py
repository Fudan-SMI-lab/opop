"""G10: is the compute ceiling REACHABLE by the backend our candidates are written in?

The calibration measures fp32/tf32/fp16/bf16 ceilings with `torch.matmul`, i.e. cuBLAS. Every
candidate we generate is Triton. If Triton cannot reach cuBLAS throughput, then
`pct_of_compute_peak` measures a candidate against a roof it structurally cannot touch, and a
kernel at Triton's own limit is told it has headroom left.

An external report puts the gap at 54% vs 96.5% of theoretical on a V100 (cuBLAS vs hand-written
WMMA). That number is about a different card, a different generation and hand-written WMMA rather
than Triton's codegen, so it is a reason to measure, not a number to import. This probe measures
the ratio on OUR card, for each precision, using the same shape and the same timing method the
calibration uses, so the two are comparable and a ratio between them means something.

What the result decides:
  ratio near 1.0    the cuBLAS ceiling is reachable; no per-backend ceiling is needed and G10 can
                    be closed as a non-issue on this hardware.
  ratio well below  a per-backend ceiling IS needed, and this probe supplies its value. The
                    honest form is then "94% of the Triton-reachable roof, which is 78% of the
                    cuBLAS roof" -- both numbers, since the second is what a reviewer will ask
                    about.

Deliberately uses a straightforward tiled Triton matmul, not a tuned one: the question is what a
candidate of the kind our agents actually write can reach. A hand-optimised kernel would answer a
different question. To keep that honest the probe sweeps a few configurations and reports the BEST
of them, so the answer is not an artefact of one unlucky tile choice.

EVERY REPORTED NUMBER IS GATED ON CORRECTNESS. The first version of this probe had no correctness
check at all, and reported Triton ABOVE cuBLAS on three of four precisions -- fp16 at 174.98
TFLOP/s, which is above this card's dense tensor-core peak. Throughput from a kernel that computes
the wrong thing measures skipped work and always looks like good news, so a config whose output
misses a fp64 reference is discarded however fast it was, and a precision whose cuBLAS side misses
its own tolerance has its denominator rejected rather than its ratio reported.
"""
from __future__ import annotations

import itertools
import statistics
import sys

import torch
import triton
import triton.language as tl


@triton.jit
def _mm_kernel(A, B, C, M, N, K,
               sa0, sa1, sb0, sb1, sc0, sc1,
               BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
               GROUP: tl.constexpr, ACC_DTYPE: tl.constexpr, IPREC: tl.constexpr):
    """Standard grouped-ordering tiled matmul -- the shape an agent writes, not a tuned one."""
    pid = tl.program_id(0)
    n_m = tl.cdiv(M, BM)
    n_n = tl.cdiv(N, BN)
    per_group = GROUP * n_n
    gid = pid // per_group
    first_m = gid * GROUP
    # tl.minimum, not the Python builtin: `min(tensor, constexpr)` evaluates a comparison and then
    # calls bool() on the result, which does not mean elementwise minimum inside a kernel. It is the
    # kind of mistake that still compiles and still produces a plausible throughput number.
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
        # input_precision pinned by the caller: without it fp32 inputs silently use tf32
        # tensor cores and the "fp32" measurement is not fp32.
        acc += tl.dot(a, b, input_precision=IPREC)
    tl.store(C + rm[:, None] * sc0 + rn[None, :] * sc1, acc.to(C.dtype.element_ty),
             mask=(rm[:, None] < M) & (rn[None, :] < N))


def timed(fn, n: int = 20, warmup: int = 8) -> float:
    """Median of n CUDA-event timings. Median, not mean: the compile/cache tail is long."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(n):
        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        samples.append(s.elapsed_time(e))
    return statistics.median(samples)


# Correctness tolerance per precision, as a RELATIVE Frobenius error against a fp64 reference.
# A throughput number from a kernel that computes the wrong thing is not a ceiling -- it is the
# speed of doing less work, and it always looks like good news. The first run of this probe reported
# Triton above cuBLAS on three of four precisions with no correctness check anywhere in it.
_TOL = {"fp32": 2e-6, "tf32": 5e-3, "fp16": 5e-2, "bf16": 8e-2}


def rel_err(got: torch.Tensor, want64: torch.Tensor) -> float:
    """Relative Frobenius error against a fp64 reference, computed in fp64."""
    g = got.to(torch.float64)
    return float(torch.linalg.norm(g - want64) / torch.linalg.norm(want64))


def main() -> int:
    if not torch.cuda.is_available():
        print("no CUDA device")
        return 2
    dev = torch.device("cuda:0")
    torch.cuda.set_device(dev)
    name = torch.cuda.get_device_properties(0).name
    free, _ = torch.cuda.mem_get_info(dev)
    n = 8192
    while n > 1024 and (3 * n * n * 4) > free * 0.35:
        n //= 2
    flops = 2 * n ** 3
    print("card: %s   matmul N=%d   (same shape rule as calibration)" % (name, n))
    print()

    # (precision, torch dtype, allow_tf32 for cuBLAS, triton accumulator, tl.dot input_precision)
    #
    # `input_precision` is why the fp32 row exists separately at all. `tl.dot` on fp32 inputs
    # defaults to tf32, i.e. tensor cores, so a run without it reported Triton at 87.05 TFLOP/s
    # against a cuBLAS fp32 roof of 54.95 -- ratio 1.584, which reads as "Triton is 58% faster
    # than cuBLAS at fp32" when in truth the two sides were computing in different precisions.
    # A ratio above 1.0 in a same-precision comparison is the tell, and the fix is to pin
    # input_precision="ieee" so the fp32 row really is fp32 on both sides.
    cases = [
        ("fp32", torch.float32, False, tl.float32, "ieee"),
        ("tf32", torch.float32, True, tl.float32, "tf32"),
        ("fp16", torch.float16, True, tl.float32, None),
        ("bf16", torch.bfloat16, True, tl.float32, None),
    ]
    CONFIGS = list(itertools.product((64, 128), (64, 128, 256), (32, 64), (4, 8), (2, 3, 4)))

    print("%-6s %14s %16s %8s  %9s  %s"
          % ("prec", "cuBLAS TFLOP/s", "Triton TFLOP/s", "ratio", "rel err", "best triton config"))
    rows = []
    notes: list[str] = []
    for prec, dtype, tf32, acc, iprec in cases:
        prev = torch.backends.cuda.matmul.allow_tf32
        try:
            torch.backends.cuda.matmul.allow_tf32 = tf32
            a = torch.randn(n, n, device=dev, dtype=dtype)
            b = torch.randn(n, n, device=dev, dtype=dtype)
            # fp64 reference for the correctness gate. Computed once per precision from the same
            # a/b the timed calls use, so a wrong-answer kernel cannot hide behind different data.
            ref64 = a.to(torch.float64) @ b.to(torch.float64)
            cublas_out = a @ b
            cublas_err = rel_err(cublas_out, ref64)
            cublas = flops / (timed(lambda: a @ b) * 1e-3) / 1e12
        except Exception as exc:  # noqa: BLE001 -- an unsupported dtype must not lose the rest
            print("%-6s cuBLAS FAILED %s: %s" % (prec, type(exc).__name__, str(exc)[:60]))
            torch.backends.cuda.matmul.allow_tf32 = prev
            continue
        finally:
            torch.backends.cuda.matmul.allow_tf32 = prev

        tol = _TOL[prec]
        if cublas_err > tol:
            # The denominator itself is suspect, so the ratio means nothing. Most likely the
            # allow_tf32 flag did not take effect and this row is not the precision it claims.
            notes.append("%s: cuBLAS rel err %.2e exceeds the %.0e tolerance for this precision, so "
                         "the DENOMINATOR is not %s and the ratio is void" % (prec, cublas_err, tol, prec))
            print("%-6s %14.2f %16s  %9.2e  denominator rejected" % (prec, cublas, "--", cublas_err))
            del a, b, ref64, cublas_out
            torch.cuda.empty_cache()
            continue

        best_tf, best_cfg, best_err = 0.0, None, None
        n_wrong = 0
        c = torch.empty((n, n), device=dev, dtype=dtype)
        for BM, BN, BK, warps, stages in CONFIGS:
            try:
                grid = (triton.cdiv(n, BM) * triton.cdiv(n, BN),)
                fn = lambda: _mm_kernel[grid](  # noqa: B023
                    a, b, c, n, n, n,
                    a.stride(0), a.stride(1), b.stride(0), b.stride(1),
                    c.stride(0), c.stride(1),
                    BM=BM, BN=BN, BK=BK, GROUP=8, ACC_DTYPE=acc,
                    IPREC=(iprec or "tf32"),
                    num_warps=warps, num_stages=stages)
                c.zero_()
                fn()
                torch.cuda.synchronize()
                # THE POSITIVE CONTROL. A config whose output is wrong is discarded, however fast
                # it was: its throughput measures skipped work, not a reachable ceiling.
                err = rel_err(c, ref64)
                if not (err == err) or err > tol:  # NaN-safe
                    n_wrong += 1
                    continue
                tfl = flops / (timed(fn, n=10, warmup=4) * 1e-3) / 1e12
                if tfl > best_tf:
                    best_tf, best_cfg, best_err = tfl, (BM, BN, BK, warps, stages), err
            except Exception:  # noqa: BLE001 -- infeasible configs are expected, skip them
                continue
        del a, b, c, ref64, cublas_out
        torch.cuda.empty_cache()
        if best_cfg is None:
            print("%-6s %14.2f %16s" % (prec, cublas, "NO CORRECT CONFIG"))
            notes.append("%s: every one of the %d configs either failed to compile or produced a "
                         "wrong result, so this precision contributes no measurement"
                         % (prec, len(CONFIGS)))
            continue
        if n_wrong:
            notes.append("%s: %d of %d configs were DISCARDED for a wrong result (tolerance %.0e); "
                         "without this gate the fastest of them would have set the ceiling"
                         % (prec, n_wrong, len(CONFIGS), tol))
        rows.append((prec, cublas, best_tf, best_tf / cublas))
        print("%-6s %14.2f %16.2f %8.3f  %9.2e  BM=%d BN=%d BK=%d warps=%d stages=%d"
              % (prec, cublas, best_tf, best_tf / cublas, best_err, *best_cfg))

    print()
    for note in notes:
        print("  note: %s" % note)
    if notes:
        print()
    if not rows:
        print("VERDICT: INCONCLUSIVE -- no precision produced both numbers, so nothing was")
        print("  compared. A probe that ran no positive case has not answered the question.")
        return 2
    # A same-precision ratio materially above 1.0 does not mean Triton beat cuBLAS; it means the
    # two sides were not computing the same thing. Reported loudly rather than averaged in, since
    # the first run of this probe produced exactly that (fp32 ratio 1.584 from an unpinned tl.dot).
    unfair = [r for r in rows if r[3] > 1.05]
    for prec, cb, tt, ratio in unfair:
        print("  *** %s ratio %.3f exceeds 1.0 in a SAME-PRECISION comparison, so the two sides "
              "are not computing the same thing -- check input_precision and the cuBLAS tf32 flag "
              "before reading this row." % (prec, ratio))
    if unfair:
        print()
    worst = min(rows, key=lambda r: r[3])
    print("measured on %d of 4 precisions; worst ratio %.3f on %s" % (len(rows), worst[3], worst[0]))
    if worst[3] >= 0.90:
        print("VERDICT: the cuBLAS ceiling IS broadly reachable from Triton on this card")
        print("  (worst case %.1f%%). A per-backend ceiling would change verdicts by less than the"
              % (100 * worst[3]))
        print("  measurement noise, so G10 is not a live defect HERE -- recheck on H100/A100,")
        print("  where the external 54%-vs-96.5% report originates.")
    else:
        print("VERDICT: a per-backend ceiling IS needed. Triton reaches only %.1f%% of the cuBLAS"
              % (100 * worst[3]))
        print("  roof at worst, so a Triton candidate at its own structural limit is currently")
        print("  reported as having %.0f%% of its headroom unused. Use these ratios to derive a"
              % (100 * (1 - worst[3])))
        print("  reachable ceiling, and report both numbers so the cuBLAS gap stays visible.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
