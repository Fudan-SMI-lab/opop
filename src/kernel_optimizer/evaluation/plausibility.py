"""How fast a CORRECT implementation of this task could possibly be on this box.

WHY THIS EXISTS. The anti-reward-hacking screen used to ask "is the candidate more than 10x
faster than the reference?". That constant is not a statement about any GPU or any task, and
measured across 25 runs / 6894 trials on three boxes it produced:

    rejections ...... 0        (the reject branch was unreachable on the relaxed path)
    flags ........... 3        all on L3:48, all verified correct 5/5

L3:48 flags every time because its reference materializes **40.5x** its own compulsory traffic
across 130 dispatched ops. A legal fusion of that reference is *supposed* to come out above 10x.
Meanwhile the same 10x would be far too LOOSE on a task whose reference is already lean: there,
a kernel skipping half the work lands at 2x and sails through. One constant, wrong in both
directions, and which direction depends entirely on how wasteful the reference happens to be.

WHAT REPLACES IT. A floor, from physics:

    a correct implementation must move `compulsory_bytes` -- distinct inputs, parameters and
    outputs -- and must perform `flop_count` arithmetic. It cannot move those bytes faster than
    this box's MEASURED DRAM bandwidth, and it cannot issue that arithmetic faster than this
    box's MEASURED peak. So

        floor_ms   = max(compulsory_bytes / dram_bytes_per_s, flop_count / peak_flop_per_s)
        ceiling_x  = reference_ms / floor_ms

    and a speedup above `ceiling_x` is not "suspicious", it is IMPOSSIBLE for any correct
    kernel. Nothing about the threshold is chosen; it is read off the task and the card.

On the A800, L3:48: 1.3506 GB compulsory / 1.6858 TB/s = 0.8011 ms; 30.367 GFLOP / 233.96
TFLOP/s = 0.1298 ms; floor = 0.8011 ms; eager 14.0 ms => ceiling 17.48x. The run's 14.29x is
82% of that -- close to physics, and legal. The old 10x called it a cheat.

THREE PLACES THIS DELIBERATELY REFUSES TO ANSWER, because a bound that is wrong is worse than
no bound (it flags honest kernels, and the flag then means nothing):

  * **No calibration.** No measured bandwidth, no floor. Falls back to the legacy constant.
  * **Working set inside L2.** `compulsory_bytes` is a LOGICAL count; an L2 hit never crosses
    the memory bus, so the DRAM term is not a bound at all. This is the same test
    `bottleneck.py` applies before reporting any bandwidth fraction (a purely L2-resident
    control kernel read 287% of DRAM peak on the 4090 and 104% on the A800). When it fires the
    DRAM term is DROPPED and only the arithmetic term stands.
  * **Neither term measurable.** `flop_count` is legitimately 0 for a task with no
    multiply-accumulate, and `compulsory_bytes` can be 0 if the counter failed. Zero is not a
    floor -- it would make every speedup "impossible".

WHY THE NUMERATOR IS THE SLOWEST EAGER BASELINE. Every error here should widen the bound, not
narrow it: this is a FLAG, and a false flag costs a human a re-verification while a missed one
costs nothing that correctness does not already cover. A bigger numerator and a bigger margin
both err toward silence.

WHAT THIS IS NOT. It is not an acceptance gate. Correctness decides acceptance -- recorded as
`speed-guard-must-not-override-correctness` after the 10x rule discarded five verified-correct
trials of `cand-c18203b6` at 11.1-13.9x while accepting a neighbouring point at 8.95x, biasing
the reported optimum downward. This module only ever raises a flag for a human to read.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

# How much slack the bound gets before it will flag. Every input to `floor_ms` can be wrong in
# the direction that makes an honest kernel look impossible:
#
#   * `dram_tbs` is measured with a streaming benchmark, and a kernel with a better access
#     pattern than that benchmark can exceed it;
#   * `compulsory_bytes` counts every distinct input, parameter and output ONCE, but a candidate
#     may legitimately not read one of them (a zero bias, a broadcast it folds away);
#   * the reference baseline is timed once at the start of a run, on a box that may have been
#     busier then than later.
#
# 1.5 is chosen so the bound still separates the thing it is for -- a work-SKIPPING kernel does
# not come in 50% under the floor, it comes in 50x under it (the identity-cache fixture returns
# a cached tensor, i.e. essentially free). It is NOT tuned on any task; a tighter margin would
# buy no detection power and would start flagging honest kernels on lean-reference tasks.
DEFAULT_MARGIN = 1.5


class SpeedupCeiling(BaseModel):
    """The largest speedup a correct implementation of one task could reach on one box."""

    model_config = ConfigDict(frozen=True)

    # The threshold a measured speedup is compared against: `ceiling_x * margin`.
    threshold_x: float
    # The unmargined physical ceiling, kept separately so a report can say how close a legal
    # result came to physics (L3:48's 14.29x against 17.48x) rather than only whether it flagged.
    ceiling_x: float
    floor_ms: float
    margin: float
    # "dram" | "compute" -- which term set the floor. Load-bearing for reading a flag: a
    # compute-set floor on a task with a huge byte count usually means the byte count was
    # dropped by the L2 test, and the reader should know that before trusting the number.
    binding_term: str
    reference_ms: float
    # Plain-language derivation, so a flag in a report can be checked rather than taken.
    derivation: str
    # Set when the DRAM term was discarded because the working set fits in L2.
    dram_applicable: bool = True


def speedup_ceiling(
    *,
    compulsory_bytes: int,
    flop_count: int,
    reference_ms: float,
    dram_tbs: float,
    peak_tflops: float,
    l2_bytes: int = 0,
    margin: float = DEFAULT_MARGIN,
) -> SpeedupCeiling | None:
    """Physical speedup ceiling, or None when this box/task cannot supply one.

    Every argument is a MEASUREMENT: `compulsory_bytes`/`flop_count` from the reference
    (`task_cost.py`), `dram_tbs`/`peak_tflops`/`l2_bytes` from this box's calibration,
    `reference_ms` from this run's own baseline. Nothing is derived from a datasheet and nothing
    is per-task.

    Returns None rather than a large sentinel: a caller must be able to tell "no bound could be
    computed" from "the bound is high", and those two call for different behaviour (fall back to
    the legacy constant vs. use this number).
    """
    if reference_ms <= 0:
        return None

    # The DRAM term, and the reason it may not apply. A logical byte count says nothing about
    # bus traffic once the working set fits in cache, so the term is dropped rather than
    # weakened -- the same call `bottleneck.py` makes before reporting any bandwidth fraction.
    dram_applicable = True
    dram_floor_ms = 0.0
    if compulsory_bytes > 0 and dram_tbs > 0:
        if l2_bytes > 0 and compulsory_bytes <= l2_bytes:
            dram_applicable = False
        else:
            dram_floor_ms = (compulsory_bytes / (dram_tbs * 1e12)) * 1e3

    # The arithmetic term. `peak_tflops` should be the HIGHEST ceiling this box measured, across
    # precisions: a candidate is allowed to use tensor cores, so a floor computed from the fp32
    # roof would be far too high and would flag every legal low-precision kernel. On the A800
    # that is the difference between 19.0 and 234.0 TFLOP/s -- a factor of 12.
    compute_floor_ms = 0.0
    if flop_count > 0 and peak_tflops > 0:
        compute_floor_ms = (flop_count / (peak_tflops * 1e12)) * 1e3

    floor_ms = max(dram_floor_ms, compute_floor_ms)
    if floor_ms <= 0:
        # Neither term measurable. A zero floor would make every speedup infinite-x "impossible".
        return None
    binding_term = "dram" if dram_floor_ms >= compute_floor_ms else "compute"

    ceiling_x = reference_ms / floor_ms
    bits = []
    if dram_applicable and dram_floor_ms > 0:
        bits.append("%.4f GB compulsory traffic / %.4f TB/s measured bandwidth = %.4f ms"
                    % (compulsory_bytes / 1e9, dram_tbs, dram_floor_ms))
    elif not dram_applicable:
        bits.append("DRAM term dropped: %.1f MiB compulsory traffic fits this box's measured "
                    "%.1f MiB L2, so the logical byte count is not bus traffic"
                    % (compulsory_bytes / 2**20, l2_bytes / 2**20))
    if compute_floor_ms > 0:
        bits.append("%.3f GFLOP / %.1f TFLOP/s measured peak = %.4f ms"
                    % (flop_count / 1e9, peak_tflops, compute_floor_ms))
    derivation = (
        "%s; floor = %.4f ms (%s-bound); reference %.3f ms / floor = %.2fx physical ceiling; "
        "x%.2f margin = %.2fx flag threshold"
        % ("; ".join(bits), floor_ms, binding_term, reference_ms, ceiling_x, margin,
           ceiling_x * margin)
    )
    return SpeedupCeiling(
        threshold_x=ceiling_x * margin,
        ceiling_x=ceiling_x,
        floor_ms=floor_ms,
        margin=margin,
        binding_term=binding_term,
        reference_ms=reference_ms,
        derivation=derivation,
        dram_applicable=dram_applicable,
    )


def best_measured_peak_tflops(calibration) -> float:
    """The highest arithmetic ceiling this box measured, over every precision and backend.

    The HIGHEST, deliberately, and this is the whole reason the helper exists rather than a
    caller reaching for `fp32_tflops`. A candidate may legally use any arithmetic path the
    correctness gate accepts, so the floor has to be computed against the fastest one available
    -- otherwise the bound sits 12x too high on an A800 (19.0 fp32 vs 234.0 bf16) and flags
    every legitimate low-precision kernel, which is the exact failure the 10x constant already
    had. Recorded as `a-ceiling-is-the-max-over-measured-backends`: cuBLAS and Triton disagree
    in BOTH directions, so neither library alone is the roof.

    Takes the object rather than the numbers so a calibration gaining a new precision is
    included automatically instead of silently omitted here.
    """
    if calibration is None:
        return 0.0
    names = ("fp32_tflops", "tf32_tflops", "fp16_tflops", "bf16_tflops",
             "fp32_triton_tflops", "tf32_triton_tflops", "fp16_triton_tflops",
             "bf16_triton_tflops")
    peaks = []
    for name in names:
        try:
            value = float(getattr(calibration, name, 0.0) or 0.0)
        except (TypeError, ValueError):
            continue
        if value > 0:
            peaks.append(value)
    return max(peaks) if peaks else 0.0
