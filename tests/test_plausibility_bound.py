"""A1: the implausible-speedup threshold must be DERIVED, never a constant.

WHAT WAS WRONG WITH 10x, measured rather than argued. Across 25 runs / 6894 trials on three
boxes the old constant rejected NOTHING and flagged three verified-correct winners -- all three
on L3:48, which flags every single time because its reference materializes 40.5x its own
compulsory traffic across 130 dispatched ops. A legal fusion of that reference is *supposed* to
land above 10x. On the A800 the physical ceiling for that task is 17.48x and the run reached
14.29x: 82% of physics, and the constant called it a cheat.

The same constant is far too LOOSE in the other direction. On a task whose reference is already
lean (fusion headroom near 1.0), a kernel that skips half the work lands at 2x and sails through.
One number, wrong in both directions, and which direction depends entirely on how wasteful the
reference happens to be -- a property of the benchmark, not of the candidate.

THE REPLACEMENT is a floor: a correct implementation must move `compulsory_bytes` at no more than
this box's measured bandwidth, and must issue `flop_count` at no more than its measured peak.
`reference_ms / floor_ms` is then a ceiling no correct kernel can exceed. Nothing here is chosen.

THREE REFUSALS ARE TESTED AS HARD AS THE HAPPY PATH, because a bound that is wrong is worse than
no bound -- it flags honest kernels, and then the flag means nothing to whoever reads it.
"""

from __future__ import annotations

from kernel_optimizer.evaluation.plausibility import (
    DEFAULT_MARGIN,
    best_measured_peak_tflops,
    speedup_ceiling,
)

# The A800's real numbers, from `run-l3-48-20260911-052647`'s own events.jsonl -- task cost from
# TASK_COST_MEASURED, ceilings from CALIBRATION_LOADED, baseline from BASELINE_DONE. Copied from
# the log rather than invented, because a fixture agreed with the reader is proof of nothing
# (recorded as `a-fixture-invented-to-match-the-reader-proves-nothing`).
L3_48_A800 = dict(
    compulsory_bytes=1_350_565_888,
    flop_count=30_366_760_960,
    reference_ms=14.0,
    dram_tbs=1.6858134468378299,
    peak_tflops=233.95617083795432,   # bf16, the highest this box measured
)


def test_the_l3_48_case_that_the_10x_constant_got_wrong():
    """The measured case. 14.29x must sit UNDER the derived bound, where 10x rejected it."""
    c = speedup_ceiling(**L3_48_A800)
    assert c is not None
    # 1.3506 GB / 1.6858 TB/s = 0.8011 ms.
    assert abs(c.floor_ms - 0.8011) < 0.002, c.floor_ms
    assert c.binding_term == "dram", (
        "this task is bandwidth-floored: 0.80 ms of traffic against 0.13 ms of arithmetic")
    # 14.0 / 0.8011 = 17.48x.
    assert abs(c.ceiling_x - 17.48) < 0.05, c.ceiling_x
    # And the run's actual result is legal.
    assert 14.294 < c.threshold_x, (
        "the run's measured 14.29x must fall below the flag threshold, or this change has "
        "reproduced the exact failure it exists to remove")
    # While the old constant would have flagged it, which is the whole point.
    assert 14.294 > 10.0


def test_a_work_skipping_kernel_is_still_caught():
    """The bound must still separate the thing it is for.

    A kernel that skips the work does not come in 50% under the floor, it comes in orders of
    magnitude under: the identity-cache fixture returns a cached tensor, so its timed loop is
    essentially free. Tested at the floor itself and at 100x beyond it.
    """
    c = speedup_ceiling(**L3_48_A800)
    # A "kernel" that runs 100x faster than physics allows.
    cheat_ms = c.floor_ms / 100
    assert L3_48_A800["reference_ms"] / cheat_ms > c.threshold_x, (
        "a kernel 100x faster than the physical floor must be flagged")
    # And one exactly at the floor -- the fastest a correct kernel could be -- must NOT be.
    assert L3_48_A800["reference_ms"] / c.floor_ms < c.threshold_x, (
        "a kernel exactly at the physical floor is legal and must not be flagged")


def test_the_margin_widens_the_bound_rather_than_narrowing_it():
    """Every uncertainty must err toward silence, since a false flag costs a human's time."""
    c = speedup_ceiling(**L3_48_A800)
    assert c.threshold_x > c.ceiling_x, "the margin must be applied outward"
    assert c.margin == DEFAULT_MARGIN
    tight = speedup_ceiling(**L3_48_A800, margin=1.0)
    assert tight.threshold_x == tight.ceiling_x, "margin 1.0 means the bare physical ceiling"


def test_no_calibration_yields_no_bound_rather_than_a_fallback_constant():
    """Refusal 1. A box with no measured bandwidth and no measured peak has no ceiling.

    None, not a large number: the caller must be able to tell "no bound could be computed" from
    "the bound is high", because those call for different behaviour -- no flag at all versus a
    flag against this number.
    """
    assert speedup_ceiling(compulsory_bytes=1_350_565_888, flop_count=30_366_760_960,
                           reference_ms=14.0, dram_tbs=0.0, peak_tflops=0.0) is None


def test_an_uncountable_task_cost_yields_no_bound():
    """Refusal 2. Zero is not a floor -- it would make every speedup infinitely 'impossible'."""
    assert speedup_ceiling(compulsory_bytes=0, flop_count=0, reference_ms=14.0,
                           dram_tbs=1.686, peak_tflops=234.0) is None
    # A missing reference is equally fatal: there is nothing to take a ratio against.
    assert speedup_ceiling(**{**L3_48_A800, "reference_ms": 0.0}) is None


def test_a_task_with_no_arithmetic_still_gets_a_bandwidth_bound():
    """`flop_count = 0` is a legitimate measurement (maxpool, elementwise), not a failure.

    Such a task's ceiling is bandwidth, so the bound must still be computable from the DRAM term
    alone -- refusing here would leave exactly the tasks whose floor is cleanest unprotected.
    """
    c = speedup_ceiling(**{**L3_48_A800, "flop_count": 0})
    assert c is not None and c.binding_term == "dram"
    assert abs(c.floor_ms - 0.8011) < 0.002


def test_an_l2_resident_working_set_drops_the_dram_term_instead_of_trusting_it():
    """Refusal 3, and the subtlest. `compulsory_bytes` is a LOGICAL count.

    Once the working set fits in L2 the bytes need never cross the memory bus, so bandwidth is
    not a floor at all -- an L2-resident control kernel read 287% of DRAM peak on the 4090 and
    104% on the A800 (`pct-of-dram-peak-counts-l2-hits-as-dram-traffic`). The DRAM term is
    DROPPED rather than weakened, and the arithmetic term stands alone.
    """
    # 8 MB of traffic against a 40 MiB L2.
    c = speedup_ceiling(compulsory_bytes=8_000_000, flop_count=30_366_760_960,
                        reference_ms=14.0, dram_tbs=1.686, peak_tflops=234.0,
                        l2_bytes=41_943_040)
    assert c is not None
    assert c.dram_applicable is False, "an L2-resident working set must not yield a DRAM floor"
    assert c.binding_term == "compute", (
        "with the DRAM term dropped the arithmetic term must be the one that binds")
    # 30.367 GFLOP / 234 TFLOP/s = 0.1298 ms -- and NOT the 0.0047 ms the dropped DRAM term
    # would have produced, which would have put the ceiling at 2950x and flagged nothing ever.
    assert abs(c.floor_ms - 0.1298) < 0.001, c.floor_ms
    assert "L2" in c.derivation, "the report must say the term was dropped and why"


def test_the_same_task_above_l2_keeps_its_dram_term():
    """The positive control for the test above -- otherwise it could pass by dropping always."""
    c = speedup_ceiling(**L3_48_A800, l2_bytes=41_943_040)
    assert c.dram_applicable is True, (
        "1.35 GB against a 40 MiB L2 must keep the DRAM floor: 1.35 GB does not fit in 40 MiB")
    assert c.binding_term == "dram"


def test_a_bound_with_no_l2_measurement_still_uses_dram():
    """`l2_bytes = 0` means unmeasured. Treating unmeasured as "fits" would silently disable the
    DRAM floor on every box whose calibration predates the L2 probe -- the A800's cached
    calibration is exactly such a box (`l2_bytes: None` on disk)."""
    c = speedup_ceiling(**L3_48_A800, l2_bytes=0)
    assert c.dram_applicable is True and c.binding_term == "dram"


def test_the_peak_is_the_max_over_measured_precisions_not_fp32():
    """A candidate may legally use tensor cores, so the floor must use the fastest path.

    On the A800 that is 234.0 bf16 against 19.0 fp32 -- a factor of 12. Computing the arithmetic
    floor from fp32 would put it 12x too high and flag every legitimate low-precision kernel,
    which is the failure mode the 10x constant already had.
    """
    class Cal:
        fp32_tflops = 19.0
        tf32_tflops = 111.8
        fp16_tflops = 226.2
        bf16_tflops = 234.0
        fp32_triton_tflops = 0.0
        tf32_triton_tflops = 0.0
        fp16_triton_tflops = 0.0
        bf16_triton_tflops = 0.0

    assert best_measured_peak_tflops(Cal()) == 234.0


def test_a_triton_ceiling_can_win_because_neither_library_is_the_roof():
    """Measured on box 1, Triton beats cuBLAS at fp16/bf16 by 7.9-9.6% while losing at fp32, so
    the roof is the max over measured PATHS, not over precisions of one library
    (`a-ceiling-is-the-max-over-measured-backends`)."""
    class Cal:
        fp32_tflops = 54.8
        tf32_tflops = 88.0
        fp16_tflops = 158.0
        bf16_tflops = 164.0
        fp32_triton_tflops = 46.1
        tf32_triton_tflops = 0.0
        fp16_triton_tflops = 173.2      # above the cuBLAS figure
        bf16_triton_tflops = 0.0

    assert best_measured_peak_tflops(Cal()) == 173.2


def test_no_calibration_object_at_all_reports_zero_rather_than_raising():
    """The no-calibration path must reach `speedup_ceiling` and be refused there, not explode
    on the way. A diagnostic must never end a run."""
    assert best_measured_peak_tflops(None) == 0.0

    class Empty:
        pass

    assert best_measured_peak_tflops(Empty()) == 0.0
    # A calibration carrying junk in one field must not poison the max.
    class Junk:
        fp32_tflops = "not a number"
        bf16_tflops = 234.0

    assert best_measured_peak_tflops(Junk()) == 234.0


def test_the_derivation_carries_every_number_a_reader_needs_to_check_it():
    """A flag whose threshold cannot be checked by hand is what the 10x constant was.

    Three L3:48 runs each flagged a verified-correct winner and each cost a manual
    re-verification, because the report said only "excessive".
    """
    c = speedup_ceiling(**L3_48_A800)
    for fragment in ("1.3506", "1.6858", "0.8011", "30.367", "234.0", "14.000", "17.48"):
        assert fragment in c.derivation, (fragment, c.derivation)


def test_the_bound_is_frozen_so_a_consumer_cannot_edit_the_threshold_it_was_handed():
    """The threshold and its derivation must travel together and unmodified: a caller that could
    raise `threshold_x` while leaving `derivation` describing the old number would produce a
    report justifying a flag that was never applied."""
    import pydantic
    import pytest

    c = speedup_ceiling(**L3_48_A800)
    with pytest.raises(pydantic.ValidationError):
        c.threshold_x = 99.0
