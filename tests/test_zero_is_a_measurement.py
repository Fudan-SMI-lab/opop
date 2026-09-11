"""A measured ZERO must not be recorded as "not measured".

FOUND ON DISK, not by reading code. Box 2's `run-l3-43-20260911-052630` (the treatment arm) has 11 of
128 `DIMENSION_STATE` records reading

    {"dimension_id": "n_spills", "measured": null,
     "ceiling_provenance": "spill count not read from the compiler"}

while the same candidate's own best complete trial carries `"n_spills": 0` in its profile. Six
candidates, every one of them measured and clean, described to the agent as unmeasured. The vector
arm's prompt then said, verbatim:

    - **n_spills** - NOT MEASURED on this candidate. Do not read this as headroom.

about a candidate with zero spills -- which is the single most useful thing that dimension can say.

The cause was `if n_spills:` in `bottleneck.py`'s evidence assembly. `dimensions.py` handles zero
correctly (`verdict="slack"` when `float(spills) == 0`), so the loss was entirely in the falsy drop
one layer up. The three candidates whose spills were 2, 12 and 26 came through fine, which is why
this looked like an occasional collector failure rather than a systematic one: it fires on exactly
the healthy candidates.

THE SAME EXPRESSION IS CORRECT ELSEWHERE, which is why this is worth a test file rather than a
one-line edit. `near_limit.append(f"spills={n_spills}")` is also guarded by truthiness and must stay
that way -- zero spills is not NEAR the spill limit, it is at it from the good side. Recording a
measurement must keep a zero; flagging a problem must drop it. Both are asserted below so a later
"consistency" pass cannot unify them.
"""

from __future__ import annotations

from kernel_optimizer.evaluation.bottleneck import DevicePeaks, classify


def _kw(**over):
    """The keyword set `classify` really takes -- signature copied from the function, not guessed.

    Ceilings arrive as a `DevicePeaks`, not as loose `dram_peak_tbs`/`compute_peak_tflops` floats; a
    first version invented those two names and every test raised TypeError, which is the loud
    version of the failure. Peaks are the box-1 measured figures so the numbers are real.
    """
    # A shape that reaches `resource_limited`, deliberately. `classify` RETURNS EARLY on
    # `compute_bound`/`memory_bound`, before the `near_limit` assembly two of these tests are
    # about -- with L3:43's real task cost (412 GFLOP over 3 ms = 137 TFLOP/s against a 54.7
    # ceiling) it short-circuits and `at_limit` is None, so the positive control below failed on
    # correct code. That was the fixture's shape, not a defect: the `ev[...]` assignment under test
    # runs on every path, the `near_limit` list only on this one.
    base = dict(
        gpu_ms=3.0,
        cpu_issue_ms=None,
        byte_count=13_505_658,
        flop_count=40_000_000_000,
        peaks=DevicePeaks(dram_tbs=0.9094, fp32_tflops=54.7),
        n_regs=155,
        n_spills=0,
        shared_bytes=17408,
        max_regs_per_thread=255,
        max_shared_bytes=101376,
        occupancy={"occupancy": 0.208, "limiter": "shared_memory"},
    )
    base.update(over)
    return base


def test_zero_spills_is_recorded_as_a_measurement_not_as_unmeasured():
    """The defect. Six candidates on box 2 were told their spills were unmeasured while their own
    best trial had measured them at 0."""
    v = classify(**_kw(n_spills=0))

    assert "n_spills" in v.evidence, (
        "a measured zero must appear in the evidence, or the dimension record downstream says "
        "'spill count not read from the compiler' about a candidate that WAS read: %r"
        % sorted(v.evidence))
    assert v.evidence["n_spills"] == 0
    assert "n_spills" not in (v.unmeasured or []), (
        "and it must not be listed as unmeasured: %r" % (v.unmeasured,))


def test_a_nonzero_spill_count_still_comes_through():
    """The direction that already worked -- kept so the fix is not a swap."""
    v = classify(**_kw(n_spills=12))
    assert v.evidence["n_spills"] == 12
    assert "n_spills" not in (v.unmeasured or [])


def test_a_genuinely_absent_spill_count_is_still_unmeasured():
    """The distinction the fix must preserve: `None` means the compiler was never asked, and that is
    a real reader gap the prompt should keep reporting. Collapsing zero and None in the OTHER
    direction would be just as wrong -- it would claim a clean kernel where nothing was measured."""
    v = classify(**_kw(n_spills=None))
    assert "n_spills" not in v.evidence, (
        "an absent count must not be invented as 0: %r" % v.evidence.get("n_spills"))


def test_zero_spills_does_NOT_appear_in_the_near_limit_list():
    """The same expression, correct as truthiness. `near_limit` names what is at or near a limit;
    zero spills is not near the spill limit. A later pass that "unified" the two reads would put
    `spills=0` in the at_limit list of every clean candidate."""
    v = classify(**_kw(n_spills=0))
    at = " ".join(v.evidence.get("at_limit") or [])
    assert "spills=0" not in at, (
        "zero spills must not be reported as a resource at its limit: %r"
        % v.evidence.get("at_limit"))
    # Not vacuous: this fixture DOES reach the branch, and `at_limit` is non-empty for another
    # reason (occupancy), so an empty list would not silently satisfy the assertion.
    assert v.evidence.get("at_limit"), (
        "the fixture must reach the resource branch or this asserts nothing: kind=%s" % v.kind)


def test_a_real_spill_count_DOES_appear_in_the_near_limit_list():
    """And the positive control for the assertion above, so it cannot pass by `at_limit` being empty
    for an unrelated reason -- the recorded `probe-needs-a-positive-control` failure."""
    v = classify(**_kw(n_spills=12))
    at = " ".join(v.evidence.get("at_limit") or [])
    assert "spills=12" in at, (
        "a real spill count must still be flagged: %r" % v.evidence.get("at_limit"))


def test_zero_registers_is_also_recorded_rather_than_dropped():
    """`n_regs` carried the same falsy drop. A zero-register kernel is not a real thing, so this
    costs nothing today -- but the bug is the truthiness, and it is fixed once rather than waiting
    for a dimension where zero is common."""
    v = classify(**_kw(n_regs=0))
    assert v.evidence.get("n_regs") == 0, (
        "a measured zero register count must be recorded: %r" % sorted(v.evidence))
