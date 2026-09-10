"""S1b acceptance: J1b-1..11, each written so it FAILS on a wrong implementation.

The history here is that two versions of this criterion were argued to be safe and measured to be
dangerous, so these tests assert BEHAVIOUR against the specific wrong variants rather than asserting
that the current code does what it does. Where a test has a positive control, the control is a
variant that really was proposed and really was refused.

Numbers quoted in the assertions come from docs/result-s1b-risk-analysis.md and are reproduced by
scripts/audit_s1b_{risk,scope,worst_case}.py against box 1's five L3 runs.
"""

from __future__ import annotations

import pytest

from kernel_optimizer.models.core import LatencyStats, ParamSet, TrialRecord
from kernel_optimizer.tuning.deweight import (
    DEFAULT_FLOOR,
    DEFAULT_STRENGTH,
    DeweightLedger,
)


def _trial(candidate_id: str, values: dict, ok: bool,
           failure_kind: str | None = "correctness_mismatch") -> TrialRecord:
    return TrialRecord(
        trial_id="tr-x", candidate_id=candidate_id, space_id="sp-1",
        params=ParamSet(values=values),
        status="complete" if ok else "fail",
        failure_kind=None if ok else failure_kind,
        latency_ms=LatencyStats(mean=1.0, median=1.0, min=1.0, max=1.0, std=0.0,
                                n_samples=20) if ok else None,
    )


def _fail_n(ledger: DeweightLedger, n: int, candidate_id: str = "c1",
            values: dict | None = None) -> None:
    for _ in range(n):
        ledger.observe(_trial(candidate_id, values or {"DOT_PRECISION": "tf32"}, ok=False))


# ---------------------------------------------------------------- J1b-1 / J1b-8  the floor

def test_j1b8_floor_is_seven_and_fires_only_at_the_floor():
    """The floor is 7 -- one more than the longest failure prefix ever observed before a pass."""
    assert DEFAULT_FLOOR == 7
    ledger = DeweightLedger()
    for i in range(1, 7):
        _fail_n(ledger, 1)
        assert not ledger.is_deweighted("c1", "DOT_PRECISION", "tf32"), (
            f"fired after {i} failures; the floor is {DEFAULT_FLOOR}. A value that later passes was "
            "measured to accumulate at most 6 failures first, so firing below 7 is the "
            "over-interception this floor exists to avoid"
        )
    _fail_n(ledger, 1)
    assert ledger.is_deweighted("c1", "DOT_PRECISION", "tf32")


def test_j1b1_a_value_that_fails_six_times_then_passes_is_never_deweighted():
    """The positive control for the floor: the worst real case measured, 6 failures then a pass.

    In the corpus exactly one value reached 6 failures before passing. At any floor of 6 or below
    it would have been deweighted while holding real passes -- which is what 78.5% mis-judgement at
    a floor of 1, and 2.3% at 6, are made of.
    """
    ledger = DeweightLedger()
    _fail_n(ledger, 6)
    ledger.observe(_trial("c1", {"DOT_PRECISION": "tf32"}, ok=True))
    assert not ledger.is_deweighted("c1", "DOT_PRECISION", "tf32")
    # and it must stay un-fired however many failures follow
    _fail_n(ledger, 20)
    assert not ledger.is_deweighted("c1", "DOT_PRECISION", "tf32"), (
        "a value with a recorded pass fired after further failures: retraction must be permanent, "
        "otherwise the rule re-fires on a value it has already been proven wrong about"
    )


def test_a_floor_of_six_would_have_mis_killed_that_value():
    """Reverse control: the same history under the refused lower floor DOES mis-kill.

    Without this, `test_j1b1...` would pass on an implementation that never fires at all.
    """
    ledger = DeweightLedger(floor=6)
    _fail_n(ledger, 6)
    assert ledger.is_deweighted("c1", "DOT_PRECISION", "tf32"), (
        "floor=6 did not fire on 6 failures, so this test is not exercising the boundary it claims"
    )


# ---------------------------------------------------------------- J1b-3  retraction

def test_j1b3_retraction_is_permanent_and_recorded():
    ledger = DeweightLedger()
    _fail_n(ledger, DEFAULT_FLOOR)
    assert ledger.is_deweighted("c1", "DOT_PRECISION", "tf32")
    ledger.observe(_trial("c1", {"DOT_PRECISION": "tf32"}, ok=True))
    assert not ledger.is_deweighted("c1", "DOT_PRECISION", "tf32")
    snap = ledger.snapshot()
    assert snap["n_retracted"] == 1
    assert snap["n_fired"] == 0
    assert any("tf32" in entry for entry in snap["retracted"])


def test_j1b3_without_retraction_the_mechanism_mis_kills():
    """Positive control for retraction: 14 of 18 candidate values were alive in SOME space.

    Simulated here as the minimal case -- a value dead long enough to fire, then passing. Under a
    no-retraction rule every later pass is a mis-kill; the ledger must instead read zero.
    """
    ledger = DeweightLedger()
    _fail_n(ledger, DEFAULT_FLOOR)
    mis_kills = 0
    for _ in range(5):
        rec = _trial("c1", {"DOT_PRECISION": "tf32"}, ok=True)
        if ledger.is_deweighted("c1", "DOT_PRECISION", "tf32"):
            mis_kills += 1
        ledger.observe(rec)
    assert mis_kills == 1, (
        "expected exactly the one pass that arrives while the rule is still fired; got "
        f"{mis_kills}, which means retraction did not take effect on the first pass"
    )


# ---------------------------------------------------------------- J1b-7  scope: not cross-candidate

def test_j1b7_evidence_does_not_pool_across_candidates():
    """The principle error that a whole revision turned on.

    Cross-candidate pooling made `DOT_PRECISION=tf32` a rule built from ten candidates' failures,
    but the root cause is the candidate's own uncompensated `dot`. A candidate that compensated
    correctly would be pre-judged by nine others' history. So six failures under c1 plus six under
    c2 must fire for NEITHER, even though twelve pooled failures would clear any floor of 12.
    """
    ledger = DeweightLedger()
    _fail_n(ledger, 6, candidate_id="c1")
    _fail_n(ledger, 6, candidate_id="c2")
    assert not ledger.is_deweighted("c1", "DOT_PRECISION", "tf32")
    assert not ledger.is_deweighted("c2", "DOT_PRECISION", "tf32")
    assert ledger.snapshot()["n_fired"] == 0, (
        "a rule fired on evidence pooled across candidates: 12 failures split 6/6 must fire for "
        "neither, since each candidate has its own source and its own uncompensated-dot question"
    )


def test_j1b7_a_candidates_own_evidence_still_pools_across_its_spaces():
    """The other half: per-candidate must NOT collapse to per-space.

    Per-space pooling was measured to save 63 failing trials against per-candidate's 206, because a
    value appears about 10 times in a 40-trial space and a safe floor cannot fire until its trials
    are nearly spent. So evidence from two different spaces of the SAME candidate must add up.
    """
    ledger = DeweightLedger()
    for i in range(DEFAULT_FLOOR):
        rec = _trial("c1", {"DOT_PRECISION": "tf32"}, ok=False)
        # alternate the space id: same candidate, different spaces
        ledger.observe(rec.model_copy(update={"space_id": f"sp-{i % 2}"}))
    assert ledger.is_deweighted("c1", "DOT_PRECISION", "tf32"), (
        "evidence did not pool across the candidate's spaces, which reduces S1b to the per-space "
        "variant and its measured 63-trial ceiling"
    )


# ---------------------------------------------------------------- J1b-4  key includes the knob

def test_j1b4_the_same_dtype_word_under_two_knobs_is_two_different_keys():
    """`DOT_PRECISION=tf32` is 0-for-164 while `COMPUTE_DTYPE=tf32` passed 209 times.

    So pooling by dtype VALUE rather than by (knob, value) would deweight a value that passes
    constantly. Asserted as behaviour: firing one must not fire the other.
    """
    ledger = DeweightLedger()
    _fail_n(ledger, DEFAULT_FLOOR, values={"DOT_PRECISION": "tf32"})
    assert ledger.is_deweighted("c1", "DOT_PRECISION", "tf32")
    assert not ledger.is_deweighted("c1", "COMPUTE_DTYPE", "tf32"), (
        "firing on DOT_PRECISION=tf32 also fired COMPUTE_DTYPE=tf32; the key must include the knob "
        "because the same precision word measurably behaves oppositely under different knobs"
    )


# ---------------------------------------------------------------- J1b-5  not across runs

def test_j1b5_a_fresh_ledger_holds_no_evidence():
    """Evidence is per run. Cross-run leave-one-out measured a 23.83% mis-kill rate."""
    first = DeweightLedger()
    _fail_n(first, DEFAULT_FLOOR)
    assert first.is_deweighted("c1", "DOT_PRECISION", "tf32")
    second = DeweightLedger()
    assert not second.is_deweighted("c1", "DOT_PRECISION", "tf32")
    assert second.snapshot()["n_fired"] == 0


# ---------------------------------------------------------------- J1b-11  deweight, not removal

def test_j1b11_strength_is_four_and_keeps_a_nonzero_probability():
    """1/4, not 1/8: at 1/8 a fired value's expected further draws fall to a median 0.8, under 1."""
    assert DEFAULT_STRENGTH == 4
    ledger = DeweightLedger()
    _fail_n(ledger, DEFAULT_FLOOR)
    p = ledger.accept_probability("c1", {"DOT_PRECISION": "tf32"})
    assert p == pytest.approx(0.25)
    assert p > 0.0, "removal, not down-weighting: a wrong call could never be corrected"


def test_j1b11_a_deweighted_value_is_still_drawn_sometimes():
    """Behavioural form of the same thing: over many draws, some must be accepted."""
    ledger = DeweightLedger(seed=1234)
    _fail_n(ledger, DEFAULT_FLOOR)
    values = {"DOT_PRECISION": "tf32"}
    accepted = sum(0 if ledger.should_reject("c1", values) else 1 for _ in range(400))
    assert accepted > 0, "a deweighted value was never accepted in 400 draws: that is removal"
    # ~1/4 of draws; the band is wide enough not to be a flaky assertion on the RNG
    assert 60 <= accepted <= 140, (
        f"accepted {accepted}/400, expected about 100 (1/{DEFAULT_STRENGTH}); the suppression "
        "strength does not match the configured one"
    )


def test_an_ordinary_configuration_is_never_rejected():
    ledger = DeweightLedger(seed=7)
    _fail_n(ledger, DEFAULT_FLOOR)
    for _ in range(200):
        assert not ledger.should_reject("c1", {"DOT_PRECISION": "fp16"})


def test_suppression_is_not_compounded_across_two_fired_values():
    """Two fired values in one configuration stay at 1/strength, not 1/strength**2.

    Compounding is a strength the analysis never measured -- the replay treated a fired value as
    binary -- and it suppresses harder, i.e. toward the over-interception this mechanism was
    interrogated for. So the uncompounded form is deliberate and asserted.
    """
    ledger = DeweightLedger()
    _fail_n(ledger, DEFAULT_FLOOR, values={"DOT_PRECISION": "tf32"})
    _fail_n(ledger, DEFAULT_FLOOR, values={"COMPUTE_DTYPE": "bf16"})
    p = ledger.accept_probability("c1", {"DOT_PRECISION": "tf32", "COMPUTE_DTYPE": "bf16"})
    assert p == pytest.approx(1.0 / DEFAULT_STRENGTH)


# ---------------------------------------------------------------- pool boundaries

@pytest.mark.parametrize("kind", ["runtime_error", "infeasible_shared_memory",
                                  "materialize_error", "oom", "timeout"])
def test_only_correctness_mismatch_counts_toward_firing(kind):
    """Other failure kinds say nothing about whether this VALUE is correct.

    `infeasible_shared_memory` is already fully handled at compile time (113 of 113 such trials in
    five runs were caught by the screen before any launch), and a `runtime_error` can be a
    transient or a whole-candidate defect. Counting either would attribute an unrelated failure to
    the value and is exactly how a mechanism like this starts over-intercepting.
    """
    ledger = DeweightLedger()
    for _ in range(DEFAULT_FLOOR * 3):
        ledger.observe(_trial("c1", {"DOT_PRECISION": "tf32"}, ok=False, failure_kind=kind))
    assert not ledger.is_deweighted("c1", "DOT_PRECISION", "tf32")


def test_observe_is_order_independent():
    """A pass anywhere in the history prevents firing, whatever the order.

    This is what makes rebuilding the ledger from the event log on resume give the same state as
    the uninterrupted run, rather than merely a similar one.
    """
    late = DeweightLedger()
    _fail_n(late, 20)
    late.observe(_trial("c1", {"DOT_PRECISION": "tf32"}, ok=True))
    early = DeweightLedger()
    early.observe(_trial("c1", {"DOT_PRECISION": "tf32"}, ok=True))
    _fail_n(early, 20)
    assert not late.is_deweighted("c1", "DOT_PRECISION", "tf32")
    assert not early.is_deweighted("c1", "DOT_PRECISION", "tf32")


# ---------------------------------------------------------------- J1b-9  the denominator

def test_j1b9_snapshot_carries_the_rule_count():
    """0 mis-kills over 4 rules and over 33 rules differ by an order of magnitude in evidence.

    So the rule count is not decoration: it is the denominator any claim about this mechanism has
    to be divided by, and it must be in the journalled record.
    """
    ledger = DeweightLedger()
    _fail_n(ledger, DEFAULT_FLOOR, values={"DOT_PRECISION": "tf32"})
    snap = ledger.snapshot()
    assert snap["n_fired"] == 1
    assert snap["floor"] == DEFAULT_FLOOR
    assert snap["strength"] == DEFAULT_STRENGTH
    assert "fired" in snap and snap["fired"]


def test_snapshot_of_an_untouched_ledger_reads_as_nothing_fired():
    """"Nothing fired" must be distinguishable from "the switch is off"."""
    snap = DeweightLedger().snapshot()
    assert snap["n_fired"] == 0
    assert snap["n_retracted"] == 0
    assert snap["floor"] == DEFAULT_FLOOR


def test_bad_parameters_are_refused():
    with pytest.raises(ValueError):
        DeweightLedger(floor=0)
    with pytest.raises(ValueError):
        DeweightLedger(strength=0)
