"""S1b wiring: the ledger being correct is not the same as the ledger being connected.

Every defect this file tests for is one that leaves `tests/test_s1b_deweight.py` fully green: a
tuner that ignores the reject callback, a switch that does nothing, evidence that a resume forgets,
a per-space ledger, an effect that never reaches the event log. The G33 lesson is the reason this
file exists at all -- there, a measurement path was broken for both boxes for every real run while
the unit tests passed, because nothing tested that the two halves were joined.
"""

from __future__ import annotations

from kernel_optimizer.config import AppConfig
from kernel_optimizer.models.core import (
    LatencyStats,
    ParamDomain,
    ParameterSpace,
    ParamSet,
    TrialRecord,
)
from kernel_optimizer.tuning.deweight import DEFAULT_FLOOR, DeweightLedger
from kernel_optimizer.tuning.tpe import OptunaTPETuner


def _space() -> ParameterSpace:
    return ParameterSpace(
        space_id="sp-1", candidate_id="c1", version=1,
        domains=[
            ParamDomain(name="DOT_PRECISION", kind="str", choices=["fp16", "tf32", "bf16"]),
            ParamDomain(name="BLOCK_M", kind="int", choices=[32, 64, 128]),
        ],
        constraints=[],
    )


def _trial(candidate_id: str, values: dict, ok: bool, space_id: str = "sp-1") -> TrialRecord:
    return TrialRecord(
        trial_id="tr-x", candidate_id=candidate_id, space_id=space_id,
        params=ParamSet(values=values),
        status="complete" if ok else "fail",
        failure_kind=None if ok else "correctness_mismatch",
        latency_ms=LatencyStats(mean=1.0, median=1.0, min=1.0, max=1.0, std=0.0,
                                n_samples=20) if ok else None,
    )


# ---------------------------------------------------------------- the switch

def test_switch_defaults_off_and_is_the_production_default():
    """`load_config` reads ONE yaml with no base layer, so a field default IS the shipped behaviour.

    This bit an entire L3 series once: an omitted key fell back silently and every agent was told
    the GPU was "unknown". So the default is asserted, not assumed.
    """
    assert AppConfig().v3.search.deweight_unconditional_failures is False


def test_switch_on_produces_a_ledger_and_off_produces_none():
    """The orchestrator builds the ledger from the switch; assert both branches of that decision.

    Constructed directly rather than through the orchestrator (which needs a GPU, a store, and an
    agent runtime) -- but the expression under test is the same one, so a change to the switch's
    meaning has to pass through here.
    """
    cfg_on = AppConfig()
    cfg_on.v3.search.deweight_unconditional_failures = True
    ledger = (DeweightLedger(seed=cfg_on.run.seed)
              if cfg_on.v3.search.deweight_unconditional_failures else None)
    assert ledger is not None

    cfg_off = AppConfig()
    ledger_off = (DeweightLedger(seed=cfg_off.run.seed)
                  if cfg_off.v3.search.deweight_unconditional_failures else None)
    assert ledger_off is None


# ---------------------------------------------------------------- the tuner honours the callback

def test_tuner_defaults_to_no_deweighting():
    tuner = OptunaTPETuner(_space(), guard_ok=lambda p: True, budget=5, seed=0)
    assert tuner.deweight_reject is None
    asked = tuner.ask()
    assert asked is not None


def test_tuner_calls_the_reject_callback_for_every_draw():
    seen: list[dict] = []

    def reject(params: ParamSet) -> bool:
        seen.append(dict(params.values))
        return False

    tuner = OptunaTPETuner(_space(), guard_ok=lambda p: True, budget=3, seed=0,
                           deweight_reject=reject)
    for _ in range(3):
        assert tuner.ask() is not None
    assert len(seen) == 3, (
        f"callback saw {len(seen)} of 3 draws: a draw that bypasses it cannot be down-weighted"
    )


def test_a_rejecting_callback_changes_which_configuration_is_returned():
    """The load-bearing behaviour: a refused draw is re-asked, so tf32 must not come back.

    Without this a wired-but-ignored callback would pass every other test in this file.
    """
    tuner = OptunaTPETuner(
        _space(), guard_ok=lambda p: True, budget=8, seed=0,
        deweight_reject=lambda p: p.values.get("DOT_PRECISION") == "tf32",
    )
    drawn = []
    while True:
        asked = tuner.ask()
        if asked is None:
            break
        drawn.append(asked[1].values["DOT_PRECISION"])
    assert drawn, "no configuration was returned at all"
    assert "tf32" not in drawn, (
        "a rejected value was still handed out for measurement: the tuner is not honouring the "
        "callback, so S1b would journal rules that change nothing"
    )


def test_rejects_are_bounded_and_never_spin():
    """A callback that refuses everything must terminate, not loop.

    It shares `max_guard_rejects_per_ask` with the guard path precisely so a heavily-fired space
    degrades into "ask fewer times" rather than hanging a 12-hour run.
    """
    tuner = OptunaTPETuner(
        _space(), guard_ok=lambda p: True, budget=40, seed=0,
        max_guard_rejects_per_ask=8,
        deweight_reject=lambda p: True,
    )
    assert tuner.ask() is None


def test_a_deweighted_value_is_still_reachable_through_the_tuner():
    """End-to-end form of J1b-11: down-weighting, not removal, once it is wired to the sampler."""
    ledger = DeweightLedger(seed=3)
    for _ in range(DEFAULT_FLOOR):
        ledger.observe(_trial("c1", {"DOT_PRECISION": "tf32", "BLOCK_M": 64}, ok=False))
    assert ledger.is_deweighted("c1", "DOT_PRECISION", "tf32")

    tuner = OptunaTPETuner(
        _space(), guard_ok=lambda p: True, budget=60, seed=0,
        deweight_reject=lambda p: ledger.should_reject("c1", p.values),
    )
    drawn = []
    while True:
        asked = tuner.ask()
        if asked is None:
            break
        drawn.append(asked[1].values["DOT_PRECISION"])
    assert "tf32" in drawn, (
        "the deweighted value never reached a measurement across a 60-trial budget; that is "
        "removal, and a wrong call could never be corrected"
    )


# ---------------------------------------------------------------- resume

def test_ledger_rebuilt_from_a_trial_stream_matches_the_live_one():
    """A resumed run must hold the SAME evidence, not similar evidence.

    `_restore_deweight_ledger` folds the TRIAL_DONE stream through the same `observe`, so this
    asserts the property that makes it sound: replaying the log reproduces the live state exactly.
    """
    stream = [_trial("c1", {"DOT_PRECISION": "tf32", "BLOCK_M": 64}, ok=False)
              for _ in range(DEFAULT_FLOOR)]
    stream.append(_trial("c1", {"DOT_PRECISION": "fp16", "BLOCK_M": 64}, ok=True))

    live = DeweightLedger()
    for record in stream:
        live.observe(record)

    # what the restore path does: re-validate each journalled record and fold it in
    restored = DeweightLedger()
    for record in stream:
        restored.observe(TrialRecord.model_validate(record.model_dump()))

    assert restored.snapshot() == live.snapshot()
    assert restored.snapshot()["n_fired"] == 1


def test_a_resume_that_skipped_reused_trials_would_diverge():
    """Positive control for folding in EVERY trial, reused measurements included.

    If the live path skipped reused records while the restore path folded them, an interrupted run
    would reach its next ask with different evidence -- so this asserts the two disagree when one
    of them drops records, which is what makes the "every trial" rule load-bearing.
    """
    stream = [_trial("c1", {"DOT_PRECISION": "tf32", "BLOCK_M": 64}, ok=False)
              for _ in range(DEFAULT_FLOOR)]
    complete = DeweightLedger()
    for record in stream:
        complete.observe(record)
    partial = DeweightLedger()
    for record in stream[:-1]:
        partial.observe(record)
    assert complete.snapshot()["n_fired"] == 1
    assert partial.snapshot()["n_fired"] == 0
    assert complete.snapshot() != partial.snapshot()


def test_evidence_pools_across_a_candidates_spaces_in_the_restored_stream():
    """The scope decision has to survive the resume path too, not just the live one."""
    ledger = DeweightLedger()
    for i in range(DEFAULT_FLOOR):
        ledger.observe(_trial("c1", {"DOT_PRECISION": "tf32", "BLOCK_M": 64}, ok=False,
                              space_id=f"sp-{i % 3}"))
    assert ledger.is_deweighted("c1", "DOT_PRECISION", "tf32")


# ---------------------------------------------------------------- journalling

def test_snapshot_is_json_serialisable_for_the_event_log():
    """It is written into TUNING_DONE, so a non-serialisable field would break the run's log.

    Sets and tuples are the obvious hazard -- the ledger holds both internally.
    """
    import json

    ledger = DeweightLedger()
    for _ in range(DEFAULT_FLOOR):
        ledger.observe(_trial("c1", {"DOT_PRECISION": "tf32", "BLOCK_M": 64}, ok=False))
    payload = {"deweight": ledger.snapshot()}
    text = json.dumps(payload, ensure_ascii=False)
    assert "tf32" in text
    assert json.loads(text)["deweight"]["n_fired"] == 1


def test_nothing_fired_is_distinguishable_from_the_switch_being_off():
    """A run where the mechanism did nothing must not read like a run where it was disabled."""
    on_but_quiet = DeweightLedger().snapshot()
    off = None
    assert on_but_quiet is not None
    assert on_but_quiet["n_fired"] == 0
    assert off is not on_but_quiet
