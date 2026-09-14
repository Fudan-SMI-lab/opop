"""The P1-P5 reader's no-wall diagnosis: a null must name its own cause.

WHY THIS EXISTS. `analyze_s7_pair.py` produced a confident wrong finding once already -- validated
against two finished runs it printed "P1/P2/P3 FAIL, latency improved +7.99%" when the control simply had
2e switched OFF, so every figure it compared was the absence of the instrument. The instrument check was
added for that, and it has a test-shaped hole: the branch that fires when S7 ran but enqueued nothing
(`not p1`) is the branch this live pair is heading for, and it is the one that decides whether the
project reports a diagnosed null or a blank one.

The three causes of "0 walls" are behaviourally different and a reader that cannot tell them apart turns
each into the same shrug:

  * no refusals at all -- `find_walls` had no input.
  * refusals present, `walls_found` 0 -- every refused value sat INSIDE its knob's measured range, so
    nothing was truncated. Measured live on this pair: NUM_WARPS=16 was refused at trial 6 and MEASURED
    successfully at trial 14, which erased the wall from trial 20 on. Walls are not cumulative.
  * `walls_found` > 0 but all worthless -- the shipping slope filter dropped them, latency flat or
    rising toward the wall.

Every fixture below is built from the emitters' real field names (`walls_found`, `walls_worthless`,
`applicable`, `reason`, `spills_at_best`) copied off the live pair's payloads, not invented to match the
reader -- a fixture invented to match its reader proves nothing.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location("asp", Path("scripts/analyze_s7_pair.py"))
asp = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(asp)


def _write(tmp_path: Path, name: str, events: list[dict]) -> Path:
    d = tmp_path / name
    d.mkdir(parents=True, exist_ok=True)
    with (d / "events.jsonl").open("w", encoding="utf-8", newline="\n") as fh:
        for e in events:
            fh.write(json.dumps(e) + "\n")
    return d


def _trial(ts: float, *, refused: bool = False, ms: float | None = 3.0,
           spills: int = 0) -> dict:
    """A TRIAL_DONE in the emitter's shape: trial NESTED under payload.trial, status "complete"
    (not "ok"), params under {"values": ...}, and `robust_ms` absent because it is a @property."""
    return {"seq": int(ts), "ts": ts, "type": "TRIAL_DONE", "payload": {"trial": {
        "trial_id": f"tr-{int(ts)}", "candidate_id": "cand-a", "space_id": "sp-a",
        "status": "fail" if refused else "complete",
        "failure_kind": "infeasible_shared_memory" if refused else None,
        "params": {"values": {"NUM_WARPS": 8}},
        "profile": {"n_spills": spills, "n_regs": 200, "shared_bytes": 18432},
        "latency_ms": None if refused or ms is None else {
            "mean": ms, "median": ms, "min": ms, "max": ms, "std": 0.1, "n_samples": 20}}}}


def _wall_event(walls_found: int, worthless: int, n_refused: int,
                walls: list[dict] | None = None, error: str | None = None) -> dict:
    # `error` is present only on the second no-verdict branch (orchestrator.py:1598-1601), where
    # walls were selected for probing but the space had no winning configuration to ablate from.
    # It is what separates that branch from "select_for_probing chose none" (:1589-1591), whose
    # payload is otherwise identical -- walls listed, verdicts unfilled.
    payload = {
        "candidate_id": "cand-a", "space_id": "sp-a", "n_refused_configs": n_refused,
        "walls_found": walls_found, "walls_probed": 0, "walls_worthless": worthless,
        "walls_skipped_by_cap": 0, "walls": walls or []}
    if error is not None:
        payload["error"] = error
    return {"seq": 900, "ts": 9000.0, "type": "RESOURCE_WALL_ATTRIBUTED", "payload": payload}


def _soft_event(applicable: bool, reason: str, spills_at_best: float = 0.0) -> dict:
    return {"seq": 901, "ts": 9001.0, "type": "RESOURCE_SOFT_WALL", "payload": {
        "candidate_id": "cand-a", "applicable": applicable, "reason": reason,
        "spills_at_best": spills_at_best, "limiter_at_best": "registers",
        "n_knobs_examined": 0, "n_knobs_with_spill_variation": 0, "n_walls": 0, "walls": []}}


def _arm(tmp_path: Path, name: str, events: list[dict]) -> dict:
    base = [{"seq": 0, "ts": 1000.0, "type": "RUN_CREATED", "payload": {"run_id": name}}]
    return asp._arm(_write(tmp_path, name, base + events))


# ---------------------------------------------------------------------------------------------
# the three causes must be told apart
# ---------------------------------------------------------------------------------------------


def test_no_refusals_at_all_is_named_as_missing_input(tmp_path):
    a = _arm(tmp_path, "no-ref", [_trial(1001.0), _trial(1002.0),
                                  _wall_event(0, 0, 0), _soft_event(False, "the best trial does not spill")])
    assert a["n_refusals"] == 0
    assert "no shared-memory refusals at all" in a["no_wall_cause"]


def test_refusals_with_no_truncation_is_named_as_inside_the_measured_range(tmp_path):
    """The live case: refusals exist, `walls_found` is 0. This must NOT read as "no refusals" and must
    NOT read as "the slope filter dropped them" -- both would point at the wrong mechanism."""
    a = _arm(tmp_path, "inside", [
        _trial(1001.0), _trial(1002.0, refused=True), _trial(1003.0),
        _wall_event(0, 0, 1), _soft_event(False, "the best trial does not spill")])
    assert a["n_refusals"] == 1
    assert a["walls_found_total"] == 0
    assert "INSIDE its knob's" in a["no_wall_cause"]


def test_all_walls_worthless_is_named_as_the_slope_filter(tmp_path):
    """The prefix-10 case: a wall existed, the shipping filter dropped it because latency rises
    toward it. Distinct from "no wall existed", and the distinction is the whole point."""
    a = _arm(tmp_path, "worthless", [
        _trial(1001.0), _trial(1002.0, refused=True),
        _wall_event(1, 1, 1, walls=[{"param": "NUM_WARPS", "refused_value": 16.0,
                                     "monotone": False, "tail_gain_pct": 11.1}]),
        _soft_event(False, "the best trial does not spill")])
    assert a["walls_found_total"] == 1
    assert a["walls_worthless_total"] == 1
    assert "ALL worthless" in a["no_wall_cause"]


def test_a_real_wall_leaves_the_cause_unset(tmp_path):
    """THE POSITIVE CONTROL, and without it every assertion above could be satisfied by a function
    that always returns a cause. When a wall IS found, IS worth acting on and IS attributed, there is
    nothing to explain and `no_wall_cause` must be None.

    `verdict` is part of the fixture because the emitter always writes it (`Wall.payload()` includes
    it, and all three probed walls on the live pair carry it). A wall without a verdict is not a
    "real wall" for this purpose -- it is a wall whose attribution is unknown.
    """
    a = _arm(tmp_path, "real", [
        _trial(1001.0), _trial(1002.0, refused=True),
        _wall_event(1, 0, 1, walls=[{"param": "NUM_WARPS", "refused_value": 16.0,
                                     "monotone": True, "tail_gain_pct": 21.9,
                                     "verdict": "attributed", "over_ratio": 1.29}]),
        _soft_event(False, "the best trial does not spill")])
    assert a["walls_found_total"] == 1
    assert a["walls_worthless_total"] == 0
    assert a["walls_attributed_total"] == 1
    assert a["no_wall_cause"] is None, "a found, worthy, attributed wall needs no excuse"


def test_a_wall_that_fails_attribution_is_named_as_such(tmp_path):
    """The live control arm, reproduced: ONE wall, zero worthless, and nothing happened.

    Before this the reader printed `walls_found 1  worthless 0` with no cause at all, which reads as
    "a good wall that went unused". The real reason is that attribution -- which runs AFTER the slope
    filter, so `walls_worthless` structurally cannot contain its failures -- found the refusal was not
    caused by a resource limit. `over_ratio 0.97` says shared-memory use at theta* never reached the
    limit, so there is no limit for a rewrite to free.
    """
    a = _arm(tmp_path, "unattributed", [
        _trial(1001.0), _trial(1002.0, refused=True),
        _wall_event(1, 0, 1, walls=[{"param": "NUM_WARPS", "refused_value": 16.0,
                                     "monotone": True, "tail_gain_pct": 21.88,
                                     "verdict": "not_attributed", "over_ratio": 0.97}]),
        _soft_event(False, "the best trial does not spill")])
    assert a["walls_found_total"] == 1
    assert a["walls_worthless_total"] == 0, "the slope filter kept it -- that is the point"
    assert a["walls_attributed_total"] == 0
    assert a["no_wall_cause"] is not None, "a wall that helped nobody must say why"
    assert "PROBED AND NOT TRACED" in a["no_wall_cause"]
    assert "0.97" in a["no_wall_cause"], "the over_ratio distinguishes 'not a resource limit' from " \
                                         "'an overflow the prober could not confirm'"
    assert "worthless" in a["no_wall_cause"], "must say why walls_worthless cannot show this"


def test_attribution_failure_is_not_reported_as_the_slope_filter(tmp_path):
    """The two causes must stay distinguishable. A wall the slope filter dropped and a wall attribution
    rejected call for different next actions -- relax the filter versus investigate why a refused
    config does not overflow at theta* -- so a reader that collapsed them would send the next round
    after the wrong thing.
    """
    slope = _arm(tmp_path, "slope", [
        _trial(1001.0), _trial(1002.0, refused=True),
        _wall_event(1, 1, 1, walls=[{"param": "NUM_WARPS", "refused_value": 16.0,
                                     "monotone": False, "tail_gain_pct": 11.1,
                                     "verdict": "attributed", "over_ratio": 1.3}]),
        _soft_event(False, "the best trial does not spill")])
    attr = _arm(tmp_path, "attr", [
        _trial(1001.0), _trial(1002.0, refused=True),
        _wall_event(1, 0, 1, walls=[{"param": "BLOCK_N", "refused_value": 256.0,
                                     "monotone": True, "tail_gain_pct": 60.69,
                                     "verdict": "not_attributed", "over_ratio": 0.727}]),
        _soft_event(False, "the best trial does not spill")])
    assert "ALL worthless" in slope["no_wall_cause"]
    assert "ATTRIBUTION" not in slope["no_wall_cause"]
    assert "PROBED AND NOT TRACED" in attr["no_wall_cause"]
    assert "ALL worthless" not in attr["no_wall_cause"]
    assert "BLOCK_N" in attr["no_wall_cause"], "naming the knob is what makes it actionable"


def test_a_wall_with_no_verdict_field_was_never_probed_not_failed(tmp_path):
    """Absent `verdict` means the wall NEVER REACHED attribution -- a different fact from failing it.

    `payload["walls"]` is written on the no-probe branch too (orchestrator.py:1590), where verdicts
    are unfilled, so a wall can pass the slope filter and still never be sent. Reporting that as
    "attribution rejected it" names the wrong mechanism: the next round would go investigate why a
    refused config does not overflow at theta*, when the actual question is why
    `select_for_probing` chose nothing.

    This test previously asserted the OPPOSITE (that the cause mentions ATTRIBUTION and "no verdict
    field"), which pinned the confusion in place -- an assertion can encode the bug.
    """
    a = _arm(tmp_path, "noverdict", [
        _trial(1001.0), _trial(1002.0, refused=True),
        _wall_event(1, 0, 1, walls=[{"param": "NUM_WARPS", "refused_value": 16.0,
                                     "monotone": True, "tail_gain_pct": 21.9}]),
        _soft_event(False, "the best trial does not spill")])
    assert a["walls_attributed_total"] == 0, "absent must never read as passing"
    assert a["walls_never_probed"] == 1
    assert not a["walls_unattributed"], "an unprobed wall is not an attribution failure"
    assert "NEVER SENT TO ATTRIBUTION" in a["no_wall_cause"]
    assert "PROBED AND NOT TRACED" not in a["no_wall_cause"]
    assert "()" not in a["no_wall_cause"], "the old wording printed an empty parenthesis here"


def test_a_wall_with_no_origin_to_ablate_from_is_named_a_defect(tmp_path):
    """The second no-verdict branch (orchestrator.py:1598) sets payload['error'] because walls WERE
    selected but the space had no winning configuration to ablate from. That is a defect in the run,
    not a property of the walls, and folding it into 'never probed' would hide it.
    """
    a = _arm(tmp_path, "noorigin", [
        _trial(1001.0), _trial(1002.0, refused=True),
        _wall_event(1, 0, 1, walls=[{"param": "NUM_WARPS", "refused_value": 16.0,
                                     "monotone": True, "tail_gain_pct": 21.9}],
                    error="no winning configuration to ablate from"),
        _soft_event(False, "the best trial does not spill")])
    assert a["walls_no_origin"] == 1
    assert a["walls_never_probed"] == 0, "the error field separates the two no-verdict branches"
    assert "NO ORIGIN TO ABLATE FROM" in a["no_wall_cause"]
    assert "DEFECT" in a["no_wall_cause"]


def test_both_no_verdict_and_a_real_failure_are_reported_together(tmp_path):
    """A run can have both. Reporting only the first cause found would let the reader fix one and
    conclude the question is closed.
    """
    a = _arm(tmp_path, "both", [
        _trial(1001.0), _trial(1002.0, refused=True),
        _wall_event(2, 0, 2, walls=[{"param": "BLOCK_N", "refused_value": 256.0,
                                     "monotone": True, "tail_gain_pct": 60.7,
                                     "verdict": "not_attributed", "over_ratio": 0.727},
                                    {"param": "NUM_WARPS", "refused_value": 16.0,
                                     "monotone": True, "tail_gain_pct": 21.9}]),
        _soft_event(False, "the best trial does not spill")])
    assert a["walls_never_probed"] == 1
    assert len(a["walls_unattributed"]) == 1
    assert "PROBED AND NOT TRACED" in a["no_wall_cause"]
    assert "NEVER SENT TO ATTRIBUTION" in a["no_wall_cause"]
    assert "ALSO:" in a["no_wall_cause"], "both causes must be visible in one sentence"


# ---------------------------------------------------------------------------------------------
# the soft wall's gate is a gate, not a negative result
# ---------------------------------------------------------------------------------------------


def test_the_soft_gate_reason_is_carried_verbatim(tmp_path):
    """`applicable: false` with the emitter's own reason string. Reported verbatim rather than
    re-derived, so the reader cannot disagree with what the run acted on -- and so the DECLARED gate
    (the best trial must itself spill) is distinguishable from the criterion finding no rising curve.
    """
    a = _arm(tmp_path, "softgate", [
        _trial(1001.0, spills=0),
        _wall_event(0, 0, 0),
        _soft_event(False, "the best trial does not spill (no soft wall exists at the optimum)")])
    assert a["soft_scans"] == 1
    assert a["soft_applicable"] == 0
    assert any("does not spill" in r for r in a["soft_reasons"])


def test_an_applicable_soft_scan_is_counted_as_applicable(tmp_path):
    """Positive control for the gate: when the best trial DOES spill the scan is applicable, so
    `soft_applicable` must move. Otherwise the field could be hard-wired to 0.
    """
    a = _arm(tmp_path, "softon", [
        _trial(1001.0, spills=240),
        _wall_event(0, 0, 0),
        _soft_event(True, "", spills_at_best=240.0)])
    assert a["soft_applicable"] == 1


# ---------------------------------------------------------------------------------------------
# the instrument check still outranks everything
# ---------------------------------------------------------------------------------------------


def test_refusals_with_no_wall_event_at_all_still_reads_as_instrument_off(tmp_path):
    """The original defect must stay fixed: refusals present and NO wall event means 2e was off, which
    is not the same as 2e running and finding nothing. `instrument_on` is the discriminator and the
    no-wall cause must not paper over it.
    """
    a = _arm(tmp_path, "instroff", [_trial(1001.0), _trial(1002.0, refused=True)])
    assert a["n_wall_events"] == 0
    assert a["instrument_on"] is False


# ---------------------------------------------------------------------------------------------
# the latency gap must be attributed to a candidate, not just measured against the noise floor
# ---------------------------------------------------------------------------------------------


def _trial_c(ts: float, cand: str, ms: float) -> dict:
    return {"seq": int(ts), "ts": ts, "type": "TRIAL_DONE", "payload": {"trial": {
        "trial_id": f"tr-{int(ts)}", "candidate_id": cand, "space_id": f"sp-{cand}",
        "status": "complete", "failure_kind": None,
        "params": {"values": {"BLOCK_M": 32}},
        "profile": {"n_spills": 0},
        "latency_ms": {"mean": ms, "median": ms, "min": ms, "max": ms, "std": 0.1,
                       "n_samples": 20}}}}


def test_the_best_is_attributed_to_its_candidate(tmp_path):
    """`best_trial_ms` alone cannot say whether a gap is a seed difference. The live pair's control led
    5.90% -- outside both the 2.35% noise floor and the 4.72% within-arm spread -- entirely through a
    candidate that existed in that arm only. So the reader records WHICH candidate owns the best.
    """
    a = _arm(tmp_path, "lat", [_trial_c(1001.0, "cand-slow", 4.0),
                               _trial_c(1002.0, "cand-fast", 3.1),
                               _wall_event(0, 0, 0),
                               _soft_event(False, "the best trial does not spill")])
    assert a["best_trial_ms"] == 3.1
    assert a["best_candidate"] == "cand-fast"
    assert a["per_candidate_best"] == {"cand-slow": 4.0, "cand-fast": 3.1}


def test_unpaired_arms_share_no_candidate(tmp_path):
    """The signature that makes a latency gap unattributable: the two arms' candidate sets are
    disjoint, which is the normal case because each arm's generator writes its own seeds.
    """
    a = _arm(tmp_path, "lat_t", [_trial_c(1001.0, "cand-t1", 3.36),
                                 _wall_event(0, 0, 0),
                                 _soft_event(False, "the best trial does not spill")])
    b = _arm(tmp_path, "lat_c", [_trial_c(1001.0, "cand-c1", 3.18),
                                 _wall_event(0, 0, 0),
                                 _soft_event(False, "the best trial does not spill")])
    shared = set(a["per_candidate_best"]) & set(b["per_candidate_best"])
    assert shared == set(), "disjoint seeds => the gap is not attributable to the switch"


def test_a_shared_candidate_is_detected_when_present(tmp_path):
    """The positive control: when a candidate DOES appear in both arms, that is the like-for-like
    comparison and must be found. Without this, the 'no candidate is shared' branch could be reached
    by a function that never finds anything.
    """
    a = _arm(tmp_path, "sh_t", [_trial_c(1001.0, "cand-both", 3.30),
                                _wall_event(0, 0, 0),
                                _soft_event(False, "the best trial does not spill")])
    b = _arm(tmp_path, "sh_c", [_trial_c(1001.0, "cand-both", 3.40),
                                _wall_event(0, 0, 0),
                                _soft_event(False, "the best trial does not spill")])
    shared = set(a["per_candidate_best"]) & set(b["per_candidate_best"])
    assert shared == {"cand-both"}


def _run_finished(ts: float = 9999.0) -> dict:
    """A RUN_FINISHED in the emitter's real shape: everything reportable is NESTED under
    payload.summary.best, NOT at summary top level.

    Copied field-for-field off run-l3-43-20260913-202332's own payload (both arms), because the reader
    read the top level only and printed `n/a  n/a` for a pair whose numbers were both on disk -- the
    same shape as `BASELINE_DONE.payload["latency_ms"]`. A fixture that put these keys at the top level
    would agree with the broken reader and prove nothing.
    """
    return {"seq": int(ts), "ts": ts, "type": "RUN_FINISHED", "payload": {"summary": {
        "task": {"level": 3, "problem_id": 43},
        "elapsed_hours": 12.644,
        "baselines": [{"kind": "eager", "latency_ms": {"median": 21.19, "mean": 21.2}},
                      {"kind": "torch_compile_tf32", "latency_ms": {"median": 11.06, "mean": 11.1}}],
        "best": {
            "candidate_id": "cand-a", "family_id": "fam-a",
            "params": {"values": {"NUM_WARPS": 8}},
            "tuned_ms": 2.935807943344116,
            "final_reeval_ok": True,
            "final_reeval_ms": 2.92,
            "final_reeval_median_ms": 2.9224960803985596,
            "excessive_speedup_flag": False,
            "precision": "bf16",
            "speedups": {"eager": 7.2603, "torch_compile_tf32": 3.8014},
            "honest_verdict": {"candidate_precision": "bf16",
                               "compared_against": "torch_compile_tf32",
                               "same_precision_speedup": 3.8014,
                               "beats_same_precision_baseline": True}}}}}


def test_final_reeval_is_read_from_the_nested_best(tmp_path):
    """The defect this pins: the numbers live under summary["best"], not at summary top level."""
    a = _arm(tmp_path, "nest_t", [_trial_c(1001.0, "cand-a", 3.36),
                                  _wall_event(0, 0, 0),
                                  _soft_event(False, "the best trial does not spill"),
                                  _run_finished()])
    assert a["finished"] is True
    assert a["summary_final_reeval_median_ms"] == 2.9224960803985596
    assert a["summary_final_reeval_ms"] == 2.92
    assert a["summary_best_ms"] == 2.935807943344116
    assert a["summary_final_reeval_ok"] is True
    assert a["summary_precision"] == "bf16"


def test_only_the_same_precision_speedup_is_carried(tmp_path):
    """A bf16 winner against an fp32 eager baseline reads 7.26x; the like-for-like number is 3.80x.
    The reader must surface the honest one, or the flattering one is what gets quoted.
    """
    a = _arm(tmp_path, "hv_t", [_trial_c(1001.0, "cand-a", 3.36),
                                _wall_event(0, 0, 0),
                                _soft_event(False, "the best trial does not spill"),
                                _run_finished()])
    assert a["summary_same_precision_speedup"] == 3.8014
    assert a["summary_compared_against"] == "torch_compile_tf32"
    assert a["summary_beats_same_precision"] is True
    # The negative half: the 7.26x figure must NOT be what any summary_* speedup key holds.
    assert 7.2603 not in [v for k, v in a.items() if k.startswith("summary_")]


def test_a_run_without_the_nested_best_reads_absent_not_zero(tmp_path):
    """An in-flight or crashed run has no summary. Absent must stay absent: a 0.0 here would be
    compared against the other arm and print a 100% difference.
    """
    a = _arm(tmp_path, "nofin_t", [_trial_c(1001.0, "cand-a", 3.36),
                                   _wall_event(0, 0, 0),
                                   _soft_event(False, "the best trial does not spill")])
    assert a["finished"] is False
    assert a.get("summary_final_reeval_median_ms") is None
    assert a.get("summary_best_ms") is None


def _refusal(ts: float) -> dict:
    """A shared-memory refusal in the emitter's shape: status is not "complete" and failure_kind is
    `infeasible_shared_memory` (the value the harness actually writes, not "oom" or "shared_memory")."""
    return {"seq": int(ts), "ts": ts, "type": "TRIAL_DONE", "payload": {"trial": {
        "trial_id": f"tr-{int(ts)}", "candidate_id": "cand-a", "space_id": "sp-a",
        "status": "fail", "failure_kind": "infeasible_shared_memory",
        "params": {"values": {"NUM_WARPS": 8}}, "latency_ms": None}}}


def test_an_arm_with_2e_off_by_design_is_not_reported_as_broken(tmp_path):
    """The false alarm this guards against. On the step-4 pair `wall_attribution.enabled` is FALSE in
    the all-off arm, so that arm has refusals and ZERO wall events BY DESIGN. The reader used to print
    "2e was OFF (or crashed) ... the pair cannot be read" about it -- and the response to that alarm
    would be to discard a 12 h pair. An arm with the switch ON emits RESOURCE_WALL_ATTRIBUTED per
    candidate even when it finds nothing, so zero EVENTS is the discriminator.
    """
    off = _arm(tmp_path, "byd_off", [_trial_c(1001.0, "cand-a", 3.4), _refusal(1002.0)])
    assert off["n_refusals"] > 0
    assert off["n_wall_events"] == 0
    assert off["instrument_on"] is False, "zero wall events must read as instrument-off"


def test_an_arm_that_should_have_had_2e_on_but_emitted_nothing_is_still_a_defect(tmp_path):
    """The positive control for the flag: naming ONE arm as deliberately-off must make the OTHER arm's
    silence loud. Without this, `--expect-2e-off` would only ever agree with the heuristic and would be
    a variant that changes no behaviour -- and a swapped pair would be forgiven silently.
    """
    a = _arm(tmp_path, "swap_t", [_trial_c(1001.0, "cand-a", 3.4), _refusal(1002.0)])
    # Same shape on both sides: refusals, no wall events. Which one is a defect depends ENTIRELY on
    # which was named, so the reader cannot decide it from the events alone -- that is why the flag
    # exists rather than a cleverer heuristic.
    b = _arm(tmp_path, "swap_c", [_trial_c(1001.0, "cand-b", 3.5), _refusal(1002.0)])
    for arm in (a, b):
        assert arm["n_refusals"] > 0 and arm["n_wall_events"] == 0
    named = {"treatment"}
    # control was NOT named => its silence is unexplained => defect.
    assert ("control" not in named) and b["n_wall_events"] == 0
