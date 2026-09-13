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
                walls: list[dict] | None = None) -> dict:
    return {"seq": 900, "ts": 9000.0, "type": "RESOURCE_WALL_ATTRIBUTED", "payload": {
        "candidate_id": "cand-a", "space_id": "sp-a", "n_refused_configs": n_refused,
        "walls_found": walls_found, "walls_probed": 0, "walls_worthless": worthless,
        "walls_skipped_by_cap": 0, "walls": walls or []}}


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
    that always returns a cause. When a wall IS found and IS worth acting on, there is nothing to
    explain and `no_wall_cause` must be None.
    """
    a = _arm(tmp_path, "real", [
        _trial(1001.0), _trial(1002.0, refused=True),
        _wall_event(1, 0, 1, walls=[{"param": "NUM_WARPS", "refused_value": 16.0,
                                     "monotone": True, "tail_gain_pct": 21.9}]),
        _soft_event(False, "the best trial does not spill")])
    assert a["walls_found_total"] == 1
    assert a["walls_worthless_total"] == 0
    assert a["no_wall_cause"] is None, "a found, worthy wall needs no excuse"


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
