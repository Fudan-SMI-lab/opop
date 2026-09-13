"""The arm-parity checker's budget question, and the in-flight case that made it lie.

WHY THIS EXISTS. `check_arm_search_parity.py` had no tests, and it produced a confident wrong verdict on
the live S7 pair: "PER-SPACE BUDGET DIFFERS: 15 vs 7 trials per space -- PARITY NOT OK", on two arms
both configured `trials_per_space: 40` that had closed no space at all. The number was real (one arm had
15 trials in its open space, the other 7) and the conclusion drawn from it was not.

The mechanism of the error is worth stating because it is not a typo. `_mode` was chosen deliberately
over `max` and `min`, and its docstring justified it as stable mid-run -- which is true ONCE SEVERAL
SPACES HAVE CLOSED, because the mode then belongs to the closed ones and the single in-flight space is
outvoted. Early in a run each arm has exactly one space, still open, so the mode IS the in-flight count
and the reasoning inverts. A guard that only checked "the mode is not the max or the min" would have
passed.

So the tests below pin the distinction the fix rests on: a trial count is "the budget" only after that
space's TUNING_DONE, and before that it is progress. An unanswerable question must read as unanswerable
rather than as agreement (which would hide a real confound) or as a difference (which invents one).
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "casp", Path("scripts/check_arm_search_parity.py"))
casp = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(casp)


def _write(tmp_path: Path, name: str, events: list[dict]) -> Path:
    d = tmp_path / name
    d.mkdir(parents=True, exist_ok=True)
    with (d / "events.jsonl").open("w", encoding="utf-8", newline="\n") as fh:
        for e in events:
            fh.write(json.dumps(e) + "\n")
    return d


def _trial(ts: float, space: str, ok: bool = True) -> dict:
    """A TRIAL_DONE shaped like the emitter's: the trial is NESTED under `payload.trial`, and its
    status is "complete" rather than "ok" -- both spellings this project has got wrong before."""
    return {"seq": int(ts), "ts": ts, "type": "TRIAL_DONE", "payload": {"trial": {
        "trial_id": f"tr-{ts}", "candidate_id": "cand-x", "space_id": space,
        "status": "complete" if ok else "fail",
        "failure_kind": None if ok else "infeasible_shared_memory",
        "params": {"values": {"BLOCK_M": 32}},
        "latency_ms": {"mean": 2.0, "median": 2.0, "min": 2.0, "max": 2.0, "n_samples": 20,
                       "std": 0.1} if ok else None}}}


def _arm(tmp_path: Path, name: str, spaces: dict[str, int], *, closed: tuple[str, ...] = (),
         finished: bool = False) -> dict:
    """One arm: `spaces` maps space_id -> trial count; `closed` lists the spaces that emitted
    TUNING_DONE; `finished` adds RUN_FINISHED."""
    ev: list[dict] = [{"seq": 0, "ts": 1000.0, "type": "RUN_CREATED", "payload": {"run_id": name}}]
    ts = 1001.0
    for sid, n in spaces.items():
        for _ in range(n):
            ev.append(_trial(ts, sid))
            ts += 30.0
        if sid in closed:
            ev.append({"seq": int(ts), "ts": ts, "type": "TUNING_DONE",
                       "payload": {"candidate_id": "cand-x", "space_id": sid, "best_ms": 2.0}})
            ts += 1.0
    if finished:
        ev.append({"seq": int(ts), "ts": ts, "type": "RUN_FINISHED", "payload": {"summary": {}}})
    return casp.read_arm(_write(tmp_path, name, ev))


# ---------------------------------------------------------------------------------------------
# the defect: an in-flight pair must not be reported as a budget difference
# ---------------------------------------------------------------------------------------------


def test_an_in_flight_pair_does_not_report_a_budget_difference(tmp_path):
    """The live S7 pair, reproduced: one space each, still open, 15 trials against 7.

    This is the exact input that produced "PER-SPACE BUDGET DIFFERS: 15 vs 7 ... PARITY NOT OK" while
    both arms were configured for 40 and neither had finished a space.
    """
    a = _arm(tmp_path, "control", {"sp-c": 15})
    b = _arm(tmp_path, "treatment", {"sp-t": 7})
    okay, notes = casp.parity_verdict(a, b, "CONTROL", "TREATMENT")
    joined = " | ".join(notes)
    assert "PER-SPACE BUDGET DIFFERS" not in joined, \
        "an open space's trial count is progress, not the budget"
    assert "NOT YET ANSWERABLE" in joined
    assert "IN FLIGHT" in joined, "a provisional verdict must say so"


def test_a_real_budget_difference_is_still_caught(tmp_path):
    """The positive control, and it is what makes the test above meaningful: with the spaces CLOSED,
    unequal budgets must still fail. Otherwise the fix could be "never check the budget".
    """
    a = _arm(tmp_path, "control2", {"sp-a": 40, "sp-b": 40},
             closed=("sp-a", "sp-b"), finished=True)
    b = _arm(tmp_path, "treatment2", {"sp-c": 20, "sp-d": 20},
             closed=("sp-c", "sp-d"), finished=True)
    okay, notes = casp.parity_verdict(a, b, "CONTROL", "TREATMENT")
    assert okay is False
    assert any("PER-SPACE BUDGET DIFFERS: 40 vs 20" in n for n in notes)


def test_equal_budgets_over_closed_spaces_agree(tmp_path):
    a = _arm(tmp_path, "c3", {"sp-a": 40, "sp-b": 40}, closed=("sp-a", "sp-b"), finished=True)
    b = _arm(tmp_path, "t3", {"sp-c": 40, "sp-d": 40}, closed=("sp-c", "sp-d"), finished=True)
    okay, notes = casp.parity_verdict(a, b, "CONTROL", "TREATMENT")
    joined = " | ".join(notes)
    assert "modal count agrees at 40" in joined
    assert "IN FLIGHT" not in joined


def test_the_open_space_is_excluded_but_the_closed_ones_still_answer(tmp_path):
    """The case the original docstring was actually reasoning about: several closed spaces plus one in
    flight. The closed ones answer the question and the open one must not disturb it -- including when
    the open one's count would, on its own, look like a different budget.
    """
    a = _arm(tmp_path, "c4", {"sp-a": 40, "sp-b": 40, "sp-open": 3}, closed=("sp-a", "sp-b"))
    b = _arm(tmp_path, "t4", {"sp-c": 40, "sp-d": 40, "sp-open2": 19}, closed=("sp-c", "sp-d"))
    okay, notes = casp.parity_verdict(a, b, "CONTROL", "TREATMENT")
    joined = " | ".join(notes)
    assert "modal count agrees at 40" in joined
    assert "PER-SPACE BUDGET DIFFERS" not in joined
    assert "IN FLIGHT" in joined, "still provisional -- neither run has RUN_FINISHED"


def test_a_space_that_overshot_its_budget_does_not_fire(tmp_path):
    """Why the mode rather than the max, kept as a test now that there is a suite: a timeout still
    counts as a trial, so one space reading 41 against a nominal 40 must not be a parity failure.
    """
    a = _arm(tmp_path, "c5", {"sp-a": 40, "sp-b": 41}, closed=("sp-a", "sp-b"), finished=True)
    b = _arm(tmp_path, "t5", {"sp-c": 40, "sp-d": 40}, closed=("sp-c", "sp-d"), finished=True)
    okay, notes = casp.parity_verdict(a, b, "CONTROL", "TREATMENT")
    assert not any("PER-SPACE BUDGET DIFFERS" in n for n in notes)


def test_closed_spaces_are_read_from_tuning_done_and_not_guessed(tmp_path):
    """`closed_spaces` must come from the event that means it. A checker that inferred "closed" from
    "has >= trials_per_space trials" would need to know the budget it is trying to measure.
    """
    a = _arm(tmp_path, "c6", {"sp-a": 40, "sp-open": 40}, closed=("sp-a",))
    assert a["closed_spaces"] == {"sp-a"}
    assert casp._closed_counts(a) == [40]
    assert sorted(a["per_space"].values()) == [40, 40], \
        "both spaces have 40 trials; only the TUNING_DONE tells them apart"


def test_finished_is_read_from_run_finished(tmp_path):
    live = _arm(tmp_path, "c7", {"sp-a": 5})
    done = _arm(tmp_path, "c8", {"sp-a": 5}, closed=("sp-a",), finished=True)
    assert live["finished"] is False
    assert done["finished"] is True
