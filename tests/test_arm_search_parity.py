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
         finished: bool = False, expansions: int = 0) -> dict:
    """One arm: `spaces` maps space_id -> trial count; `closed` lists the spaces that emitted
    TUNING_DONE; `finished` adds RUN_FINISHED; `expansions` adds that many SPACE_EXPANDED events.

    SPACE_EXPANDED's payload is copied from the live emitter, which carries `candidate_id` and nothing
    else -- the reader only counts these events, and inventing a richer payload would let a fixture
    prove a field the emitter does not send (`a-fixture-invented-to-match-the-reader-proves-nothing`).
    """
    ev: list[dict] = [{"seq": 0, "ts": 1000.0, "type": "RUN_CREATED", "payload": {"run_id": name}}]
    ts = 1001.0
    for _ in range(expansions):
        ev.append({"seq": int(ts), "ts": ts, "type": "SPACE_EXPANDED",
                   "payload": {"candidate_id": "cand-x"}})
        ts += 1.0
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


# ---------------------------------------------------------------------------------------------
# the second defect: equal per-space budgets do not make the TOTAL budgets equal
# ---------------------------------------------------------------------------------------------


def test_an_extra_space_from_a_k_expansion_is_caught(tmp_path):
    """The live S7 pair at 2.9 h, reproduced with its real numbers: the control arm closed 4 spaces and
    the treatment arm 3, because the control arm made 2 K expansions and each expansion publishes a
    SECOND space over the same candidate -- and a space is charged another whole `trials_per_space`.

    Every check that existed passed on this input: the per-space budget agreed at 40, the median gap was
    within tolerance, and the checker printed `expansions 0` and `2` as decoration before concluding the
    latency difference was "attributable to the switch". 200 trials against 120 is not attributable to
    anything but the extra search.
    """
    a = _arm(tmp_path, "kc", {"sp-a": 40, "sp-b": 40, "sp-c": 40, "sp-d": 40},
             closed=("sp-a", "sp-b", "sp-c", "sp-d"), finished=True, expansions=2)
    b = _arm(tmp_path, "kt", {"sp-e": 40, "sp-f": 40, "sp-g": 40},
             closed=("sp-e", "sp-f", "sp-g"), finished=True)
    okay, notes = casp.parity_verdict(a, b, "CONTROL", "TREATMENT")
    joined = " | ".join(notes)
    assert okay is False, "160 vs 120 trials is one whole space of extra search"
    assert "TOTAL SEARCH DIFFERS" in joined
    assert "K expansion" in joined, "the mechanism must be named, not just the totals"
    # The per-space test must still AGREE -- that is the whole point: the old checks cannot see this.
    assert "modal count agrees at 40" in joined
    assert "PER-SPACE BUDGET DIFFERS" not in joined


def test_a_mid_run_total_difference_is_recorded_but_does_not_flip_the_verdict(tmp_path):
    """Two arms merely OUT OF STEP must not read as a confound. Mid-run the trailing arm is always
    behind on totals by construction -- it has not opened its remaining spaces yet -- so judging this
    before RUN_FINISHED would make every live pair fail. It is reported, and marked PROVISIONAL.
    """
    a = _arm(tmp_path, "mc", {"sp-a": 40, "sp-b": 40, "sp-c": 40, "sp-d": 40, "sp-e": 34},
             closed=("sp-a", "sp-b", "sp-c", "sp-d"), expansions=2)
    b = _arm(tmp_path, "mt", {"sp-f": 40, "sp-g": 40, "sp-h": 40},
             closed=("sp-f", "sp-g", "sp-h"))
    okay, notes = casp.parity_verdict(a, b, "CONTROL", "TREATMENT")
    joined = " | ".join(notes)
    assert "TOTAL SEARCH DIFFERS" in joined, "the asymmetry must still be visible mid-run"
    assert "PROVISIONAL" in joined
    assert okay is True, "an out-of-step live pair is not yet a confound"


def test_equal_totals_do_not_fire_even_with_equal_expansions(tmp_path):
    """The negative control for the new check. Expansions are not themselves a fault -- both arms
    expanding twice and landing on the same total is parity, and must read as such.
    """
    a = _arm(tmp_path, "ec", {"sp-a": 40, "sp-b": 40, "sp-c": 40},
             closed=("sp-a", "sp-b", "sp-c"), finished=True, expansions=2)
    b = _arm(tmp_path, "et", {"sp-d": 40, "sp-e": 40, "sp-f": 40},
             closed=("sp-d", "sp-e", "sp-f"), finished=True, expansions=2)
    okay, notes = casp.parity_verdict(a, b, "CONTROL", "TREATMENT")
    joined = " | ".join(notes)
    assert okay is True
    assert "TOTAL SEARCH DIFFERS" not in joined
    assert "total search: 120 vs 120" in joined

def test_a_sub_space_total_difference_does_not_fire(tmp_path):
    """The unit is ONE SPACE, derived from the modal budget rather than a percentage I picked. A few
    trials of slack -- a timeout that still counted, an arm one trial into its next space -- is not an
    extra space's worth of search and must not be reported as one.
    """
    a = _arm(tmp_path, "sc", {"sp-a": 40, "sp-b": 40, "sp-c": 41},
             closed=("sp-a", "sp-b", "sp-c"), finished=True)
    b = _arm(tmp_path, "st", {"sp-d": 40, "sp-e": 40, "sp-f": 40},
             closed=("sp-d", "sp-e", "sp-f"), finished=True)
    okay, notes = casp.parity_verdict(a, b, "CONTROL", "TREATMENT")
    assert okay is True
    assert "TOTAL SEARCH DIFFERS" not in " | ".join(notes)


def test_the_total_check_is_silent_when_the_per_space_budget_is_unknown(tmp_path):
    """With no closed space there is no modal budget, so there is no unit to measure a gap in. The
    check must abstain rather than fall back to a made-up unit -- a difference reported in units the
    run has not established would be a number with no denominator.
    """
    a = _arm(tmp_path, "uc", {"sp-a": 15})
    b = _arm(tmp_path, "ut", {"sp-b": 7})
    okay, notes = casp.parity_verdict(a, b, "CONTROL", "TREATMENT")
    joined = " | ".join(notes)
    assert "TOTAL SEARCH" not in joined
    assert "NOT YET ANSWERABLE" in joined


def test_a_landed_rewrite_is_not_hidden_by_an_unclosed_round(tmp_path):
    """Loop C in progress: rewrites have landed, no family round has closed yet.

    The checker printed "rewrite rounds 0" for BOTH arms of the live pair while three rewrites were
    already being tuned, because it counted only FAMILY_ROUND_RECORDED -- which fires once a round is
    tuned and reconciled, so it is legitimately zero mid-loop-C. A reader seeing only that concludes
    loop C never ran, which is the same defect as counting wall-search events instead of walls.
    """
    ev = [{"seq": 0, "ts": 1000.0, "type": "RUN_CREATED", "payload": {"run_id": "lc"}},
          _trial(1001.0, "sp-a"),
          {"seq": 1002, "ts": 1002.0, "type": "TUNING_DONE",
           "payload": {"candidate_id": "cand-x", "space_id": "sp-a", "best_ms": 2.0}},
          {"seq": 1003, "ts": 1003.0, "type": "REWRITE_PRODUCED",
           "payload": {"candidate_id": "cand-r1", "family_id": "fam-1", "hypothesis_id": "H1"}},
          {"seq": 1004, "ts": 1004.0, "type": "REWRITE_PRODUCED",
           "payload": {"candidate_id": "cand-r2", "family_id": "fam-1", "hypothesis_id": "H2"}}]
    a = casp.read_arm(_write(tmp_path, "loopc", ev))
    assert a["rewrites"] == 2, "REWRITE_PRODUCED is the event that says a rewrite exists"
    assert a["rounds"] == 0, "no family round has closed -- that is correct, not a contradiction"


def test_a_closed_round_is_counted_separately_from_the_rewrite(tmp_path):
    """The positive control for the split: once the round closes, BOTH counts are non-zero and they do
    not collapse into one another. Without this, "count REWRITE_PRODUCED as rounds" would also pass.
    """
    ev = [{"seq": 0, "ts": 1000.0, "type": "RUN_CREATED", "payload": {"run_id": "lc2"}},
          {"seq": 1, "ts": 1001.0, "type": "REWRITE_PRODUCED",
           "payload": {"candidate_id": "cand-r1", "family_id": "fam-1"}},
          {"seq": 2, "ts": 1002.0, "type": "FAMILY_ROUND_RECORDED",
           "payload": {"family_id": "fam-1", "round": 1}}]
    a = casp.read_arm(_write(tmp_path, "loopc2", ev))
    assert (a["rewrites"], a["rounds"]) == (1, 1)


def test_expansions_are_counted_from_the_event(tmp_path):
    """`expansions` must come from SPACE_EXPANDED itself. Inferring it from "two spaces share a
    candidate_id" would work on today's logs and break the moment a candidate is re-parameterized for
    any other reason.
    """
    a = _arm(tmp_path, "xc", {"sp-a": 40}, closed=("sp-a",), expansions=3)
    assert a["expansions"] == 3
    b = _arm(tmp_path, "xt", {"sp-b": 40}, closed=("sp-b",))
    assert b["expansions"] == 0


# ---------------------------------------------------------------------------------------------
# Three tests added because a revert check (scripts/probes/revert_check_total_search.py) showed the
# first six could not tell the shipped check from three plausible alternatives. In each case the fault
# was that the fixtures above are too TIDY -- space count, trial count and modal budget all move
# together, so implementations that key on different quantities agree on them. These pull those
# quantities apart. Without them the suite passed on all three variants.
# ---------------------------------------------------------------------------------------------


def test_a_whole_extra_space_fires_even_when_it_is_a_small_share_of_a_long_run(tmp_path):
    """Discriminates the unit-of-one-space from a fixed percentage tolerance.

    A 25%-style tolerance passes every test above, because those runs are short enough that one space is
    a large fraction of them. On a long run it does not: 240 vs 200 trials is a 20% difference and a
    WHOLE extra space of search. The quantity that matters is how much search one arm got that the other
    did not, and that does not shrink because the run was long.
    """
    a = _arm(tmp_path, "lc", {f"sp-a{i}": 40 for i in range(6)},
             closed=tuple(f"sp-a{i}" for i in range(6)), finished=True, expansions=1)
    b = _arm(tmp_path, "lt", {f"sp-b{i}": 40 for i in range(5)},
             closed=tuple(f"sp-b{i}" for i in range(5)), finished=True)
    okay, notes = casp.parity_verdict(a, b, "CONTROL", "TREATMENT")
    joined = " | ".join(notes)
    assert okay is False, "240 vs 200 is one full space of extra search, at only 20%"
    assert "TOTAL SEARCH DIFFERS" in joined
    assert "1.0 whole space" in joined


def test_equal_space_counts_with_unequal_search_still_fires(tmp_path):
    """Discriminates trials from SPACES. Counting spaces looks equivalent and is not: a space that
    closed early bought less search than a full one. Here both arms closed 4 spaces, but one arm's
    spaces were cut short (a wall-clock stop, an arm that ran out of time) and it got 40 fewer trials.
    An implementation that compares `len(per_space)` sees parity.
    """
    a = _arm(tmp_path, "qc", {"sp-a": 40, "sp-b": 40, "sp-c": 40, "sp-d": 40},
             closed=("sp-a", "sp-b", "sp-c", "sp-d"), finished=True)
    # `sp-h` closed too, but on a partial budget -- the arm ran out of wall clock inside it.
    b = _arm(tmp_path, "qt", {"sp-e": 40, "sp-f": 40, "sp-g": 40, "sp-h": 1},
             closed=("sp-e", "sp-f", "sp-g", "sp-h"), finished=True)
    assert len(a["per_space"]) == len(b["per_space"]) == 4, "the space COUNTS are equal"
    okay, notes = casp.parity_verdict(a, b, "CONTROL", "TREATMENT")
    joined = " | ".join(notes)
    assert okay is False, "160 vs 121 trials is a space's worth of search despite equal space counts"
    assert "TOTAL SEARCH DIFFERS" in joined


def test_one_overshooting_space_does_not_shift_the_unit(tmp_path):
    """Discriminates the MODAL budget from the max. A timeout still counts as a trial, so one space can
    read 41 -- and `max` would then price the gap in units of 41. Here the true gap is exactly one
    40-trial space (161 vs 121); with the unit at 41 the same gap measures 0.98 spaces and the check
    goes silent. `_mode` is already the reader's choice for the per-space budget, and this pins that the
    total check spends the same unit rather than deriving its own.
    """
    a = _arm(tmp_path, "oc", {"sp-a": 40, "sp-b": 41, "sp-c": 40, "sp-d": 40},
             closed=("sp-a", "sp-b", "sp-c", "sp-d"), finished=True, expansions=1)
    b = _arm(tmp_path, "ot", {"sp-e": 40, "sp-f": 41, "sp-g": 40},
             closed=("sp-e", "sp-f", "sp-g"), finished=True)
    assert casp._mode(casp._closed_counts(a)) == 40, "the modal budget is 40, not the 41 outlier"
    okay, notes = casp.parity_verdict(a, b, "CONTROL", "TREATMENT")
    joined = " | ".join(notes)
    assert okay is False, "161 vs 121 is one 40-trial space; only a unit of 41 hides it"
    assert "TOTAL SEARCH DIFFERS" in joined
    assert "modal budget of 40" in joined
