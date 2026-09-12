"""1c. The report must say which of the four loops ran -- and box 1's own log is the control.

WHY A POSITIVE CONTROL IS THE ONLY VALID TEST HERE. This section exists to detect a **silent**
failure: box 1's E1 control arm finished normally after 12.506 h, wrote a `RUN_FINISHED`, produced
four families and a 4.0852 ms winner, and never once called the rewriter. Zero agent calls is not
an error, so nothing in the existing report could show it, and running the new checker over data I
believe to be clean would prove nothing -- the recorded
`a-clean-run-is-not-evidence-a-checker-works` failure, where a checker exited 0 on code containing
four real bugs.

So the fixture is the REAL failing log: `tests/fixtures/box1_control_skeleton.jsonl`, extracted
from `run-l3-43-20260911-230217` on box 1. Only `ts`, `type` and the handful of payload fields this
module reads were kept -- the timing structure is byte-for-byte the run's own, which is what the
checker is being asked to read. The known answers it must reproduce:

    span                    12.506 h
    loop C (rewrite)        NEVER RAN  -- 0 REWRITE_PRODUCED, 0 rewriter calls
    loop D (novelty)        NEVER RAN
    largest agent-free gap  6.10 h, 50.8% of the span, all 40 trials in it one candidate
    the candidate           cand-941ea454

Every figure above was derived independently (by a throwaway script, before this module existed),
so a disagreement means the reader is wrong -- this is a control with a known answer, not a
regression snapshot of whatever the code happens to do.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from kernel_optimizer.reporting.completeness import completeness_lines

FIXTURE = Path(__file__).parent / "fixtures" / "box1_control_skeleton.jsonl"


@dataclass
class _Ev:
    seq: int
    ts: float
    type: str
    payload: dict[str, Any]


def _load(path: Path = FIXTURE) -> list[_Ev]:
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            out.append(_Ev(r["seq"], r["ts"], r["type"], r.get("payload") or {}))
    return out


def _ev(ts: float, t: str, payload: dict | None = None, seq: int = 0) -> _Ev:
    return _Ev(seq, ts, t, payload or {})


# --------------------------------------------------------- the positive control


def test_the_fixture_really_is_the_defective_run():
    """Check the control before trusting it. A fixture that had been quietly regenerated from a
    healthy run would make every assertion below vacuous -- the recorded
    `a-fixture-invented-to-match-the-reader-proves-nothing` failure."""
    events = _load()
    span_h = (events[-1].ts - events[0].ts) / 3600.0
    types = {}
    for e in events:
        types[e.type] = types.get(e.type, 0) + 1

    assert len(events) == 473, len(events)
    assert span_h == pytest.approx(12.506, abs=0.01), span_h
    assert types.get("REWRITE_PRODUCED", 0) == 0, "the control's defect IS the absent rewrite"
    assert types.get("NOVELTY_PRODUCED", 0) == 0
    assert types["TRIAL_DONE"] == 280
    assert types["RUN_FINISHED"] == 1, "it finished NORMALLY -- that is what makes it silent"


def test_it_reports_that_loop_C_never_ran():
    """The headline requirement. Must name loop C explicitly, not merely omit it."""
    lines = completeness_lines(_load(), {"wall_clock_hours": 12.0})
    text = "\n".join(lines)

    assert "loop C (rewrite): **NEVER RAN**" in text, text
    assert "loop D (novelty): **NEVER RAN**" in text, text
    assert "loop B (tuning): **ran**" in text, text
    assert "NEVER EXECUTED" in text.upper(), (
        "a reader skimming must not have to infer it from an absence: %s" % text)


def test_it_reports_the_610_hour_agent_free_window_and_what_was_in_it():
    """The second half of the finding, and the part that took a throwaway script to discover.

    BOTH denominators are asserted, because writing this test is what exposed that they differ.
    The 50.8% on record is the window against the 12 h BUDGET; against the run's actual 12.506 h
    SPAN the same window is 48.8%. Neither is wrong and the gap between them is the per-candidate
    budget overrun -- so the section must name which denominator each figure uses rather than
    printing one bare percentage. Every finished run overran (4.2 / 2.9 / 9.7%), so the two
    numbers never coincide in practice.
    """
    lines = completeness_lines(_load(), {"wall_clock_hours": 12.0})
    text = "\n".join(lines)

    assert "6.10 h" in text, f"the measured gap is 6.10 h: {text}"
    assert "48.8% of the run's 12.51 h span" in text, (
        f"the span-relative fraction, with its denominator named: {text}")
    assert "50.8% of the 12.0 h budget" in text, (
        f"the budget-relative fraction is the 50.8 pct on record: {text}")
    assert "cand-941ea454" in text, (
        f"it must name the candidate to look at, not just report silence: {text}")
    assert "ONE candidate" in text, text
    assert "single tuning pass" in text, text
    assert "**this single window is 49% of the run.**" in text, (
        f"past the 33 pct threshold the section must say so explicitly: {text}")


def test_it_says_the_clock_was_spent_rather_than_a_freeze_firing_early():
    """The two causes need different fixes, so the section must say which one this is. Box 1 spent
    12.506 h of a 12 h budget -- the clock WAS spent, so the cause is where the time went."""
    lines = completeness_lines(_load(), {"wall_clock_hours": 12.0})
    text = "\n".join(lines)

    assert "clock was spent" in text, text
    assert "where the time went" in text, text
    assert "NOT spent" not in text, "the opposite diagnosis must not also appear: %s" % text


# --------------------------------------------------------- the negative controls
# A checker that fires on everything is as useless as one that fires on nothing, so these
# construct healthy runs and require silence on the alarm lines.


def test_a_run_that_used_all_four_loops_raises_nothing():
    t = 1_000_000.0
    events = [_ev(t, "RUN_CREATED")]
    for i, (module, produced) in enumerate([
        ("repair", "REPAIR_PRODUCED"), ("rewriter", "REWRITE_PRODUCED"),
        ("novelty", "NOVELTY_PRODUCED"),
    ]):
        events += [
            _ev(t + 60 * (3 * i + 1), "AGENT_CALL_STARTED", {"module": module}),
            _ev(t + 60 * (3 * i + 2), "AGENT_CALL_FINISHED", {"module": module}),
            _ev(t + 60 * (3 * i + 3), produced, {}),
        ]
    events.append(_ev(t + 600, "TUNING_DONE", {}))
    events.append(_ev(t + 660, "RUN_FINISHED", {"summary": {"elapsed_hours": 0.2}}))

    text = "\n".join(completeness_lines(events, {"wall_clock_hours": 12.0}))

    assert "NEVER RAN" not in text, text
    assert "NEVER EXECUTED" not in text.upper(), text
    for letter in "ABCD":
        assert f"loop {letter}" in text, text


def test_a_short_agent_free_window_is_reported_but_not_flagged():
    """The 33% threshold. The window is always reported -- it is a budget fact -- but the
    "this single window is N% of the run" alarm must stay quiet at a normal fraction.

    The first version of this test was itself wrong, and the checker caught it: its "healthy" run
    put a 47-minute gap into a 60-minute span, so the alarm fired at 94% and the assertion failed.
    The alarm was right. A gap fixture has to be built by ARITHMETIC -- keep every gap under a
    third of the span -- rather than by writing plausible-looking timestamps, which is the same
    lesson as `a-fixture-invented-to-match-the-reader-proves-nothing`.
    """
    t = 1_000_000.0
    # A 1-hour run with agent calls every ~5 minutes: the largest gap is 300 s = 8.3% of the span.
    events: list[_Ev] = [_ev(t, "RUN_CREATED")]
    for i in range(12):
        s = t + 300.0 * i
        events += [
            _ev(s + 1, "AGENT_CALL_STARTED", {"module": "rewriter"}),
            _ev(s + 30, "AGENT_CALL_FINISHED", {"module": "rewriter"}),
            _ev(s + 60, "TRIAL_DONE", {"trial": {"candidate_id": "c1"}}),
        ]
    events += [
        _ev(t + 3600, "REWRITE_PRODUCED", {}),
        _ev(t + 3601, "TUNING_DONE", {}),
        _ev(t + 3602, "RUN_FINISHED", {"summary": {"elapsed_hours": 1.0}}),
    ]

    text = "\n".join(completeness_lines(events, {"wall_clock_hours": 1.0}))

    assert "longest window with **no agent running**" in text
    assert "this single window is" not in text, (
        f"the alarm must not fire below the threshold: {text}")


def test_a_long_agent_call_is_not_counted_as_a_gap():
    """The direction that matters. A 30-minute rewriter call is the loop WORKING; measuring gaps
    between FINISHED events would count that duration as starvation and invert the meaning."""
    t = 1_000_000.0
    events = [
        _ev(t, "RUN_CREATED"),
        _ev(t + 10, "AGENT_CALL_STARTED", {"module": "rewriter"}),
        _ev(t + 1810, "AGENT_CALL_FINISHED", {"module": "rewriter"}),   # a 30-min call
        _ev(t + 1820, "REWRITE_PRODUCED", {}),
        _ev(t + 1830, "AGENT_CALL_STARTED", {"module": "analyst"}),
        _ev(t + 1840, "AGENT_CALL_FINISHED", {"module": "analyst"}),
        _ev(t + 1850, "RUN_FINISHED", {"summary": {"elapsed_hours": 0.5}}),
    ]

    text = "\n".join(completeness_lines(events, {"wall_clock_hours": 12.0}))

    assert "this single window is" not in text, (
        "the 30-minute call is work, not a gap: %s" % text)
    # The largest true gap here is 10 s, well under a minute.
    assert "0.00 h" in text or "0.01 h" in text, text


def test_an_agent_call_that_produced_nothing_is_distinguished_from_never_calling():
    """Two different failures with two different fixes: a rewriter that was called and timed out
    (the artifact-rescue / timeout work) vs a rewriter never reached (the budget work). Collapsing
    them would send the reader to the wrong place."""
    t = 1_000_000.0
    events = [
        _ev(t, "RUN_CREATED"),
        _ev(t + 10, "AGENT_CALL_STARTED", {"module": "rewriter"}),
        _ev(t + 1810, "AGENT_CALL_FAILED", {"module": "rewriter"}),
        _ev(t + 1820, "TUNING_DONE", {}),
        _ev(t + 1830, "RUN_FINISHED", {"summary": {"elapsed_hours": 0.5}}),
    ]

    text = "\n".join(completeness_lines(events, {"wall_clock_hours": 12.0}))

    assert "loop C (rewrite): **ran**" in text, (
        "the loop WAS entered -- reporting NEVER RAN here would point at the wrong cause: %s"
        % text)
    assert "CALLED BUT PRODUCED NOTHING" in text, text


def test_no_agent_calls_at_all_is_its_own_message():
    """A run that never called any agent (a crash during the baseline, say) must not be described
    through the gap machinery -- there are no calls to measure between."""
    t = 1_000_000.0
    events = [_ev(t, "RUN_CREATED"), _ev(t + 3600, "BASELINE_DONE", {})]

    text = "\n".join(completeness_lines(events, {}))

    assert "no agent call of any kind" in text, text


def test_empty_events_return_nothing():
    assert completeness_lines([], {}) == []


# --------------------------------------------------------- wiring


def test_the_section_is_actually_in_the_report():
    """A section no report calls is not implemented. Same discipline that caught
    `conversion_verdict` being journalled and read zero times."""
    from kernel_optimizer.reporting import report as report_mod

    src = Path(report_mod.__file__).read_text(encoding="utf-8")
    assert "completeness_lines(" in src, "the report must call it"
    assert "from kernel_optimizer.reporting.completeness import completeness_lines" in src


def test_it_reads_only_the_event_log():
    """`kernel-opt report` must keep regenerating purely from events.jsonl, so this module may not
    touch the filesystem, the config beyond `budgets`, or the store."""
    from kernel_optimizer.reporting import completeness as mod

    src = Path(mod.__file__).read_text(encoding="utf-8")
    for forbidden in ("open(", "read_text", "Path(", "subprocess", "RunStore"):
        assert forbidden not in src, (
            "a read-only report section must not reach outside the event list: %r" % forbidden)
