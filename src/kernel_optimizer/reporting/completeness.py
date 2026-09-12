"""Did this run reach every stage of the process, or did it stop somewhere silently?

WHY THIS MODULE EXISTS. Box 1's E1 control arm (`run-l3-43-20260911-230217`) finished normally
after 12.506 h, wrote a `RUN_FINISHED`, produced four families with a best of 4.0852 ms -- and
never once called the rewriter. Its report looked completely ordinary, because **zero agent calls
is not an error**: no exception, no timeout, no rejection, nothing for any existing section to
show. The finding needed a throwaway script against `events.jsonl`, and what it found was a
**6.10 h window with no agent call at all** -- 50.8% of the budget -- containing exactly 40
`TRIAL_DONE` events, all for one candidate.

Why it must exist BEFORE any large-scale benchmark rather than after. At scale this failure mode
lands on some fraction of tasks and not others, depending only on whether one candidate happened
to compile slowly. A run that never entered the rewrite loop and a run whose rewrites did not help
produce the same shape of report, so "no rewriting happened" reads as "rewriting does not work" --
the wrong conclusion, drawn from the framework's own budget accident, and drawn silently.

Read-only over the event log. Nothing here changes what the search does, so it is safe to apply to
runs that have already finished (and it is validated by doing exactly that -- see
`tests/test_process_completeness.py`, whose positive control is box 1's own log).
"""

from __future__ import annotations

from typing import Any

# The four loops, by the events that prove each one ran. Loop A is the repair loop, so a
# REPAIR_PRODUCED is its evidence; B is tuning; C is structural rewriting; D is novelty.
#
# Keyed on the PRODUCING event rather than on the agent call, because an agent call that timed out
# proves the loop was ENTERED but produced nothing -- a distinction this section has to keep, since
# "entered and produced nothing" and "never entered" have different causes and different fixes.
_LOOPS: tuple[tuple[str, str, tuple[str, ...], tuple[str, ...]], ...] = (
    ("A", "repair", ("REPAIR_PRODUCED",), ("repair",)),
    ("B", "tuning", ("TUNING_DONE",), ()),
    ("C", "rewrite", ("REWRITE_PRODUCED",), ("rewriter",)),
    ("D", "novelty", ("NOVELTY_PRODUCED",), ("novelty",)),
)


def _fmt_h(seconds: float) -> str:
    return f"{seconds / 3600.0:.2f} h"


def _agent_activity(events: list) -> list[tuple[float, str, str]]:
    """Every agent call boundary, as (ts, module, kind). Both STARTED and FINISHED.

    Both ends, deliberately. A gap measured between FINISHED events would count a long call's own
    duration as a gap, which reverses the meaning: a 30-minute rewriter call is the loop WORKING,
    while 6 hours with nothing between a FINISHED and the next STARTED is the loop starved. So the
    gap this module reports is FINISHED -> next STARTED, i.e. time when no agent was running.
    """
    out: list[tuple[float, str, str]] = []
    for e in events:
        if e.type in ("AGENT_CALL_STARTED", "AGENT_CALL_FINISHED", "AGENT_CALL_FAILED"):
            kind = "start" if e.type == "AGENT_CALL_STARTED" else "end"
            out.append((float(e.ts), str((e.payload or {}).get("module") or "?"), kind))
    out.sort(key=lambda r: r[0])
    return out


def _what_happened_in(events: list, t0: float, t1: float) -> str:
    """Describe the largest agent-free window by what the harness was doing instead.

    Counting the events inside the window is what turns "6.10 h of silence" into a diagnosis: 40
    `TRIAL_DONE` for a single candidate says the run was tuning, not hung, and names the candidate
    to look at. A window containing NOTHING says something else entirely (a wedged worker), and the
    two must not print the same sentence.
    """
    inside = [e for e in events if t0 < float(e.ts) < t1]
    if not inside:
        return "no events at all — the harness was not making progress of any kind"
    counts: dict[str, int] = {}
    for e in inside:
        counts[e.type] = counts.get(e.type, 0) + 1
    trials = [e for e in inside if e.type == "TRIAL_DONE"]
    parts = [f"{n} {t}" for t, n in sorted(counts.items(), key=lambda kv: -kv[1])[:4]]
    desc = ", ".join(parts)
    if trials:
        cands = {(e.payload.get("trial") or {}).get("candidate_id") for e in trials}
        cands.discard(None)
        if len(cands) == 1:
            only = next(iter(cands))
            desc += (f" — ALL {len(trials)} trials belong to ONE candidate (`{only}`), "
                     f"so this window is a single tuning pass")
        else:
            desc += f" — trials span {len(cands)} candidates"
    return desc


def completeness_lines(events: list, budgets: dict[str, Any] | None = None) -> list[str]:
    """The report section. Returns [] only when there are no events at all.

    Three questions, in the order a reader asks them:
      1. Which of the four loops ran, and how many rounds did each get?
      2. If a loop never ran, was there budget left when the run ended?
      3. Where did the time go that no loop was using -- the longest agent-free window, and
         what the harness was doing inside it?
    """
    if not events:
        return []
    budgets = budgets or {}
    span_t0 = float(events[0].ts)
    span_t1 = float(events[-1].ts)
    span_s = max(0.0, span_t1 - span_t0)

    lines = ["## Process completeness\n"]

    # ---- 1. per-loop execution
    counts: dict[str, int] = {}
    for e in events:
        counts[e.type] = counts.get(e.type, 0) + 1
    calls_by_module: dict[str, int] = {}
    for e in events:
        if e.type == "AGENT_CALL_STARTED":
            m = str((e.payload or {}).get("module") or "?")
            calls_by_module[m] = calls_by_module.get(m, 0) + 1

    never_ran: list[str] = []
    for letter, label, produced_types, modules in _LOOPS:
        produced = sum(counts.get(t, 0) for t in produced_types)
        called = sum(calls_by_module.get(m, 0) for m in modules)
        if produced or called:
            detail = f"{produced} produced"
            if modules:
                detail += f", {called} agent call{'s' if called != 1 else ''}"
                if called and not produced:
                    # Entered but empty-handed: a different failure from never entering, and the
                    # one that A2's artifact rescue and the agent timeout work address.
                    detail += " — CALLED BUT PRODUCED NOTHING"
            lines.append(f"- loop {letter} ({label}): **ran** — {detail}")
        else:
            never_ran.append(f"{letter} ({label})")
            lines.append(f"- loop {letter} ({label}): **NEVER RAN** — no "
                         f"{'/'.join(produced_types)}"
                         + (f" and no {'/'.join(modules)} call" if modules else ""))

    # ---- 2. was budget left when a loop never ran
    if never_ran:
        elapsed = None
        for e in reversed(events):
            if e.type == "RUN_FINISHED":
                elapsed = ((e.payload or {}).get("summary") or {}).get("elapsed_hours")
                break
        wc = budgets.get("wall_clock_hours")
        lines.append("")
        lines.append(f"> **Loop {' and '.join(never_ran)} never executed.** This is not reported "
                     f"anywhere else in this report: a loop that never runs raises no error and "
                     f"produces no rejection, so the run looks complete. A result from this run "
                     f"cannot be read as evidence about what those loops do.")
        if wc and elapsed is not None:
            # Both directions are informative and they point at different causes, so name which
            # one this is rather than printing the numbers and leaving it to the reader.
            if float(elapsed) >= float(wc) * 0.98:
                lines.append(f"> The clock was spent ({elapsed} h of {wc} h), so the cause is "
                             f"**where the time went**, not an early freeze — see the window "
                             f"below.")
            else:
                lines.append(f"> The clock was NOT spent ({elapsed} h of {wc} h), so a freeze "
                             f"rule ended the run before the loop was reached.")

    # ---- 3. the largest agent-free window
    acts = _agent_activity(events)
    lines.append("")
    if not acts:
        lines.append(f"- **no agent call of any kind** in {_fmt_h(span_s)} of wall clock — this "
                     f"run exercised none of the LLM loops")
        return lines + [""]

    # FINISHED -> next STARTED, plus the two open ends (run start -> first call, last call -> run
    # end). The open ends matter: a run that spent its first 6 hours before the first agent call
    # has the same problem as one that spent 6 hours in the middle.
    gaps: list[tuple[float, float, float, str]] = []
    prev_end = span_t0
    prev_label = "run start"
    for ts, module, kind in acts:
        if kind == "start":
            if ts > prev_end:
                gaps.append((ts - prev_end, prev_end, ts, f"{prev_label} -> {module} call"))
        else:
            prev_end = ts
            prev_label = f"{module} call"
    if span_t1 > prev_end:
        gaps.append((span_t1 - prev_end, prev_end, span_t1, f"{prev_label} -> run end"))

    n_calls = sum(1 for _, _, k in acts if k == "start")
    lines.append(f"- agent calls: **{n_calls}** over {_fmt_h(span_s)} of wall clock")
    if gaps:
        gaps.sort(key=lambda g: -g[0])
        worst, t0, t1, what = gaps[0]
        pct = (worst / span_s * 100.0) if span_s > 0 else 0.0
        # BOTH denominators, named. The span and the budget are different numbers whenever a run
        # overruns -- box 1's 6.10 h window is 48.8% of its 12.506 h span and 50.8% of its 12 h
        # budget -- and a bare percentage invites the reader to assume whichever they had in mind.
        # Every finished run so far overran (4.2 / 2.9 / 9.7%), so they never coincide in practice.
        wc = budgets.get("wall_clock_hours")
        of_budget = (f", {worst / (float(wc) * 3600.0) * 100.0:.1f}% of the {wc} h budget"
                     if wc else "")
        lines.append(f"- longest window with **no agent running**: **{_fmt_h(worst)}** "
                     f"({pct:.1f}% of the run's {_fmt_h(span_s)} span{of_budget}), {what}")
        lines.append(f"  - inside it: {_what_happened_in(events, t0, t1)}")
        # A threshold, so the line is a finding and not just a number. A third of the span in one
        # agent-free window is far outside normal: on the three finished runs the treatment arms'
        # largest windows are a small fraction of the span, while the control arm's was ~49%.
        if pct >= 33.0:
            lines.append(f"  - **this single window is {pct:.0f}% of the run.** The stage that "
                         f"follows it received whatever was left, which is why a loop can be "
                         f"skipped without any error being raised.")
    return lines + [""]
