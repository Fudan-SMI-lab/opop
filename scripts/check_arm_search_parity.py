"""Did the two arms get comparable SEARCH? The load-bearing precondition J2-5 does not state.

`scripts/audit_arm_comparability.py` verified the two arms' CONFIGS match field-by-field. That is
necessary and not sufficient: the budget has two limits, `trials_per_space` and `wall_clock_hours`.
Equal per-space trial counts only settle the first. If one box is slower, it reaches fewer SPACES and
fewer rewrite rounds inside the same 12 h, and a latency difference then has two explanations -- the
switch, or the extra search -- which is the ambiguity the paired cross-box design exists to remove
(see `scripts/compare_calibrations.py`, which settles the *denominator* half of the same question).

TWO LOOP-C COUNTS, and why both are printed. `rewrites landed` counts REWRITE_PRODUCED; `family rounds
CLOSED` counts FAMILY_ROUND_RECORDED, which fires only once a family's round has been tuned and
reconciled. Mid-loop-C the second is legitimately zero while the first is not -- this checker printed
"rewrite rounds 0" for BOTH arms of the live pair while three rewrites had landed and were being tuned,
which reads as "loop C never ran". Same shape as counting wall-search events instead of walls.

WHAT THIS DOES NOT MEASURE, and why it matters more than it sounds
-----------------------------------------------------------------
Inter-event gaps are NOT per-trial costs. `max_shared_jobs: 2` puts `compile-screen` and `prescreen`
on the shared lane while `eval` takes the exclusive lock, so a trial's compile overlaps the previous
trial's timing. A gap therefore measures "time until the next TRIAL_DONE landed", which under
concurrency can be near zero for a trial that did real work -- and a first pass at this comparison
produced per-config medians of 0.0 s and a ratio of 192988x from dividing by one.

So the comparable quantity is the AGGREGATE: trials completed per wall-clock hour, which is
insensitive to how the work is interleaved. Per-trial numbers are printed only as a distribution
with the tail called out, because the useful question about them is "is the total dominated by a few
pathological configs" -- and that question is answerable from the gap sum even when individual gaps
are not attributable.

    python scripts/check_arm_search_parity.py <control_run_dir> <treatment_run_dir>
"""

from __future__ import annotations

import argparse
import collections
import json
import statistics
import sys
from pathlib import Path

# A rate difference this large would let the faster arm reach a whole extra space inside 12 h.
_RATE_TOL = 1.15

# How far apart the arms' TOTAL search may be, measured in space-equivalents (trials / modal per-space
# budget). Below 1.0 on purpose: at exactly 1.0 an arm that lost a whole space to the wall clock -- so
# its last space closed on a partial budget -- measures 0.98 spaces short and passes. The slack it does
# allow is the few trials a closed space can carry over its nominal budget (a timeout still counts as a
# trial, so 41 against 40 is normal).
_SPACE_TOL = 0.9


def _events(run_dir: Path) -> list[dict]:
    p = run_dir / "events.jsonl"
    if not p.exists():
        raise SystemExit("not a run dir (no events.jsonl): %s" % run_dir)
    out = []
    with p.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                e = json.loads(line)
            except ValueError:
                continue
            if isinstance(e.get("ts"), (int, float)):
                out.append(e)
    if not out:
        raise SystemExit("no timestamped events in %s" % run_dir)
    return out


def read_arm(run_dir: Path) -> dict:
    evs = _events(run_dir)
    span = evs[-1]["ts"] - evs[0]["ts"]

    per_space: collections.Counter = collections.Counter()
    kinds: collections.Counter = collections.Counter()
    gaps: list[tuple[float, str, dict]] = []
    ok = 0
    prev = evs[0]["ts"]
    agent_s = 0.0
    open_calls: dict[str, float] = {}
    rounds = 0
    rewrites = 0
    spaces_expanded = 0
    spaces_rejected = 0
    expansion_refusals: dict[str, int] = {}
    closed_spaces: set = set()
    finished = False
    for e in evs:
        t = e.get("type")
        p = e.get("payload") or {}
        if t == "TRIAL_DONE":
            tr = p.get("trial") or p
            per_space[tr.get("space_id")] += 1
            fk = tr.get("failure_kind") or "ok"
            kinds[fk] += 1
            if fk == "ok" and tr.get("status") == "complete":
                ok += 1
            gaps.append((e["ts"] - prev, fk, (tr.get("params") or {}).get("values") or {}))
        elif t == "AGENT_CALL_STARTED":
            open_calls[str(p.get("call_id") or e.get("seq"))] = e["ts"]
        elif t in ("AGENT_CALL_FINISHED", "AGENT_CALL_FAILED"):
            k = str(p.get("call_id") or "")
            st = open_calls.pop(k, None)
            if st is None and open_calls:
                # `call_id` is present on both sides in v3; the FIFO fallback exists so an older
                # log does not silently contribute 0 h of agent time and read as "all GPU".
                st = open_calls.pop(min(open_calls, key=lambda x: open_calls[x]))
            if st is not None:
                agent_s += e["ts"] - st
        elif t == "FAMILY_ROUND_RECORDED":
            rounds += 1
        elif t == "REWRITE_PRODUCED":
            # Counted SEPARATELY from `rounds`, because they answer different questions and conflating
            # them made this checker print "rewrite rounds 0" for both arms while three rewrites had
            # already landed and were being tuned. FAMILY_ROUND_RECORDED fires when a family's ROUND
            # closes -- after the rewrite has been tuned and reconciled -- so mid-loop-C it is legitimately
            # zero. A reader seeing only that concludes loop C never ran. Same defect shape as counting
            # wall-search events instead of walls.
            rewrites += 1
        elif t == "SPACE_EXPANDED":
            spaces_expanded += 1
        elif t == "SPACE_EXPANSION_REJECTED":
            # Counted so "this arm expanded less" and "this arm was refused more" are separable.
            # They are different findings: the first says the arms searched unequally, the second
            # would say the switch changed what the expander is allowed to do. On the S7 pair both
            # arms were refused exactly once, both for `no_new_choices` -- the expansion returned a
            # domain set identical to the one it started from, so re-tuning would spend another
            # whole budget on the same searchable space. That is correct behaviour, not a defect,
            # and the gap in ACCEPTED expansions comes from the arms having different numbers of
            # candidates reach closure.
            spaces_rejected += 1
            reason = str((p.get("reason") or "?"))
            expansion_refusals[reason] = expansion_refusals.get(reason, 0) + 1
        elif t == "TUNING_DONE":
            sid = p.get("space_id")
            if sid:
                closed_spaces.add(sid)
        elif t == "RUN_FINISHED":
            finished = True
        prev = e["ts"]

    n = sum(per_space.values())
    gs = sorted(g for g, _, _ in gaps)
    tail_n = max(1, len(gs) // 10)
    return {
        "span_h": span / 3600.0,
        "trials": n,
        "ok": ok,
        "kinds": dict(kinds),
        "spaces": len(per_space),
        "per_space": dict(per_space),
        "rounds": rounds,
        "rewrites": rewrites,
        "expansions": spaces_expanded,
        "expansions_refused": spaces_rejected,
        "expansion_refusals": dict(expansion_refusals),
        "trials_per_h": n / (span / 3600.0) if span else 0.0,
        "agent_frac": agent_s / span if span else 0.0,
        "gap_median": statistics.median(gs) if gs else 0.0,
        "gap_sum_h": sum(gs) / 3600.0,
        "tail_share": (sum(sorted(gs, reverse=True)[:tail_n]) / sum(gs)) if sum(gs) else 0.0,
        # Key on the gap alone: a bare `sorted` on the tuples falls through to comparing the params
        # dict whenever two trials share a gap and a failure kind, which raises TypeError.
        "worst": sorted(gaps, key=lambda t: t[0], reverse=True)[:3],
        # Which spaces have actually CLOSED. A space's trial count only means "the budget" once tuning
        # over it has finished; before that it means "progress so far". Read from TUNING_DONE, which is
        # emitted exactly once per space when its tuning pass ends.
        "closed_spaces": closed_spaces,
        "finished": finished,
    }


def _closed_counts(a: dict) -> list[int]:
    """Trial counts of the spaces that have FINISHED tuning, in ascending order.

    Separated from `per_space` because the budget question is only answerable about closed spaces. On a
    live pair this is often empty, and an empty list is the correct answer -- see `_mode`.
    """
    return sorted(n for sid, n in a["per_space"].items() if sid in a["closed_spaces"])


def _mode(counts: list[int]) -> int | None:
    """The per-space budget as the arms actually applied it, over CLOSED spaces only.

    NOT `max`: a space can exceed the nominal budget (a timeout still counts as a trial), so the max
    reports 41-vs-40 and fires. NOT `min`: the smallest space is the one in flight, so the min reports
    1-vs-40 on every live invocation. The mode is the number most spaces agree on, which is what "the
    budget" means.

    THE MODE IS NOT ENOUGH ON ITS OWN, and this was found on a live pair rather than by reading the
    code. The docstring above used to justify the mode as "stable mid-run", which assumed several
    CLOSED spaces plus one in flight -- the mode then belongs to the closed ones. Early in a run each
    arm has exactly ONE space, still open, so the mode IS the in-flight count: the live S7 pair read
    "PER-SPACE BUDGET DIFFERS: 15 vs 7" and declared PARITY NOT OK when both arms were configured
    `trials_per_space: 40` and neither had emitted a single TUNING_DONE. That is a plausible number
    pointing at the wrong cause -- the same shape as trusting an in-flight run's `ended` rows -- and it
    would have read as a fatal confound on a pair that was in fact fine.

    So the caller passes only closed spaces, and an empty list returns None, which the verdict reports
    as "not yet answerable" rather than as agreement or as a difference.
    """
    if not counts:
        return None
    return collections.Counter(counts).most_common(1)[0][0]


def parity_verdict(a: dict, b: dict, la: str, lb: str) -> tuple[bool, list[str]]:
    """Three independent questions: was the per-space budget applied equally, did each arm get the same
    TOTAL number of spaces to spend it on, and did the wall clock buy comparable search? A no on any one
    makes a latency difference unattributable, and the middle one is the one this file originally
    collected the data for (`expansions`, `per_space`) without ever asking."""
    notes: list[str] = []
    okay = True

    # In-flight runs first, because every verdict below means something different for one. Neither the
    # budget question nor the rate question is settled while trials are still landing, and reporting a
    # provisional answer in the same words as a final one is how a live snapshot gets quoted as a result.
    live = not (a["finished"] and b["finished"])
    if live:
        notes.append("IN FLIGHT: %s / %s have not written RUN_FINISHED, so every verdict below is "
                     "PROVISIONAL -- re-run at the end before quoting any of it"
                     % ("finished" if a["finished"] else la,
                        "finished" if b["finished"] else lb))

    # Budget over CLOSED spaces only. A space's trial count is "the budget" once its tuning pass has
    # ended and "progress so far" before that; conflating the two made this checker report
    # "15 vs 7 trials per space, PARITY NOT OK" on a healthy pair whose arms were both set to 40 and
    # had closed no space at all.
    ca, cb = _closed_counts(a), _closed_counts(b)
    ma, mb = _mode(ca), _mode(cb)
    if ma is None or mb is None:
        notes.append("per-space budget: NOT YET ANSWERABLE -- closed spaces %d vs %d (a space's trial "
                     "count is the budget only after its TUNING_DONE; in-flight counts are progress). "
                     "Open-space progress so far: %s vs %s"
                     % (len(a["closed_spaces"]), len(b["closed_spaces"]),
                        sorted(a["per_space"].values()), sorted(b["per_space"].values())))
    elif ma != mb:
        okay = False
        notes.append("PER-SPACE BUDGET DIFFERS: %d vs %d trials per closed space (modal). The budget "
                     "itself is being applied unequally, which is worse than a rate difference"
                     % (ma, mb))
    else:
        notes.append("per-space budget: %s vs %s over closed spaces (modal count agrees at %s)"
                     % (ca, cb, ma))

    # THE THIRD QUESTION, and the one this checker's own opening paragraph promised but never asked:
    # equal per-space budgets do not make the TOTAL budgets equal. `trials_per_space` is charged PER
    # SPACE, and a K expansion publishes a SECOND space over the same candidate -- so each expansion
    # hands that arm another whole `trials_per_space` of search. On the live S7 pair the treatment arm
    # closed 3 spaces (3x40 = 120 trials) while the control arm closed 4 and opened a 5th (5x40 = 200)
    # off the back of 2 expansions the treatment arm never made. Both arms passed the per-space budget
    # test at 40 and the rate test on the median, and the checker printed `expansions 0` and `2` as
    # decoration while concluding "a latency difference between these arms is attributable to the
    # switch". It is not: one arm simply got more search.
    #
    # THE UNIT IS ONE SPACE, not a percentage I would have to pick: the size of a space is the modal
    # budget this same function just measured, so the threshold comes from the data. But the comparison
    # is made in SPACE-EQUIVALENTS (`trials / unit`) rather than on the raw trial difference, because a
    # space that closed early contributes a FRACTION of a space of search -- 1 trial is 1/40 of one, not
    # one. Comparing raw totals against `unit` made a 40-trial shortfall read as 39 and go silent, which
    # is how the arm that lost a whole space to the wall clock would have passed.
    #
    # _SPACE_TOL is below 1.0 for exactly that reason: it must catch a space's worth of search that was
    # truncated rather than never opened. It is not a sensitivity knob -- it is the allowance for the
    # few trials of slack each closed space can carry (a timeout still counts as a trial, so 41 against
    # a nominal 40 is normal), and it must stay under 1 or a truncated space hides in the rounding.
    #
    # WHY IT ONLY FAILS AT THE END. The trailing arm keeps opening spaces, so mid-run the totals differ
    # by construction on any pair that is merely out of step. Before RUN_FINISHED this reports the
    # asymmetry and names its mechanism without flipping the verdict -- the same discipline as
    # "NOT YET ANSWERABLE" above, and the reason a live snapshot cannot be quoted as a confound.
    unit = ma if (ma is not None and ma == mb) else None
    sa, sb = len(a["per_space"]), len(b["per_space"])
    ta, tb = a["trials"], b["trials"]
    ea, eb = a["expansions"], b["expansions"]
    gap_spaces = abs(ta - tb) / unit if unit else 0.0
    if unit and gap_spaces >= _SPACE_TOL:
        mech = ""
        if ea != eb:
            mech = (" %s made %d K expansion(s) against %d, and each expansion publishes another space "
                    "over the same candidate => another %d trials of budget" %
                    (la if ea > eb else lb, max(ea, eb), min(ea, eb), unit))
        msg = ("TOTAL SEARCH DIFFERS: %d vs %d trials over %d vs %d spaces -- a gap of %d trials, "
               "which is %.1f whole space(s) at the modal budget of %d.%s The per-space budget being "
               "equal does not make the arms' total search equal, and the arm with more search has an "
               "advantage no switch accounts for." %
               (ta, tb, sa, sb, abs(ta - tb), gap_spaces, unit, mech))
        if live:
            notes.append("PROVISIONAL, " + msg + " Mid-run the trailing arm may still catch up, so "
                         "this is recorded rather than judged; re-run at the end.")
        else:
            okay = False
            notes.append(msg)
    elif unit:
        notes.append("total search: %d vs %d trials over %d vs %d spaces (%.2f space-equivalents "
                     "apart, tolerance %.2f); K expansions %d vs %d"
                     % (ta, tb, sa, sb, gap_spaces, _SPACE_TOL, ea, eb))

    ra, rb = a["trials_per_h"], b["trials_per_h"]
    if ra > 0 and rb > 0:
        ratio = max(ra, rb) / min(ra, rb)
        faster = la if ra > rb else lb
        if ratio > _RATE_TOL:
            # Is the rate gap a TAIL (a few pathological configs, a property of what TPE sampled)
            # or a shifted BODY (every trial slower, a property of the box)? Only the second
            # compounds over a 12 h run. The discriminator is the MEDIAN gap, which is robust to a
            # bounded number of outliers by construction -- no threshold to tune. A first draft used
            # `tail_share > 0.5` and it failed on both sides: the live pair read 68% (correct by
            # luck) while a synthetic single-timeout arm read 48% and was misreported as a confound,
            # because tail_share depends on how many trials the run happens to have.
            gm_a, gm_b = a["gap_median"], b["gap_median"]
            body = max(gm_a, gm_b) / min(gm_a, gm_b) if gm_a > 0 and gm_b > 0 else None
            worse = a if ra < rb else b
            if body is not None and body <= _RATE_TOL:
                notes.append(
                    "rate %.1f vs %.1f trials/h (%.2fx, %s faster) -- but the MEDIAN gap is %.1f vs "
                    "%.1f s (%.2fx, within tolerance), so the body of the distribution agrees and "
                    "this is a TAIL: %s's slowest single trial was %.0f s (%s). A tail does not "
                    "compound over the wall clock"
                    % (ra, rb, ratio, faster, gm_a, gm_b, body,
                       la if ra < rb else lb, worse["worst"][0][0], worse["worst"][0][1]))
            else:
                okay = False
                notes.append(
                    "RATE DIFFERS: %.1f vs %.1f trials/h (%.2fx, %s faster) and the MEDIAN gap "
                    "differs too (%.1f vs %.1f s, %s) => the body of the distribution is shifted, "
                    "not a few bad configs, so the faster arm fits ~%.0f%% more search into the "
                    "same wall clock"
                    % (ra, rb, ratio, faster, gm_a, gm_b,
                       "%.2fx" % body if body is not None else "one side has no gaps",
                       100 * (ratio - 1)))
        else:
            notes.append("rate %.1f vs %.1f trials/h (%.2fx, within %.0f%%): the wall clock buys "
                         "comparable search" % (ra, rb, ratio, 100 * (_RATE_TOL - 1)))

    # An unequal ACCEPTED-expansion count has two very different explanations, and the checker
    # already prints the accepted count as though it had one. Say which: if both arms were refused
    # the same number of times for the same reason, the gap is in how many candidates reached
    # closure (a candidate-supply difference); if the refusals themselves differ, the expander is
    # behaving differently between arms, which a switch could be responsible for.
    fa, fb = a["expansions_refused"], b["expansions_refused"]
    if ea != eb or fa or fb:
        same_reasons = a["expansion_refusals"] == b["expansion_refusals"]
        msg = ("K expansions ACCEPTED %d vs %d, REFUSED %d vs %d (%s vs %s)"
               % (ea, eb, fa, fb,
                  a["expansion_refusals"] or "-", b["expansion_refusals"] or "-"))
        if fa == fb and same_reasons and ea != eb:
            msg += (". Equal refusals for the same reason, so the accepted gap is a difference in "
                    "how many CANDIDATES reached closure, not in what the expander allowed")
        elif not same_reasons:
            msg += (". The refusal REASONS differ between arms -- check whether the switch changed "
                    "what the expander is permitted to do before reading the accepted counts")
        notes.append(msg)
    return okay, notes


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("control", type=Path)
    ap.add_argument("treatment", type=Path)
    args = ap.parse_args(argv)

    rows = [("CONTROL", read_arm(args.control)), ("TREATMENT", read_arm(args.treatment))]
    for lbl, r in rows:
        print("%-10s span %.2f h   %d trials (%d ok) over %d spaces   %.1f trials/h" % (
            lbl, r["span_h"], r["trials"], r["ok"], r["spaces"], r["trials_per_h"]))
        print("%-10s rewrites landed %d   family rounds CLOSED %d   space expansions %d   "
              "agent %.0f%% of wall clock" % (
                  "", r["rewrites"], r["rounds"], r["expansions"], 100 * r["agent_frac"]))
        print("%-10s failure kinds %s" % ("", r["kinds"]))
        print("%-10s inter-event gap: median %.1f s, sum %.2f h, top 10%% hold %.0f%%" % (
            "", r["gap_median"], r["gap_sum_h"], 100 * r["tail_share"]))
        print("%-10s   (gaps are NOT per-trial costs: max_shared_jobs=2 overlaps compile with "
              "timing)" % "")
        # Spelled out because the short form above did not stop me from quoting a 1854 s gap as a trial
        # that had breached the 1800 s deadline. It had not: the pair's slowest actual trial was
        # 1733.8 s and `job_timed_out` was false on all 255 trials. A gap spans whatever else ran in
        # the shared channel, so it is an UPPER BOUND on the trial's cost, never the cost itself.
        print("%-10s   -- a gap is an upper bound; for a trial's own cost read `job_wall_s` on the "
              "trial record, and never compare a gap against the deadline" % "")
        for g, fk, vals in r["worst"][:2]:
            keep = {k: vals.get(k) for k in ("COMPUTE_DTYPE", "DOT_MODE") if k in vals}
            print("%-10s   slowest: %.0f s  %-14s %s" % ("", g, fk, keep))
        print()

    print("=" * 78)
    okay, notes = parity_verdict(rows[0][1], rows[1][1], "control", "treatment")
    for n in notes:
        print("  * %s" % n)
    print()
    if okay:
        # A bare "PARITY OK" under a note that begins "PROVISIONAL, TOTAL SEARCH DIFFERS" is the kind of
        # line that gets quoted on its own. Mid-run findings are deliberately not judged (the trailing
        # arm may catch up), but the closing line must not read as though nothing was found.
        pending = [n for n in notes if n.startswith("PROVISIONAL,")]
        if pending:
            print("PARITY OK ON WHAT IS ANSWERABLE NOW -- but %d finding(s) above are PROVISIONAL and "
                  "would fail this check at the end of the run. Do not quote this line without them."
                  % len(pending))
            return 0
        print("PARITY OK -- a latency difference between these arms is attributable to the switch.")
        return 0
    print("PARITY NOT OK -- record this beside the J2-5 verdict. A difference in the final result "
          "has more than one explanation, and the fix is not a tighter tolerance: it is reporting "
          "the confound alongside the number.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
