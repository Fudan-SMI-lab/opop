"""S7 acceptance: run the SHIPPING SlopeGuide over the real corpus and report what it would have done.

WHY THIS EXISTS SEPARATELY FROM `s7_feasibility_replay.py`. The feasibility replay asked whether a WALL is
available early and stable enough to act on (yes at 74%, stable at 53%). It says nothing about whether
this implementation can turn such a wall into a legal, useful, non-redundant point -- and every one of the
gates between "a wall exists" and "a trial is enqueued" can silently swallow all of them:

  * the walled knob's values may ALL be drawn already, in which case there is nothing to propose;
  * the only undrawn choices may lie past the refusal, or on the wrong side of the optimum;
  * the proposed point may be one the tuner has already asked;
  * the incumbent may not be a point OF the space being tuned (after an expansion re-tune).

So the number this reports is the FIRING RATE of the shipping mechanism on real data, decomposed by which
gate declined -- not a performance claim, which only the 12h pair can make.

WHAT IT REPLAYS, and the discipline is the same as the feasibility probe's. Each candidate's trials are
taken IN THE EVENT LOG'S ORDER, and at every `recompute_every` boundary the per-choice median table is
rebuilt FROM THE PREFIX and handed to the shipping `SlopeGuide.suggest`. Using `STATS_DONE` instead would
tell the replay what the sampler could not have known -- the wall would look available at trial 10 because
its evidence arrived at trial 80.

WHAT IT CANNOT SAY, stated because the gap is the whole reason the 12h pair exists. A replay knows what the
trials WERE; it cannot know what the sampler would have drawn afterwards had a point been inserted. So
"would this have found a better configuration" is unanswerable here by construction, and every number below
is about the mechanism firing, not about it helping.

The HARD criterion is additionally unanswerable on this host for a second, independent reason: 0 of 151
local candidates carry an `infeasible_shared_memory` record, because every locally backed-up run predates
the shared-memory screen. The probe reports that absence explicitly rather than printing a zero that could
be read as "the mechanism never fires".
"""
from __future__ import annotations

import argparse
import statistics
import sys
from collections import defaultdict
from pathlib import Path

SRC = Path(__file__).resolve().parents[2] / "src"
sys.path.insert(0, str(SRC))

from kernel_optimizer.models.core import (  # noqa: E402
    ParamDomain,
    ParameterSpace,
    ParamSet,
)
from kernel_optimizer.models.reports import ParamStat, TuningStats  # noqa: E402
from kernel_optimizer.store import read as store_read  # noqa: E402
from kernel_optimizer.tuning import slope_guide as sg  # noqa: E402


def _stats_from(trials: list[dict], cid: str) -> TuningStats:
    """The per-choice MEDIAN table, rebuilt from a PREFIX of the trials.

    Rebuilt rather than taken from `STATS_DONE` for the reason in the module docstring. Median rather than
    mean because that is the framework's own objective (93.2% rank-correctness against 64.8%).
    """
    by_knob: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for t in trials:
        if t.get("status") != "complete":
            continue
        ms = store_read.latency_ms_of(t)
        if ms is None:
            continue
        for knob, raw in (((t.get("params") or {}).get("values")) or {}).items():
            by_knob[str(knob)][str(raw)].append(ms)
    out = []
    for knob, table in sorted(by_knob.items()):
        med = {v: statistics.median(xs) for v, xs in table.items()}
        if not med:
            continue
        best = min(med, key=lambda v: med[v])
        out.append(ParamStat(name=knob, best_value=best, latency_by_value=med))
    return TuningStats(candidate_id=cid, space_id="sp",
                       n_complete=sum(1 for t in trials if t.get("status") == "complete"),
                       n_fail=sum(1 for t in trials if t.get("status") != "complete"),
                       param_stats=out)


def _space_from(trials: list[dict], cid: str,
                published: dict[str, ParameterSpace] | None = None) -> ParameterSpace:
    """The candidate's space: the PUBLISHED one when the log has it, else reconstructed from draws.

    PREFER THE PUBLISHED SPACE, and the difference is not cosmetic. `SPACE_PUBLISHED` carries the full
    declared `domains`, including choices the tuner never drew -- and "a choice that exists but was never
    drawn" is exactly what this mechanism proposes. Reconstructing from draws makes every such choice
    invisible, so `_toward_wall` finds nothing and the knob is skipped for "no undrawn value". Measured on
    box 4's run, where the log DOES carry the spaces: 9 knobs were skipped that way before this function
    read them.

    The reconstruction stays as the fallback because the local 19-run corpus predates the event. Its
    direction of error has to be stated wherever it is used: choices are a SUBSET of what was declared, so
    every "no undrawn value" it reports is an upper bound and the firing rate a LOWER bound.
    """
    if published and cid in published:
        return published[cid]
    seen: dict[str, set] = defaultdict(set)
    for t in trials:
        for knob, raw in (((t.get("params") or {}).get("values")) or {}).items():
            seen[str(knob)].add(raw)
    domains = []
    for knob, vals in sorted(seen.items()):
        try:
            ordered = sorted(vals, key=lambda v: (isinstance(v, str), v))
        except TypeError:
            ordered = sorted(vals, key=repr)
        kind = "str" if any(isinstance(v, str) for v in vals) else "int"
        domains.append(ParamDomain(name=knob, kind=kind, choices=list(ordered)))
    return ParameterSpace(space_id="sp", candidate_id=cid, source_sha="0" * 8, domains=domains)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--every", type=int, default=10)
    ap.add_argument("--cap", type=int, default=2)
    ap.add_argument("--soft", action="store_true",
                    help="also let the spill wall steer, as v3.slope_guide.use_soft_wall does")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    out: list[str] = []

    def say(s: str = "") -> None:
        out.append(s)

    n_cands = 0
    n_refusing = 0
    n_fired = 0
    per_cand: list[tuple[str, str, int, int, int, dict]] = []
    totals: dict[str, int] = defaultdict(int)
    first_fire_frac: list[float] = []
    gains: list[float] = []
    knob_counts: dict[str, int] = defaultdict(int)
    source_counts: dict[str, int] = defaultdict(int)

    for rd in args.runs:
        try:
            events = store_read.read_events(rd)
        except Exception as exc:  # noqa: BLE001
            say(f"!! {rd}: {type(exc).__name__}: {exc}")
            continue
        seq: dict[str, list[dict]] = defaultdict(list)
        # The DECLARED space per candidate, when the log carries it. A candidate can publish more than
        # one (an expansion re-tune republishes), and the LAST one is the space its final trials ran
        # under -- which is the one whose choices those trials' walls have to be resolved against.
        published: dict[str, ParameterSpace] = {}
        n_published = 0
        for ev in events:
            if ev.get("type") == "SPACE_PUBLISHED":
                raw = (ev.get("payload") or ev).get("space") or {}
                try:
                    sp = ParameterSpace.model_validate(raw)
                except Exception:  # noqa: BLE001
                    continue
                published[str(sp.candidate_id)] = sp
                n_published += 1
                continue
            if ev.get("type") != "TRIAL_DONE":
                continue
            t = store_read.trial_of(ev)
            cid = str(t.get("candidate_id") or "")
            if cid:
                seq[cid].append(t)
        totals["n_spaces_published"] += n_published

        for cid, trials in sorted(seq.items()):
            if len(trials) < args.every:
                continue
            n_cands += 1
            if cid in published:
                totals["n_declared_space"] += 1
            if any(t.get("failure_kind") == "infeasible_shared_memory" for t in trials):
                n_refusing += 1
            space = _space_from(trials, cid, published)
            guide = sg.SlopeGuide(space=space, recompute_every=args.every,
                                  max_enqueued_per_recompute=args.cap,
                                  use_soft_wall=args.soft)
            n_sugg = 0
            first_at: int | None = None
            # The SAME dedup set the tuner keeps: every point already asked, so a suggestion cannot be a
            # re-measurement. Built from the prefix, like everything else here.
            drawn_keys: set[str] = set()
            for i, t in enumerate(trials, start=1):
                vals = ((t.get("params") or {}).get("values")) or {}
                if vals:
                    drawn_keys.add(ParamSet(values=vals).key())
                if not guide.due(i):
                    continue
                prefix = trials[:i]
                got = guide.suggest(_stats_from(prefix, cid), prefix, set(drawn_keys))
                if got and first_at is None:
                    first_at = i
                for s in got:
                    n_sugg += 1
                    gains.append(s.tail_gain_pct)
                    knob_counts[s.knob] += 1
                    source_counts[s.source] += 1
            snap = guide.snapshot()
            for key in ("n_recomputes", "n_skipped_no_wall", "n_skipped_no_value_toward_wall",
                        "n_skipped_already_proposed", "n_skipped_incomplete_incumbent"):
                totals[key] += snap[key]
            totals["n_suggested"] += n_sugg
            if n_sugg:
                n_fired += 1
                if first_at is not None:
                    first_fire_frac.append(first_at / len(trials))
            per_cand.append((Path(rd).name, cid, len(trials), snap["n_recomputes"], n_sugg, snap))

    say("=" * 100)
    say("S7 ACCEPTANCE -- the SHIPPING SlopeGuide replayed over each candidate's real trial ORDER")
    say(f"  recompute_every={args.every}  max_enqueued_per_recompute={args.cap}  "
        f"use_soft_wall={args.soft}")
    say("=" * 100)
    say("Per-choice medians are rebuilt FROM THE PREFIX at every recompute, so the mechanism sees only")
    say("what the sampler could have seen. The dedup set is the prefix's own asked points.")
    say()
    say("WHAT THIS CANNOT ANSWER, by construction: a replay knows what the trials WERE, not what the")
    say("sampler would have drawn after a point was inserted. So this is a FIRING RATE, not a")
    say("performance claim -- the 12h pair is the only thing that can price the mechanism.")
    say()
    say(f"candidates with >= {args.every} trials: {n_cands}")
    say(f"  ...of which any trial was refused for shared memory: {n_refusing}")
    if not n_refusing and not args.soft:
        say()
        say("  HARD CRITERION UNANSWERABLE ON THIS CORPUS. `find_walls`' only input is the refused")
        say("  parameter sets, and there are none: every locally backed-up run predates the")
        say("  shared-memory screen. The zero below is the absence of INPUT, not a property of the")
        say("  mechanism. Re-run with --soft for the criterion this corpus can exercise.")
    say()
    say(f"recomputes performed:            {totals['n_recomputes']}")
    say(f"candidates where it EVER fired:  {n_fired}/{n_cands}"
        + (f" = {100.0 * n_fired / n_cands:.1f}%" if n_cands else ""))
    say(f"points it would have enqueued:   {totals['n_suggested']}")
    if first_fire_frac:
        say(f"first firing, as a fraction of the candidate's own sequence: median "
            f"{statistics.median(first_fire_frac):.2f}  "
            f"(0.50 or less means budget remained to spend on it)")
    if gains:
        say(f"tail gain of the acted-on knob: median {statistics.median(gains):+.1f}%  "
            f"range {min(gains):+.1f}% .. {max(gains):+.1f}%")
    say()
    say("WHICH GATE DECLINED (these are the numbers a P4 reading needs). NOTE THE TWO DENOMINATORS:")
    say(f"  no actionable wall at all, PER RECOMPUTE:      {totals['n_skipped_no_wall']}"
        f" of {totals['n_recomputes']}")
    say(f"  no undrawn value on the walled side, PER KNOB: {totals['n_skipped_no_value_toward_wall']}")
    say(f"  the point was already asked, PER KNOB:         {totals['n_skipped_already_proposed']}")
    say(f"  incumbent not a point of the space, PER RECOMPUTE: "
        f"{totals['n_skipped_incomplete_incumbent']}")
    say("  The per-knob counters can exceed the number of recomputes, and a recompute whose only")
    say("  walled knob had nothing undrawn increments BOTH -- the first because the row was dropped")
    say("  before the wall list was tested for emptiness. So they are not a partition and must not be")
    say("  summed.")
    say()
    n_decl = totals["n_declared_space"]
    say(f"WHERE THE CHOICE LISTS CAME FROM: {n_decl} of {n_cands} candidates had a published")
    say("  `SPACE_PUBLISHED` and were resolved against their DECLARED domains.")
    if n_decl < n_cands:
        say(f"  The remaining {n_cands - n_decl} were RECONSTRUCTED from drawn values, so their choice")
        say("  lists are a SUBSET of what was declared: for those candidates every 'no undrawn value'")
        say("  above is an upper bound and the firing rate a LOWER bound -- the conservative direction.")
    if n_decl:
        say("  For the declared ones there is no such bias: a choice that exists but was never drawn is")
        say("  exactly what this mechanism proposes, and reconstruction hides it.")
    if knob_counts:
        say()
        say("KNOBS IT WOULD HAVE PUSHED (top 12):")
        for knob, n in sorted(knob_counts.items(), key=lambda kv: -kv[1])[:12]:
            say(f"  {n:5d}  {knob}")
        say(f"  by criterion: {dict(sorted(source_counts.items()))}")
    fired_rows = [r for r in per_cand if r[4]]
    if fired_rows:
        say()
        say("PER CANDIDATE, where it fired (first 25):")
        say("-" * 100)
        say(f"{'run':24s} {'candidate':15s} {'trials':>7s} {'recomp':>7s} {'enq':>5s}   knobs")
        say("-" * 100)
        for run, cid, n, nrec, nsug, snap in fired_rows[:25]:
            knobs = ",".join(sorted(snap["knobs_pushed"]))[:38]
            say(f"{run[:24]:24s} {cid[:15]:15s} {n:7d} {nrec:7d} {nsug:5d}   {knobs}")
        say("-" * 100)
    say()
    say("HOW TO READ THIS FOR THE S7 DECISION:")
    say("  a low firing rate with 'no wall to act on' dominant  => the criterion, not the wiring, is")
    say("     the binding constraint, and S7 inherits 2e's coverage problem rather than fixing it")
    say("  a low rate with 'no undrawn value' dominant          => the mechanism is redundant with")
    say("     naive coverage: the sampler had already measured everything on the walled side")
    say("  a healthy rate with an early median first-firing     => there is room for the 12h pair to")
    say("     measure something, and P1-P4 become answerable")

    text = "\n".join(out) + "\n"
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
        print(f"wrote {args.out}")
    else:
        sys.stdout.buffer.write(text.encode("utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
