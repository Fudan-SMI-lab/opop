"""Item 2 acceptance: run the SHIPPING `find_soft_walls` over the real corpus.

WHY THIS EXISTS SEPARATELY FROM THE UNIT TESTS. The unit tests fix the criterion's behaviour on
constructed curves. This asks a different question -- what does the shipping function actually produce
on the trials the framework really ran -- and it is the question the previous exploratory probe
(`soft_wall_shape.py`) answered with a REIMPLEMENTATION of the criterion. A number produced by a
reimplementation says nothing about the code that will run: `a-test-that-copies-the-loop-does-not-test-it`.

FOUR NUMBERS ARE REPORTED, and three of them are denominators:

  applicable        candidates whose BEST trial spills at all. Measured before: 24 of 56. A candidate
                    whose winner does not spill has no wall AT THE POINT THE AGENT REWRITES FROM, and
                    that is a structural non-event, not a negative result.
  walls             (candidate, knob) pairs that pass all three conditions.
  candidates hit    how many candidates carry at least one.
  hard walls        the SAME candidates replayed through the shipping `find_walls` -- the comparison
                    that the analysis document lists as acceptance item #1, because I nearly claimed
                    "soft walls find an order of magnitude more" from numbers measured on two DIFFERENT
                    corpora. On this corpus the hard wall has no input at all (every local run predates
                    the shared-memory screen, so `infeasible_shared_memory` appears zero times), and
                    the honest output is that the ratio is UNDEFINED here and has to be measured on one
                    run with both mechanisms live.
"""
from __future__ import annotations

import argparse
import statistics
import sys
from collections import defaultdict
from pathlib import Path

SRC = Path(__file__).resolve().parents[2] / "src"
sys.path.insert(0, str(SRC))

from kernel_optimizer.evaluation import soft_wall, wall_attribution  # noqa: E402
from kernel_optimizer.models.reports import ParamStat, TuningStats  # noqa: E402
from kernel_optimizer.store import read as store_read  # noqa: E402


def _stats_for(trials: list[dict], cid: str) -> TuningStats:
    """A minimal TuningStats carrying the knob NAMES, which is all `find_soft_walls` reads from it.

    Rebuilt from the trials rather than taken from `STATS_DONE`, deliberately: a candidate whose space
    was EXPANDED emits several STATS_DONE records and any one of them names a subset of the knobs, so
    the scan would silently skip the knobs added by the expansion.
    """
    names: set[str] = set()
    for t in trials:
        names.update(((t.get("params") or {}).get("values") or {}).keys())
    return TuningStats(
        candidate_id=cid, space_id="sp", n_complete=len(trials), n_fail=0,
        param_stats=[ParamStat(name=n, best_value="?") for n in sorted(names)])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    out: list[str] = []

    def say(s: str = "") -> None:
        out.append(s)

    tot = defaultdict(int)
    gains: list[float] = []
    examples: list[str] = []
    per_run: list[tuple[str, dict]] = []

    for rd in args.runs:
        try:
            events = store_read.read_events(rd)
        except Exception as exc:  # noqa: BLE001
            say(f"!! {rd}: {type(exc).__name__}: {exc}")
            continue
        by_cand: dict[str, list[dict]] = defaultdict(list)
        refused: dict[str, list[dict]] = defaultdict(list)
        for ev in events:
            if ev.get("type") != "TRIAL_DONE":
                continue
            t = store_read.trial_of(ev)
            cid = str(t.get("candidate_id") or "")
            if not cid:
                continue
            by_cand[cid].append(t)
            if t.get("failure_kind") == "infeasible_shared_memory":
                vals = (t.get("params") or {}).get("values")
                if vals:
                    refused[cid].append(vals)

        c = defaultdict(int)
        for cid, trials in sorted(by_cand.items()):
            done = [t for t in trials if t.get("status") == "complete"]
            if len(done) < 8:
                continue
            c["candidates"] += 1
            # ---- the SHIPPING soft-wall criterion, not a copy of it
            scan = soft_wall.find_soft_walls(_stats_for(done, cid), trials)
            if scan.applicable:
                c["applicable"] += 1
                c["walls"] += len(scan.walls)
                if scan.walls:
                    c["candidates_hit"] += 1
                    gains.extend(w.tail_gain_pct for w in scan.walls)
                    if len(examples) < 10:
                        w = scan.walls[0]
                        examples.append(
                            f"  {Path(rd).name[:26]:26s} {cid[:13]} {w.param:16s} "
                            f"spills {w.spills_by_value[0]:.0f}->{w.spills_by_value[-1]:.0f} "
                            f"over {len(w.ran_values)} values, tail {w.tail_gain_pct:+.1f}%, "
                            f"limiter={w.limiter_at_best}")
                else:
                    c["applicable_no_wall"] += 1
            else:
                c["not_applicable"] += 1
                if "does not spill" in scan.reason:
                    c["na_winner_zero_spill"] += 1
                else:
                    c["na_other"] += 1
            # ---- the hard wall on the SAME candidate, same denominator
            if refused.get(cid):
                c["candidates_with_refusals"] += 1
                hard = wall_attribution.find_walls(_stats_for(done, cid), refused[cid])
                worthy, _ = wall_attribution.select_for_probing(hard, 8)
                c["hard_walls_found"] += len(hard)
                c["hard_probe_worthy"] += len(worthy)
                if worthy:
                    c["hard_candidates_hit"] += 1
        per_run.append((Path(rd).name, dict(c)))
        for k, v in c.items():
            tot[k] += v

    say("=" * 100)
    say("ITEM 2 ACCEPTANCE -- the SHIPPING find_soft_walls on the real corpus")
    say("=" * 100)
    say("(the shipping function, not a reimplementation of its criterion: a number produced by a")
    say(" copy of the logic says nothing about the code that will run)")
    say()
    say("-" * 100)
    say(f"{'run':28s} {'cands':>6s} {'applic':>7s} {'walls':>6s} {'hit':>5s} "
        f"{'na:0spill':>10s} {'refusals':>9s} {'hardW':>6s}")
    say("-" * 100)
    for name, c in per_run:
        say(f"{name[:28]:28s} {c.get('candidates', 0):6d} {c.get('applicable', 0):7d} "
            f"{c.get('walls', 0):6d} {c.get('candidates_hit', 0):5d} "
            f"{c.get('na_winner_zero_spill', 0):10d} "
            f"{c.get('candidates_with_refusals', 0):9d} {c.get('hard_probe_worthy', 0):6d}")
    say("-" * 100)
    say(f"{'TOTAL':28s} {tot['candidates']:6d} {tot['applicable']:7d} {tot['walls']:6d} "
        f"{tot['candidates_hit']:5d} {tot['na_winner_zero_spill']:10d} "
        f"{tot['candidates_with_refusals']:9d} {tot['hard_probe_worthy']:6d}")
    say()

    n = tot["candidates"]
    if n:
        say(f"APPLICABILITY: {tot['applicable']}/{n} = {100.0 * tot['applicable'] / n:.1f}% of")
        say(f"  candidates have a spilling best trial. The other {tot['not_applicable']} are NOT a")
        say(f"  negative result: {tot['na_winner_zero_spill']} have a winner that does not spill at")
        say(f"  all, so no knob can be walled by spilling at the point the agent rewrites from.")
        say()
        say("  ** APPLICABILITY IS NOT A CONSTANT, AND THE EARLIER FIGURE WAS FROM A DIFFERENT")
        say("     CORPUS. ** The soft-wall analysis recorded 43% (24 of 56 candidates) from FIVE")
        say("     backed-up runs spanning a 4090 and an A800; this run of the shipping function")
        say(f"     over {n} candidates in the local corpus gives "
            f"{100.0 * tot['applicable'] / n:.1f}%. Both are real; they are not the same")
        say("     population, and quoting one as the other is the cross-corpus error this project")
        say("     has already recorded. What travels is the SHAPE of the result -- a large share of")
        say("     winners do not spill, so the detector's denominator must always be published --")
        say("     not the percentage. Per-task it splits further: on the L3:48 runs here 0 of 8")
        say("     candidates are applicable at all.")
        say()
        if tot["applicable"]:
            say(f"AMONG THE APPLICABLE: {tot['candidates_hit']}/{tot['applicable']} carry at least "
                f"one wall ({100.0 * tot['candidates_hit'] / tot['applicable']:.1f}%), "
                f"{tot['walls']} (candidate, knob) pairs in total;")
            say(f"  {tot['applicable_no_wall']} spill at the optimum but no knob's curve rises "
                f"monotonically with latency still improving -- 'no wall' as distinct from "
                f"'not applicable'.")
        if gains:
            say(f"  tail gain of the walls found: median {statistics.median(gains):+.1f}%, "
                f"max {max(gains):+.1f}%, min {min(gains):+.1f}%")
    say()
    # Per task, because the pooled rate hides a 0% task. Pooling classes with different base rates
    # has produced a two-way Simpson reversal in this project before, and a mechanism that never fires
    # on one of the three benchmark tasks is a fact a reader needs before planning an experiment on it.
    say("BY TASK (the pooled rate hides a task where the mechanism never fires):")
    by_task: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for name, c in per_run:
        parts = name.split("-")
        task = f"{parts[1]}:{parts[2]}" if len(parts) > 2 else name
        for k in ("candidates", "applicable", "walls", "candidates_hit"):
            by_task[task][k] += c.get(k, 0)
    for task in sorted(by_task):
        t = by_task[task]
        if not t["candidates"]:
            continue
        say(f"  {task:8s} candidates {t['candidates']:3d}   applicable {t['applicable']:3d} "
            f"({100.0 * t['applicable'] / t['candidates']:5.1f}%)   "
            f"walls {t['walls']:3d}   candidates hit {t['candidates_hit']:3d}")
    say()
    say("HARD vs SOFT ON THIS SAME CORPUS (acceptance item #1 of the analysis document):")
    say(f"  candidates with any shared-memory refusal recorded: {tot['candidates_with_refusals']}")
    say(f"  hard walls found: {tot['hard_walls_found']}, probe-worthy: {tot['hard_probe_worthy']}, "
        f"candidates hit: {tot['hard_candidates_hit']}")
    if not tot["candidates_with_refusals"]:
        say("  => THE RATIO IS UNDEFINED ON THIS CORPUS, and that is the finding, not a gap to fill")
        say("     with a number from elsewhere. Every locally backed-up run predates the")
        say("     shared-memory screen, so the hard wall's ONLY input -- trials recorded as")
        say("     `infeasible_shared_memory` -- is absent. I nearly wrote 'soft walls find an order")
        say("     of magnitude more' from a soft count on THIS corpus against a hard count from a")
        say("     DIFFERENT one; comparing across corpora is a mistake this project has recorded")
        say("     (`box1-and-box4-have-different-cpus-so-trial-counts-are-not-comparable`).")
        say("     The comparison has to be re-run on one run with both mechanisms live.")
    if examples:
        say()
        say("EXAMPLES (highest-slope wall per candidate, first 10):")
        out.extend(examples)

    text = "\n".join(out) + "\n"
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
        print(f"wrote {args.out}")
    else:
        sys.stdout.buffer.write(text.encode("utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
