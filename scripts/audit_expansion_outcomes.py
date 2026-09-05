"""Audit: what does a K expansion (improvement K) actually do to the reported best?

Written after I asserted -- twice, wrongly -- that a flat re-tune is "the default outcome"
and that a particular 28.6 -> 28.0 was "the first expansion to improve a best". Both claims
came from generalizing over the three cases I happened to have investigated, which were
selected BECAUSE they were flat. This script prints the denominator so that mistake is a
command away from being caught.

It also flags which post-expansion bests were CACHE REPLAYS of the pre-expansion winner
(reused_measurement), because a flat outcome is uninformative about the widened region
whether or not it was replayed.

And it ATTRIBUTES each expansion: among freshly measured trials, was the best reached by a
value K ADDED, or by one already in the domain? A re-tune re-runs 40 trials over the whole
space, so it can improve the reported best without the widened region contributing anything
-- which makes improved% an UPPER BOUND on K's contribution and the attribution the honest
per-case test. Measured so far: 71% improved, but only 48% attributable to an added value.

The PER-KNOB section goes one level finer, because an expansion widens several knobs at once
and is scored as ONE number -- so a productive widening and a counterproductive one inside the
same expansion are indistinguishable in the headline. It compares MEDIANS (not minima: the
"did not use it" pool is ~5x larger, median 19 vs 4 samples, so min() over it is lower almost
mechanically -- a min-based version reported 73% "hurt" purely from that asymmetry versus 46%
on medians), plus the size-independent question of whether the expansion's own winner used the
knob's added value. Splitting by direction and knob kind is where the signal lives: tile sizes
widened UPWARD help 76%, warp counts help 2 of 16 and 0 of 7 downward.

See docs/finding-k-retune-cannot-disconfirm-its-incumbent.md
    docs/measurement-per-knob-expansion-attribution.md

Usage:  python scripts/audit_expansion_outcomes.py [runs/run-... ...]
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from statistics import median
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def main(argv: list[str]) -> int:
    runs = [Path(a) for a in argv] or sorted(REPO.glob("runs/run-l3-*"))
    outcomes: Counter[str] = Counter()
    attributed: Counter[str] = Counter()
    cached_by_outcome: Counter[str] = Counter()
    per_knob: Counter[str] = Counter()
    winner_used: Counter[str] = Counter()
    by_direction: Counter[tuple] = Counter()
    by_kind: Counter[tuple] = Counter()
    rows: list[tuple] = []
    knob_rows: list[tuple] = []

    for run in runs:
        ev_path = run / "events.jsonl"
        if not ev_path.exists():
            continue
        events = [json.loads(line) for line in ev_path.read_text(encoding="utf-8").splitlines()
                  if line.strip()]

        seq: dict[str, list[tuple[str, float | None]]] = defaultdict(list)
        complete: dict[tuple[str, str], list[dict]] = defaultdict(list)
        domains: dict[str, list[tuple[str, dict]]] = defaultdict(list)
        cached: set[str] = set()
        for e in events:
            if e["type"] == "TRIAL_DONE":
                t = e["payload"].get("trial") or e["payload"]
                if e["payload"].get("reused_measurement"):
                    cached.add(t.get("trial_id"))
                if t.get("status") == "complete" and t.get("latency_ms"):
                    complete[(t.get("candidate_id"), t.get("space_id"))].append(t)
            elif e["type"] == "TUNING_DONE":
                seq[e["payload"].get("candidate_id")].append(
                    (e["payload"].get("space_id"), e["payload"].get("best_ms")))
            elif e["type"] == "SPACE_PUBLISHED":
                sp = e["payload"].get("space") or {}
                if sp.get("candidate_id"):
                    domains[sp["candidate_id"]].append(
                        (sp.get("space_id"),
                         {d["name"]: d["choices"] for d in sp.get("domains", [])}))

        # space_id -> the values this space added relative to the space published before it
        added_by_space: dict[str, dict[str, list]] = {}
        prev_dom_of: dict[str, dict] = {}
        for cid, pubs in domains.items():
            for (_, prev_dom), (sid, dom) in zip(pubs, pubs[1:]):
                prev_repr = {k: {repr(c) for c in v} for k, v in prev_dom.items()}
                added = {k: [c for c in v if repr(c) not in prev_repr.get(k, set())]
                         for k, v in dom.items()}
                added_by_space[sid] = {k: v for k, v in added.items() if v}
                prev_dom_of[sid] = prev_dom

        for cid, spaces in seq.items():
            for (prev_sid, prev_ms), (sid, ms) in zip(spaces, spaces[1:]):
                if prev_ms is None or ms is None:
                    continue
                trials = complete.get((cid, sid), [])
                best = min(trials, key=lambda t: t["latency_ms"]["mean"]) if trials else None
                was_cached = bool(best and best.get("trial_id") in cached)
                if ms < prev_ms:
                    kind = "improved"
                elif ms == prev_ms:
                    kind = "flat"
                else:
                    kind = "worse"
                outcomes[kind] += 1
                if was_cached:
                    cached_by_outcome[kind] += 1
                pct = (prev_ms - ms) / prev_ms * 100.0

                # The attribution question: did a value K ADDED produce the new best, or did
                # the re-tune simply find a better point inside the original domain? Only
                # fresh (uncached) measurements count on either side.
                added = added_by_space.get(sid) or {}
                new_best = old_best = None
                if added:
                    for t in trials:
                        if t.get("trial_id") in cached:
                            continue
                        vals = t["params"]["values"]
                        used_new = any(vals.get(k) in v for k, v in added.items())
                        lat = t["latency_ms"]["mean"]
                        if used_new:
                            new_best = lat if new_best is None else min(new_best, lat)
                        else:
                            old_best = lat if old_best is None else min(old_best, lat)
                if new_best is not None and old_best is not None:
                    attributed[("new" if new_best < old_best else "old")] += 1

                # PER-KNOB attribution. An expansion widens several knobs at once and is
                # scored as one number, so a productive knob and a counterproductive one
                # inside the same expansion are indistinguishable in the headline.
                #
                # Compare MEDIANS, not minima. The "did not use it" pool is every other
                # trial and is typically 5x larger (median 19 vs 4 samples), so min() over
                # the larger pool is lower almost mechanically -- a min-based comparison
                # reports ~73% "hurt" purely from that asymmetry, versus 46% on medians.
                fresh = [t for t in trials if t.get("trial_id") not in cached]
                fresh_best = (min(fresh, key=lambda t: t["latency_ms"]["mean"])
                              if fresh else None)
                for knob, vals in added.items():
                    with_v = [t["latency_ms"]["mean"] for t in fresh
                              if t["params"]["values"].get(knob) in vals]
                    without_v = [t["latency_ms"]["mean"] for t in fresh
                                 if t["params"]["values"].get(knob) not in vals]
                    # Did the expansion's actual winner use one of this knob's new values?
                    # That is the decision the search acted on, independent of sample size.
                    used_by_winner = bool(
                        fresh_best and fresh_best["params"]["values"].get(knob) in vals)
                    if not with_v or not without_v:
                        knob_rows.append((run.name, cid, sid, knob, vals, None, None, None,
                                          len(with_v), len(without_v), used_by_winner))
                        continue
                    mw, mwo = median(with_v), median(without_v)
                    helped = mw < mwo
                    per_knob["helped" if helped else ("tied" if mw == mwo else "hurt")] += 1
                    winner_used["yes" if used_by_winner else "no"] += 1
                    # Direction (did the added values go above or below the old domain?) and
                    # what the knob controls. This is where the signal actually lives.
                    old_nums = [x for x in (prev_dom_of.get(sid) or {}).get(knob, [])
                                if isinstance(x, int) and not isinstance(x, bool)]
                    new_nums = [x for x in vals
                                if isinstance(x, int) and not isinstance(x, bool)]
                    if old_nums and new_nums:
                        direction = "UP" if max(new_nums) > max(old_nums) else "DOWN"
                        kind_of = ("WARPS" if "WARP" in knob else
                                   "STAGES" if "STAGE" in knob else "TILE")
                        by_direction[(direction, helped)] += 1
                        by_kind[(direction, kind_of, helped)] += 1
                    knob_rows.append((run.name, cid, sid, knob, vals, mw, mwo,
                                      (mwo - mw) / mwo * 100.0,
                                      len(with_v), len(without_v), used_by_winner))

                rows.append((run.name, cid, prev_sid, prev_ms, sid, ms, kind, was_cached, pct,
                             new_best, old_best))

    total = sum(outcomes.values())
    if not total:
        print("No K expansion with a before/after TUNING_DONE pair found.")
        return 0

    print(f"{total} K expansions with a before/after pair:\n")
    for kind in ("improved", "flat", "worse"):
        n = outcomes[kind]
        print(f"  {kind:9s} {n:3d}  ({n / total * 100:4.1f}%)   "
              f"post-expansion best was a cache replay: {cached_by_outcome[kind]}")

    n_attr = attributed["new"] + attributed["old"]
    if n_attr:
        print(f"\nATTRIBUTION -- of the {n_attr} expansions where both a new-choice and an "
              f"old-choice trial were freshly measured,\nthe best came from a value K ADDED in "
              f"{attributed['new']} ({attributed['new'] / n_attr * 100:.0f}%) and from the "
              f"original domain in {attributed['old']}.")
        print("A 'improved' row whose best came from the ORIGINAL domain is a benefit of "
              "re-running the\ntuner, not of widening it -- so the improved% above is an UPPER "
              "BOUND on K's contribution.")
    print()
    for row in sorted(rows, key=lambda r: (r[6], -r[8])):
        run, cid, ps, pm, s, m, kind, wc, pct, nb, ob = row
        tag = "  [best is a CACHE REPLAY]" if wc else ""
        if nb is not None and ob is not None:
            who = "new-choice WON" if nb < ob else "old-choice won"
            tag += f"  [fresh: new={nb} old={ob} -> {who}]"
        print(f"  {kind:9s} {pct:+6.1f}%  {run}  {cid}  {ps}={pm} -> {s}={m}{tag}")

    n_knob = sum(per_knob.values())
    if n_knob:
        print(f"\nPER-KNOB -- an expansion widens several knobs at once and is scored as ONE "
              f"number,\nso a productive knob and a counterproductive one in the same expansion "
              f"are\nindistinguishable in the headline. Comparing, per widened knob, the MEDIAN "
              f"of fresh\ntrials that used one of its added values against the median of those "
              f"that did not:\n")
        for verdict in ("helped", "tied", "hurt"):
            n = per_knob[verdict]
            print(f"  {verdict:7s} {n:3d}  ({n / n_knob * 100:4.1f}%)")
        n_w = sum(winner_used.values())
        print(f"\n  And the size-independent test -- did the expansion's own winning trial use "
              f"one of\n  the knob's added values?  YES {winner_used['yes']} "
              f"({winner_used['yes'] / n_w * 100:.0f}%)   NO {winner_used['no']} "
              f"({winner_used['no'] / n_w * 100:.0f}%)")
        print("  So most widened knobs are not the reason their expansion improved, even when it "
              "did.")

        if by_direction:
            print("\n  BY DIRECTION -- did the added values go above or below the old domain?")
            print(f"    {'dir':<6} {'helped':>7} {'hurt/tied':>10} {'rate':>6}")
            for d in ("UP", "DOWN"):
                h, x = by_direction[(d, True)], by_direction[(d, False)]
                if h + x:
                    print(f"    {d:<6} {h:>7} {x:>10} {100 * h / (h + x):>5.0f}%")
        if by_kind:
            print("\n  BY DIRECTION AND KNOB KIND -- this is where the signal lives:")
            print(f"    {'dir':<6} {'kind':<8} {'helped':>7} {'hurt':>6} {'rate':>6}")
            for d in ("UP", "DOWN"):
                for k in ("TILE", "WARPS", "STAGES"):
                    h, x = by_kind[(d, k, True)], by_kind[(d, k, False)]
                    if h + x:
                        print(f"    {d:<6} {k:<8} {h:>7} {x:>6} {100 * h / (h + x):>5.0f}%")

        print("\n  worst widened knobs (added values' median slower than not using them):")
        rated = [r for r in knob_rows if r[7] is not None]
        for run, cid, sid, knob, vals, mw, mwo, delta, nw, nwo, won in sorted(
                rated, key=lambda r: r[7])[:12]:
            print(f"    {delta:+6.1f}%  {cid}  {sid}  {knob} += {vals}   "
                  f"(median with={mw} n={nw}, without={mwo} n={nwo})")
        print("\n  best widened knobs:")
        for run, cid, sid, knob, vals, mw, mwo, delta, nw, nwo, won in sorted(
                rated, key=lambda r: -r[7])[:8]:
            tag = "  <- winner used it" if won else ""
            print(f"    {delta:+6.1f}%  {cid}  {sid}  {knob} += {vals}   "
                  f"(median with={mw} n={nw}, without={mwo} n={nwo}){tag}")
        n_unrated = len(knob_rows) - len(rated)
        if n_unrated:
            print(f"\n  ({n_unrated} widened knobs unrated: one side never measured fresh.)")
    return 0
    print()
    print("A flat outcome says nothing about the widened region -- judge an expansion by the")
    print("trials that used NEW choices. But flat is NOT the usual result: most expansions")
    print("improve the reported best, so do not predict flatness from a handful of examples.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
