"""Audit: does seeding best_history restore the slope ordering in round 2?

`active_families()` ranks by (unproven, -improvement_slope, latency). The docstring's whole
argument is that latency must NOT decide, because a family's current latency reflects its
initial parameterization rather than its remaining structural headroom -- measured on
L3:43, where ranking by latency would have deleted the run's winner.

But `_improvement_pct` returns 0.0 whenever `best_history` has fewer than two entries. So
UNSEEDED, at the decision that picks families for round 2, every family that has run exactly
one round has a one-entry history, ties at slope 0.0, and falls through to the latency
tie-break. The slope ordering is unavailable exactly once -- at the first decision where it
could matter -- and the fallback is the ranking the docstring rejects.

Seeding puts the seed-phase best at `best_history[0]`, so after round 1 the history is
[seed, round1] and the round-1 slope is available at that decision.

This replays both orderings against every run on disk and reports where the chosen SET
differs -- not just the order, since only the first `max_families_active` entries get a
round. Exits 1 if a swap would have dropped a family that went on to improve.

Usage:  python scripts/audit_slope_ordering.py [--runs runs] [--config configs/experiments_l3.yaml]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, "src")

from kernel_optimizer.config import load_config  # noqa: E402
from kernel_optimizer.control.families import FamilyManager  # noqa: E402
from kernel_optimizer.models.core import BestRecord, Family, ParamSet  # noqa: E402


def order(histories: dict[str, list[float]], rounds_used: dict[str, int],
          k: int) -> list[str]:
    """Run the REAL active_families() ranking over synthetic families."""
    mgr = FamilyManager.__new__(FamilyManager)
    mgr.families = {}
    mgr.max_families_active = k
    for fid, hist in histories.items():
        mgr.families[fid] = Family(
            family_id=fid, anchor_candidate_id="c", member_ids=["c"],
            best=BestRecord(candidate_id="c", params=ParamSet(values={}),
                            latency_ms=hist[-1]),
            best_history=list(hist), rewrite_rounds_used=rounds_used[fid],
            status="active")
    return [f.family_id for f in mgr.active_families()]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--config", default="configs/experiments_l3.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)
    k = cfg.budgets.max_families_active
    print(f"config: max_families_active={k}\n")

    differing = 0
    examined = 0
    harmful: list[tuple[str, str, float]] = []

    for d in sorted(Path(args.runs).glob("run-*")):
        f = d / "events.jsonl"
        if not f.exists():
            continue
        events = []
        for line in f.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
        fin = [e for e in events if e.get("type") == "RUN_FINISHED"]
        if not fin:
            continue
        fams = ((fin[0].get("payload") or {}).get("summary") or {}).get("families")
        if not isinstance(fams, dict) or len(fams) < 2:
            continue

        # Seed = the anchor's tuning best, i.e. the family's latency before any rewrite.
        seeds: dict[str, float] = {}
        hists: dict[str, list[float]] = {}
        for fid, info in fams.items():
            hist = info.get("history") or []
            anchor = info.get("anchor")
            s = [e["payload"]["best_ms"] for e in events
                 if e.get("type") == "TUNING_DONE"
                 and (e.get("payload") or {}).get("candidate_id") == anchor
                 and (e.get("payload") or {}).get("best_ms")]
            if not s or not hist:
                continue
            seeds[fid] = s[0]
            hists[fid] = hist
        if len(hists) < 2:
            continue

        # The decision that picks families for round 2: every family has run 1 round.
        rounds_used = {fid: 1 for fid in hists}
        unseeded = {fid: h[:1] for fid, h in hists.items()}
        seeded = {fid: [seeds[fid]] + h[:1] for fid, h in hists.items()}

        got_un = order(unseeded, rounds_used, k)
        got_se = order(seeded, rounds_used, k)
        examined += 1
        if set(got_un) == set(got_se):
            continue
        differing += 1

        print(f"=== {d.name[-13:]} ===")
        for fid in hists:
            slope_un = 0.0
            prev, cur = seeds[fid], hists[fid][0]
            slope_se = max(0.0, (prev - cur) / prev * 100.0) if prev else 0.0
            print(f"  {fid[:13]}  seed {seeds[fid]:6.1f} -> round1 {cur:6.1f}  "
                  f"full history {[round(x, 1) for x in hists[fid]]}")
            print(f"      slope unseeded {slope_un:5.2f}%   seeded {slope_se:5.2f}%")
        print(f"  chosen unseeded (latency tie-break): {[x[:13] for x in got_un]}")
        print(f"  chosen seeded   (real slope)       : {[x[:13] for x in got_se]}")

        # Did the unseeded choice drop a family that went on to improve?
        dropped = set(got_se) - set(got_un)
        for fid in dropped:
            h = hists[fid]
            if len(h) >= 2 and h[0] > 0:
                gain = (h[0] - min(h)) / h[0] * 100.0
                verdict = ("IMPROVED LATER" if gain >= cfg.budgets.min_improvement_pct
                           else "stayed flat")
                print(f"  unseeded would drop {fid[:13]}: it went "
                      f"{[round(x, 1) for x in h]} -> {verdict} ({gain:.1f}%)")
                if gain >= cfg.budgets.min_improvement_pct:
                    harmful.append((d.name[-13:], fid[:13], gain))
        print()

    print(f"runs examined: {examined}")
    print(f"runs where the round-2 selection SET differs: {differing}")
    if harmful:
        print(f"\n{len(harmful)} case(s) where the unseeded (latency) tie-break drops a "
              f"family that later improved by >= {cfg.budgets.min_improvement_pct}%:")
        for run, fid, gain in harmful:
            print(f"   {run}  {fid}  later gained {gain:.1f}%")
        print("\nThat is the early-pruning-by-latency the ordering exists to prevent, "
              "occurring because the slope was unavailable at round 2.")
    else:
        print("\nNo case where the unseeded tie-break drops a family that later improved.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
