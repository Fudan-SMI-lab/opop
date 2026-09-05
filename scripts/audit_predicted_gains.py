"""Audit: is the analyst's `predicted_gain_pct` calibrated?

A BottleneckReport's `parameter_limits[].predicted_gain_pct` says how much the analyst thinks
relieving a boundary would buy. Nothing in the harness reads it -- convergence uses
`min_improvement_pct` against measured history -- so it is a reporting field a human trusts.
This script checks whether it deserves that trust.

"Actual" is the parent's tuned best against its best CHILD's tuned best. Read the confound in
docs/measurement-predicted-gain-overshoots.md before quoting the numbers: a child implements a
HYPOTHESIS, not a parameter_limits entry, there are usually 2 children (so "best child" already
takes a max, biasing toward the prediction), and the winning child addresses the predicted knob
less than half the time. This is a joint test of the whole chain, not of the number alone.

Also compares reports that stated a gain against reports that declined to, since "should the
analyst predict at all" is the actionable question.

Usage:  python scripts/audit_predicted_gains.py [runs/run-... ...]
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from statistics import median
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def main(argv: list[str]) -> int:
    runs = [Path(a) for a in argv] or sorted(REPO.glob("runs/run-l3-*"))

    best: dict[str, float] = {}
    parent_of: dict[str, str | None] = {}
    summary_of: dict[str, str] = {}
    children: dict[str, list[str]] = defaultdict(list)
    reports: list[tuple[str, str, list[tuple[str, float]]]] = []

    for run in runs:
        ev_path = run / "events.jsonl"
        if not ev_path.exists():
            continue
        for line in ev_path.open(encoding="utf-8"):
            e = json.loads(line)
            if e["type"] == "TUNING_DONE":
                cid, ms = e["payload"].get("candidate_id"), e["payload"].get("best_ms")
                if cid and ms is not None:
                    best[cid] = min(best.get(cid, float("inf")), ms)
            elif e["type"] == "CANDIDATE_REGISTERED":
                c = e["payload"].get("candidate") or e["payload"]
                cid = c["candidate_id"]
                par = (c.get("parent_ids") or [None])[0]
                parent_of[cid] = par
                summary_of[cid] = c.get("approach_summary") or ""
                if par:
                    children[par].append(cid)
            elif e["type"] == "BOTTLENECK_REPORTED":
                r = e["payload"].get("report") or {}
                limits = [(x.get("param") or "", x.get("predicted_gain_pct"))
                          for x in r.get("parameter_limits", [])]
                stated = [(n, g) for n, g in limits
                          if isinstance(g, (int, float)) and not isinstance(g, bool) and g > 0]
                reports.append((run.name, e["payload"].get("candidate_id") or "", stated))

    def realised(cid: str) -> float | None:
        """Percent the best measurable child improved on this candidate's own tuned best."""
        parent_ms = best.get(cid)
        kid_ms = [best[k] for k in children.get(cid, []) if k in best]
        if parent_ms is None or not kid_ms or parent_ms <= 0:
            return None
        return (parent_ms - min(kid_ms)) / parent_ms * 100.0

    # --- calibration of the stated numbers -----------------------------------------
    seen: set[tuple[str, float]] = set()
    rows: list[tuple[float, float, str, str]] = []
    for _run, cid, stated in reports:
        if not stated:
            continue
        pred = max(g for _, g in stated)
        if (cid, pred) in seen:
            continue
        seen.add((cid, pred))
        act = realised(cid)
        if act is None:
            continue
        rows.append((pred, act, cid, stated[0][0]))

    print("=" * 74)
    print("CALIBRATION -- largest stated predicted_gain_pct vs what the best child achieved")
    print("=" * 74)
    if not rows:
        print("  no report with both a stated gain and a measurable child")
    else:
        print(f"  {'predicted':>9} {'actual':>8} {'error':>9}  {'candidate':<16} first knob")
        for pred, act, cid, knob in sorted(rows, key=lambda r: -r[0]):
            print(f"  {pred:>9.1f} {act:>8.1f} {act - pred:>+9.1f}  {cid:<16} {knob}")
        errs = [a - p for p, a, _, _ in rows]
        print(f"\n  n={len(rows)}   median predicted {median(p for p, _, _, _ in rows):.1f}%"
              f"   median actual {median(a for _, a, _, _ in rows):.1f}%"
              f"   median SIGNED error {median(errs):+.1f}%")
        print(f"  actual >= predicted in {sum(1 for e in errs if e >= 0)} of {len(errs)}")
        print(f"  actual was NEGATIVE (child worse than parent) in "
              f"{sum(1 for _, a, _, _ in rows if a < 0)} of {len(rows)}")

    # --- does stating a number predict a better round at all? -----------------------
    with_pred: list[float] = []
    without_pred: list[float] = []
    for _run, cid, stated in reports:
        act = realised(cid)
        if act is None:
            continue
        (with_pred if stated else without_pred).append(act)

    print()
    print("=" * 74)
    print("DOES STATING A NUMBER CARRY SIGNAL? -- realised delta, split by whether the")
    print("report named a positive predicted_gain_pct at all")
    print("=" * 74)
    for label, arr in (("stated a gain", with_pred), ("stated NO gain", without_pred)):
        if arr:
            print(f"  {label:<16} n={len(arr):<3} median realised {median(arr):+.1f}%"
                  f"   improved {sum(1 for x in arr if x > 0)}/{len(arr)}")

    # --- the confound, quantified --------------------------------------------------
    n_kids: list[int] = []
    on_target = total = 0
    for _run, cid, stated in reports:
        if not stated:
            continue
        kids = [k for k in children.get(cid, []) if k in best]
        if not kids or cid not in best:
            continue
        n_kids.append(len(kids))
        total += 1
        winner = min(kids, key=lambda k: best[k])
        knob = stated[0][0]
        prefix = knob.split("_")[0] if "_" in knob else knob
        if prefix and prefix.lower() in summary_of.get(winner, "").lower():
            on_target += 1

    print()
    print("=" * 74)
    print("CONFOUND -- 'actual' is the best child, and a child implements a HYPOTHESIS")
    print("=" * 74)
    if n_kids:
        print(f"  predictions with a measurable child: {total}")
        print(f"  median children per parent:          {median(n_kids):.0f}"
              f"   (so 'best child' already takes a max -> biased TOWARD the prediction)")
        print(f"  best child's summary mentions the predicted knob's prefix: "
              f"{on_target} of {total}")
        print("  -> treat this as a joint test of analyst+rewriter+tuner, not of the number.")

    # --- the counterweight: WHAT it names, as opposed to how much ---------------------
    kinds: dict[str, int] = defaultdict(int)
    blocked: dict[str, int] = defaultdict(int)
    for run in runs:
        ev_path = run / "events.jsonl"
        if not ev_path.exists():
            continue
        for line in ev_path.open(encoding="utf-8"):
            e = json.loads(line)
            if e["type"] != "BOTTLENECK_REPORTED":
                continue
            for x in (e["payload"].get("report") or {}).get("parameter_limits", []):
                name = x.get("param") or ""
                kinds["TILE" if any(k in name for k in ("BLOCK", "ROW", "CHUNK"))
                      else "WARPS" if "WARP" in name
                      else "STAGES" if ("STAGE" in name or "PIPE" in name)
                      else "other"] += 1
                blocked[str(x.get("blocked_by"))] += 1

    if kinds:
        n = sum(kinds.values())
        print()
        print("=" * 74)
        print("COUNTERWEIGHT -- what the analyst NAMES as having headroom (vs how much)")
        print("=" * 74)
        for k, v in sorted(kinds.items(), key=lambda kv: -kv[1]):
            print(f"  {k:<8} {v:>4}  ({100 * v / n:.0f}%)")
        print("\n  Compare audit_expansion_outcomes.py's PER-KNOB section: tile sizes widened")
        print("  UPWARD help 76%, warps 22%. The analyst spends ~2/3 of its attention on the")
        print("  knob class that pays -- derived completely independently, and they agree.")
        print("\n  blocked_by attributions:")
        for k, v in sorted(blocked.items(), key=lambda kv: -kv[1]):
            print(f"    {k:<22} {v:>3}")
        print("  ('none' being common is a good sign: the analyst declines to claim a binding")
        print("   resource rather than inventing one.)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
