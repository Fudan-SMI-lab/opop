"""Is H4 a one-run accident? Test "the attributable wall is the weakest" on more runs, offline.

H4 (measured on arm 3): the one wall 2e could ATTRIBUTE had a 7.5% tail slope, while slopes of 60.7%,
25.6% and 10.7% went unattributed. If that holds generally, the mechanism systematically steers the
rewrite toward its least valuable finding -- which would be a design fault, not bad luck.

arm 3 is the only run with 2e switched on, so n=1 from the events alone. But `find_walls` is pure and
the inputs it needs (TuningStats, plus the refused parameter sets) are journalled by EVERY run. So the
slope distribution can be recovered offline on runs that never ran 2e, which is a bigger sample for
the question "where do the steep slopes sit relative to the walls".

WHAT THIS CAN AND CANNOT SHOW. Offline it can recover which knobs were TRUNCATED and their slopes --
that is `find_walls`, pure arithmetic. It CANNOT recover which would have been ATTRIBUTED, because
attribution needs the compile probe that only runs live. So the test is one-directional: it measures
whether the steepest-slope walls tend to be the ones a rewrite could plausibly fix (over_ratio near 1,
which is 2e's own feasibility criterion), or whether steep slopes systematically sit on walls far over
the limit. The first would mean arm 3 was unlucky; the second, that the mechanism selects against its
own payoff by construction.

Uses the SHIPPED find_walls -- a second implementation could disagree with the one that produced
arm 3's numbers.
"""
import json
import os
import sys
from collections import defaultdict

if len(sys.argv) < 3 or "=" in sys.argv[1]:
    print(__doc__)
    print("usage: h4_is_the_attributable_wall_the_weakest.py <checkout-src> label=<run_dir> ...")
    raise SystemExit(2)
sys.path.insert(0, sys.argv[1])
from kernel_optimizer.evaluation import wall_attribution  # noqa: E402
from kernel_optimizer.models.reports import TuningStats  # noqa: E402

print(f"wall finder: {wall_attribution.__file__}")


def analyse(rd, label):
    evs = []
    with open(os.path.join(rd, "events.jsonl"), encoding="utf-8") as fh:
        for ln in fh:
            ln = ln.strip()
            if ln:
                try:
                    evs.append(json.loads(ln))
                except Exception:
                    pass
    trials = defaultdict(list)
    for e in evs:
        if e.get("type") != "TRIAL_DONE":
            continue
        t = (e.get("payload") or {}).get("trial") or {}
        if t.get("candidate_id"):
            trials[t["candidate_id"]].append(t)
    stats_for = {}
    pending = None
    for e in evs:
        if e.get("type") == "TUNING_DONE":
            pending = (e.get("payload") or {}).get("candidate_id")
        elif e.get("type") == "STATS_DONE" and pending:
            raw = (e.get("payload") or {}).get("stats")
            if raw:
                try:
                    stats_for[pending] = TuningStats.model_validate(raw)
                except Exception:
                    pass
            pending = None

    rows = []
    for cid, stats in stats_for.items():
        refused = [(t.get("params") or {}).get("values") or {} for t in trials.get(cid, [])
                   if t.get("failure_kind") == "infeasible_shared_memory"]
        if not refused:
            continue
        for w in wall_attribution.find_walls(stats, refused):
            probe_worthy = bool(w.monotone and w.tail_gain_pct > 0.0)
            rows.append({"cand": cid, "param": w.param, "gain": float(w.tail_gain_pct),
                         "monotone": bool(w.monotone), "worthy": probe_worthy})
    print(f"\n=== {label}: {len(rows)} walls found offline")
    if not rows:
        return []
    rows.sort(key=lambda r: -r["gain"])
    for r in rows:
        print(f"    {r['gain']:+7.1f}%  {r['cand']} {r['param']}"
              f"{'' if r['monotone'] else '  (non-monotone)'}"
              f"{'  <- probe-worthy' if r['worthy'] else ''}")
    worthy = [r for r in rows if r["worthy"]]
    if worthy and len(rows) > len(worthy):
        best_w = max(r["gain"] for r in worthy)
        best_all = max(r["gain"] for r in rows)
        print(f"    steepest probe-worthy {best_w:+.1f}% vs steepest overall {best_all:+.1f}%"
              + ("  => the steepest slope was NOT probe-worthy" if best_all > best_w + 1e-9
                 else "  => the steepest slope WAS probe-worthy"))
    return rows


allrows = []
for spec in sys.argv[2:]:
    label, rd = spec.split("=", 1)
    allrows += analyse(rd, label)

print()
print("=== pooled")
worthy = [r for r in allrows if r["worthy"]]
unworthy = [r for r in allrows if not r["worthy"]]
print(f"probe-worthy walls: {len(worthy)}   others: {len(unworthy)}")
if worthy:
    gs = sorted(r["gain"] for r in worthy)
    print(f"  probe-worthy slopes: {[round(g,1) for g in gs]}")
if unworthy:
    gs = sorted(r["gain"] for r in unworthy)
    print(f"  other slopes:        {[round(g,1) for g in gs]}")
print()
print("READING: 2e's filter keeps only monotone positive-slope walls, and only a probe-worthy wall")
print("can ever be ATTRIBUTED. If the steepest slopes repeatedly fail that filter, the mechanism")
print("cannot reach its own best findings -- a design limit rather than one run's bad luck.")
