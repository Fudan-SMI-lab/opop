"""How often does the median table's winner rest on too few samples to mean anything?

`latency_by_value` is a per-choice median, and `best_value` / `effect_pct` are derived from it
and shipped to the analyst in stats.json. A choice sampled once or twice can win that table on
luck alone, which inflates `effect_pct` and names the wrong `best_value`.

Found on the run's own best candidate. cand-0d0dcd49, FINAL_BLOCK:

    value    n   median      min
       64    8    13.55     9.33
      128   14    12.00     9.23
      256    5    13.60     9.47
      512    6    13.45     9.14   <- the winning trial
     1024    2    10.95    10.50   <- median's pick; effect_pct 24.2 came from this row

1024 was sampled twice, caught a quiet pair, and beat the table by 1.05 ms. Its own best trial
is worse than every other value's best, and it appears in none of the space's top 8 trials.

The EXPANSION consequence is already fixed: `fix-boundary-direction-follows-the-winning-trial.md`
anchors at_boundary/boundary_direction on the winning trial, and on this space the new rule
requests nothing at all instead of extending toward 1024. What this script measures is the
RESIDUAL exposure -- `best_value` and `effect_pct` still come from the thin median and still
reach the analyst prompt.

Counts a knob as thin-won when the median's winner has n<=2 while some other value has n>=5,
i.e. a demonstrably better-sampled alternative existed.

Reads events.jsonl only; no GPU.
"""
from __future__ import annotations

import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, "src")
sys.stdout.reconfigure(encoding="utf-8")

from kernel_optimizer.models.core import ParameterSpace, TrialRecord  # noqa: E402

MIN_COMPLETE = 10   # ignore barely-tuned spaces
THIN = 2            # winner sampled this few times ...
WELL = 5            # ... while another value had at least this many

thin_rows, total = [], 0
for run in sorted(Path("runs").glob("run-l3-*")):
    f = run / "events.jsonl"
    if not f.exists():
        continue
    spaces, trials = [], defaultdict(list)
    for ln in f.read_text(encoding="utf-8").splitlines():
        if not ln.strip():
            continue
        try:
            e = json.loads(ln)
        except json.JSONDecodeError:
            continue
        p = e.get("payload", {})
        if e["type"] == "SPACE_PUBLISHED":
            try:
                spaces.append(ParameterSpace.model_validate(p["space"]))
            except Exception:
                pass
        elif e["type"] == "TRIAL_DONE":
            try:
                t = TrialRecord.model_validate(p["trial"])
                trials[t.space_id].append(t)
            except Exception:
                pass
    for s in spaces:
        ts = [t for t in trials.get(s.space_id, [])
              if t.status == "complete" and t.latency_ms]
        if len(ts) < MIN_COMPLETE:
            continue
        for d in s.domains:
            per = defaultdict(list)
            for t in ts:
                per[repr(t.params.values.get(d.name))].append(t.latency_ms.mean)
            if len(per) < 3:
                continue
            total += 1
            meds = {k: statistics.median(v) for k, v in per.items()}
            mins = {k: min(v) for k, v in per.items()}
            win = min(meds, key=meds.get)
            if len(per[win]) > THIN or max(
                    (len(v) for k, v in per.items() if k != win), default=0) < WELL:
                continue
            better = sum(1 for k, v in mins.items() if k != win and v < mins[win])
            lo, hi = min(meds.values()), max(meds.values())
            thin_rows.append({
                "run": run.name, "space": s.space_id, "param": d.name,
                "winner": win, "n": len(per[win]),
                "median": meds[win], "min": mins[win],
                "better_mins": better, "others": len(per) - 1,
                "effect_pct": (hi - lo) / lo * 100.0 if lo > 0 else 0.0,
                # what effect_pct would be with the thin row dropped
                "effect_wo": ((max(v for k, v in meds.items() if k != win) -
                               min(v for k, v in meds.items() if k != win)) /
                              min(v for k, v in meds.items() if k != win) * 100.0),
            })

if not total:
    print("no knobs to score")
    raise SystemExit(0)

print(f"knobs scored: {total}")
print(f"thin-won (winner n<={THIN} while another value had n>={WELL}): "
      f"{len(thin_rows)} ({len(thin_rows)/total*100:.1f}%)\n")

overstated = [r for r in thin_rows if r["better_mins"] > 0]
print(f"of those, the thin winner's OWN BEST TRIAL is beaten by at least one other value: "
      f"{len(overstated)} ({len(overstated)/len(thin_rows)*100:.0f}%)")
print("  -- i.e. the median names a value the objective disagrees with\n")

infl = [r for r in thin_rows if r["effect_pct"] > r["effect_wo"]]
if infl:
    d = [r["effect_pct"] - r["effect_wo"] for r in infl]
    print(f"effect_pct inflated by the thin row on {len(infl)} knobs, "
          f"median +{statistics.median(d):.1f} points (max +{max(d):.1f})")

print("\nworst instances by inflation:")
for r in sorted(thin_rows, key=lambda r: -(r["effect_pct"] - r["effect_wo"]))[:8]:
    print(f"  {r['run'][8:]} {r['space']} {r['param']:18s} winner={r['winner']:>6s} n={r['n']}"
          f"  effect {r['effect_pct']:.1f}% -> {r['effect_wo']:.1f}% without it"
          f"  better_mins={r['better_mins']}/{r['others']}")

print("""
NOT proposed as a fix. Dropping or down-weighting thin rows would change what every analyst
call sees, and the analyst's own prompt already tells it to reason over trials.csv rather than
trust a summary row. The expansion consequence -- the one with a measured cost -- is already
addressed by anchoring at_boundary on the winning trial. This records the residual so a future
change to stats.json starts from a number rather than an anecdote.
""")
