"""How much wall clock does the running L3 chain still need?

Answers it from finished runs, not from `wall_clock_hours`: the budget is a CAP that no
completed L3 run has ever hit. A run ends when its families freeze `budget_exhausted`.

Method: a POSITION-MATCHED reference class. The outer loop emits one family
CONVERGENCE_DECIDED per family per round, and a full run shows the interleaved pattern
r0,r0,r1,r1,r2,r2,r3,r3 -- so "time remaining after the Nth decision" is directly measurable
in every past run and directly comparable to where the live run stands now.

Two contaminants had to be excluded, both found by inspecting the decision lists rather than
by trusting a pooled average:

  * The last two decisions of a full run share a timestamp (both families freeze together),
    so naive round-to-round gaps include a 0-minute "round". This is why an earlier pooled
    version reported a first quartile of 1.4 min.
  * Runs whose decisions all carry `rewrite_rounds_used: 0` stopped early on the
    empty-family bug (a family with no surviving candidates never set `progressed`). They
    finished in 1.2-2.6h without spending their rewrite budget, and averaging them in
    halves the estimate. The reference class is therefore runs that reached
    `rewrite_rounds_used == 3`.

The two estimators printed at the end are independent (per-round cost vs. whole-run tail); a
disagreement between them is the honest error bar.
"""
from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

RUNS = Path("runs")
FULL_ROUNDS = 3  # rewrite_rounds_per_family; a run that reaches this spent its budget


def scan(run: Path) -> dict | None:
    f = run / "events.jsonl"
    if not f.exists():
        return None
    evs = []
    for ln in f.read_text(encoding="utf-8").splitlines():
        if ln.strip():
            try:
                evs.append(json.loads(ln))
            except json.JSONDecodeError:
                pass
    if not evs:
        return None
    decisions, origins, seeds_done = [], {}, None
    for e in evs:
        p = e.get("payload", {})
        if e["type"] == "CANDIDATE_REGISTERED":
            c = p.get("candidate", p)
            origins[c.get("candidate_id")] = c.get("origin")
        elif e["type"] == "TUNING_DONE" and origins.get(p.get("candidate_id")) == "seed":
            seeds_done = e["ts"]
        elif e["type"] == "CONVERGENCE_DECIDED" and p.get("decision", {}).get("scope") == "family":
            decisions.append((e["ts"], p["decision"]["evidence"].get("rewrite_rounds_used")))
    return {
        "name": run.name,
        "task": run.name.split("-")[2],
        "t0": evs[0]["ts"], "last": evs[-1]["ts"],
        "total_h": (evs[-1]["ts"] - evs[0]["ts"]) / 3600,
        "finished": any(e["type"] == "RUN_FINISHED" for e in evs),
        "seeds_done": seeds_done,
        "decisions": decisions,
        "max_rounds": max((r for _, r in decisions if r is not None), default=-1),
    }


runs = [r for r in (scan(p) for p in sorted(RUNS.glob("run-l3-*"))) if r]
live = [r for r in runs if not r["finished"] and time.time() - r["last"] < 1800]
# Spent its rewrite budget => a valid model for a run that will do the same.
ref = [r for r in runs if r["finished"] and r["max_rounds"] >= FULL_ROUNDS]
early = [r for r in runs if r["finished"] and r["max_rounds"] < FULL_ROUNDS]

print(f"=== reference class: finished runs that reached rewrite_rounds_used={FULL_ROUNDS}")
for r in sorted(ref, key=lambda r: r["total_h"]):
    print(f"  {r['name']:34s} {r['total_h']:5.2f}h  decisions={len(r['decisions'])}")
full = [r["total_h"] for r in ref]
print(f"  n={len(full)} median={statistics.median(full):.2f}h "
      f"range {min(full):.2f}-{max(full):.2f}h")
print(f"\n=== EXCLUDED: {len(early)} finished runs stopped early with rewrite budget unspent")
for r in sorted(early, key=lambda r: r["total_h"]):
    print(f"  {r['name']:34s} {r['total_h']:5.2f}h  max_rounds_used={r['max_rounds']}")

# per-round cost, with the simultaneous final freeze collapsed
gaps = []
for r in ref:
    marks = ([r["seeds_done"]] if r["seeds_done"] else []) + [t for t, _ in r["decisions"]]
    dedup = [t for i, t in enumerate(marks) if i == 0 or t - marks[i - 1] > 60]
    gaps += [(b - a) / 60 for a, b in zip(dedup, dedup[1:])]
print(f"\n=== per-round wall time in the reference class (min), simultaneous freezes collapsed")
print(f"  n={len(gaps)} median={statistics.median(gaps):.1f} "
      f"mean={statistics.mean(gaps):.1f} range {min(gaps):.1f}-{max(gaps):.1f}")

for r in live:
    el = (time.time() - r["t0"]) / 3600
    n = len(r["decisions"])
    print(f"\n=== in flight: {r['name']}")
    print(f"  elapsed {el:.2f}h, {n} family decisions, last event "
          f"{(time.time()-r['last'])/60:.1f} min ago")
    owed = 2 * (FULL_ROUNDS + 1) - n   # 2 active families x (rounds + the r0 decision)
    # (a) per-round cost x rounds owed
    med, mean = statistics.median(gaps), statistics.mean(gaps)
    print(f"  (a) {owed} decisions owed x median {med:.0f} min -> {owed*med/60:.1f}h "
          f"(at the mean {mean:.0f} min -> {owed*mean/60:.1f}h)")
    # (b) position-matched tail: how long past its Nth decision each reference run ran
    tails = [(r2["last"] - r2["decisions"][n - 1][0]) / 3600 for r2 in ref
             if len(r2["decisions"]) >= n]
    here = (time.time() - r["decisions"][n - 1][0]) / 3600 if n else 0.0
    print(f"  (b) past decision #{n}: reference tails "
          f"{'/'.join(f'{t:.1f}' for t in sorted(tails))}h, median "
          f"{statistics.median(tails):.1f}h; this run is {here:.1f}h in")
    print(f"      -> {max(0.0, statistics.median(tails)-here):.1f}h more "
          f"(quartile spread {max(0.0, statistics.quantiles(tails, n=4)[0]-here):.1f}"
          f"-{max(0.0, statistics.quantiles(tails, n=4)[2]-here):.1f}h)")
    print(f"  wall-clock cap: hard stop {12.0 - el:.1f}h from now (never yet binding)")

print("\n=== remaining queue: scripts/queue_glm_first.ps1")
print("  1. glm level3:21  (configs/experiments_l3_glm.yaml)")
print("  2. gpt level3:43")
print("  3. gpt level3:48")
print("\n  The GLM arm has NO finished run, so its duration is unpredicted. Its budgets are")
print("  identical to the gpt arm's, but agent latency, retry rate and whether its candidates")
print("  pass the PARAMS contract are all unmeasured -- one smoke call is the only evidence.")
print("  A gpt run on the same task is the closest available prior, quoted below for scale only.")
gpt21 = [r["total_h"] for r in ref if r["task"] == "21"]
if gpt21:
    print(f"    gpt on level3:21: n={len(gpt21)} median {statistics.median(gpt21):.2f}h "
          f"range {min(gpt21):.2f}-{max(gpt21):.2f}h")
print("\n  the two gpt tasks that follow:")
for task in ("43", "48"):
    ts = [r["total_h"] for r in ref if r["task"] == task]
    if ts:
        print(f"    level3:{task}: n={len(ts)} median {statistics.median(ts):.2f}h "
              f"range {min(ts):.2f}-{max(ts):.2f}h")
per_task = {t: [r["total_h"] for r in ref if r["task"] == t] for t in ("43", "48")}
if all(per_task.values()):
    lo = sum(min(v) for v in per_task.values())
    mid = sum(statistics.median(v) for v in per_task.values())
    hi = sum(max(v) for v in per_task.values())
    print(f"    both gpt tasks: {mid:.1f}h at the medians (range {lo:.1f}-{hi:.1f}h)")
