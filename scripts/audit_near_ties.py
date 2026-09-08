#!/usr/bin/env python3
"""Is this run's tuning distinguishing configurations, or ranking noise?

Motivation (from an earlier finding on L3:48): when a trial's own sample std is a large
fraction of its mean, and many trials sit within that spread of each other, the "improvement"
from one configuration to the next is not a ranking -- it is the same number measured twice.
A run can then spend its whole budget and report a winner chosen by chance.

This is a RESULT-VALIDITY question, unlike budget waste, so it is worth checking during a run.

Prints, for the completed trials: the best time, how many trials fall within one combined
standard error of it (a near-tie band), the median per-trial CV, and the spread of the band.
A large near-tie count with a per-trial CV comparable to the band width means the winner is
not distinguishable from its rivals.
"""
import json
import math
import pathlib
import statistics
import sys

run = pathlib.Path(sys.argv[1])
rows = []
for ln in (run / "events.jsonl").read_text(encoding="utf-8").splitlines():
    try:
        e = json.loads(ln)
    except Exception:  # noqa: BLE001
        continue
    if e.get("type") != "TRIAL_DONE":
        continue
    t = (e.get("payload") or {}).get("trial") or {}
    if t.get("status") != "complete":
        continue
    lat = t.get("latency_ms") or {}
    m = lat.get("median") or lat.get("mean")
    if m is None:
        continue
    rows.append({
        "ms": m,
        "std": lat.get("std"),
        "n": lat.get("n") or lat.get("n_samples"),
        "params": (t.get("params") or {}).get("values") or {},
    })

if not rows:
    print("no completed trials yet")
    sys.exit(0)

rows.sort(key=lambda r: r["ms"])
best = rows[0]
print(f"run              : {run.name}")
print(f"completed trials : {len(rows)}")
print(f"best             : {best['ms']:.4f} ms  (std {best['std']}, n {best['n']})")

cvs = [r["std"] / r["ms"] for r in rows if r["std"] and r["ms"]]
if cvs:
    print(f"per-trial CV     : median {100*statistics.median(cvs):.1f}%  "
          f"max {100*max(cvs):.1f}%")

# Near-tie band: within one combined standard error of the best.
def sem(r):
    if r["std"] and r["n"]:
        return r["std"] / math.sqrt(r["n"])
    return 0.0


b_sem = sem(best)
band = []
for r in rows:
    comb = math.sqrt(b_sem ** 2 + sem(r) ** 2)
    if r["ms"] - best["ms"] <= comb:
        band.append(r)

print(f"\nnear-tie band (within 1 combined SEM of the best):")
print(f"  trials in band : {len(band)} of {len(rows)} ({100*len(band)/len(rows):.0f}%)")
if len(band) > 1:
    lo, hi = band[0]["ms"], band[-1]["ms"]
    print(f"  band spread    : {lo:.4f} - {hi:.4f} ms ({100*(hi-lo)/lo:.2f}%)")
    print(f"  combined SEM   : {math.sqrt(2)*b_sem:.4f} ms "
          f"({100*math.sqrt(2)*b_sem/best['ms']:.2f}% of best)")

# Do the band members agree on the categorical knobs? If they disagree, the winning VALUES
# are not what the timing is selecting.
if len(band) > 2:
    keys = {k for r in band for k, v in r["params"].items() if isinstance(v, str)}
    print("\n  categorical agreement within the band:")
    for k in sorted(keys):
        vals = {}
        for r in band:
            v = r["params"].get(k)
            if isinstance(v, str):
                vals[v] = vals.get(v, 0) + 1
        top = max(vals.values()) if vals else 0
        verdict = "decided" if top == len(band) else f"split {vals}"
        print(f"    {k:<20} {verdict}")

print()
if len(band) > max(3, 0.25 * len(rows)):
    print("VERDICT: many near-ties -- the winner is not clearly distinguishable from its")
    print("         rivals at this sample count. Treat per-config 'gains' inside the band as")
    print("         unmeasured, and quote final_reeval_ms (100 samples) not tuned_ms.")
else:
    print("VERDICT: the best is separated from the field by more than the measurement error.")
