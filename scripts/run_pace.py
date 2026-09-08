#!/usr/bin/env python3
"""Per-phase wall clock for a run, and what remains against the config budgets.

Purpose: estimate completion from THIS run's own observed pace, not from a guess.
Prints each step's duration and the per-candidate/per-round cost, so the remaining
budget can be multiplied by a measured number.
"""
import collections
import json
import pathlib
import sys

run = pathlib.Path(sys.argv[1])
evs = []
for ln in (run / "events.jsonl").read_text(encoding="utf-8").splitlines():
    try:
        evs.append(json.loads(ln))
    except Exception:  # noqa: BLE001
        continue

t0 = evs[0]["ts"]
print(f"run: {run.name}   elapsed {(evs[-1]['ts'] - t0) / 60:.1f} min")
print()

# STEP_DONE carries step_key; time each step by the gap since the previous step.
print("steps completed (cumulative min -> duration):")
prev = t0
for e in evs:
    if e.get("type") == "STEP_DONE":
        key = (e.get("payload") or {}).get("step_key")
        print(f"  {(e['ts'] - t0) / 60:7.1f}  (+{(e['ts'] - prev) / 60:5.1f})  {key}")
        prev = e["ts"]

# agent calls: which module, how long
print("\nagent calls:")
started = {}
durs = collections.defaultdict(list)
for e in evs:
    t = e.get("type")
    p = e.get("payload") or {}
    if t == "AGENT_CALL_STARTED":
        started[p.get("call_id") or p.get("module")] = (e["ts"], p.get("module"))
    elif t in ("AGENT_CALL_FINISHED", "AGENT_CALL_FAILED"):
        k = p.get("call_id") or p.get("module")
        if k in started:
            ts, mod = started.pop(k)
            durs[mod or p.get("module")].append(e["ts"] - ts)
            print(f"  {(ts - t0) / 60:7.1f}  {mod or p.get('module'):<16} "
                  f"{(e['ts'] - ts) / 60:5.1f} min  {'OK' if t.endswith('FINISHED') else 'FAILED'}")
for k, (ts, mod) in started.items():
    print(f"  {(ts - t0) / 60:7.1f}  {mod:<16} IN FLIGHT ({(evs[-1]['ts'] - ts) / 60:.1f} min so far)")

print("\nmedian agent-call cost by module:")
for mod, ds in sorted(durs.items()):
    ds = sorted(ds)
    print(f"  {mod:<16} n={len(ds)}  median {ds[len(ds) // 2] / 60:.1f} min  "
          f"total {sum(ds) / 60:.1f} min")

# tuning cost
tr = [e for e in evs if e.get("type") == "TRIAL_DONE"]
if len(tr) > 1:
    span = tr[-1]["ts"] - tr[0]["ts"]
    print(f"\ntrials: {len(tr)}  span {span / 60:.1f} min  "
          f"~{span / max(len(tr) - 1, 1):.1f} s/trial")

# where are we in the outer loop
print("\nouter-loop progress:")
c = collections.Counter(e.get("type") for e in evs)
for k in ("CANDIDATE_REGISTERED", "FAMILY_SEEDED", "TUNING_DONE", "CONVERGENCE_DECIDED",
          "REWRITE_PRODUCED", "REWRITE_REJECTED", "NOVELTY_PRODUCED", "NOVELTY_REJECTED",
          "RUN_FINISHED"):
    print(f"  {k:<24} {c.get(k, 0)}")
for e in evs:
    if e.get("type") == "CONVERGENCE_DECIDED":
        p = e.get("payload") or {}
        print(f"    verdict: {json.dumps(p)[:220]}")
