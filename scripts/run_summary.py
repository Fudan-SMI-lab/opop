#!/usr/bin/env python3
"""Summarize a run's events.jsonl from disk. Read-only, on-disk truth only.

Usage: run_summary.py <run_dir>

Exists because inline heredocs through ssh keep breaking on nested quotes; a file on the
remote takes its argument and has no quoting problem.
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
    except Exception:  # noqa: BLE001 - a truncated tail line is normal mid-run
        continue

c = collections.Counter(e.get("type") for e in evs)
span = (evs[-1]["ts"] - evs[0]["ts"]) / 60 if len(evs) > 1 else 0.0
print(f"run      : {run.name}")
print(f"events   : {len(evs)}   elapsed: {span:.1f} min   last: {evs[-1].get('type')}")
print()
for k, v in c.most_common():
    print(f"  {v:>5}  {k}")

# trial outcomes
st = collections.Counter()
best = None
for e in evs:
    if e.get("type") == "TRIAL_DONE":
        t = (e.get("payload") or {}).get("trial") or {}
        st[(t.get("status"), t.get("failure_kind"))] += 1
        lat = t.get("latency_ms") or {}
        m = lat.get("median")
        if t.get("status") == "complete" and m is not None:
            if best is None or m < best:
                best = m
if st:
    print("\ntrials:")
    for (status, kind), n in st.most_common():
        print(f"  {n:>5}  {status}" + (f" / {kind}" if kind else ""))
if best is not None:
    print(f"\nbest trial median: {best:.4f} ms")

# prescreen verdicts (F5)
pre = [e["payload"] for e in evs if e.get("type") == "SPACE_PRESCREENED"]
if pre:
    print("\nprescreen (F5):")
    for p in pre:
        print(f"  {p.get('candidate_id')}: probed={p.get('configs_probed')} "
              f"infeasible={p.get('infeasible')}")
fail = [e["payload"] for e in evs if e.get("type") == "PRESCREEN_FAILED"]
for p in fail:
    print(f"  PRESCREEN_FAILED {p.get('candidate_id')}: {str(p.get('detail'))[:120]}")

# calibration
for e in evs:
    if e.get("type") in ("CALIBRATION_LOADED", "CALIBRATION_MEASURED"):
        p = e["payload"]
        print(f"\ncalibration ({e['type']}, source={p.get('source')}):")
        for k in ("dram_tbs", "fp32_tflops", "tf32_tflops", "fp16_tflops", "bf16_tflops"):
            print(f"  {k:<14} {p.get(k)}")

# agent failures
for e in evs:
    if e.get("type") in ("AGENT_CALL_FAILED", "RUN_FINISHED"):
        print(f"\n{e['type']}: {json.dumps(e.get('payload'))[:400]}")
