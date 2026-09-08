#!/usr/bin/env python3
"""Is this run making progress, or stalled? Answers from event timestamps only.

A stalled orchestrator looks identical to a busy one from `pgrep`: the process is alive, the
GPU may be idle because it is mid agent call, and the log's last line can be minutes or hours
old either way. What distinguishes them is how long the newest event has been the newest event,
compared against the longest agent call this run has actually completed.

Usage: run_progress.py <run_dir> [expected_ceiling_s]
"""
import json
import pathlib
import sys
import time

run = pathlib.Path(sys.argv[1])
ceiling = float(sys.argv[2]) if len(sys.argv) > 2 else 1800.0

evs = []
for ln in (run / "events.jsonl").read_text(encoding="utf-8").splitlines():
    try:
        evs.append(json.loads(ln))
    except Exception:  # noqa: BLE001
        continue

now = time.time()
last = evs[-1]
age = now - last["ts"]


def hhmm(ts):
    return time.strftime("%H:%M:%S", time.localtime(ts))


print(f"run           : {run.name}")
print(f"events        : {len(evs)}")
print(f"last event    : {last.get('type')} at {hhmm(last['ts'])}  ({age/60:.1f} min ago)")
print(f"now           : {hhmm(now)}")

# What is in flight, and for how long?
started = {}
completed = []
for e in evs:
    t, p = e.get("type"), (e.get("payload") or {})
    if t == "AGENT_CALL_STARTED":
        started[p.get("call_id") or p.get("module")] = (e["ts"], p.get("module"))
    elif t in ("AGENT_CALL_FINISHED", "AGENT_CALL_FAILED"):
        k = p.get("call_id") or p.get("module")
        if k in started:
            ts, mod = started.pop(k)
            completed.append((e["ts"] - ts, mod))

if started:
    print("\nin flight:")
    for k, (ts, mod) in started.items():
        print(f"  {mod:<16} started {hhmm(ts)}, running {(now-ts)/60:.1f} min "
              f"(ceiling {ceiling/60:.0f} min)")
else:
    print("\nin flight: nothing -- the orchestrator is between agent calls")

if completed:
    ds = sorted(d for d, _ in completed)
    print(f"\ncompleted agent calls: n={len(ds)}  "
          f"median {ds[len(ds)//2]/60:.1f} min  max {ds[-1]/60:.1f} min")
    slowest = max(completed)
    print(f"  slowest: {slowest[1]} at {slowest[0]/60:.1f} min")

# The verdict.
print()
longest = max((d for d, _ in completed), default=0.0)
if started:
    inflight = max(now - ts for ts, _ in started.values())
    if inflight > ceiling:
        print(f"VERDICT: STALLED -- an agent call has run {inflight/60:.1f} min, past the "
              f"{ceiling/60:.0f} min ceiling. The abort should have fired.")
    elif inflight > max(longest * 2, 600):
        print(f"VERDICT: SLOW -- {inflight/60:.1f} min in flight against a "
              f"{longest/60:.1f} min slowest-completed. Plausible but watch it.")
    else:
        print(f"VERDICT: HEALTHY -- {inflight/60:.1f} min in flight is within this run's "
              f"own range (slowest completed {longest/60:.1f} min).")
else:
    # "No agent call in flight" does NOT mean nothing is happening: a GPU job holds no
    # AGENT_CALL_STARTED event, and F5's batch prescreen legitimately runs for minutes on a
    # multi-kernel candidate (measured on L3:21: 76-260 s per prescreen, one observed at 19 min
    # while ptxas was actively compiling). An earlier version of this check ignored that and
    # cried SUSPICIOUS twice on a perfectly healthy run, which is worse than useless -- a
    # monitor that fires on healthy states trains you to ignore it.
    #
    # So look for the worker process before judging, and report a live worker as WORKING with
    # what it is doing. Only silence with NO worker and NO agent call is actually suspicious.
    worker = ""
    try:
        import subprocess  # noqa: PLC0415

        out = subprocess.run(["pgrep", "-af", "worker_main"], capture_output=True,
                             text=True, timeout=10).stdout
        for line in out.splitlines():
            if "pgrep" in line or "run_progress" in line:
                continue
            worker = line.strip()
            break
    except Exception:  # noqa: BLE001 — no pgrep, or not on the box; fall through
        worker = ""

    if worker:
        job = ""
        for tok in worker.split():
            if tok.endswith(".json") and "--out" not in tok:
                job = pathlib.Path(tok).name
        print(f"VERDICT: WORKING -- a GPU worker is running ({job or 'job unknown'}); "
              f"no event for {age/60:.1f} min is expected while it compiles.")
    elif age > 900:
        print(f"VERDICT: SUSPICIOUS -- no agent call, NO GPU worker, and no event for "
              f"{age/60:.1f} min. This one is worth investigating.")
    else:
        print(f"VERDICT: HEALTHY -- last event {age/60:.1f} min ago, between calls.")
