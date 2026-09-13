"""Is this arm alive and progressing, or stalled? Read from events.jsonl only.

Answers the one question a monitor cannot: a run past its wall-clock budget is either (a) inside a
long tuning pass, because the budget check is per CANDIDATE and cannot interrupt one, or (b) stalled.
The discriminator is the AGE of the newest event, not the elapsed total -- so print it.
"""
import json
import os
import sys
import time
from collections import Counter

for spec in sys.argv[1:]:
    label, rd = spec.split("=", 1)
    path = os.path.join(rd, "events.jsonl")
    if not os.path.isfile(path):
        print(f"=== {label}: NO events.jsonl at {rd}")
        continue
    evs = []
    for ln in open(path, encoding="utf-8"):
        ln = ln.strip()
        if ln:
            try:
                evs.append(json.loads(ln))
            except Exception:
                pass
    now = time.time()
    c = Counter(e.get("type") for e in evs)
    print(f"=== {label} :: {os.path.basename(rd)}")
    print(f"  events {len(evs)}   span {(evs[-1]['ts'] - evs[0]['ts']) / 3600.0:.3f}h   "
          f"newest event {(now - evs[-1]['ts']) / 60.0:.1f} min old")
    print("  tail:")
    for e in evs[-8:]:
        p = e.get("payload") or {}
        extra = ""
        for k in ("step_key", "candidate_id", "family_id", "elapsed_hours", "round"):
            if p.get(k) is not None:
                extra += f" {k}={p[k]}"
        print(f"    {(now - e['ts']) / 60.0:8.1f}min  {e.get('type'):28s}{extra}")
    for k in ("TRIAL_DONE", "REWRITE_PRODUCED", "FAMILY_ROUND_RECORDED", "WALL_CLOCK_REACHED",
              "CONVERGENCE_DECIDED", "RUN_FINISHED", "RUN_INTERRUPTED", "AGENT_CALL_STARTED",
              "AGENT_CALL_FINISHED", "RESOURCE_WALL_ATTRIBUTED"):
        if c.get(k):
            print(f"  {k:26s} {c[k]}")

    # AGENT_CALL_FAILED fires PER ATTEMPT (`agents/base.py:145`) and again at the end with
    # `final: True` (`base.py:289`). So the raw count is attempts-that-needed-a-retry, not calls
    # that failed, and subtracting it from AGENT_CALL_STARTED gives nonsense -- it printed "-1 calls
    # in flight" for arm 3, which is how this was found. `reporting/report.py:542` filters on
    # `final`; every reader must. A retried-then-succeeded call is a COST (it burns the timeout),
    # not a loss, and reporting it as a loss would make the arms look unequal when they are not.
    final_failed = [e for e in evs if e.get("type") == "AGENT_CALL_FAILED"
                    and (e.get("payload") or {}).get("final")]
    retried = c.get("AGENT_CALL_FAILED", 0) - len(final_failed)
    print(f"  agent calls               : {c.get('AGENT_CALL_STARTED', 0)} started, "
          f"{c.get('AGENT_CALL_FINISHED', 0)} finished, {len(final_failed)} failed FINALLY, "
          f"{retried} retried attempt(s)")
    for e in final_failed:
        p = e.get("payload") or {}
        print(f"      FINAL FAIL {p.get('module')} {str(p.get('error'))[:120]}")

    # A call that STARTED and never reached a terminal event is the one stall this harness can
    # actually have: the read timeout is per-READ idle, not a call ceiling (measured 4057 s), so a
    # call can sit for hours without tripping anything. Paired by call_id, because counting cannot.
    open_calls: dict[str, dict] = {}
    for e in evs:
        t = e.get("type")
        p = e.get("payload") or {}
        cid = p.get("call_id")
        if not cid:
            continue
        if t == "AGENT_CALL_STARTED":
            open_calls[cid] = e
        elif t == "AGENT_CALL_FINISHED" or (t == "AGENT_CALL_FAILED" and p.get("final")):
            open_calls.pop(cid, None)
    if open_calls:
        print(f"  *** {len(open_calls)} agent call(s) in flight")
        for cid, e in open_calls.items():
            p = e.get("payload") or {}
            print(f"      {(now - e['ts']) / 60.0:8.1f}min  module={p.get('module')} {cid}")
