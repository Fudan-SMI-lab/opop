"""Is this arm alive and progressing, or stalled? Read from events.jsonl only.

Answers the one question a monitor cannot: a run past its wall-clock budget is either (a) inside a
long tuning pass, because the budget check is per CANDIDATE and cannot interrupt one, or (b) stalled.
The discriminator is the AGE of the newest event, not the elapsed total -- so print it.
"""
import json
import os
import subprocess
import sys
import time
from collections import Counter


def _proc_age_s(pattern: str) -> float | None:
    """Age in seconds of a process matching `pattern`, or None if none is running.

    events.jsonl alone cannot separate "the process died" from "the process is doing something
    that writes no event": both look like a stale newest-event age. This closes that gap.

    `[p]attern` so the grep does not match its own command line -- a probe that names its target
    matches itself, which has produced 20+ false hits in this project.
    """
    if not pattern:
        return None
    pat = "[" + pattern[0] + "]" + pattern[1:]
    try:
        r = subprocess.run(
            ["bash", "-lc", "ps -eo etimes,args | grep -- '%s' | head -1" % pat],
            capture_output=True, timeout=20)
    except (subprocess.SubprocessError, OSError):
        return None
    txt = r.stdout.decode("utf-8", "replace").strip()
    if not txt:
        return None
    try:
        return float(txt.split(None, 1)[0])
    except (IndexError, ValueError):
        return None


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
    age_min = (now - evs[-1]["ts"]) / 60.0
    print(f"  events {len(evs)}   span {(evs[-1]['ts'] - evs[0]['ts']) / 3600.0:.3f}h   "
          f"newest event {age_min:.1f} min old")
    # A stale event age has two causes and they call for opposite actions. Ask the process table.
    # The arm name is the directory holding the run dir, taken with basename/dirname rather than
    # index arithmetic on split("/") -- a trailing slash or an extra segment silently shifts an
    # index and would hand the grep a pattern that never matches, printing "not matched" forever.
    arm_name = os.path.basename(os.path.dirname(os.path.normpath(rd)))
    proc_age = _proc_age_s(arm_name)
    if proc_age is not None:
        print(f"  process ALIVE, {proc_age / 3600.0:.2f}h old")
    elif age_min > 30.0:
        print("  *** NO MATCHING PROCESS and the newest event is over 30 min old -- this arm looks")
        print("      DEAD, not slow. Check for an orphaned `opencode serve`: the orchestrator has")
        print("      no signal handler, so SIGTERM skips OpencodeServer.stop().")
    else:
        print("  process not matched (the pattern may not fit this launch) -- events are recent,")
        print("  so absence of a match is not evidence of death here")
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
            age_s = now - e["ts"]
            # Price the wait against the deadline that actually applies, so "busy" and "past every
            # deadline the harness has" are not both printed as a number of minutes. A flat trial
            # count during a loop-C agent call is normal and says nothing on its own -- the arm is
            # not producing trials because it is not tuning.
            verdict = ("within the 1800 s request deadline"
                       if age_s <= 1800.0 else
                       f"PAST the 1800 s request deadline by {age_s - 1800.0:.0f} s")
            print(f"      {age_s / 60.0:8.1f}min  module={p.get('module')} {cid}")
            print(f"                {verdict}")
        print("      (a run in loop C emits no TRIAL_DONE, so a flat trial count here is the")
        print("       agent working, not a stall. The discriminator is this age, not the count.)")
