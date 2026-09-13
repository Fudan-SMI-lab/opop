"""One arm's result, read from its own events.jsonl -- never from a report or a notification.

Prints what an arm comparison needs: the ending, the event SPAN (not the budget, since every finished
run so far overran), loop-C activity per family, the winner with its independent re-eval, and the
2e counters when the arm ran it.
"""
import json
import os
import sys
import time
from collections import Counter, defaultdict

rd = sys.argv[1]
label = sys.argv[2] if len(sys.argv) > 2 else os.path.basename(rd)
evs = []
for ln in open(os.path.join(rd, "events.jsonl"), encoding="utf-8"):
    ln = ln.strip()
    if ln:
        try:
            evs.append(json.loads(ln))
        except Exception:
            pass
c = Counter(e.get("type") for e in evs)
span = (evs[-1]["ts"] - evs[0]["ts"]) / 3600.0

print(f"=== {label} :: {os.path.basename(rd)} ===")
term = [e for e in evs if e.get("type") in ("RUN_FINISHED", "RUN_INTERRUPTED")]
print(f"ending          : {term[-1]['type'] if term else 'STILL RUNNING'}")
wc = [e for e in evs if e.get("type") == "WALL_CLOCK_REACHED"]
if wc:
    p = wc[-1].get("payload") or {}
    print(f"                  wall clock {p.get('elapsed_hours')}h of {p.get('budget_hours')}h")
# The span, because a resumed or overrunning run's own clock is not the budget.
print(f"event span      : {span:.3f}h   events {len(evs)}")
print(f"trials          : {c.get('TRIAL_DONE', 0)}")

fams = defaultdict(lambda: {"rewrites": 0, "rounds": 0})
for e in evs:
    p = e.get("payload") or {}
    f = p.get("family_id")
    if not f:
        continue
    if e.get("type") == "REWRITE_PRODUCED":
        fams[f]["rewrites"] += 1
    if e.get("type") == "FAMILY_ROUND_RECORDED":
        fams[f]["rounds"] += 1
print(f"loop C          : {c.get('REWRITE_PRODUCED', 0)} rewrites, "
      f"{c.get('FAMILY_ROUND_RECORDED', 0)} rounds recorded, over {len(fams)} families")
for f, v in sorted(fams.items()):
    print(f"                  {f}: {v['rewrites']} rewrites / {v['rounds']} rounds")

fin = term[-1] if term and term[-1]["type"] == "RUN_FINISHED" else None
if fin:
    s = (fin.get("payload") or {}).get("summary") or {}
    best = s.get("best") or {}
    for k in ("candidate_id", "family_id", "best_ms", "final_reeval_ms",
              "same_precision_speedup", "compared_against", "beats_same_precision_baseline"):
        if k in s or k in best:
            print(f"  {k:30s} {s.get(k, best.get(k))}")

wall = [e for e in evs if e.get("type") == "RESOURCE_WALL_ATTRIBUTED"]
if wall:
    found = sum(int((e.get("payload") or {}).get("walls_found") or 0) for e in wall)
    refused = sum(int((e.get("payload") or {}).get("n_refused_configs") or 0) for e in wall)
    probed = sum(int((e.get("payload") or {}).get("walls_probed") or 0) for e in wall)
    attributed = sum(1 for e in wall for w in ((e.get("payload") or {}).get("walls") or [])
                     if w.get("verdict") == "attributed")
    print(f"2e              : {len(wall)} events, {refused} refused configs, "
          f"{found} walls found, {probed} probed, {attributed} ATTRIBUTED")
    for e in wall:
        for w in ((e.get("payload") or {}).get("walls") or []):
            print(f"                  {(e.get('payload') or {}).get('candidate_id')} "
                  f"{w.get('param')} ran={w.get('ran_values')} refused={w.get('refused_value')} "
                  f"monotone={w.get('monotone')} tail={w.get('tail_gain_pct')} "
                  f"verdict={w.get('verdict')}")
# AGENT_CALL_FAILED is deliberately EXCLUDED from this catch-all and reported separately below,
# because it fires per ATTEMPT (`agents/base.py:145`) as well as finally (`base.py:289`). Counting it
# raw turned one retried-then-succeeded analyst call into "arm3: 1 failed call", which I reported as
# a difference between the arms. `reporting/report.py:542` filters on `final`; so must every reader.
bad = {t: n for t, n in c.items()
       if ("FAIL" in t or "REJECT" in t or "ERROR" in t) and t != "AGENT_CALL_FAILED"}
if bad:
    print(f"problems        : {bad}")

final_failed = [e for e in evs if e.get("type") == "AGENT_CALL_FAILED"
                and (e.get("payload") or {}).get("final")]
retried = c.get("AGENT_CALL_FAILED", 0) - len(final_failed)
print(f"agent calls     : {c.get('AGENT_CALL_STARTED', 0)} started, "
      f"{c.get('AGENT_CALL_FINISHED', 0)} finished, {len(final_failed)} failed FINALLY, "
      f"{retried} retried attempt(s) —— 重试是成本(吃掉超时)而不是损失")
for e in final_failed:
    p = e.get("payload") or {}
    print(f"                  FINAL FAIL {p.get('module')} {str(p.get('error'))[:160]}")
