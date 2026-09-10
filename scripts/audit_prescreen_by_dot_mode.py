"""Is the pre-screen removing split3 from the search, or just its infeasible corners?

The pre-screen exists because 180 of 1004 trials on L3:43 were shared-memory failures carrying
their own proof; refusing them before paying for a launch is correct. But split3 doubles the
operand tiles, so a screen that rejects EVERY split3 config would silently delete the contract
change's main remedy from the space -- the tuner would then report on a knob value it never
measured, which is the same defect as the (split3, tf32) fall-through.
"""
import json, pathlib, sys, collections
d = sys.argv[1]
tag = sys.argv[2] if len(sys.argv) > 2 else pathlib.Path(d).name
scr = collections.Counter(); ran = collections.Counter()
for l in open(d + "/events.jsonl", encoding="utf-8"):
    e = json.loads(l); t = e["type"]; p = e.get("payload") or {}
    if t == "CONFIG_SCREENED_INFEASIBLE":
        v = p.get("params") or {}
        scr[v.get("DOT_MODE")] += 1
    elif t == "TRIAL_DONE":
        tr = p.get("trial") or {}
        v = ((tr.get("params") or {}).get("values")) or {}
        if tr.get("status") == "complete" and not tr.get("failure_kind"):
            ran[v.get("DOT_MODE")] += 1
modes = set(scr) | set(ran)
print("%s  by DOT_MODE: screened-out vs completed" % tag)
for m in sorted(modes, key=str):
    s, r = scr[m], ran[m]
    tot = s + r
    print("   %-8s screened=%-4d completed=%-4d  screened share %.0f%%" % (
        m, s, r, 100*s/tot if tot else 0))
if ran.get("split3", 0) == 0 and scr.get("split3", 0) > 0:
    print("   => split3 NEVER completed a trial: the remedy is absent from the measurements")
else:
    print("   => split3 completed %d trials: it IS being measured" % ran.get("split3", 0))
