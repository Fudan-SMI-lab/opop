"""Is a value's failure intrinsic to the value, or CONDITIONAL on the other knobs?

For a given (candidate, knob, value), split its trials by the other knobs' values and show
pass rate per co-value. If failure is conditional, retiring the value is over-blocking:
the value works, just not in combination with certain partners.
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

run = Path(sys.argv[1])
target_knob = sys.argv[2]
target_val = sys.argv[3]

# trials for this knob=value: list of (ok, other_params)
rows = []
with open(run / "events.jsonl", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        try:
            e = json.loads(line)
        except Exception:
            continue
        if e.get("type") != "TRIAL_DONE":
            continue
        tr = (e.get("payload") or {}).get("trial") or {}
        vals = (tr.get("params") or {}).get("values") or {}
        if str(vals.get(target_knob)) != target_val:
            continue
        ok = (tr.get("status") or "").lower() == "complete"
        rows.append((ok, tr.get("candidate_id"), tr.get("failure_kind"),
                     {k: v for k, v in vals.items() if k != target_knob}))

if not rows:
    print("no trials with %s=%s in %s" % (target_knob, target_val, run.name))
    sys.exit(0)

n_ok = sum(1 for r in rows if r[0])
print("=== %s: %s=%s -> %d/%d passed" % (run.name, target_knob, target_val, n_ok, len(rows)))
print()

# per co-knob co-value pass rate
co = defaultdict(lambda: [0, 0])
for ok, _cand, _fk, others in rows:
    for k, v in others.items():
        slot = co[(k, str(v))]
        slot[0 if ok else 1] += 1

print("pass rate of %s=%s, split by each OTHER knob's value:" % (target_knob, target_val))
lastk = None
for (k, v) in sorted(co):
    ok, bad = co[(k, v)]
    if k != lastk:
        print("  --- %s" % k)
        lastk = k
    flag = ""
    if ok + bad >= 3:
        if ok == 0:
            flag = "   <-- NEVER passes with this partner"
        elif bad == 0:
            flag = "   <-- ALWAYS passes with this partner"
    print("      %-14s %2d/%2d%s" % (v, ok, ok + bad, flag))

print()
fk = defaultdict(int)
for ok, _c, k, _o in rows:
    if not ok:
        fk[k] += 1
print("failure kinds:", dict(fk))
