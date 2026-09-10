"""Does the freed budget have a CONSUMER? Did any space exhaust trials_per_space?

If a space stops for a reason other than running out of trials, budget returned by S1 is not spent.
"""
import json, sys, glob, os
from collections import Counter, defaultdict

per_space = defaultdict(Counter)
run_of = {}
for r in sorted(glob.glob(sys.argv[1] + "/run-l3-*")):
    ev = os.path.join(r, "events.jsonl")
    if not os.path.exists(ev): continue
    for line in open(ev, encoding="utf-8"):
        try: e = json.loads(line)
        except Exception: continue
        if e.get("type") == "TRIAL_DONE":
            tr = (e.get("payload") or {}).get("trial") or (e.get("payload") or {})
            sid = tr.get("space_id")
            if sid:
                per_space[sid]["n"] += 1
                per_space[sid][tr.get("failure_kind") or "complete"] += 1
                run_of[sid] = os.path.basename(r)

BUDGET = 40
hit, under = 0, 0
dist = Counter()
for sid, c in per_space.items():
    dist[c["n"]] += 1
    if c["n"] >= BUDGET: hit += 1
    else: under += 1
print("spaces: %d ; reached trials_per_space=%d: %d ; stopped short: %d" % (len(per_space), BUDGET, hit, under))
print()
print("trial-count distribution (how many spaces ran exactly N trials):")
for n in sorted(dist):
    print("  %3d trials : %2d spaces %s" % (n, dist[n], "*" * dist[n]))
print()
print("spaces WITH infeasible trials, and whether they hit the budget:")
print("  %-14s %-30s %5s %6s %6s" % ("space", "run", "n", "infeas", "hit?"))
for sid, c in sorted(per_space.items()):
    if c.get("infeasible_shared_memory"):
        print("  %-14s %-30s %5d %6d %6s" % (sid[:14], run_of[sid][:30], c["n"],
                                             c["infeasible_shared_memory"],
                                             "YES" if c["n"] >= BUDGET else "no"))
