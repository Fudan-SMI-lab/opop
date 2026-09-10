"""Does the in-sampler guard already suppress dead points? And is the screen's cache warm?

If `_shared_memory_ok` already rejects a dead draw INSIDE ask(), then those points never become
trials -- and the 113 that DID become trials are the ones the cache had no opinion on. That is the
number S1 has to beat, and a domain shrink built on the same cache cannot see them either.
"""
import json, sys, glob, os
from collections import Counter

tot = Counter()
for r in sorted(glob.glob(sys.argv[1] + "/run-l3-*")):
    ev = os.path.join(r, "events.jsonl")
    if not os.path.exists(ev): continue
    n_screen_fail = n_prescreen = 0
    probed = infeas_found = 0
    for line in open(ev, encoding="utf-8"):
        try: e = json.loads(line)
        except Exception: continue
        t = e.get("type"); p = e.get("payload") or {}
        if t == "PRESCREEN_DONE" or t == "SPACE_PRESCREENED":
            n_prescreen += 1
            probed += int(p.get("configs_probed") or 0)
            infeas_found += int(p.get("infeasible") or 0)
        tot[t] += 1
    print("%-32s prescreen events %2d  configs_probed %4d  found infeasible %3d" % (
        os.path.basename(r), n_prescreen, probed, infeas_found))
print()
print("=== every event type containing PRESCREEN / SCREEN ===")
for k, v in sorted(tot.items()):
    if "SCREEN" in k.upper() or "PRESCR" in k.upper():
        print("  %-40s %5d" % (k, v))
