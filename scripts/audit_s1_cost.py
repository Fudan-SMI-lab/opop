"""What does an `infeasible_shared_memory` trial COST now, versus a real trial?

`ts` is a float epoch (verified on disk), not an ISO string. An earlier version of this script
parsed it as ISO and got None for every event -- n=0 across the board, which reads as "no data"
rather than "the reader is wrong". The same silent-zero shape as G10/G26/G33.
"""
import json, sys, glob, os, statistics as st

def parse(ts):
    if ts is None: return None
    try: return float(ts)
    except Exception: return None

infeas, mismatch, complete = [], [], []
for r in sorted(glob.glob(sys.argv[1] + "/run-l3-*")):
    ev = os.path.join(r, "events.jsonl")
    if not os.path.exists(ev): continue
    prev_t = None
    for line in open(ev, encoding="utf-8"):
        try: e = json.loads(line)
        except Exception: continue
        t = parse(e.get("ts"))
        if t is None: continue
        if e.get("type") == "TRIAL_DONE":
            tr = (e.get("payload") or {}).get("trial") or (e.get("payload") or {})
            if prev_t is not None:
                gap = t - prev_t
                if 0 <= gap < 600:
                    fk = tr.get("failure_kind")
                    if fk == "infeasible_shared_memory": infeas.append(gap)
                    elif fk == "correctness_mismatch": mismatch.append(gap)
                    elif not fk: complete.append(gap)
            prev_t = t
        elif e.get("type") in ("SPACE_PUBLISHED", "QUICKTEST_DONE"):
            prev_t = t

def show(name, xs):
    if not xs: print("  %-28s n=0" % name); return
    print("  %-28s n=%4d  median %7.2f s  mean %7.2f s  total %8.1f s" % (
        name, len(xs), st.median(xs), st.mean(xs), sum(xs)))

print("Inter-TRIAL_DONE gap, by outcome (bounds the per-trial cost):")
show("infeasible_shared_memory", infeas)
show("correctness_mismatch", mismatch)
show("complete", complete)
print()
if infeas and complete:
    print("A screened-infeasible trial costs %.1f%% of a completed one (median)." %
          (100.0 * st.median(infeas) / st.median(complete)))
    print("Total wall clock inside screened-infeasible trials: %.1f s = %.3f h" %
          (sum(infeas), sum(infeas)/3600.0))
