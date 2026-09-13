"""C2's negative result: WHY did the steer not convert? Measure the four candidate causes.

THE FINDING TO EXPLAIN. 2e's mechanism works (the attributed wall was freed, 1/1 against arm 2's 0/7)
yet arm 3 finished 8.0% SLOWER than the same configuration without 2e, and its winner came from a
family the wall text never reached.

THE STRUCTURAL FACT THAT FRAMES ALL OF IT (`orchestrator.py:1710`, in the code's own words): 2e runs
"outside the tuning loop, so it cannot touch the trial budget or what the sampler sees". So the slope
it measures is delivered ONLY as prompt text, AFTER that candidate's tuning has finished. Four
measurable consequences, and this script prices each:

  H1 TOO LATE. The wall is found after tuning ends, so the run learns which knob is truncated only
     once it has stopped sampling that knob. Measured as: how many trials did the candidate spend
     before the wall was known, and how many after (it is zero after, by construction -- the point is
     the size of the number that came before).

  H2 NEVER STEERS THE SAMPLER. `find_walls` reads `stats.param_stats[].latency_by_value` -- the same
     per-choice median table the tuner already built. So the slope EXISTS inside the tuning loop and
     is discarded there. Measured as: at the moment tuning ended, which knob had the steepest tail
     slope, and how many of that candidate's trials had actually explored it?

  H3 REACHES ONE FAMILY ONLY. The text goes to the rewriter for the walled candidate's family. Every
     other family in the run gets nothing, so with 4 families the steer covers at most a quarter of
     the rewrite budget. Measured as: families with a wall against families total.

  H4 THE SLOPE IS SMALL WHERE IT IS ATTRIBUTABLE. `select_for_probing` keeps only monotone,
     positive-slope walls, and attribution needs the compile probe to agree. Measured as: the tail
     gain of the wall that got ATTRIBUTED against the ones that did not.

     H4 WAS TESTED AND DOES NOT GENERALISE. On arm 3 it looks damning -- the attributed wall had a
     7.5% slope while 60.7%, 25.6% and 10.7% went unattributed. But pooled over four runs with the
     shipped `find_walls` offline (`h4_is_the_attributable_wall_the_weakest.py`), probe-worthy slopes
     are 7.5, 25.6, 25.6, 25.7, 45.3 and 56.0% against non-worthy ones of -359.2, -177.8, -0.7, 10.7,
     11.2, 12.3, 16.3 and 57.9%: the filter keeps the steep ones about as often as it drops them, and
     it was the steepest available in 2 of the 3 runs that had any. So arm 3's 7.5% is THAT RUN's bad
     luck, not a design that selects against its own payoff. Kept in this script because the
     per-run reading is still worth seeing -- but it must not be reported as a general cause.

Read-only, from events.jsonl. No GPU.
"""
import json
import os
import sys
from collections import defaultdict


def load(rd):
    out = []
    with open(os.path.join(rd, "events.jsonl"), encoding="utf-8") as fh:
        for ln in fh:
            ln = ln.strip()
            if ln:
                try:
                    out.append(json.loads(ln))
                except Exception:
                    pass
    return out


rd = sys.argv[1]
evs = load(rd)
t0 = evs[0]["ts"]

# Walls, with the timestamp at which each became known.
walls = []
for e in evs:
    if e.get("type") != "RESOURCE_WALL_ATTRIBUTED":
        continue
    p = e.get("payload") or {}
    for w in (p.get("walls") or []):
        walls.append({"ts": e["ts"], "cand": p.get("candidate_id"), **w})

fam_of = {}
for e in evs:
    if e.get("type") == "CANDIDATE_REGISTERED":
        c = (e.get("payload") or {}).get("candidate") or {}
        if c.get("candidate_id"):
            fam_of[c["candidate_id"]] = c.get("family_id")

trials = defaultdict(list)
for e in evs:
    if e.get("type") != "TRIAL_DONE":
        continue
    t = (e.get("payload") or {}).get("trial") or {}
    if t.get("candidate_id"):
        trials[t["candidate_id"]].append((e["ts"], t))

print(f"=== {os.path.basename(rd)}")
print(f"walls found {len(walls)}; attributed "
      f"{sum(1 for w in walls if w.get('verdict') == 'attributed')}")
print()

# ---- H1: the wall arrives after that candidate's tuning is over.
print("H1  TOO LATE -- trials spent before vs after the wall was known")
for w in walls:
    cid = w["cand"]
    xs = trials.get(cid, [])
    before = sum(1 for ts, _ in xs if ts <= w["ts"])
    after = sum(1 for ts, _ in xs if ts > w["ts"])
    print(f"    {cid} {w.get('param')}: {before} trials BEFORE the wall was known, "
          f"{after} after   (verdict {w.get('verdict')})")
print("    => the sampler never sees the wall for the candidate the wall is about.")
print()

# ---- H2: the slope existed inside the tuning loop and was thrown away.
print("H2  NEVER STEERS THE SAMPLER -- the slope came from the tuner's OWN per-choice table")
for w in walls:
    cid = w["cand"]
    knob = w.get("param")
    xs = trials.get(cid, [])
    # How much of this candidate's sampling actually touched the walled knob's extreme values?
    at_top = 0
    ran = [float(v) for v in (w.get("ran_values") or []) if isinstance(v, (int, float))]
    top = max(ran) if ran else None
    for _, t in xs:
        vals = (t.get("params") or {}).get("values") or {}
        if knob in vals and top is not None:
            try:
                if float(vals[knob]) >= top:
                    at_top += 1
            except (TypeError, ValueError):
                pass
    print(f"    {cid} {knob}: tail slope {w.get('tail_gain_pct')}%, ran {w.get('ran_values')}, "
          f"{at_top}/{len(xs)} trials at the top measured value {top}")
print("    => `find_walls` reads stats.param_stats[].latency_by_value, which the tuner BUILT.")
print("       The steepest-slope knob is therefore knowable DURING tuning and is used only after.")
print()

# ---- H3: coverage across families.
fams_all = {f for f in fam_of.values() if f}
fams_walled = {fam_of.get(w["cand"]) for w in walls if fam_of.get(w["cand"])}
fams_attr = {fam_of.get(w["cand"]) for w in walls
             if w.get("verdict") == "attributed" and fam_of.get(w["cand"])}
print("H3  REACHES ONE FAMILY ONLY")
print(f"    families in the run: {len(fams_all)}  {sorted(fams_all)}")
print(f"    families with ANY wall: {len(fams_walled)}  {sorted(fams_walled)}")
print(f"    families with an ATTRIBUTED wall (the only ones whose rewriter got the text): "
      f"{len(fams_attr)}  {sorted(fams_attr)}")
if fams_all:
    print(f"    => the steer covered {100.0 * len(fams_attr) / len(fams_all):.0f}% of families; "
          f"the other {len(fams_all) - len(fams_attr)} rewrote with no wall information at all.")
print()

# ---- H4: is the attributable wall the least valuable one?
print("H4  SLOPE WHERE IT IS ATTRIBUTABLE")
attr = [w for w in walls if w.get("verdict") == "attributed"]
other = [w for w in walls if w.get("verdict") != "attributed"]
def gains(ws):
    return sorted(round(float(w.get("tail_gain_pct") or 0.0), 1) for w in ws)
print(f"    ATTRIBUTED tail gains: {gains(attr)}")
print(f"    not attributed / not probed: {gains(other)}")
if attr and other:
    best_attr = max(float(w.get("tail_gain_pct") or 0) for w in attr)
    best_other = max(float(w.get("tail_gain_pct") or 0) for w in other)
    print(f"    best attributable slope {best_attr:.1f}% against best non-attributable "
          f"{best_other:.1f}%")
    if best_other > best_attr:
        print("    => in THIS run the steepest slopes were not the attributable ones.")
        print("       NOT a general cause: pooled over four runs the probe-worthy slopes are")
        print("       7.5/25.6/25.6/25.7/45.3/56.0% against non-worthy -359/-178/-0.7/10.7/11.2/")
        print("       12.3/16.3/57.9%, and the filter kept the steepest available in 2 of 3 runs.")
        print("       See h4_is_the_attributable_wall_the_weakest.py -- this is arm3's bad luck.")
