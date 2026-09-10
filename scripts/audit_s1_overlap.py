"""Can a CONDITIONAL domain shrink separate feasible from infeasible at all?

S1 path A shrinks the domain of one knob given the others already bound. That is only sound if, for
some binding, a value is ALWAYS infeasible. Test it directly: for every (space, knob, value), count
feasible and infeasible recorded configurations, and then ask the conditional question -- given each
partner assignment actually observed, is the value ever cleanly dead?
"""
import json, sys, glob, os
from collections import defaultdict

from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))  # sibling audit scripts
from replay_sampler import _load_space_trials

spaces = {}
for r in sorted(glob.glob(sys.argv[1] + "/run-l3-*")):
    spaces.update(_load_space_trials(Path(r)))

def parse(k):
    return dict(part.partition("=")[::2] for part in k.split("|"))

tot_pairs = tot_clean = 0
tot_uncond = 0
per_space_clean = {}
for sid, s in spaces.items():
    if not s["infeasible"] or not s["table"]:
        continue
    feas = [parse(k) for k in s["table"]]
    infs = [parse(k) for k in s["infeasible"]]
    knobs = sorted(s["domains"])
    clean = uncond = pairs = 0
    for knob in knobs:
        vals = {str(v) for v in s["domains"][knob]}
        for v in vals:
            f = [c for c in feas if c.get(knob) == v]
            i = [c for c in infs if c.get(knob) == v]
            if not i:
                continue
            pairs += 1
            if not f:
                uncond += 1          # infeasible beside every observed partner
            # CONDITIONAL: is there a single OTHER knob whose value makes (knob=v) always dead?
            found = False
            for other in knobs:
                if other == knob: continue
                for ov in {c.get(other) for c in i if c.get(other) is not None}:
                    fi = [c for c in f if c.get(other) == ov]
                    ii = [c for c in i if c.get(other) == ov]
                    if ii and not fi:
                        found = True; break
                if found: break
            if found:
                clean += 1
    tot_pairs += pairs; tot_clean += clean; tot_uncond += uncond
    per_space_clean[sid] = (pairs, clean, uncond)

print("Over %d spaces with recorded infeasible points:" % len(per_space_clean))
print("  (knob,value) pairs that ever appear in an INFEASIBLE config : %d" % tot_pairs)
print("  of those, UNCONDITIONALLY dead (never in a feasible config)  : %d  (%.1f%%)" % (
    tot_uncond, 100.0*tot_uncond/max(1,tot_pairs)))
print("  of those, CONDITIONALLY dead given one other knob's value    : %d  (%.1f%%)" % (
    tot_clean, 100.0*tot_clean/max(1,tot_pairs)))
print()
print("NOTE: 'conditionally dead' here is the MOST OPTIMISTIC reading -- it counts a pair as")
print("separable if ANY single partner value co-occurs only with failures, on as few as ONE")
print("observation. It is an upper bound on what a shrink could act on, not an estimate.")
