"""How much of that 95.6% survives an evidence requirement, and what would it MIS-KILL?

The over-interception question, answered by leave-one-out on real history: build the conditional
rule from all-but-one run, apply it to the held-out run, and count feasible configurations it would
have removed. A rule that kills points which actually ran is the failure mode to fear.
"""
import json, sys, glob, os
from collections import defaultdict
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))  # sibling audit scripts
from replay_sampler import _load_space_trials

runs = sorted(glob.glob(sys.argv[1] + "/run-l3-*"))
def parse(k): return dict(part.partition("=")[::2] for part in k.split("|"))

def rules_from(spaces, min_ev):
    """{(knob,val,other,oval)} judged dead: >= min_ev infeasible co-occurrences, 0 feasible."""
    out = set()
    for sid, s in spaces.items():
        if not s["infeasible"]: continue
        feas = [parse(k) for k in s["table"]]
        infs = [parse(k) for k in s["infeasible"]]
        knobs = sorted(s["domains"])
        for knob in knobs:
            for other in knobs:
                if other == knob: continue
                pair_i = defaultdict(int); pair_f = defaultdict(int)
                for c in infs:
                    if knob in c and other in c: pair_i[(c[knob], c[other])] += 1
                for c in feas:
                    if knob in c and other in c: pair_f[(c[knob], c[other])] += 1
                for (v, ov), n in pair_i.items():
                    if n >= min_ev and pair_f.get((v, ov), 0) == 0:
                        out.add((knob, v, other, ov))
    return out

print("%-8s %-10s %-14s %-16s %s" % ("min_ev", "rules", "held-out feas", "MIS-KILLED", "mis-kill rate"))
print("-"*72)
for min_ev in (1, 2, 3, 5, 8):
    tot_rules = tot_feas = tot_kill = 0
    for held in runs:
        train = {}
        for r in runs:
            if r != held: train.update(_load_space_trials(Path(r)))
        rules = rules_from(train, min_ev)
        test = _load_space_trials(Path(held))
        for sid, s in test.items():
            for k in s["table"]:            # FEASIBLE, measured points in the held-out run
                c = parse(k)
                tot_feas += 1
                if any(c.get(kn) == v and c.get(ot) == ov for (kn, v, ot, ov) in rules):
                    tot_kill += 1
        tot_rules += len(rules)
    print("%-8d %-10d %-14d %-16d %.2f%%" % (
        min_ev, tot_rules // max(1,len(runs)), tot_feas, tot_kill,
        100.0*tot_kill/max(1,tot_feas)))
print()
print("MIS-KILLED = a configuration that RAN in the held-out run, removed by a rule learned")
print("elsewhere. Non-zero at any threshold means the mechanism kills reachable points.")
