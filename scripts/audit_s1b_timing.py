"""Why does S1b save so little even when it fires safely? Measure the timing, do not assert it.

At floor N=8 the rule mis-kills nothing and still fires on 26 values -- but saves only 59 failing
trials. The suspected mechanism is arithmetic rather than statistical: to be safe the rule must see N
failures of a value, while the value only APPEARS about budget/|choices| times in the whole space, so
most of its trials are already spent by the time the rule may act.

If that is the mechanism it is a hard ceiling, not a tuning problem: raising N to buy safety directly
costs benefit, and the two cannot both be had inside a 40-trial budget.
"""
import sys, glob
from pathlib import Path
from statistics import median

sys.path.insert(0, str(Path(__file__).resolve().parent))  # sibling audit scripts
from audit_s1b_prospective import load_ordered, first_fire_index  # noqa: E402

runs = sorted(glob.glob(sys.argv[1] + "/run-l3-*"))

print("%-8s %-7s %-14s %-14s %-14s %s" % (
    "floor N", "fired", "appearances", "before fire", "after fire", "max saveable share"))
print("-" * 90)
for floor in (3, 6, 8, 12):
    fired = 0
    tot_app = tot_before = tot_after = 0
    shares = []
    for r in runs:
        for sid, s in load_ordered(Path(r)).items():
            seq = s["seq"]
            if len(seq) < 5:
                continue
            for knob, choices in s["domains"].items():
                for value in {str(c) for c in choices}:
                    i = first_fire_index(seq, knob, value, floor, True, True)
                    if i is None:
                        continue
                    app = [j for j, (cfg, _) in enumerate(seq) if cfg.get(knob) == value]
                    before = [j for j in app if j <= i]
                    after = [j for j in app if j > i]
                    fired += 1
                    tot_app += len(app)
                    tot_before += len(before)
                    tot_after += len(after)
                    if app:
                        shares.append(len(after) / len(app))
    print("%-8d %-7d %-14d %-14d %-14d %s" % (
        floor, fired, tot_app, tot_before, tot_after,
        ("%.1f%% (median %.0f%%)" % (100.0 * tot_after / tot_app,
                                     100.0 * median(shares))) if tot_app else "n/a"))

print()
print("appearances = every trial in the space using that value. 'before fire' is evidence the rule")
print("had to spend to become safe; 'after fire' is all it could ever avoid. The last column is the")
print("CEILING on S1b's benefit for the values it fires on -- independent of implementation.")
print()

# The arithmetic that predicts it, checked against the corpus rather than assumed.
sizes = []
for r in runs:
    for sid, s in load_ordered(Path(r)).items():
        if len(s["seq"]) < 5:
            continue
        for knob, choices in s["domains"].items():
            sizes.append(len(set(str(c) for c in choices)))
if sizes:
    print("knob domain sizes: median %.0f choices, so a value appears about 40/%.0f = %.1f times in a"
          % (median(sizes), median(sizes), 40.0 / median(sizes)))
    print("40-trial budget IF sampling were uniform. A floor of N=8 therefore cannot fire before the")
    print("value's trials are mostly spent -- which is what the table above shows, measured.")
    print()
    print("CAVEAT on that arithmetic: TPE does NOT sample uniformly, and a value that keeps failing")
    print("gets sampled MORE by a sampler that reports failures as FAIL (Optuna prunes those points,")
    print("so the share is not 1/|choices|). The measured columns above are the load-bearing ones;")
    print("this line only says why the direction is expected.")
