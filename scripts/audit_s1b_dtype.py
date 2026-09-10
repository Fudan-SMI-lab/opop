"""S1b's own acceptance criterion: does it save the tf32 class, and at what count floor?

The plan's J for S1b is "tf32 的 172 个 trial 至少省 100" -- the motivating case is a dtype value
that is 0-for-172, unconditionally, in every partner combination. So ask the rule directly: which
values does it fire on, how many failing trials does each save, and is the dtype class among them?

Reported per count floor, because the previous measurement showed the floor is what carries the
safety: the plan's median-M and live-control conditions add ~nothing over a plain count threshold.
"""
import json, sys, glob
from collections import defaultdict
from pathlib import Path
from statistics import median

sys.path.insert(0, str(Path(__file__).resolve().parent))  # sibling audit scripts
from audit_s1b_prospective import load_ordered, first_fire_index  # noqa: E402

runs = sorted(glob.glob(sys.argv[1] + "/run-l3-*"))

# Which knob names denote a precision/dtype choice. Kept as a substring list rather than an exact
# set because knob naming is agent-authored and varies per candidate; printed so it can be checked.
DTYPE_HINTS = ("PRECISION", "DTYPE", "PREC", "ACC_TYPE", "ACCUM")


def is_dtype(knob: str) -> bool:
    return any(h in knob.upper() for h in DTYPE_HINTS)


print("dtype knob name hints: %s\n" % (DTYPE_HINTS,))

# First: the raw class size, so the 172 figure can be checked on this corpus.
cls_trials = defaultdict(lambda: [0, 0])   # value -> [pass, fail]
dtype_knobs = set()
for r in runs:
    for sid, s in load_ordered(Path(r)).items():
        for knob in s["domains"]:
            if not is_dtype(knob):
                continue
            dtype_knobs.add(knob)
            for cfg, ok in s["seq"]:
                v = cfg.get(knob)
                if v is None:
                    continue
                cls_trials[(knob, v)][0 if ok else 1] += 1

print("dtype-like knobs found: %s" % sorted(dtype_knobs))
print("%-28s %-8s %-8s %s" % ("(knob, value)", "pass", "fail", "verdict"))
print("-" * 64)
for (knob, v), (p, f) in sorted(cls_trials.items(), key=lambda kv: -kv[1][1]):
    if p + f < 5:
        continue
    print("%-28s %-8d %-8d %s" % ("%s=%s" % (knob, v), p, f,
                                  "0-for-%d, unconditional candidate" % f if p == 0 else "passes sometimes"))

print()
print("Now: at each count floor, which values fire, how many failing trials are saved, and how many")
print("of those saved are the dtype class?")
print()
print("%-8s %-7s %-13s %-13s %-13s %s" % (
    "floor N", "fired", "saved fails", "of which dtype", "later PASS", "value mis-kill"))
print("-" * 84)
for floor in (3, 6, 8, 12, 20):
    fired = saved = saved_dtype = lp = vwp = 0
    fired_dtype = []
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
                    tail = [(cfg, ok) for cfg, ok in seq[i + 1:] if cfg.get(knob) == value]
                    p = sum(1 for _, ok in tail if ok)
                    fired += 1
                    lp += p
                    saved += len(tail) - p
                    if p:
                        vwp += 1
                    if is_dtype(knob):
                        saved_dtype += len(tail) - p
                        fired_dtype.append("%s=%s(+%d/-%d)" % (knob, value, p, len(tail) - p))
    print("%-8d %-7d %-13d %-13d %-13d %s" % (
        floor, fired, saved, saved_dtype, lp,
        ("%d/%d = %.1f%%" % (vwp, fired, 100.0 * vwp / fired)) if fired else "n/a"))
    if fired_dtype:
        print("         dtype values fired: %s" % ", ".join(sorted(set(fired_dtype))[:6]))

print()
print("READ THIS AGAINST THE CRITERION. S1b's J is 'save at least 100 of tf32's 172 trials'. The")
print("'saved fails' column is the TOTAL over all knobs; 'of which dtype' is the part the criterion")
print("is actually about. A large total carried by non-dtype knobs does not satisfy it -- those are")
print("the conditionally-failing values whose removal is the risk, not the benefit.")
