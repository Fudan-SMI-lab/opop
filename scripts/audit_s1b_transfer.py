"""S1b's last rescue path: accumulate evidence ACROSS spaces within a run, not inside one space.

The timing measurement showed S1b's ceiling is arithmetic -- a value appears ~10 times in a 40-trial
space, so a safe floor of N=8 leaves almost nothing to save. Pooling a knob's evidence across the
spaces of one run would break that ceiling: DOT_PRECISION=tf32 is 0-for-164 over the whole corpus, so
a rule learned in space 1 could act from trial 1 of space 2.

That is only sound if the conclusion actually TRANSFERS between candidates. It may not: the root
cause on record is that the CANDIDATE puts tf32 in an uncompensated `else` fallback, which is a
property of that candidate's code, not of tf32. So test transfer directly, per (knob, value):
how many spaces is it alive in, how many dead in.

The corpus already shows the shape of the trap -- the same dtype word behaves oppositely under
different knobs (DOT_PRECISION=tf32 0-for-164; COMPUTE_DTYPE=tf32 passes 209 times) -- so pooling by
dtype rather than by (knob, value) would be catastrophic. This measures pooling by (knob, value),
which is the defensible version.
"""
import sys, glob
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # sibling audit scripts
from audit_s1b_prospective import load_ordered  # noqa: E402

runs = sorted(glob.glob(sys.argv[1] + "/run-l3-*"))

# Per run, per (knob, value): the per-space pass/fail split, so transfer can be judged run-locally
# (cross-RUN transfer is out of scope by the plan's own "conclusions must not carry across").
print("%-22s %-26s %-8s %-8s %-9s %s" % (
    "run", "(knob, value)", "spaces", "dead sp", "alive sp", "verdict"))
print("-" * 100)
tot_pool_ok = tot_pool_bad = 0
saveable_if_pooled = 0
for r in runs:
    spaces = load_ordered(Path(r))
    per_kv = defaultdict(lambda: defaultdict(lambda: [0, 0]))   # (knob,val) -> sid -> [pass, fail]
    for sid, s in spaces.items():
        for cfg, ok in s["seq"]:
            for knob, v in cfg.items():
                per_kv[(knob, v)][sid][0 if ok else 1] += 1
    for (knob, v), by_space in sorted(per_kv.items()):
        if len(by_space) < 2:
            continue                                    # pooling needs >=2 spaces to mean anything
        dead = [sid for sid, (p, f) in by_space.items() if p == 0 and f > 0]
        alive = [sid for sid, (p, f) in by_space.items() if p > 0]
        if not dead:
            continue                                    # never a candidate for pooling
        if alive:
            tot_pool_bad += 1
            verdict = "TRANSFER FAILS -- pooling would kill %d live space(s)" % len(alive)
        else:
            tot_pool_ok += 1
            # what pooling could save: every failing trial in every space but the FIRST dead one
            fails = sorted((f for sid, (p, f) in by_space.items()), reverse=True)
            saveable_if_pooled += sum(fails) - max(fails)
            verdict = "transfers (dead in all %d)" % len(dead)
        if len(dead) + len(alive) >= 2 and (alive or sum(f for _, (p, f) in by_space.items()) >= 8):
            print("%-22s %-26s %-8d %-8d %-9d %s" % (
                r.rsplit("/", 1)[-1][:22], "%s=%s" % (knob, v),
                len(by_space), len(dead), len(alive), verdict))

print()
print("(knob,value) pairs appearing in >=2 spaces with >=1 dead space:")
print("  transfer HOLDS (dead everywhere) : %d" % tot_pool_ok)
print("  transfer FAILS (alive somewhere) : %d" % tot_pool_bad)
print("  failing trials pooling could save (optimistic, over all runs): %d" % saveable_if_pooled)
print()
print("If transfer FAILS for a meaningful share, cross-space pooling is not available and S1b is")
print("stuck with the per-space arithmetic ceiling measured in audit_s1b_timing.py. Note the count above")
print("is itself OPTIMISTIC about pooling: 'dead in all spaces' is judged with full hindsight over")
print("the whole run, while a live rule would have to decide from a prefix.")
