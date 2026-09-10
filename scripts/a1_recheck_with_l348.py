"""Re-score A-1 with the L3:48 runs included and the label computed both ways.

Two defects in the earlier A-1 analysis, both found by looking at what the corpus contained
rather than at the analysis code:

  1. THE CORPUS OMITTED L3:48. The multi-binding counts (40 candidates, then 51) ran on L3:21
     and L3:43 only, where dram pressure sits at 6-17%. L3:48 is the one task where DRAM
     genuinely binds -- 93.7-95.5% of a measured roof -- and it is the case that MOTIVATED
     A-1 (DRAM 93.7% AND occupancy 17%, two dimensions demanding opposite rewrites). A
     finding about "which binding dimension to attack" computed without the DRAM-bound task
     is not a finding about the situation it claims to describe. The give-away was the swing
     table: dram_bandwidth ranged over only 0.022-0.165 across the whole corpus, which is
     impossible for a task set that includes a bandwidth-bound kernel.

  2. THE LABEL MAY BE A CONFOUND. Q2's label is argmax |Delta pressure| across dimensions.
     Every scored rewrite HELPED -- that was the filter -- so gpu_ms fell, so dram and compute
     pressure ROSE by identity (byte_count / gpu_ms with a task-constant numerator). If those
     two columns swing harder than the compile-time ones, the argmax picks a latency column
     almost every time, and "only 4 of 14 moved a binding dimension" is measuring the
     identity rather than the rewrites.

Both are checked here. Rules and controls are re-scored under the compile-time-only label,
which is the only label that cannot move for free when a kernel gets faster.

Zero GPU. Reads the row+lineage dumps produced on each box.
"""
import json
import random
import sys
from collections import Counter

SHARED_LIMIT, REG_LIMIT = 101376, 255
NEAR = 0.85        # a dimension counts as binding at this pressure
MIN_GAIN = 5.0     # per-trial std reaches 16% of the mean; below this is not evidence (D-3)
RELIEF = 0.05      # pressure drop that counts as relieving a dimension

data = {"rows": [], "lineage": {}, "runmeta": {}}
for p in sys.argv[1:]:
    d = json.load(open(p, encoding="utf-8"))
    data["rows"] += d["rows"]
    data["lineage"].update(d["lineage"])
    data["runmeta"].update(d["runmeta"])


def num(*v):
    for x in v:
        if isinstance(x, (int, float)):
            return x
    return None


def indep(r):
    """The three dimensions read at compile/launch time. None = not measured, never 'slack'."""
    ev, pf = r["ev"], r["prof"]
    occ = num(ev.get("occupancy"), pf.get("occupancy"))
    nr = num(ev.get("n_regs"), pf.get("n_regs"))
    sb = num(pf.get("shared_bytes"), ev.get("shared_bytes"))
    return {"occupancy": (1 - occ) if occ is not None else None,
            "register_pressure": nr / REG_LIMIT if nr is not None else None,
            "shared_capacity": min(sb, SHARED_LIMIT) / SHARED_LIMIT if sb is not None else None}


def alld(r):
    d = dict(indep(r))
    for n, f in (("dram_bandwidth", "pct_of_dram_peak"), ("compute", "pct_of_compute_peak")):
        v = r["ev"].get(f)
        d[n] = min(v / 100.0, 1.0) if isinstance(v, (int, float)) else None
    return d


print("=" * 100)
print("CORPUS -- which tasks are actually in it")
print("=" * 100)
for rn, m in sorted(data["runmeta"].items()):
    dr = [r["ev"].get("pct_of_dram_peak") for r in data["rows"] if r["run"] == rn]
    dr = [x for x in dr if isinstance(x, (int, float))]
    print("  %-32s classified=%-3d finished=%d  dram%% %.1f - %.1f"
          % (rn[:32], m["n_classified"], m["finished"],
             min(dr) if dr else 0, max(dr) if dr else 0))
rows_all = data["rows"]
print("\n  classified candidates: %d   (the earlier A-1 corpora were 40 and 51, both WITHOUT "
      "L3:48)" % len(rows_all))

print("\n" + "=" * 100)
print("Q1  MULTI-BINDING WITH L3:48 INCLUDED")
print("=" * 100)
for label, fn in (("all five (latency counted twice)", alld), ("compile-time only", indep)):
    print("  -- %s" % label)
    for th in (0.70, 0.85):
        hist, combos = Counter(), Counter()
        for r in rows_all:
            b = tuple(sorted(d for d, v in fn(r).items() if v is not None and v >= th))
            hist[len(b)] += 1
            if len(b) >= 2:
                combos[b] += 1
        tot = sum(hist.values())
        multi = sum(v for k, v in hist.items() if k >= 2)
        print("     th %.2f  multi %2d/%d (%.1f%%)   %s"
              % (th, multi, tot, 100 * multi / tot,
                 " ".join("%dd:%d" % (k, hist[k]) for k in sorted(hist))))
        for c, n in combos.most_common(3):
            print("             %2d x %s" % (n, " + ".join(c)))

by = {(r["run"], r["cid"]): r for r in rows_all}
lab = []
for rn, par in data["lineage"].items():
    for cid, pid in par.items():
        kc, kp = (rn, cid), (rn, pid)
        if kc not in by or kp not in by:
            continue
        rc, rp = by[kc], by[kp]
        pm, cm = rp.get("best_ms"), rc.get("best_ms")
        if not (isinstance(pm, (int, float)) and isinstance(cm, (int, float))):
            continue
        lab.append(dict(run=rn, parent=pid, child=cid, pm=pm, cm=cm,
                        gain=100 * (pm - cm) / pm, p=rp, c=rc))

print("\n" + "=" * 100)
print("Q2  THE LABEL: 'which dimension moved most', computed both ways")
print("=" * 100)
print("  parent->child pairs, both classified and timed: %d" % len(lab))
scored = [l for l in lab if l["gain"] >= MIN_GAIN]
print("  of those, gain >= %.1f%%: %d" % (MIN_GAIN, len(scored)))

for label, fn in (("ALL FIVE", alld), ("COMPILE-TIME ONLY", indep)):
    print("\n  -- argmax |Delta pressure| under %s" % label)
    cnt, inset, n = Counter(), 0, 0
    for l in scored:
        up, uc = fn(l["p"]), fn(l["c"])
        dl = {d: (uc[d] - up[d]) for d in up
              if up.get(d) is not None and uc.get(d) is not None}
        if not dl:
            continue
        n += 1
        moved = max(dl, key=lambda d: abs(dl[d]))
        cnt[moved] += 1
        if moved in [d for d, v in up.items() if v is not None and v >= NEAR]:
            inset += 1
    for d, c in cnt.most_common():
        print("       %-20s x%-3d (%.0f%%)" % (d, c, 100 * c / n if n else 0))
    print("       most-moved dim WAS binding in the parent: %d/%d (%.1f%%)"
          % (inset, n, 100 * inset / n if n else 0))

print("\n" + "=" * 100)
print("Q3  ROBUST FORM (no cross-dimension argmax), compile-time dims only")
print("=" * 100)
rb = rnone = ro = 0
for l in scored:
    up, uc = indep(l["p"]), indep(l["c"])
    dl = {d: (uc[d] - up[d]) for d in up if up.get(d) is not None and uc.get(d) is not None}
    binding = [d for d, v in up.items() if v is not None and v >= NEAR]
    if any(dl.get(d) is not None and dl[d] <= -RELIEF for d in binding):
        rb += 1
    elif not any(v <= -RELIEF for v in dl.values()):
        rnone += 1
    else:
        ro += 1
print("  of %d rewrites with gain >= %.1f%%:" % (len(scored), MIN_GAIN))
print("    %2d relieved a BINDING compile-time dimension by >= %.2f" % (rb, RELIEF))
print("    %2d relieved nothing by that much" % rnone)
print("    %2d relieved only NON-binding dimensions" % ro)

print("\n" + "=" * 100)
print("Q4  RULES VS CONTROLS, re-scored on the compile-time label")
print("=" * 100)
multi = []
for l in scored:
    up = indep(l["p"])
    b = [d for d, v in up.items() if v is not None and v >= NEAR]
    if len(b) < 2:
        continue
    uc = indep(l["c"])
    dl = {d: (uc[d] - up[d]) for d in up if up.get(d) is not None and uc.get(d) is not None}
    multi.append(dict(l, binding=b, u=up, deltas=dl,
                      moved=max(dl, key=lambda d: abs(dl[d])) if dl else None))
print("  scoreable (multi-binding parent on compile-time dims AND gain >= %.1f%%): %d"
      % (MIN_GAIN, len(multi)))
if len(multi) < 5:
    print("  *** TOO FEW TO SCORE ANY RULE. That is the finding, not licence to invent one:")
    print("      a rule fitted on %d cases cannot be told apart from chance." % len(multi))
if multi:
    def score(pick):
        return sum(1 for l in multi if pick(l) == l["moved"])
    r1 = score(lambda l: max(l["binding"], key=lambda d: l["u"][d]))
    r2 = score(lambda l: min(l["binding"], key=lambda d: l["u"][d]))
    print("    R1 tightest wall first        %d/%d (%.1f%%)" % (r1, len(multi), 100 * r1 / len(multi)))
    print("    R2 most remaining room        %d/%d (%.1f%%)" % (r2, len(multi), 100 * r2 / len(multi)))
    for d in ("occupancy", "register_pressure", "shared_capacity"):
        c = sum(1 for l in multi if l["moved"] == d)
        print("    C1 always pick %-18s %d/%d (%.1f%%)" % (d, c, len(multi), 100 * c / len(multi)))
    random.seed(0)
    tr = [sum(1 for l in multi if random.choice(l["binding"]) == l["moved"]) for _ in range(2000)]
    print("    C2 uniform random (2000 draws) %.1f%%" % (100 * sum(tr) / len(tr) / len(multi)))
