"""Is latency SMOOTH along an ordered knob's choices? This decides whether order-aware methods help.

WHY THIS IS THE PREREQUISITE QUESTION. Our knobs are ordered (BLOCK_N 16/32/64/128) but declared as
unordered categoricals, and the sampler was measured to be order-blind (the top value of a walled knob
is sampled at mean normalised position 0.52 -- no outward walk). The obvious fix is to give the
sampler the order: Optuna's `categorical_distance_func`, an ordinal/integer parameter, or a GP with a
kernel over the index.

But every one of those fixes assumes the objective VARIES SMOOTHLY along the order. If latency is
jagged in the choice index -- 16 fast, 32 slow, 64 fast -- then "64 was good so try 128" is not an
inference, it is a superstition, and an order-aware kernel would confidently interpolate nonsense. The
project has already refuted two plausible signals by checking their premise instead of assuming it, so
this premise gets checked too.

WHAT IS MEASURED, per knob, over the per-choice median latency curve the tuner itself built:

  monotone         is the curve non-increasing or non-decreasing across the ordered choices?
  unimodal         does it fall then rise (or rise then fall) with a single turning point? A unimodal
                   curve is still learnable by an order-aware model; a multi-modal one is not.
  sign flips       how many times the step-to-step direction changes. 0-1 is smooth, many is jagged.
  neighbour vs far is |latency(i) - latency(i+1)| typically SMALLER than |latency(i) - latency(j)| for
                   distant j? This is the actual content of "distance in the index means something".
                   If neighbours are no more similar than strangers, the order carries no information
                   and no distance function can help.

The last one is the decisive test, and it is reported as a ratio so it can be read against 1.0:
below 1 means neighbours really are more alike, at or above 1 means the ordering is decorative.
"""
import json
import os
import statistics
import sys
from collections import defaultdict

if len(sys.argv) < 3:
    print(__doc__)
    print("usage: is_latency_smooth_along_the_order.py <checkout-src> label=<run_dir> ...")
    raise SystemExit(2)
sys.path.insert(0, sys.argv[1])
from kernel_optimizer.models.reports import TuningStats  # noqa: E402


def as_num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


curves = []
for spec in sys.argv[2:]:
    label, rd = spec.split("=", 1)
    evs = []
    with open(os.path.join(rd, "events.jsonl"), encoding="utf-8") as fh:
        for ln in fh:
            ln = ln.strip()
            if ln:
                try:
                    evs.append(json.loads(ln))
                except Exception:
                    pass
    spaces = {}
    for e in evs:
        if e.get("type") != "SPACE_PUBLISHED":
            continue
        sp = (e.get("payload") or {}).get("space") or {}
        if sp.get("candidate_id"):
            spaces[sp["candidate_id"]] = {d["name"]: (list(d.get("choices") or []), d.get("kind"))
                                          for d in (sp.get("domains") or [])
                                          if isinstance(d, dict)}
    stats_for, pend = {}, None
    for e in evs:
        if e.get("type") == "TUNING_DONE":
            pend = (e.get("payload") or {}).get("candidate_id")
        elif e.get("type") == "STATS_DONE" and pend:
            raw = (e.get("payload") or {}).get("stats")
            if raw:
                try:
                    stats_for[pend] = TuningStats.model_validate(raw)
                except Exception:
                    pass
            pend = None

    for cid, stats in stats_for.items():
        space = spaces.get(cid) or {}
        for ps in stats.param_stats:
            entry = space.get(ps.name)
            if not entry:
                continue
            choices, kind = entry
            nums = [as_num(c) for c in choices]
            # Only NUMERIC knobs have a meaningful order. A str knob (COMPUTE_DTYPE) is genuinely
            # unordered and must be excluded rather than given a fake index -- including it would
            # manufacture the jaggedness this probe is looking for.
            if any(n is None for n in nums) or len(nums) < 4:
                continue
            by_value = ps.latency_by_value or {}
            pts = []
            for c, n in zip(choices, nums):
                ms = by_value.get(str(c), by_value.get(c))
                if isinstance(ms, (int, float)):
                    pts.append((n, float(ms)))
            if len(pts) < 4:
                continue
            pts.sort()
            ys = [y for _, y in pts]
            steps = [b - a for a, b in zip(ys, ys[1:])]
            signs = [1 if s > 0 else (-1 if s < 0 else 0) for s in steps]
            nz = [s for s in signs if s]
            flips = sum(1 for a, b in zip(nz, nz[1:]) if a != b)
            neighbour = statistics.fmean(abs(s) for s in steps)
            far = statistics.fmean(abs(ys[i] - ys[j])
                                   for i in range(len(ys)) for j in range(i + 2, len(ys)))
            curves.append({
                "label": label, "cand": cid, "knob": ps.name, "n": len(pts),
                "monotone": flips == 0,
                "unimodal": flips <= 1,
                "flips": flips,
                "ratio": (neighbour / far) if far else None,
            })

print(f"numeric knobs with >= 4 measured choices: {len(curves)}")
if not curves:
    raise SystemExit(0)
mono = sum(1 for c in curves if c["monotone"])
uni = sum(1 for c in curves if c["unimodal"])
print(f"  monotone            {mono:3d} / {len(curves)}  ({100.0*mono/len(curves):.0f}%)")
print(f"  unimodal (<=1 flip) {uni:3d} / {len(curves)}  ({100.0*uni/len(curves):.0f}%)")
fl = sorted(c["flips"] for c in curves)
print(f"  direction flips: p50 {fl[len(fl)//2]}, max {max(fl)}")
ratios = [c["ratio"] for c in curves if c["ratio"] is not None]
if ratios:
    r = sorted(ratios)
    p50 = r[len(r) // 2]
    print()
    print("DECISIVE TEST -- mean |neighbour gap| / mean |distant gap|:")
    print(f"  p50 {p50:.3f}   p10 {r[max(0,int(0.1*(len(r)-1)))]:.3f}   "
          f"p90 {r[min(len(r)-1,int(0.9*(len(r)-1)))]:.3f}")
    print(f"  below 1.0 on {sum(1 for x in r if x < 1.0)} of {len(r)} knobs")
    print("  => " + ("neighbours ARE more alike than strangers: the ordering carries information, so "
                     "an order-aware kernel / distance function has something to learn"
                     if p50 < 0.85 else
                     "neighbours are NO more alike than strangers: the ordering is decorative here "
                     "and an order-aware model would interpolate noise"))
print()
print("worst-behaved knobs (most direction flips) -- these are what an order-aware model must survive:")
for c in sorted(curves, key=lambda c: -c["flips"])[:8]:
    print(f"  {c['label']:6s} {c['cand']} {c['knob']:20s} {c['n']} pts, {c['flips']} flips, "
          f"ratio {c['ratio']:.2f}" if c["ratio"] is not None else "")
