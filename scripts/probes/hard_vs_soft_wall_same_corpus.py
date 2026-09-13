"""Hard wall vs soft wall on THE SAME data. Without this the "an order of magnitude more" claim is void.

WHY IT HAD TO BE MEASURED THIS WAY. The soft-wall criterion found 49 (candidate, knob) pairs across
five backed-up runs. I was about to compare that against "the hard wall attributed only a handful
across four runs" -- but those five runs PREDATE 2e, so they contain zero RESOURCE_WALL_ATTRIBUTED
events (verified: 0 in all of them). Comparing a number measured on this corpus against a number
measured on a different corpus would be exactly the mistake recorded in
`box1-and-box4-have-different-cpus-so-trial-counts-are-not-comparable`.

So the hard wall is replayed OFFLINE on the same five runs with the shipping `find_walls` and
`select_for_probing`, giving both criteria the same denominator. The probe step is skipped -- it needs
a GPU and the compiler -- so the hard-wall figure here is "probe-worthy walls", i.e. an UPPER bound on
what would have been attributed. That favours the hard wall, which is the right direction for a
comparison whose conclusion is that the soft wall finds more.
"""
import json
import os
import statistics
import sys
from collections import defaultdict

if len(sys.argv) < 3:
    print(__doc__)
    print("usage: hard_vs_soft_wall_same_corpus.py <checkout-src> <run_dir> [<run_dir>...]")
    raise SystemExit(2)

sys.path.insert(0, sys.argv[1])
from kernel_optimizer.evaluation import wall_attribution  # noqa: E402
from kernel_optimizer.models.reports import TuningStats  # noqa: E402


def robust_ms(lat):
    if not isinstance(lat, dict):
        return None
    for k in ("median", "mean"):
        v = lat.get(k)
        if isinstance(v, (int, float)) and v > 0:
            return float(v)
    return None


def as_num(v):
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        try:
            return float(v)
        except ValueError:
            return None
    return None


hard_worthy = hard_found = 0
soft_worthy = soft_found = 0
soft_candidates = hard_candidates = 0
n_cands = 0
inapplicable_zero_spill = 0

for rd in sys.argv[2:]:
    path = os.path.join(rd, "events.jsonl")
    if not os.path.exists(path):
        continue
    evs = []
    with open(path, encoding="utf-8") as fh:
        for ln in fh:
            ln = ln.strip()
            if ln:
                try:
                    evs.append(json.loads(ln))
                except Exception:
                    pass

    trials = defaultdict(list)
    for e in evs:
        if e.get("type") != "TRIAL_DONE":
            continue
        t = (e.get("payload") or {}).get("trial") or {}
        if t.get("candidate_id"):
            trials[t["candidate_id"]].append(t)

    stats_for, pending = {}, None
    for e in evs:
        if e.get("type") == "TUNING_DONE":
            pending = (e.get("payload") or {}).get("candidate_id")
        elif e.get("type") == "STATS_DONE" and pending:
            raw = (e.get("payload") or {}).get("stats")
            if raw:
                try:
                    stats_for[pending] = TuningStats.model_validate(raw)
                except Exception:
                    pass
            pending = None

    for cid, ts in sorted(trials.items()):
        done = [t for t in ts if t.get("status") == "complete"]
        if len(done) < 8:
            continue
        n_cands += 1

        # ---- HARD WALL, replayed with the shipping functions
        stats = stats_for.get(cid)
        if stats is not None:
            refused = [(t.get("params") or {}).get("values") or {} for t in ts
                       if t.get("failure_kind") == "infeasible_shared_memory" and t.get("params")]
            if refused:
                walls = wall_attribution.find_walls(stats, refused)
                worthy, _ = wall_attribution.select_for_probing(walls, 8)
                hard_found += len(walls)
                hard_worthy += len(worthy)
                if worthy:
                    hard_candidates += 1

        # ---- SOFT WALL (n_spills), same criterion as scripts/probes/soft_wall_shape.py
        win = min(done, key=lambda t: (robust_ms(t.get("latency_ms") or {}) or 9e9))
        win_sp = (win.get("profile") or {}).get("n_spills")
        if isinstance(win_sp, (int, float)) and win_sp == 0:
            inapplicable_zero_spill += 1

        knobs = set()
        for t in done:
            knobs.update(((t.get("params") or {}).get("values") or {}).keys())
        hit_here = 0
        for knob in sorted(knobs):
            buckets = defaultdict(list)
            for t in done:
                x = as_num(((t.get("params") or {}).get("values") or {}).get(knob))
                if x is None:
                    continue
                sp = (t.get("profile") or {}).get("n_spills")
                ms = robust_ms(t.get("latency_ms") or {})
                if isinstance(sp, (int, float)) and ms is not None:
                    buckets[x].append((float(sp), ms))
            if len(buckets) < 3:
                continue
            xs = sorted(buckets)
            sig = [statistics.median(s for s, _ in buckets[x]) for x in xs]
            lat = [statistics.median(m for _, m in buckets[x]) for x in xs]
            monotone = all(sig[i + 1] >= sig[i] for i in range(len(sig) - 1)) and sig[-1] > sig[0]
            if not monotone or sig[-1] <= 0:
                continue
            soft_found += 1
            tail = lat[-3:]
            gain = (tail[0] - tail[-1]) / tail[0] * 100.0 if tail[0] else 0.0
            if gain > 0:
                soft_worthy += 1
                hit_here += 1
        if hit_here:
            soft_candidates += 1

print(f"candidates with >=8 completed trials: {n_cands}")
print()
print("HARD WALL (shared memory), replayed offline with the shipping find_walls/select_for_probing:")
print(f"  walls found        : {hard_found}")
print(f"  probe-worthy       : {hard_worthy}   <- UPPER bound on what would be attributed")
print(f"                          (the compile probe is skipped here, and on box 2 it confirmed")
print(f"                           6 of 6 from the optimum, so the true figure is at or below this)")
print(f"  candidates with >=1: {hard_candidates} / {n_cands}")
print()
print("SOFT WALL (n_spills), same corpus, same candidates:")
print(f"  monotone-into-binding : {soft_found}")
print(f"  latency still improving: {soft_worthy}")
print(f"  candidates with >=1   : {soft_candidates} / {n_cands}")
print(f"  INAPPLICABLE (winner already at zero spills): {inapplicable_zero_spill} / {n_cands}")
print()
if hard_worthy:
    print(f"ratio soft/hard on probe-worthy findings: {soft_worthy / hard_worthy:.1f}x")
else:
    print("the hard wall found NOTHING probe-worthy in this corpus, so the ratio is undefined --")
    print("report it that way rather than dividing by zero.")
