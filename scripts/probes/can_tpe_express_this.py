"""Does TPE, as we configure it, actually behave like it understands our space?

Three checks, each about a capability the proposal needs and each answerable from finished runs. The
question is not "is TPE good" but "which of the three things the user wants can TPE express at all".

  1. ORDER-BLINDNESS. Our knobs are ordered (BLOCK_N 16/32/64/128) but declared via
     `suggest_categorical`, and TPE treats categoricals as unordered unless given a
     `categorical_distance_func`. Consequence if it matters: knowing 64 is good tells the sampler
     nothing about 128, so it cannot walk outward toward a wall -- it has to stumble on it. Measured
     here as: for the walled knobs, did sampling of the top measured value CLUSTER late (consistent
     with a directed walk) or spread evenly (consistent with unordered draws)?

  2. NO EXPLICIT UNCERTAINTY. TPE models a density RATIO l(x)/g(x), not a posterior with variance, so
     there is no term to put an exploration bonus on. Measured indirectly: how much of the budget went
     to REPEAT values of a knob after its curve was already flat there -- effort a variance-aware
     acquisition would have moved elsewhere.

  3. CONSTRAINT HANDLING AS WE USE IT. We report hard infeasibility as PRUNED so it stays in the TPE
     model. Measured: does the infeasible fraction actually DECAY within a pass? If it does, the
     pruned-trial channel is teaching the sampler; if it stays flat, the mechanism is not working and
     a real constraint model would be a genuine gain rather than a refinement.

Check 3 is the load-bearing one, because a previous measurement in this project found the failure rate
"never decayed over a run" BEFORE the PRUNED fix went in. This re-measures it AFTER.
"""
import json
import os
import statistics
import sys
from collections import defaultdict

if len(sys.argv) < 3:
    print(__doc__)
    print("usage: can_tpe_express_this.py <checkout-src> label=<run_dir> ...")
    raise SystemExit(2)
sys.path.insert(0, sys.argv[1])
from kernel_optimizer.evaluation import wall_attribution  # noqa: E402
from kernel_optimizer.models.reports import TuningStats  # noqa: E402


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


all_decay = []
all_late = []
for spec in sys.argv[2:]:
    label, rd = spec.split("=", 1)
    evs = load(rd)

    trials = defaultdict(list)
    for e in evs:
        if e.get("type") != "TRIAL_DONE":
            continue
        t = (e.get("payload") or {}).get("trial") or {}
        if t.get("candidate_id"):
            trials[t["candidate_id"]].append(t)

    spaces = {}
    for e in evs:
        if e.get("type") != "SPACE_PUBLISHED":
            continue
        sp = (e.get("payload") or {}).get("space") or {}
        if sp.get("candidate_id"):
            spaces[sp["candidate_id"]] = {d["name"]: list(d.get("choices") or [])
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

    print(f"=== {label}")
    for cid, ts in sorted(trials.items()):
        if len(ts) < 20:
            continue
        # --- check 3: does the infeasible fraction decay within the pass?
        half = len(ts) // 2
        def infeas(seq):
            n = sum(1 for t in seq if t.get("failure_kind") == "infeasible_shared_memory")
            return 100.0 * n / len(seq) if seq else 0.0
        first, second = infeas(ts[:half]), infeas(ts[half:])
        if first > 0 or second > 0:
            all_decay.append((label, cid, first, second))

        # --- check 1: when were the top values of a WALLED knob sampled?
        stats = stats_for.get(cid)
        if stats is None:
            continue
        refused = [(t.get("params") or {}).get("values") or {} for t in ts
                   if t.get("failure_kind") == "infeasible_shared_memory"]
        if not refused:
            continue
        for w in wall_attribution.find_walls(stats, refused):
            ran = [float(v) for v in (w.ran_values or []) if isinstance(v, (int, float))]
            if not ran:
                continue
            top = max(ran)
            hits = [i for i, t in enumerate(ts)
                    if str((t.get("params") or {}).get("values", {}).get(w.param)) in
                    (str(top), str(int(top)) if float(top).is_integer() else str(top))]
            if len(hits) >= 3:
                frac = [h / (len(ts) - 1) for h in hits]
                all_late.append((label, cid, w.param, len(hits), len(ts),
                                 statistics.fmean(frac)))

print()
print("CHECK 3 -- does the infeasible fraction DECAY within a tuning pass?")
print("  (reported as first-half% -> second-half%; the PRUNED channel is supposed to teach TPE)")
worse = better = same = 0
for label, cid, a, b in all_decay:
    arrow = "decayed" if b < a - 2 else ("ROSE" if b > a + 2 else "flat")
    if arrow == "decayed":
        better += 1
    elif arrow == "ROSE":
        worse += 1
    else:
        same += 1
    print(f"  {label:6s} {cid} {a:5.1f}% -> {b:5.1f}%   {arrow}")
print(f"  totals: decayed {better}, flat {same}, ROSE {worse}")
print("  => if 'decayed' does not dominate, reporting infeasibility as PRUNED is NOT measurably")
print("     steering TPE away from the infeasible region, and an explicit constraint model would be")
print("     a real gain rather than a refinement.")

print()
print("CHECK 1 -- when in the pass were a walled knob's TOP values sampled?")
print("  (mean normalised position; ~0.5 = spread evenly, >0.7 = clustered late as a directed walk)")
for label, cid, knob, hits, n, mean_frac in sorted(all_late, key=lambda r: -r[5]):
    print(f"  {label:6s} {cid} {knob:20s} {hits:3d}/{n:3d} trials at the top value, "
          f"mean position {mean_frac:.2f}")
if all_late:
    overall = statistics.fmean(r[5] for r in all_late)
    print(f"  overall mean position {overall:.2f} -- "
          + ("consistent with ORDER-BLIND sampling (no outward walk)" if 0.35 < overall < 0.65
             else "shows a positional bias worth explaining"))
