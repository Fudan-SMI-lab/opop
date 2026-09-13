"""Your proposal, tested: could the slope have redirected sampling DURING tuning?

THE PROPOSAL. 2e currently tells the model which knob is steep-but-truncated. But the slope is not
only extra information for the prompt -- it could steer the SEARCH: sample the steep knobs' space
first, and find the resource wall of the steep knob earlier.

WHAT HAS TO BE TRUE FOR THAT TO WORK, and this script tests it rather than assuming:

  1. The slope must be COMPUTABLE mid-tuning. `TuningStatsAnalyzer.analyze(space, trials)` takes a
     plain trial list, so nothing stops calling it at trial k. Verified by doing it.

  2. The mid-tuning slope must AGREE with the final one. If the ranking of knobs by tail slope at
     trial 20 does not resemble the ranking at trial 80, then steering early would chase noise -- and
     the project has already measured that four cheap signals for "remaining gain" all had rho <= 0.52,
     so this cannot be taken on faith.

  3. The wall must be DISCOVERABLE earlier. A wall needs a refused configuration; if the refusals only
     arrive late in the pass, no amount of re-ordering finds the wall sooner.

This replays each candidate's real trials in their real order and recomputes the walls at 25/50/75/100%
of the pass, using the SHIPPED analyzer and find_walls. That makes the answer a measurement of what
this run's own data would have supported, not a projection.
"""
import json
import os
import sys
from collections import defaultdict

if len(sys.argv) < 3:
    print(__doc__)
    print("usage: could_the_slope_steer_tuning.py <checkout-src> <run_dir>")
    raise SystemExit(2)
sys.path.insert(0, sys.argv[1])
from kernel_optimizer.evaluation import wall_attribution  # noqa: E402
from kernel_optimizer.models.core import (LatencyStats, ParamDomain, ParameterSpace,  # noqa: E402
                                          ParamSet, ProfileRecord, TrialRecord)
from kernel_optimizer.tuning.stats import TuningStatsAnalyzer  # noqa: E402
from kernel_optimizer.config import load_config  # noqa: E402

rd = sys.argv[2]
evs = []
with open(os.path.join(rd, "events.jsonl"), encoding="utf-8") as fh:
    for ln in fh:
        ln = ln.strip()
        if ln:
            try:
                evs.append(json.loads(ln))
            except Exception:
                pass

# Spaces, rebuilt from SPACE_PUBLISHED so the analyzer sees the real domains.
spaces = {}
for e in evs:
    if e.get("type") != "SPACE_PUBLISHED":
        continue
    sp = (e.get("payload") or {}).get("space") or {}
    cid = sp.get("candidate_id")
    doms = []
    for d in (sp.get("domains") or []):
        try:
            doms.append(ParamDomain(name=d["name"], kind=d.get("kind", "int"),
                                    choices=tuple(d.get("choices") or ())))
        except Exception:
            pass
    if cid and doms:
        spaces[cid] = ParameterSpace(space_id=sp.get("space_id", "s"), candidate_id=cid,
                                     version=int(sp.get("version") or 1),
                                     source_sha=str(sp.get("source_sha") or ""),
                                     domains=tuple(doms))

# Trials in their real order.
trials = defaultdict(list)
for e in evs:
    if e.get("type") != "TRIAL_DONE":
        continue
    t = (e.get("payload") or {}).get("trial") or {}
    cid = t.get("candidate_id")
    if cid:
        trials[cid].append(t)


def rebuild(t, cid):
    lat = t.get("latency_ms") or {}
    ls = None
    if lat.get("mean") is not None:
        ls = LatencyStats(mean=float(lat["mean"]), std=float(lat.get("std") or 0.0),
                          min=float(lat.get("min") or lat["mean"]),
                          max=float(lat.get("max") or lat["mean"]),
                          n_samples=int(lat.get("n_samples") or 1),
                          median=lat.get("median"))
    prof = None
    p = t.get("profile") or {}
    if p:
        try:
            prof = ProfileRecord(**{k: v for k, v in p.items()
                                    if k in ProfileRecord.model_fields})
        except Exception:
            prof = None
    return TrialRecord(
        trial_id=str(t.get("trial_id") or "t"), candidate_id=cid,
        space_id=str(t.get("space_id") or "s"),
        params=ParamSet(values=dict((t.get("params") or {}).get("values") or {})),
        status=str(t.get("status") or "fail"),
        failure_kind=t.get("failure_kind"), latency_ms=ls, profile=prof)


cfg = load_config(None)
analyzer = TuningStatsAnalyzer(cfg.device)

print(f"=== {os.path.basename(rd)}")
for cid, raw in trials.items():
    space = spaces.get(cid)
    if space is None or len(raw) < 8:
        continue
    recs = [rebuild(t, cid) for t in raw]
    refused_all = [dict((t.get("params") or {}).get("values") or {}) for t in raw
                   if t.get("failure_kind") == "infeasible_shared_memory"]
    if not refused_all:
        continue
    print(f"\n  {cid}: {len(recs)} trials, {len(refused_all)} shared-memory refusals")
    for frac in (0.25, 0.50, 0.75, 1.00):
        k = max(4, int(frac * len(recs)))
        prefix = recs[:k]
        refused = [dict(r.params.values) for r in prefix
                   if r.failure_kind == "infeasible_shared_memory"]
        try:
            st = analyzer.analyze(space, prefix)
        except Exception as exc:
            print(f"    at {frac:.0%} ({k} trials): analyze failed: {exc}")
            continue
        walls = wall_attribution.find_walls(st, refused) if refused else []
        worthy, _ = wall_attribution.select_for_probing(walls, 8)
        # The knob ranking by tail slope -- the thing a sampler would steer on.
        order = [(w.param, round(float(w.tail_gain_pct), 1)) for w in
                 sorted(walls, key=lambda w: -float(w.tail_gain_pct))]
        print(f"    at {frac:>4.0%} ({k:3d} trials, {len(refused):2d} refusals): "
              f"{len(walls)} walls, {len(worthy)} probe-worthy   ranking {order}")
print()
print("READING")
print("  * a wall appearing at 25-50% means the steer COULD have been applied mid-pass")
print("  * a ranking that is stable from 50% onward means steering early would not chase noise")
print("  * refusals arriving only at 100% would mean re-ordering cannot find the wall sooner")
