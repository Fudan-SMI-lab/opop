"""Do "steep slope", "unexplored", and "likely fast" actually point in DIFFERENT directions?

WHY THIS COMES FIRST, BEFORE ANY ALGORITHM CHOICE. The proposal is to combine three exploration
criteria instead of one. That is only worth building if the three DISAGREE. If the steep-slope knob is
also the under-explored one and also the one carrying the best configurations, then any single-signal
sampler already goes there and a composite acquisition function buys nothing but complexity.

This project has been bitten twice by exactly that: NVIDIA-style speed-of-light headroom turned out to
correlate rho=+1.000 with 1/latency (it restated the objective and changed no decision), and
boundary-saturation -- the literature's strongest "where is there headroom" signal -- came in at
rho -0.11..+0.24 against remaining gain. So a signal must be shown to carry information the objective
does not already carry.

THE THREE SIGNALS, computed per KNOB from one candidate's finished tuning pass:

  slope       the knob's tail gain: how much the per-choice median latency curve is still falling at
              the edge of its measured range. This is `find_walls`/`tail_gain`'s quantity, and the one
              the user wants to steer on.
  coverage    what fraction of the knob's declared choices were ever sampled. The complement is the
              "unexplored" signal -- a knob with 2 of 8 choices tried is under-explored regardless of
              how good those 2 were.
  exploit     how much better the knob's best choice is than its median choice: a knob whose choice
              matters a lot for the achieved latency is where the payoff is concentrated.

Then: Spearman correlations between the three, across all knobs of all candidates. Low correlation
means the three genuinely rank knobs differently and a composite is justified; high correlation means
one of them is standing in for another.

Also reported: whether the WALLED knob (the one whose range was truncated) is distinguishable by
coverage alone -- because if it is, a plain "sample what you haven't sampled" rule finds the wall and
no slope signal is needed.
"""
import json
import os
import statistics
import sys
from collections import defaultdict

if len(sys.argv) < 3:
    print(__doc__)
    print("usage: do_the_three_signals_disagree.py <checkout-src> label=<run_dir> ...")
    raise SystemExit(2)
sys.path.insert(0, sys.argv[1])
from kernel_optimizer.evaluation import wall_attribution  # noqa: E402
from kernel_optimizer.models.reports import TuningStats  # noqa: E402


def spearman(xs, ys):
    """Rank correlation, ties averaged. Returns None when a side is constant (undefined, not 0)."""
    n = len(xs)
    if n < 3:
        return None

    def ranks(v):
        order = sorted(range(n), key=lambda i: v[i])
        r = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r

    if len(set(xs)) < 2 or len(set(ys)) < 2:
        return None
    rx, ry = ranks(xs), ranks(ys)
    mx, my = statistics.fmean(rx), statistics.fmean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    dx = sum((a - mx) ** 2 for a in rx) ** 0.5
    dy = sum((b - my) ** 2 for b in ry) ** 0.5
    return num / (dx * dy) if dx and dy else None


rows = []          # one per (candidate, knob)
walled_rows = []   # the subset whose knob was truncated by a resource limit

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
        cid = sp.get("candidate_id")
        if cid:
            spaces[cid] = {d.get("name"): list(d.get("choices") or [])
                           for d in (sp.get("domains") or []) if isinstance(d, dict)}

    trials = defaultdict(list)
    for e in evs:
        if e.get("type") != "TRIAL_DONE":
            continue
        t = (e.get("payload") or {}).get("trial") or {}
        if t.get("candidate_id"):
            trials[t["candidate_id"]].append(t)

    stats_for = {}
    pending = None
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

    for cid, stats in stats_for.items():
        space = spaces.get(cid) or {}
        refused = [(t.get("params") or {}).get("values") or {} for t in trials.get(cid, [])
                   if t.get("failure_kind") == "infeasible_shared_memory"]
        walls = {w.param: w for w in wall_attribution.find_walls(stats, refused)} if refused else {}
        for ps in stats.param_stats:
            by_value = ps.latency_by_value or {}
            measured = [v for v, ms in by_value.items() if isinstance(ms, (int, float))]
            declared = space.get(ps.name) or []
            if len(measured) < 2 or not declared:
                continue
            lat = [float(by_value[v]) for v in measured]
            best, med = min(lat), statistics.median(lat)
            row = {
                "label": label, "cand": cid, "knob": ps.name,
                # slope: the tail gain the wall finder would compute for this knob
                "slope": float(walls[ps.name].tail_gain_pct) if ps.name in walls else 0.0,
                # coverage: fraction of declared choices ever measured
                "coverage": len(measured) / len(declared),
                # exploit: how much the best choice beats the typical choice, in percent
                "exploit": 100.0 * (med - best) / med if med else 0.0,
                "walled": ps.name in walls,
            }
            rows.append(row)
            if row["walled"]:
                walled_rows.append(row)

print(f"knob-rows: {len(rows)} over {len({(r['label'], r['cand']) for r in rows})} candidates; "
      f"walled knobs: {len(walled_rows)}")
print()

pairs = (("slope", "coverage"), ("slope", "exploit"), ("coverage", "exploit"))
print("Spearman between the three signals, over all knob-rows:")
for a, b in pairs:
    rho = spearman([r[a] for r in rows], [r[b] for r in rows])
    verdict = ("undefined (a side is constant)" if rho is None else
               "essentially independent -- a composite is justified" if abs(rho) < 0.3 else
               "moderately related" if abs(rho) < 0.6 else
               "**one is standing in for the other** -- combining them adds little")
    print(f"  {a:9s} vs {b:9s}  rho {'None' if rho is None else f'{rho:+.3f}'}   {verdict}")
print()

# Would plain "sample what you haven't sampled" have found the walled knob?
if walled_rows:
    wcov = [r["coverage"] for r in walled_rows]
    ocov = [r["coverage"] for r in rows if not r["walled"]]
    print("Is the WALLED knob distinguishable by coverage alone?")
    print(f"  walled knobs   coverage p50 {statistics.median(wcov):.2f}  n={len(wcov)}")
    if ocov:
        print(f"  other knobs    coverage p50 {statistics.median(ocov):.2f}  n={len(ocov)}")
        print("  => " + ("coverage does NOT single out the walled knob, so an "
                         "'explore the unexplored' rule would not have found it"
                         if abs(statistics.median(wcov) - statistics.median(ocov)) < 0.15
                         else "coverage DOES separate them -- a plain unexplored-first rule "
                              "might find the wall without any slope signal"))
    print()
    print("Walled knobs, with all three signals (the rows a composite would have to rank):")
    for r in sorted(walled_rows, key=lambda r: -r["slope"]):
        print(f"  {r['label']:6s} {r['cand']} {r['knob']:20s} slope {r['slope']:+7.1f}%  "
              f"coverage {r['coverage']:.2f}  exploit {r['exploit']:5.1f}%")
