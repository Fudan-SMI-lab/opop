"""The step-3 three-arm comparison, with every confound named next to the number.

WHY THIS SCRIPT AND NOT A TABLE IN A COMMIT MESSAGE. The three arms did NOT differ in one variable, so
a bare ranking would misinform. Measured differences:

  arm 1  label,  2e off, box 1 (Xeon 8358P @ 2.60 GHz)
  arm 2  vector, 2e off, box 4 GPU 1 (Xeon 8352V @ 2.10 GHz)
  arm 3  vector, 2e on,  box 4 GPU 0 (same box as arm 2)

So arm1-vs-arm2 confounds `mode` with the BOX (compile_s 0.321 s vs 0.423 s, +32%), while arm2-vs-arm3
is the clean pair: same box, same mode, 2e the only difference. Any statement about 2e must come from
the second pair; any statement about `mode` from the first is confounded and must say so.

AND THE GAPS ARE SMALL RELATIVE TO KNOWN NOISE. The project's own measurement is that an independent
re-evaluation of the SAME configuration moves +-2-4% with an unstable sign. The three final numbers
span 2.49..2.83 ms, i.e. 13.7% -- above that floor, so the ordering is probably real -- but the
arm2-vs-arm3 gap of 8.0% is only about twice the re-eval gap, on n=1 run per arm. Reported with that
comparison attached, because a reader who does not know the noise floor cannot price a 8% difference.

`final_reeval_ms` is the only latency that may be quoted (it is an independent re-measurement in a
fresh process); `tuned_ms` is the tuner's own best and is optimistically biased by selection.
"""
import json
import os
import sys
from collections import Counter

REEVAL_NOISE_PCT = 4.0  # the project's measured re-eval gap, upper end, sign unstable


def read(rd):
    evs = []
    with open(os.path.join(rd, "events.jsonl"), encoding="utf-8") as fh:
        for ln in fh:
            ln = ln.strip()
            if ln:
                try:
                    evs.append(json.loads(ln))
                except Exception:
                    pass
    c = Counter(e.get("type") for e in evs)
    fin = [e for e in evs if e.get("type") == "RUN_FINISHED"]
    out = {
        "events": len(evs),
        "span_h": (evs[-1]["ts"] - evs[0]["ts"]) / 3600.0,
        "trials": c.get("TRIAL_DONE", 0),
        "rewrites": c.get("REWRITE_PRODUCED", 0),
        "rounds": c.get("FAMILY_ROUND_RECORDED", 0),
        "finished": bool(fin),
        "wall_h": None, "best": None, "family": None, "reeval": None,
        "same_prec": None, "vs": None, "precision": None,
        "walls_attributed": 0,
    }
    wc = [e for e in evs if e.get("type") == "WALL_CLOCK_REACHED"]
    if wc:
        out["wall_h"] = (wc[-1].get("payload") or {}).get("elapsed_hours")
    if fin:
        b = ((fin[-1].get("payload") or {}).get("summary") or {}).get("best") or {}
        hv = b.get("honest_verdict") or {}
        out.update(best=b.get("candidate_id"), family=b.get("family_id"),
                   reeval=b.get("final_reeval_ms"), precision=b.get("precision"),
                   same_prec=hv.get("same_precision_speedup"),
                   vs=hv.get("compared_against"))
    for e in evs:
        if e.get("type") != "RESOURCE_WALL_ATTRIBUTED":
            continue
        for w in ((e.get("payload") or {}).get("walls") or []):
            if w.get("verdict") == "attributed":
                out["walls_attributed"] += 1
    return out


rows = []
for spec in sys.argv[1:]:
    label, rd = spec.split("=", 1)
    rows.append((label, read(rd)))

print(f"{'arm':26s} {'reeval_ms':>10s} {'same-prec':>10s} {'trials':>7s} "
      f"{'rewrites':>9s} {'span_h':>7s} {'2e walls':>9s}")
for label, r in rows:
    print(f"{label:26s} {str(r['reeval']):>10s} "
          f"{(str(r['same_prec']) + 'x') if r['same_prec'] else '?':>10s} "
          f"{r['trials']:7d} {r['rewrites']:9d} {r['span_h']:7.2f} {r['walls_attributed']:9d}")
    print(f"{'':26s} winner {r['best']} / {r['family']}   "
          f"{r['precision']} vs {r['vs']}   "
          f"{'FINISHED' if r['finished'] else 'DID NOT FINISH'}")

vals = [(label, r["reeval"]) for label, r in rows if isinstance(r["reeval"], (int, float))]
if len(vals) >= 2:
    best_label, best_ms = min(vals, key=lambda x: x[1])
    worst_label, worst_ms = max(vals, key=lambda x: x[1])
    print()
    print(f"spread: {best_label} {best_ms} ms .. {worst_label} {worst_ms} ms "
          f"= {100.0 * (worst_ms - best_ms) / best_ms:.1f}%")
    print(f"the project's measured re-eval gap on the SAME configuration is +-{REEVAL_NOISE_PCT}% "
          f"with an unstable sign, so:")
    for i in range(len(vals)):
        for j in range(i + 1, len(vals)):
            (la, a), (lb, b) = vals[i], vals[j]
            gap = 100.0 * abs(a - b) / min(a, b)
            verdict = ("larger than the noise floor" if gap > REEVAL_NOISE_PCT
                       else "WITHIN the noise floor -- not a difference")
            print(f"  {la} vs {lb}: {gap:5.1f}%  ({gap / REEVAL_NOISE_PCT:.1f}x the floor) "
                  f"-- {verdict}")
