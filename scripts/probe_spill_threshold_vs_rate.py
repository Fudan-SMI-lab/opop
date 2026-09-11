# -*- coding: utf-8 -*-
"""Is `n_spills`'s high r2 a RATE, or just "spilling at all is catastrophic"?

WHY IT MATTERS. `probe_conversion_efficiency.py` found n_spills to be the one dimension whose
latency relationship is stable across three boxes (median r2 0.573/0.733/0.800, slope sign
consistent in 97 of 99 usable series) while n_regs, shared_bytes, occupancy and threads_launched
all sit under r2 0.10 and flip sign. That looks like the one place a conversion RATE could be
estimated.

But a rate and a threshold produce the same r2 on data where the resource takes two values. If
most series contain only `0` and `some large number`, the fit is measuring "a spilling kernel is
slower", which is already known and is NOT an efficiency. A rate claim needs the relationship to
hold WITHIN the spilling regime.

So this probe splits every series three ways:
  full        every trial
  binary      collapsed to spills==0 vs spills>0 -- how much of the r2 a threshold alone explains
  positive    only trials with spills>0 -- is there still a slope once the zeros are removed?

The decisive comparison is `r2_positive` against `r2_full`. If the positive-only fit collapses,
the "rate" is a threshold and the honest form of the third design point is a CLASSIFIER, not a
regression. Also reports ms-per-spill in absolute terms so the magnitude can be sanity-checked
against the kernel's own latency.
"""
from __future__ import annotations

import io
import json
import math
import os
import sys
from collections import defaultdict


def _robust_ms(lat):
    if not isinstance(lat, dict):
        return None
    for key in ("median", "mean"):
        v = lat.get(key)
        if isinstance(v, (int, float)) and v > 0:
            return float(v)
    return None


def _ols(xs, ys):
    n = len(xs)
    if n < 3:
        return None
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((a - mx) ** 2 for a in xs)
    if sxx <= 0:
        return None
    sxy = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
    slope = sxy / sxx
    icept = my - slope * mx
    ss_tot = sum((b - my) ** 2 for b in ys)
    if ss_tot <= 0:
        return None
    ss_res = sum((b - (icept + slope * a)) ** 2 for a, b in zip(xs, ys))
    return slope, 1.0 - ss_res / ss_tot


def scan(run_dir):
    path = os.path.join(run_dir, "events.jsonl")
    if not os.path.isfile(path):
        return {}
    series = defaultdict(list)
    with io.open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if '"TRIAL_DONE"' not in line:
                continue
            try:
                ev = json.loads(line)
            except Exception:
                continue
            if ev.get("type") != "TRIAL_DONE":
                continue
            t = (ev.get("payload") or {}).get("trial") or {}
            if t.get("status") != "complete":
                continue
            ms = _robust_ms(t.get("latency_ms"))
            prof = t.get("profile")
            if ms is None or not isinstance(prof, dict):
                continue
            sp = prof.get("n_spills")
            rg = prof.get("n_regs")
            if not isinstance(sp, (int, float)):
                continue
            series[t.get("candidate_id") or "?"].append(
                {"ms": ms, "spills": float(sp),
                 "regs": float(rg) if isinstance(rg, (int, float)) else None})
    return series


def main():
    roots = sys.argv[1:]
    if not roots:
        print("usage: probe_spill_threshold_vs_rate.py <runs_dir> [...]")
        raise SystemExit(2)
    run_dirs = []
    for root in roots:
        if os.path.isfile(os.path.join(root, "events.jsonl")):
            run_dirs.append(root)
            continue
        if not os.path.isdir(root):
            continue
        for name in sorted(os.listdir(root)):
            d = os.path.join(root, name)
            if os.path.isdir(d) and os.path.isfile(os.path.join(d, "events.jsonl")):
                run_dirs.append(d)

    rows = []
    for d in run_dirs:
        for cand, trials in scan(d).items():
            if len(trials) < 8:
                continue
            xs = [t["spills"] for t in trials]
            ys = [t["ms"] for t in trials]
            distinct = sorted(set(xs))
            n_zero = sum(1 for x in xs if x == 0)
            n_pos = len(xs) - n_zero
            full = _ols(xs, ys)
            binary = _ols([1.0 if x > 0 else 0.0 for x in xs], ys) if n_zero and n_pos else None
            pos_pairs = [(x, y) for x, y in zip(xs, ys) if x > 0]
            pos = _ols([p[0] for p in pos_pairs], [p[1] for p in pos_pairs]) \
                if len({p[0] for p in pos_pairs}) >= 2 and len(pos_pairs) >= 8 else None
            med_zero = None
            med_pos = None
            if n_zero:
                z = sorted(y for x, y in zip(xs, ys) if x == 0)
                med_zero = z[len(z) // 2]
            if n_pos:
                p = sorted(y for x, y in zip(xs, ys) if x > 0)
                med_pos = p[len(p) // 2]
            rows.append({
                "run": os.path.basename(d), "candidate": cand, "n": len(trials),
                "n_distinct_spills": len(distinct),
                "spill_values": distinct[:8],
                "n_zero": n_zero, "n_pos": n_pos,
                "r2_full": full[1] if full else None,
                "slope_full_ms_per_spill": full[0] if full else None,
                "r2_binary": binary[1] if binary else None,
                "r2_positive": pos[1] if pos else None,
                "slope_positive": pos[0] if pos else None,
                "med_ms_zero": med_zero, "med_ms_pos": med_pos,
            })

    print("=" * 104)
    print("SERIES: %d  (n>=8 trials, n_spills measured)" % len(rows))
    print("=" * 104)
    only_zero = [r for r in rows if r["n_pos"] == 0]
    only_pos = [r for r in rows if r["n_zero"] == 0]
    mixed = [r for r in rows if r["n_zero"] and r["n_pos"]]
    print("  never spilled (all trials 0) .......... %3d" % len(only_zero))
    print("  always spilled (no trial 0) ........... %3d" % len(only_pos))
    print("  MIXED (both regimes present) .......... %3d" % len(mixed))

    def med(vals):
        v = sorted(x for x in vals if x is not None)
        return v[len(v) // 2] if v else None

    print()
    print("--- on the MIXED series, which is where the r2 comes from -------------------------")
    r2f = med(r["r2_full"] for r in mixed)
    r2b = med(r["r2_binary"] for r in mixed)
    print("  median r2, spills as a COUNT ............ %s" % ("%.3f" % r2f if r2f else "-"))
    print("  median r2, spills as a 0/1 THRESHOLD .... %s" % ("%.3f" % r2b if r2b else "-"))
    if r2f and r2b:
        print("  -> the count explains %+.3f beyond the threshold" % (r2f - r2b))
        print("     (a small or negative number means the 'rate' IS a threshold)")

    withpos = [r for r in rows if r["r2_positive"] is not None]
    print()
    print("--- WITHIN the spilling regime (zeros removed) ------------------------------------")
    print("  series with >=2 distinct positive spill counts and n>=8 ... %d" % len(withpos))
    if withpos:
        rp = med(r["r2_positive"] for r in withpos)
        print("  median r2 there ........................................... %.3f" % rp)
        pos_slopes = [r["slope_positive"] for r in withpos if r["slope_positive"] is not None]
        npos = sum(1 for s in pos_slopes if s > 0)
        print("  slope sign ................................................ %d+/%d-"
              % (npos, len(pos_slopes) - npos))

    print()
    print("--- absolute cost of spilling, on the mixed series --------------------------------")
    ratios = [(r["med_ms_pos"] / r["med_ms_zero"], r) for r in mixed
              if r["med_ms_zero"] and r["med_ms_pos"]]
    ratios.sort()
    if ratios:
        print("  median latency ratio (spilling / not) ..... %.3fx" % ratios[len(ratios) // 2][0])
        print("  range ..................................... %.3fx .. %.3fx"
              % (ratios[0][0], ratios[-1][0]))
        print("  series where spilling was FASTER .......... %d of %d"
              % (sum(1 for x, _ in ratios if x < 1.0), len(ratios)))

    out = os.path.join(os.getcwd(), "spill_threshold_rows.json")
    with io.open(out, "w", encoding="utf-8") as fh:
        fh.write(json.dumps(rows, indent=1, default=str))
    print()
    print("wrote %s" % out)


if __name__ == "__main__":
    main()
