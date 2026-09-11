# -*- coding: utf-8 -*-
"""Can "resource -> latency conversion efficiency" be MEASURED on the data we have?

WHY THIS PROBE EXISTS. The v3 design names three core ideas: resource-dimension alignment
(done, S2), what an optimizing action does to resources (done, S2d declarations), and
"how efficiently a resource change converts into performance". The third has only a
QUALITATIVE verdict today (`conversion.py`: improved / no_conversion / regressed / flat /
unknown) and deliberately no number, because deltas have no common unit ACROSS dimensions.

But "no cross-dimension composite" does not imply "no number at all". Inside ONE dimension the
unit is fixed, so a slope d(latency)/d(resource) is dimensionally meaningful. The question this
probe answers is whether the DATA supports estimating one -- before any design is written.

WHAT IT MEASURES, per (candidate, dimension):
  n            complete trials carrying both a latency and that dimension
  n_distinct   how many distinct values of the resource were actually observed
  spearman     rank correlation between the resource and the median latency
  slope        OLS slope in the resource's own unit (ms per unit)
  r2           how much of the latency variance that single dimension explains
  monotone     whether the per-value medians are monotone in the resource

The decisive columns are `n_distinct` and `r2`. A dimension that never varies inside a
candidate cannot have a slope (this is why the ROUND-level data is not enough: 6 rounds). A
dimension that varies but explains nothing is telling us the conversion rate is not a stable
property.

CONFOUND, stated up front and NOT solved here: a parameter change moves several resources at
once, so a single-dimension slope absorbs its neighbours' effects. That is why the probe also
reports the multi-dimension fit -- if one dimension's r2 is already close to the joint r2, the
confound is small for that candidate; if not, a univariate slope would be a fiction. Recorded
prior: the cost map is NOT separable (0 of 10 one-step deltas agreed) and even the SIGN reverses
across a wide sweep, so the expectation is that this varies by candidate.

Reads only events.jsonl. No GPU, no re-measurement.
"""
from __future__ import annotations

import io
import json
import math
import os
import sys
from collections import defaultdict

DIMS = ("n_regs", "n_spills", "shared_bytes", "occupancy", "peak_alloc_bytes",
        "candidate_aten_bytes", "candidate_aten_ops", "threads_launched")


def _robust_ms(lat):
    """The statistic the run RANKED by: median when present, else mean.

    Must match `LatencyStats.robust_ms`; using the mean here would correlate a resource against a
    number no decision ever used, and this project has measured the mean's ranking accuracy at
    64.8% against the median's 93.2%.
    """
    if not isinstance(lat, dict):
        return None
    for key in ("median", "mean"):
        v = lat.get(key)
        if isinstance(v, (int, float)) and v > 0:
            return float(v)
    return None


def _dim(profile, name):
    """One dimension off a trial profile. `occupancy` is NESTED and a flat read fakes 'unmeasured'."""
    if not isinstance(profile, dict):
        return None
    if name == "occupancy":
        occ = profile.get("occupancy")
        if isinstance(occ, dict):
            v = occ.get("occupancy")
            return float(v) if isinstance(v, (int, float)) else None
        return None
    v = profile.get(name)
    return float(v) if isinstance(v, (int, float)) else None


def _spearman(xs, ys):
    def ranks(vs):
        order = sorted(range(len(vs)), key=lambda i: vs[i])
        r = [0.0] * len(vs)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and vs[order[j + 1]] == vs[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r

    rx, ry = ranks(xs), ranks(ys)
    return _pearson(rx, ry)


def _pearson(xs, ys):
    n = len(xs)
    if n < 3:
        return None
    mx, my = sum(xs) / n, sum(ys) / n
    sxy = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
    sxx = sum((a - mx) ** 2 for a in xs)
    syy = sum((b - my) ** 2 for b in ys)
    if sxx <= 0 or syy <= 0:
        return None
    return sxy / math.sqrt(sxx * syy)


def _ols(xs, ys):
    """Slope, intercept, r2 for a single predictor. None when x does not vary."""
    n = len(xs)
    if n < 3:
        return None
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((a - mx) ** 2 for a in xs)
    if sxx <= 0:
        return None
    sxy = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
    slope = sxy / sxx
    intercept = my - slope * mx
    ss_tot = sum((b - my) ** 2 for b in ys)
    if ss_tot <= 0:
        return None
    ss_res = sum((b - (intercept + slope * a)) ** 2 for a, b in zip(xs, ys))
    return slope, intercept, 1.0 - ss_res / ss_tot


def _ols_multi(cols, ys):
    """r2 of a joint fit over several standardized predictors, by normal equations.

    Only the r2 is wanted -- how much of the latency variance ALL the varying dimensions together
    explain -- so a small Gaussian solve is enough and no library is needed (the boxes' orch venvs
    have numpy, but this must also run where they do not).
    """
    n = len(ys)
    k = len(cols)
    if k == 0 or n < k + 2:
        return None
    # Standardize so the normal equations are well conditioned across wildly different units
    # (shared_bytes ~1e4, occupancy ~1e-1).
    std_cols = []
    for c in cols:
        m = sum(c) / n
        v = math.sqrt(sum((x - m) ** 2 for x in c) / n)
        if v <= 0:
            return None
        std_cols.append([(x - m) / v for x in c])
    my = sum(ys) / n
    yc = [y - my for y in ys]
    # X'X and X'y with an intercept column dropped (data is centred).
    a = [[sum(std_cols[i][t] * std_cols[j][t] for t in range(n)) for j in range(k)]
         for i in range(k)]
    b = [sum(std_cols[i][t] * yc[t] for t in range(n)) for i in range(k)]
    # Gaussian elimination with partial pivoting, plus a ridge nudge for collinear columns.
    for i in range(k):
        a[i][i] += 1e-9
    for i in range(k):
        p = max(range(i, k), key=lambda r: abs(a[r][i]))
        if abs(a[p][i]) < 1e-12:
            return None
        a[i], a[p] = a[p], a[i]
        b[i], b[p] = b[p], b[i]
        for r in range(i + 1, k):
            f = a[r][i] / a[i][i]
            for c in range(i, k):
                a[r][c] -= f * a[i][c]
            b[r] -= f * b[i]
    coef = [0.0] * k
    for i in range(k - 1, -1, -1):
        s = b[i] - sum(a[i][j] * coef[j] for j in range(i + 1, k))
        coef[i] = s / a[i][i]
    ss_tot = sum(y * y for y in yc)
    if ss_tot <= 0:
        return None
    ss_res = sum((yc[t] - sum(coef[i] * std_cols[i][t] for i in range(k))) ** 2 for t in range(n))
    return 1.0 - ss_res / ss_tot


def scan(run_dir):
    """Per-candidate trial series from one run's events.jsonl."""
    path = os.path.join(run_dir, "events.jsonl")
    if not os.path.isfile(path):
        return {}
    series = defaultdict(list)
    with io.open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line or '"TRIAL_DONE"' not in line:
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
            if ms is None:
                continue
            prof = t.get("profile")
            row = {"ms": ms}
            for d in DIMS:
                v = _dim(prof, d)
                if v is not None:
                    row[d] = v
            series[t.get("candidate_id") or "?"].append(row)
    return series


def analyse(series, min_n=8):
    out = []
    for cand, rows in sorted(series.items()):
        if len(rows) < min_n:
            continue
        ys = [r["ms"] for r in rows]
        varying = []
        for d in DIMS:
            xs = [r.get(d) for r in rows]
            if any(x is None for x in xs):
                # Restrict to the subset where this dimension WAS measured, rather than dropping
                # the dimension: a partially-measured dimension is common (reused trials leave no
                # artefact, so ~40% of candidates lack some readings).
                pairs = [(r[d], r["ms"]) for r in rows if r.get(d) is not None]
            else:
                pairs = list(zip(xs, ys))
            if len(pairs) < min_n:
                continue
            px = [p[0] for p in pairs]
            py = [p[1] for p in pairs]
            distinct = len(set(px))
            fit = _ols(px, py) if distinct >= 2 else None
            sp = _spearman(px, py) if distinct >= 2 else None
            # Monotonicity of the per-value medians: a stable conversion rate implies it.
            by_val = defaultdict(list)
            for a, b in pairs:
                by_val[a].append(b)
            meds = [(_v, sorted(vs)[len(vs) // 2]) for _v, vs in sorted(by_val.items())]
            mono = None
            if len(meds) >= 3:
                ups = sum(1 for i in range(len(meds) - 1) if meds[i + 1][1] > meds[i][1])
                downs = sum(1 for i in range(len(meds) - 1) if meds[i + 1][1] < meds[i][1])
                mono = (ups == 0 or downs == 0)
            out.append({"candidate": cand, "dim": d, "n": len(pairs), "distinct": distinct,
                        "spearman": sp, "slope": fit[0] if fit else None,
                        "r2": fit[2] if fit else None, "monotone": mono,
                        "lo": min(px), "hi": max(px)})
            if distinct >= 2:
                varying.append(d)
        # Joint fit over every dimension that varies, on the rows where all of them are present.
        if len(varying) >= 2:
            usable = [r for r in rows if all(r.get(d) is not None for d in varying)]
            if len(usable) >= len(varying) + 3:
                cols = [[r[d] for r in usable] for d in varying]
                jr2 = _ols_multi(cols, [r["ms"] for r in usable])
                out.append({"candidate": cand, "dim": "__JOINT__(%d dims)" % len(varying),
                            "n": len(usable), "distinct": None, "spearman": None,
                            "slope": None, "r2": jr2, "monotone": None,
                            "lo": None, "hi": None})
    return out


def main():
    roots = sys.argv[1:]
    if not roots:
        print("usage: probe_conversion_efficiency.py <runs_dir> [<runs_dir> ...]")
        raise SystemExit(2)
    run_dirs = []
    for root in roots:
        if os.path.isfile(os.path.join(root, "events.jsonl")):
            run_dirs.append(root)
            continue
        for name in sorted(os.listdir(root)):
            d = os.path.join(root, name)
            if os.path.isdir(d) and os.path.isfile(os.path.join(d, "events.jsonl")):
                run_dirs.append(d)
    print("scanning %d run(s)" % len(run_dirs))
    allrows = []
    for d in run_dirs:
        series = scan(d)
        rows = analyse(series)
        n_tr = sum(len(v) for v in series.values())
        print("  %-46s %4d complete trials, %2d candidates, %3d (cand,dim) series"
              % (os.path.basename(d), n_tr, len(series), len(rows)))
        for r in rows:
            r["run"] = os.path.basename(d)
        allrows.extend(rows)

    print()
    print("=" * 108)
    print("PER-DIMENSION SUMMARY  (only series with n>=8; 'usable' = >=3 distinct values)")
    print("=" * 108)
    print("%-22s %6s %7s %8s %9s %9s %9s" %
          ("dimension", "series", "usable", "med|rho|", "med r2", "mono", "signflip"))
    for d in DIMS:
        rows = [r for r in allrows if r["dim"] == d]
        usable = [r for r in rows if (r["distinct"] or 0) >= 3 and r["r2"] is not None]
        if not rows:
            print("%-22s %6d %7s" % (d, 0, "-"))
            continue
        rhos = sorted(abs(r["spearman"]) for r in usable if r["spearman"] is not None)
        r2s = sorted(r["r2"] for r in usable if r["r2"] is not None)
        monos = [r["monotone"] for r in usable if r["monotone"] is not None]
        slopes = [r["slope"] for r in usable if r["slope"] is not None]
        pos = sum(1 for s in slopes if s > 0)
        neg = sum(1 for s in slopes if s < 0)
        print("%-22s %6d %7d %8s %9s %9s %9s" % (
            d, len(rows), len(usable),
            "%.3f" % rhos[len(rhos) // 2] if rhos else "-",
            "%.3f" % r2s[len(r2s) // 2] if r2s else "-",
            "%d/%d" % (sum(1 for m in monos if m), len(monos)) if monos else "-",
            "%d+/%d-" % (pos, neg) if slopes else "-"))

    joint = [r for r in allrows if r["dim"].startswith("__JOINT__") and r["r2"] is not None]
    if joint:
        js = sorted(r["r2"] for r in joint)
        print()
        print("JOINT fits: %d candidates, median r2 %.3f  (min %.3f, max %.3f)"
              % (len(joint), js[len(js) // 2], js[0], js[-1]))
        print("  -- compare against the per-dimension median r2 above: if a single dimension is")
        print("     already close to the joint figure, the confound is small for that candidate.")

    out_path = os.path.join(os.getcwd(), "conversion_efficiency_rows.json")
    with io.open(out_path, "w", encoding="utf-8") as fh:
        fh.write(json.dumps(allrows, indent=1, default=str))
    print()
    print("wrote %s (%d rows)" % (out_path, len(allrows)))


if __name__ == "__main__":
    main()
