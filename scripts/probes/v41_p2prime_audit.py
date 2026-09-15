"""Audit the three defects an external review alleged in v41_p2prime.py, on real logs.

WHY AN AUDIT RATHER THAN A FIX. Each allegation is about whether a published number means
what it says, so the first job is to measure the defect's size on the same runs that
produced the number -- a fix applied before the size is known cannot be checked, and a
"corrected" figure whose delta nobody measured is just a second unverified figure.

The three, each independently decidable from the journal:

  1. SIGN/DENOMINATOR MISMATCH. scanner.py computes g_d = 1 - N/F (positive = N faster).
     The reader's marginal returned (n_med - f_med)/n_med -- opposite sign AND a different
     denominator. If true, every |g_m - y| is inflated by construction.
  2. CROSS-CANDIDATE POOLING. The reader buckets every completed trial in the run by axis
     value, with no candidate or space filter, so one candidate's BK=32 latency can be
     averaged with another implementation's BK=32.
  3. HOLDOUT LEAK. TRIAL_DONE is journalled (orchestrator:1324) BEFORE the scan step folds
     the result in and emits SCAN_BLOCK_DONE (:1335), so a prefix cut at SCAN_BLOCK_DONE
     already contains the C4's OWN four trials -- including the validation pair that y is
     computed from.

Prints, per contrast, the marginal under four definitions so the reader can see which part
of the change moves which number:
    old       (n-f)/n over the whole-run pool, prefix at SCAN_BLOCK_DONE
    signfix   1 - n/f, same pool, same prefix
    scoped    1 - n/f, THIS CANDIDATE's trials only
    noleak    1 - n/f, this candidate, prefix excluding the C4's own scan trials
"""
import json, sys
from collections import defaultdict
from pathlib import Path
from statistics import median

def load(run):
    out = []
    for line in (Path(run) / "events.jsonl").open(encoding="utf-8", errors="replace"):
        line = line.strip()
        if line:
            try: out.append(json.loads(line))
            except ValueError: pass
    return out

def marg(trials, axis, f_repr, n_repr, mode):
    b = defaultdict(list)
    for t in trials:
        v = (t.get("params") or {}).get("values") or {}
        if axis not in v: continue
        lat = t.get("latency_ms") or {}
        ms = lat.get("median") or lat.get("mean")
        if ms is None or ms <= 0: continue
        b[repr(v[axis])].append(float(ms))
    if len(b) < 2: return None, "no_slope"
    if f_repr not in b or n_repr not in b: return None, "unsampled"
    if len(b[f_repr]) < 2 or len(b[n_repr]) < 2: return None, "thin"
    f, n = median(b[f_repr]), median(b[n_repr])
    if f <= 0 or n <= 0: return None, "nonpos"
    return ((n - f) / n if mode == "old" else 1.0 - n / f), None

def run(rd):
    ev = load(rd)
    trials, scan_trials = [], defaultdict(set)
    # which trial_ids belong to which scan block (both roles)
    for e in ev:
        p = e.get("payload") or {}
        if e.get("type") == "SCAN_POINT_DONE" and p.get("trial_id"):
            scan_trials[str(p.get("scan_id"))].add(str(p["trial_id"]))
    # Window 1 predates axis_f_value on SCAN_BLOCK_DONE, so the PUBLISHED numbers came
    # from the reader's wall fallback: UW_PROBE_BATCH.walls keyed (space_id, axis), then
    # re-keyed onto the scan via SCAN_BLOCK_ADMITTED. Reproduced exactly, or this audit
    # would silently grade a different set of contrasts from the one under dispute.
    wall_ep, scan_ep = {}, {}
    for e in ev:
        t, p = e.get("type"), e.get("payload") or {}
        if t == "UW_PROBE_BATCH":
            for w in p.get("walls", []):
                if w.get("f_value") is not None and w.get("n_value") is not None:
                    wall_ep[(p.get("space_id"), w.get("axis"))] = (w["f_value"], w["n_value"])
        elif t == "SCAN_BLOCK_ADMITTED" and p.get("kind") == "C4":
            scan_ep.setdefault(str(p.get("scan_id")),
                               wall_ep.get((p.get("space_id"), p.get("axis")), (None, None)))
    rows = []
    prefix, by_cand = [], defaultdict(list)
    for e in ev:
        t, p = e.get("type"), e.get("payload") or {}
        if t == "TRIAL_DONE":
            rec = p.get("trial") or p
            if rec.get("status") == "complete":
                prefix.append(rec)
                by_cand[str(rec.get("candidate_id"))].append(rec)
        elif t == "SCAN_BLOCK_DONE" and p.get("kind") == "C4":
            if not p.get("full"): continue
            f_r, n_r = p.get("axis_f_value"), p.get("axis_n_value")
            if f_r is None or n_r is None:
                f_r, n_r = scan_ep.get(str(p.get("scan_id")), (None, None))
            if f_r is None or n_r is None: continue
            cid, sid, ax = str(p.get("candidate_id")), str(p.get("scan_id")), p.get("axis")
            own = scan_trials.get(sid, set())
            pool_all = list(prefix)
            pool_cand = list(by_cand[cid])
            pool_noleak = [x for x in pool_cand if str(x.get("trial_id")) not in own]
            res = {}
            res["old"], s0 = marg(pool_all, ax, f_r, n_r, "old")
            res["signfix"], _ = marg(pool_all, ax, f_r, n_r, "new")
            res["scoped"], _ = marg(pool_cand, ax, f_r, n_r, "new")
            res["noleak"], sN = marg(pool_noleak, ax, f_r, n_r, "new")
            rows.append(dict(scan=sid, axis=ax, cid=cid, gd=p.get("g_d"), y=p.get("y"),
                             n_own_in_prefix=len(own & {str(x.get("trial_id")) for x in pool_cand}),
                             n_all=len(pool_all), n_cand=len(pool_cand), skip=s0 or sN, **res))
    return rows

def fmt(x): return "   -    " if x is None else "%+8.4f" % x

for rd in sys.argv[1:]:
    rows = run(rd)
    print("=" * 118)
    print(Path(rd).parent.name, " n_contrasts(full, endpoints journalled) =", len(rows))
    print("  %-14s %-13s %8s %8s | %8s %8s %8s %8s | own %5s %5s"
          % ("scan", "axis", "g_d", "y", "old", "signfix", "scoped", "noleak", "nAll", "nCand"))
    for r in rows:
        print("  %-14s %-13s %s %s | %s %s %s %s | %3d %5d %5d"
              % (r["scan"][:14], str(r["axis"])[:13], fmt(r["gd"]), fmt(r["y"]),
                 fmt(r["old"]), fmt(r["signfix"]), fmt(r["scoped"]), fmt(r["noleak"]),
                 r["n_own_in_prefix"], r["n_all"], r["n_cand"]))
    for key in ("old", "signfix", "scoped", "noleak"):
        pairs = [(abs(r[key] - r["y"]), (r[key] > 0) == (r["y"] > 0))
                 for r in rows if r[key] is not None and r["y"] is not None]
        errs_c = [abs(r["gd"] - r["y"]) for r in rows
                  if r[key] is not None and r["gd"] is not None and r["y"] is not None]
        if not pairs:
            print("  %-8s n=0 (never computable)" % key); continue
        e = sorted(p[0] for p in pairs)
        m = e[len(e)//2] if len(e) % 2 else (e[len(e)//2-1] + e[len(e)//2]) / 2
        ec = sorted(errs_c)
        mc = ec[len(ec)//2] if len(ec) % 2 else (ec[len(ec)//2-1] + ec[len(ec)//2]) / 2
        print("  %-8s n=%2d  median|err_m|=%.4f  vs conditioned %.4f  ratio %5.1fx  sign %d/%d"
              % (key, len(pairs), m, mc, (m / mc if mc else float("inf")),
                 sum(1 for p in pairs if p[1]), len(pairs)))
