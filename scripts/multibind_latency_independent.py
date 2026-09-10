"""Recount multi-binding with the two latency-derived dimensions removed.

WHY THIS RERUN EXISTS

`multibind_rule_check.py` counted five dimensions. Two of them are not dimensions:

    achieved_tbs    = byte_count / gpu_ms      (bottleneck.py:298)
    achieved_tflops = flop_count / gpu_ms

and `byte_count` / `flop_count` are TASK-level constants -- `orchestrator.py:1421` passes
`cost.compulsory_bytes`, computed once on the reference, identical for every candidate of a
task. So `pct_of_dram_peak` and `pct_of_compute_peak` are 1/latency on two different scales.
Verified: within a run, `gpu_ms x achieved_tbs` is constant to 0.07-0.36%.

Counting both gives latency two votes in any multi-binding tally, and inflates the rate two
ways at once: each latency column can push a candidate over the threshold on its own, and the
two of them move together, so they co-bind with each other by construction.

The dimensions that are genuinely independent of latency are the ones read at compile or
launch time:

    occupancy        low  = binding   (a launch-configuration read)
    n_regs           high = binding   (readable after one 0.4 ms launch)
    shared_bytes     high = binding   (readable at compile time)
    n_spills         any  = binding   (binary, not a fraction -- reported separately)

This script reports BOTH tallies side by side, and attributes the difference: for each
candidate it says whether it was multi-bound via independent dimensions alone, only via a
latency column, or both. That attribution is the point -- "the rate dropped" would leave open
whether multi-binding was real at all.

A note on what this does NOT invalidate. The two ratios are still the right way to ask "how
far from the physical roof is this kernel" -- L3:48 at 94.6% of a measured 924 GB/s is a real
and useful statement. What they cannot do is serve as two independent dimensions.

Zero GPU. Reads pre-dumped rows (see --dump) or events.jsonl directly.
"""
import argparse
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

SHARED_LIMIT = 101376          # box 1 and box 2 are both sm_89, opt-in limit
REG_LIMIT = 255

ap = argparse.ArgumentParser()
ap.add_argument("inputs", nargs="+", type=Path,
                help="run directories (with events.jsonl) and/or *.json row dumps")
ap.add_argument("--dump", type=Path, default=None,
                help="write the extracted rows here, so a run on a remote box can be "
                     "re-analysed locally without re-reading its events.jsonl")
args = ap.parse_args()


def rows_from_run(run):
    """One row per classified candidate: its evidence dict plus its winning trial's profile.

    Both sources are needed. `shared_bytes` and `n_spills` are absent from the classifier's
    evidence dict on some runs but present in TRIAL_DONE.profile; `occupancy` and
    `pct_of_*` are the other way round. Reading only one source silently reports the missing
    dimension as slack -- the failure mode that produced a 0% multi-binding rate the first
    time this analysis was attempted.
    """
    cls, best = {}, {}
    with open(run / "events.jsonl", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                e = json.loads(line)
            except Exception:
                continue
            t = e.get("type")
            if t == "BOTTLENECK_CLASSIFIED":
                pl = e["payload"]
                cid = pl.get("candidate_id")
                if cid:
                    cls[cid] = dict(pl.get("evidence") or {}, _kind=pl.get("kind"))
            elif t == "TRIAL_DONE":
                tr = (e.get("payload") or {}).get("trial") or {}
                if (tr.get("status") or "").lower() != "complete":
                    continue
                m = (tr.get("latency_ms") or {}).get("median")
                if not isinstance(m, (int, float)):
                    continue
                cid = tr.get("candidate_id")
                if cid not in best or m < best[cid][0]:
                    best[cid] = (m, tr.get("profile") or {})
    return [dict(run=run.name, cid=c, ev=ev, prof=best.get(c, (None, {}))[1])
            for c, ev in cls.items()]


rows = []
for p in args.inputs:
    if p.is_dir():
        if (p / "events.jsonl").exists():
            rows += rows_from_run(p)
        else:
            print("  SKIP (no events.jsonl): %s" % p)
    else:
        rows += json.load(open(p, encoding="utf-8"))
if args.dump:
    json.dump(rows, open(args.dump, "w", encoding="utf-8"), ensure_ascii=False)
    print("  wrote %d rows -> %s" % (len(rows), args.dump))


def num(*vals):
    for v in vals:
        if isinstance(v, (int, float)):
            return v
    return None


def indep(r):
    """The three latency-independent fractional dimensions. None = not measured."""
    ev, prof = r["ev"], r["prof"]
    occ = num(ev.get("occupancy"), prof.get("occupancy"))
    nr = num(ev.get("n_regs"), prof.get("n_regs"))
    sb = num(prof.get("shared_bytes"), ev.get("shared_bytes"))
    return {
        "occupancy": (1.0 - occ) if occ is not None else None,     # low occupancy = pressure
        "register_pressure": (nr / REG_LIMIT) if nr is not None else None,
        "shared_capacity": (min(sb, SHARED_LIMIT) / SHARED_LIMIT) if sb is not None else None,
    }


def latency_cols(r):
    """The two 1/latency columns, kept separate so their contribution can be attributed."""
    ev = r["ev"]
    out = {}
    for name, field in (("dram_bandwidth", "pct_of_dram_peak"),
                        ("compute", "pct_of_compute_peak")):
        v = ev.get(field)
        out[name] = min(v / 100.0, 1.0) if isinstance(v, (int, float)) else None
    return out


def alldims(r):
    d = indep(r)
    d.update(latency_cols(r))
    return d


THS = [0.70, 0.80, 0.85, 0.90]
byrun = Counter(r["run"] for r in rows)
print("=" * 100)
print("MULTI-BINDING, RECOUNTED WITHOUT THE TWO 1/LATENCY COLUMNS")
print("=" * 100)
print("\n  corpus: %d classified candidates" % len(rows))
for rn, n in sorted(byrun.items()):
    print("     %-34s %d" % (rn[:34], n))
cov = Counter()
for r in rows:
    for d, v in indep(r).items():
        if v is not None:
            cov[d] += 1
print("  coverage:  " + "   ".join("%s %d/%d" % (d, cov[d], len(rows))
                                   for d in ("occupancy", "register_pressure", "shared_capacity")))


def tally(fn, label):
    print("\n  --- %s ---" % label)
    for th in THS:
        hist, combos = Counter(), Counter()
        for r in rows:
            p = fn(r)
            b = tuple(sorted(d for d, v in p.items() if v is not None and v >= th))
            hist[len(b)] += 1
            if len(b) >= 2:
                combos[b] += 1
        tot = sum(hist.values())
        multi = sum(v for k, v in hist.items() if k >= 2)
        print("    th %.2f: %-30s multi %2d/%d (%.1f%%)"
              % (th, " ".join("%dd:%d" % (k, hist[k]) for k in sorted(hist)),
                 multi, tot, 100 * multi / tot if tot else 0.0))
        for c, n in combos.most_common(3):
            print("            %2d x %s" % (n, " + ".join(c)))


tally(alldims, "ALL FIVE (what the earlier run reported -- latency counted twice)")
tally(indep, "LATENCY-INDEPENDENT ONLY (occupancy / registers / shared)")

print("\n  --- attribution: was the multi-binding real, or carried by the latency columns? ---")
for th in (0.70, 0.85):
    neither = only_lat = only_ind = mixed = 0
    for r in rows:
        lat = [d for d, v in latency_cols(r).items() if v is not None and v >= th]
        ind = [d for d, v in indep(r).items() if v is not None and v >= th]
        if len(lat) + len(ind) < 2:
            neither += 1
        elif len(ind) >= 2 and not lat:
            only_ind += 1
        elif len(ind) <= 1 and lat:
            only_lat += 1
        else:
            mixed += 1
    print("    th %.2f of %d:  %d single-or-none   %d independent dims alone   "
          "%d needed a latency column   %d mixed"
          % (th, len(rows), neither, only_ind, only_lat, mixed))

print("\n  --- n_spills: binding is binary here, so it is never a fraction ---")
sp = [(r["run"], r["cid"], num(r["ev"].get("n_spills"), r["prof"].get("n_spills"))) for r in rows]
have = [x for x in sp if x[2] is not None]
nz = [x for x in have if x[2] > 0]
print("    measured on %d/%d candidates; %d spill" % (len(have), len(rows), len(nz)))
for a, b, c in sorted(nz, key=lambda x: -x[2])[:8]:
    print("       %-32s %-22s %d" % (a[:32], b[:22], c))

print("\n  --- identity re-verification (this is the check that started the correction) ---")
print("      if dram/compute pressure are 1/latency, gpu_ms x achieved_* is constant per run")
per = defaultdict(list)
for r in rows:
    ev = r["ev"]
    g, tbs, tfl = ev.get("gpu_ms"), ev.get("achieved_tbs"), ev.get("achieved_tflops")
    if isinstance(g, (int, float)) and isinstance(tbs, (int, float)) and tbs > 0:
        per[r["run"]].append((g * tbs, g * tfl if isinstance(tfl, (int, float)) else None))
for rn, v in sorted(per.items()):
    b = [x[0] for x in v]
    fl = [x[1] for x in v if x[1] is not None]
    sb = "%.6f-%.6f (%.2f%%)" % (min(b), max(b), 100 * (max(b) - min(b)) / max(b))
    sf = ("%.4f-%.4f (%.3f%%)" % (min(fl), max(fl), 100 * (max(fl) - min(fl)) / max(fl))
          if fl else "-")
    print("    %-32s n=%2d  gpu_ms*tbs %-26s gpu_ms*tflops %s" % (rn[:32], len(v), sb, sf))
