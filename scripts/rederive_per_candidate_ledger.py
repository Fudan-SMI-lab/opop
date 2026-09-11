#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Re-derive the CORRECT per-candidate ledger offline, from the same events.jsonl.

docs/result-s2d-pooled-ledger-defect.md states the correct ledger is re-derivable offline from
`REWRITE_PRODUCED`. This is that derivation, run on the real entry rather than on a simulation, so the
claim is measured instead of asserted. The three live runs hold the PRE-FIX orchestrator, so their
journalled entries are pooled and this is how the paper's numbers are obtained from them.

The pooled entry scores BOTH candidates' declarations against ONE conversion -- the family
incumbent's. Its fingerprint needs no recomputation: a `rel` is a property of a MEASUREMENT, so the
same `rel` appearing under two candidates' rows is the two being scored against one measurement.

MEASURED on box 2's first production entry (`run-l3-43-20260911-052630`, fam-efd15aa9 round 1):

    journalled (pooled)                    7 hits /  6 misses / 3 vacuous, candidate_id=None
    cand-2d8eaf9a  H1+H3  2.8616 ms        3 hits /  2 misses / 3 vacuous
    cand-3760b4d7  H2     3.2031 ms        7 hits /  1 miss   / 0 vacuous
    correct total                         10 hits /  3 misses / 3 vacuous

and on box 1's, the CONTROL arm's first round -- same defect, so it is not arm-specific:

    journalled (pooled)                    6 hits / 10 misses / 0 vacuous, candidate_id=None
    cand-70cbf6bc  H2         3.1842 ms    5 hits /  3 misses / 0 vacuous
    cand-d02b0742  H1+H2+H3   3.1329 ms    3 hits /  5 misses / 0 vacuous
    correct total                          8 hits /  8 misses / 0 vacuous

The pooling inflates misses (3->6 and 8->10) and loses hits on both arms, and it charges box 2's H2 --
the most accurate predictions either arm has made, 7 for 8 -- with five misses earned by the other
candidate's changes.

Three field-name traps, all hit while writing this and all worth their inline comments: expectations
live in `models.reports` (not `models.core`); `entry["conversion"]` is the verdict LABEL string, not the
deltas dict; and `occupancy` is NESTED (`profile["occupancy"]["occupancy"]`), which production reads via
a `nested` flag. Missing the third silently dropped every occupancy row as `unmeasured` while the
harness's own entry read `0.1667 -> 0.1667` -- it cost box 1 two hits and box 2 one. The parent reading
comes from the entry's own per-dimension `before` fields.

    python rederive_per_candidate_ledger.py <runs_dir> [<repo_root>]
"""
from __future__ import annotations

import io
import json
import os
import sys

RUNS = sys.argv[1]
REPO = sys.argv[2] if len(sys.argv) > 2 else "/root/autodl-tmp/work/opop"
sys.path.insert(0, os.path.join(REPO, "src"))

from kernel_optimizer.evaluation.reconcile import reconcile  # noqa: E402
from kernel_optimizer.models.reports import ResourceExpectation  # noqa: E402


def latest_run(root):
    """Accept EITHER a runs directory or a single run directory.

    Passing a run dir is the natural thing to have on hand -- every other script here takes one -- and
    this took only a runs root, so it died on `os.path.join(None, ...)`. That is a real generality gap
    rather than a usage error: a tool that reads one run should accept that run.
    """
    if os.path.exists(os.path.join(root, "events.jsonl")):
        return root
    c = []
    for d in sorted(os.listdir(root)):
        p = os.path.join(root, d)
        if os.path.isdir(p) and os.path.exists(os.path.join(p, "events.jsonl")):
            c.append((os.path.getmtime(os.path.join(p, "events.jsonl")), p))
    c.sort()
    return c[-1][1] if c else None


run = latest_run(RUNS)
if run is None:
    raise SystemExit("no run with an events.jsonl under %s" % RUNS)
evs = []
with io.open(os.path.join(run, "events.jsonl"), encoding="utf-8") as fh:
    for line in fh:
        line = line.strip()
        if line:
            try:
                evs.append(json.loads(line))
            except ValueError:
                pass
print("run: %s   events: %d" % (run, len(evs)))

# --- what each candidate DECLARED, kept per candidate (this is the fix's whole point) -------------
declared = {}          # candidate_id -> (hypothesis_id, [ResourceExpectation])
order = []
for e in evs:
    if e.get("type") != "REWRITE_PRODUCED":
        continue
    p = e.get("payload") or {}
    cid = p.get("candidate_id")
    exps = p.get("resource_expectations") or p.get("expectations") or []
    keep = []
    for x in exps:
        try:
            keep.append(ResourceExpectation(**x) if isinstance(x, dict) else x)
        except Exception as exc:                                   # noqa: BLE001
            print("  !! expectation rejected for %s: %s" % (cid, exc))
    # The family is carried too, because the parent reading a candidate is scored against is the
    # FAMILY's incumbent -- see the parent recovery below. Free here:
    # `REWRITE_PRODUCED.payload.family_id` is journalled (there is no parent field, but family_id is).
    declared[cid] = (p.get("hypothesis_id"), keep, p.get("family_id") or "")
    order.append(cid)
print("\ndeclarations: %s" % [(c, declared[c][0], len(declared[c][1]), declared[c][2])
                             for c in order])

# --- each candidate's OWN best profile, from its OWN trials ---------------------------------------
# The defect is that this step never happened: the round's single conversion was reused for every
# candidate. A candidate with no measured profile must come back UNMEASURED, not inherit one.
best = {}              # candidate_id -> (median_ms, profile dict)
for e in evs:
    if e.get("type") != "TRIAL_DONE":
        continue
    t = ((e.get("payload") or {}).get("trial")) or {}
    if t.get("status") != "complete":
        continue
    cid = t.get("candidate_id")
    lat = (t.get("latency_ms") or {}).get("median")
    prof = t.get("profile")
    if not isinstance(lat, (int, float)) or not isinstance(prof, dict):
        continue
    if cid not in best or lat < best[cid][0]:
        best[cid] = (lat, prof)
print("\ncandidates with a measured best profile: %d" % len(best))
for cid in order:
    print("  %-18s %s" % (cid, ("%.4f ms" % best[cid][0]) if cid in best else "NO PROFILE"))

# The PARENT. `REWRITE_PRODUCED` carries no parent field -- checked in the emitter rather than
# guessed: a first pass looked for parent_id / parent_candidate_id / source_candidate_id, found none
# of them, and printed an empty dict. That is the "a wrong read returns a clean None" failure again.
#
# Nor is it in `entry["conversion"]`: that field holds only the VERDICT LABEL (a string such as
# "improved"), because `conversion_verdict` returns `resource_deltas` as a sibling key and the ledger
# entry stores just the label. A second wrong read, and this one raised AttributeError instead of
# returning None -- the loud failure, which is the one you want.
#
# Per `_candidate_conversion`'s own docstring the `before` is the family incumbent's profile from
# BEFORE the round, shared by every candidate in it "and correctly so: they all restructure the same
# parent". `DimensionReconciliation` carries that reading per row as `before`, so the parent profile
# is recoverable from the pooled entry itself without re-running anything.
#
# KEYED PER FAMILY. A single flat dict was correct only while a run had ONE entry: with two, the second
# family's `before` overwrote the first's and every candidate was scored against the wrong parent.
# Measured consequence -- `cand-3760b4d7` read 1 hit / 7 misses instead of 7 / 1, an exact inversion of
# the number this script exists to produce. Each family restructures its OWN incumbent, so the parent is
# a property of the family, never of the run.
parent_by_family = {}
for e in evs:
    if e.get("type") != "EXPECTATIONS_RECONCILED":
        continue
    fid = (e.get("payload") or {}).get("family_id") or ""
    prof = parent_by_family.setdefault(fid, {})
    for row in ((e.get("payload") or {}).get("reconciliation") or {}).get("per_dimension") or []:
        if isinstance(row.get("before"), (int, float)):
            prof[row.get("dimension")] = row["before"]
for fid, prof in parent_by_family.items():
    print("\nparent profile for %s, recovered from its own entry's rows: %s" % (fid, prof))

_FIELDS = ("n_regs", "n_spills", "shared_bytes", "occupancy", "threads_launched",
           "peak_alloc_bytes", "candidate_aten_bytes", "candidate_aten_ops")


def _read(profile: dict, name: str):
    """One dimension off a trial's profile dict, matching production's `conversion._read`.

    `occupancy` is NESTED: the profile holds a dict
    `{"occupancy": 0.1667, "active_warps": 8, "limiter": "registers", ...}` under that key, and
    production reads the inner scalar via a `nested` flag on its dimension table. A first version of
    this script did `before.get(f)`, got a dict, failed the isinstance check and silently dropped the
    dimension -- so every occupancy row came back `unmeasured` while the harness's own entry correctly
    read `0.1667 -> 0.1667`. Silent, and in the direction that manufactures a missing measurement out
    of a present one.
    """
    v = profile.get(name)
    if name == "occupancy" and isinstance(v, dict):
        v = v.get("occupancy")
    return float(v) if isinstance(v, (int, float)) else None


def deltas(before: dict, after: dict) -> dict:
    """`resource_deltas` in the shape `reconcile` reads: per dimension a before/after/delta/rel."""
    out = {}
    for f in _FIELDS:
        b, a = _read(before, f), _read(after, f)
        if b is None or a is None:
            continue
        rel = abs(a - b) / abs(b) if b else (0.0 if a == b else 1.0)
        out[f] = {"before": b, "after": a, "delta": a - b, "rel": rel}
    return out


print("\n" + "=" * 78)
print("CORRECT ledger: one entry per candidate, each against ITS OWN measurement")
print("=" * 78)
totals = {"hit": 0, "miss": 0, "vacuous": 0}
for cid in order:
    hyp, exps, fid = declared[cid]
    if cid not in best:
        print("\n%s (%s): UNMEASURED -- no complete trial with a profile, so every declaration is "
              "unmeasured rather than a miss" % (cid, hyp))
        continue
    parent_profile = parent_by_family.get(fid) or {}
    if not parent_profile:
        print("\n%s (%s, %s): no parent reading recoverable for this family; cannot form a delta"
              % (cid, hyp, fid))
        continue
    d = deltas(parent_profile, best[cid][1])
    rec = reconcile(exps, d, hypothesis_id=hyp or "")
    print("\n%s (%s, %s)   its own best = %.4f ms" % (cid, hyp, fid, best[cid][0]))
    print("  hits/misses/vacuous = %d / %d / %d" % (rec.hits, rec.misses, rec.vacuous))
    totals["hit"] += rec.hits
    totals["miss"] += rec.misses
    totals["vacuous"] += rec.vacuous
    for r in rec.per_dimension:
        print("    %-24s expected=%-10s actual=%-8s %-8s rel=%.4f"
              % (r.dimension, r.expected, r.actual, r.match, r.rel or 0.0))
print("\nre-derived totals: %s" % totals)

print("\n" + "=" * 78)
print("what the run actually journalled (POOLED)")
print("=" * 78)
for e in evs:
    if e.get("type") != "EXPECTATIONS_RECONCILED":
        continue
    p = e.get("payload") or {}
    r = p.get("reconciliation") or {}
    print("  candidate_id=%r  hypothesis_id=%r  hits/misses/vacuous = %s / %s / %s"
          % (p.get("candidate_id"), r.get("hypothesis_id"),
             r.get("hits"), r.get("misses"), r.get("vacuous")))
    seen = {}
    for row in (r.get("per_dimension") or []):
        seen.setdefault(round(row.get("rel") or 0.0, 6), []).append(row.get("dimension"))
    dup = {k: v for k, v in seen.items() if len(v) > 1 and k}
    print("  rel values appearing more than once (the pooling fingerprint): %s" % dup)
