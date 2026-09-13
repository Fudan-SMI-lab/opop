"""For a FREED wall: did the child free it by lowering shared memory, or by changing what a step costs?

WHY THIS IS NEEDED. A3 came back FREED on both children of the one attributed wall, but the two
children disagree about HOW:

  cand-5bd8ddd3  shared_bytes 36864 at its optimum  (-36864 B vs the walled parent)  3.8369 ms
  cand-e5172b8a  shared_bytes 73728 at its optimum  (+0 B vs the walled parent)      3.0397 ms

The second is the interesting one and the reason this script exists. "shared_bytes at the optimum"
is measured at a DIFFERENT configuration than the wall was: the wall is a statement about BLOCK_N=128
(122880 B, 1.212x over the 101376 B limit), while the optimum sits at whatever BLOCK_N the tuner
preferred. So an unchanged optimum footprint does not mean the footprint at BLOCK_N=128 is unchanged
-- and if it IS unchanged, then BLOCK_N=128 could not have become feasible by freeing shared memory,
and "FREED" would be measuring something else (a renamed knob, a changed meaning for the same value,
a different tiling where 128 no longer implies the same footprint).

WHAT THIS READS. For the parent and each child, the shared_bytes actually recorded AT the wall's
value of the knob, from completed trials only -- the apples-to-apples comparison. Three outcomes:

  footprint fell at K=V     the rewrite did what 2e's prompt asked: same knob value, less shared
                            memory, now under the limit. This is the mechanism working.
  footprint unchanged/up    K=V became feasible for some OTHER reason. The wall was not the binding
                            constraint, or V no longer means what it meant. Attribution needs review.
  no completed trial at K=V in the parent, expected -- that is what "refused" means. Stated rather
                            than left blank so the asymmetry between parent and child is legible.

The parent by construction has NO completed trial at K=V, so its footprint there comes from the
2e probe's own measurement (`max_shared` on the wall record), which is the only number that exists
for a configuration the compiler refused.
"""
import json
import os
import sys
from collections import defaultdict


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


rd = sys.argv[1]
evs = load(rd)

children = defaultdict(list)
fam_of = {}
for e in evs:
    if e.get("type") != "CANDIDATE_REGISTERED":
        continue
    c = (e.get("payload") or {}).get("candidate") or {}
    if c.get("candidate_id"):
        fam_of[c["candidate_id"]] = c.get("family_id")
        for p in (c.get("parent_ids") or []):
            children[p].append(c["candidate_id"])

trials = defaultdict(list)
for e in evs:
    if e.get("type") != "TRIAL_DONE":
        continue
    t = (e.get("payload") or {}).get("trial") or {}
    if t.get("candidate_id"):
        trials[t["candidate_id"]].append(t)


def eq(a, b):
    try:
        return float(a) == float(b)
    except (TypeError, ValueError):
        return a == b


def at_knob(cid, knob, value):
    """(shared_bytes, latency) over COMPLETED trials holding knob == value."""
    rows = []
    for t in trials.get(cid, []):
        vals = (t.get("params") or {}).get("values") or {}
        if knob not in vals or not eq(vals[knob], value):
            continue
        if t.get("status") != "complete":
            continue
        prof = t.get("profile") or {}
        lat = t.get("latency_ms") or {}
        med = lat.get("median")
        ms = med if isinstance(med, (int, float)) and med > 0 else lat.get("mean")
        rows.append((prof.get("shared_bytes"), ms, dict(vals)))
    return rows


def descend(cid):
    out, stack = [], list(children.get(cid, []))
    while stack:
        x = stack.pop(0)
        out.append(x)
        stack.extend(children.get(x, []))
    return out


for e in evs:
    if e.get("type") != "RESOURCE_WALL_ATTRIBUTED":
        continue
    p = e.get("payload") or {}
    cid = p.get("candidate_id")
    for w in (p.get("walls") or []):
        if w.get("verdict") != "attributed":
            continue
        knob, V = w.get("param"), w.get("refused_value")
        limit = w.get("limit")
        print(f"=== {cid} {knob}={V}   limit {limit} B")
        print(f"  parent at {knob}={V}: REFUSED, so no completed trial exists. The 2e probe measured "
              f"{w.get('max_shared')} B ({w.get('over_ratio')}x over) -- the only figure available "
              f"for a configuration the compiler rejected.")
        for k in descend(cid):
            rows = at_knob(k, knob, V)
            if not rows:
                print(f"  {k}: no COMPLETED trial at {knob}={V}")
                continue
            shareds = [s for s, _, _ in rows if isinstance(s, (int, float))]
            best = min((r for r in rows if isinstance(r[1], (int, float))),
                       key=lambda r: r[1], default=None)
            print(f"  {k}: {len(rows)} completed trial(s) at {knob}={V}")
            if shareds:
                lo, hi = min(shareds), max(shareds)
                verdict = ("footprint FELL below the limit"
                           if isinstance(limit, (int, float)) and hi <= limit
                           else "footprint still at/over the limit -- freed for another reason")
                print(f"      shared_bytes {lo}..{hi} B  vs wall's {w.get('max_shared')} B "
                      f"and limit {limit} B  ==> {verdict}")
            if best:
                print(f"      fastest at {knob}={V}: {round(best[1], 4)} ms, "
                      f"shared {best[0]} B, params {json.dumps(best[2], ensure_ascii=False)}")
