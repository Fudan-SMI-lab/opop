"""A3: after an ATTRIBUTED wall reached the rewriter, did the rewrite actually FREE that dimension?

This is the question step 3's third arm exists to answer, and `wall_report.py` cannot answer it: that
section says which knob each wall belongs to and what it costs, but stops at the diagnosis. A3 is
about the verb -- `for_prompt` steers a rewrite, and the claim under test is that the steer works.

WHAT COUNTS AS FREED, and why it needs three outcomes rather than two. For an attributed wall on
candidate C, knob K, refused value V:

  FREED          a later candidate in C's family has K = V in a trial that COMPLETED. The compiler
                 accepted what it refused before, so the rewrite bought the range back.
  STILL WALLED   K = V is still refused (`infeasible_shared_memory`) in the child.
  DIMENSION GONE the child's space has no K at all. This is NOT "not freed": the rewriter restructured
                 the kernel and the knob ceased to exist. Reported separately because collapsing it
                 into either verdict would be the mistake -- counted as a failure it slanders a
                 rewrite that may have removed the constraint by removing the tile loop; counted as a
                 success it credits one for deleting the evidence.
  NOT TRIED      K = V is in the child's space but no trial ever sampled it. TPE is not a uniform
                 sampler, so absence is not refusal -- this must not be read as either verdict.

`shared_bytes` at the child's optimum is reported alongside, since freeing the wall is supposed to
work BY lowering it: a child that reports K=V feasible while using MORE shared memory means the wall
was never the binding constraint and the attribution should be re-examined.

READ-ONLY, and from `events.jsonl` alone -- no GPU, no re-run. Safe to use on a live run; the
answer simply grows as more rewrites land.
"""
import json
import os
import sys
from collections import defaultdict


def load(rd):
    evs = []
    with open(os.path.join(rd, "events.jsonl"), encoding="utf-8") as fh:
        for ln in fh:
            ln = ln.strip()
            if ln:
                try:
                    evs.append(json.loads(ln))
                except Exception:
                    pass
    return evs


rd = sys.argv[1]
evs = load(rd)

# Lineage. `parent_ids` is what makes "a later candidate in the same family" precise: a family can
# contain candidates that are not descendants of the walled one, and those say nothing about A3.
fam_of, parents, children = {}, {}, defaultdict(list)
order = []
for e in evs:
    if e.get("type") != "CANDIDATE_REGISTERED":
        continue
    c = (e.get("payload") or {}).get("candidate") or {}
    cid = c.get("candidate_id")
    if not cid:
        continue
    fam_of[cid] = c.get("family_id")
    parents[cid] = list(c.get("parent_ids") or [])
    order.append(cid)
    for p in parents[cid]:
        children[p].append(cid)

# Every trial, grouped by candidate: what was sampled, what completed, what the compiler refused.
trials = defaultdict(list)
for e in evs:
    if e.get("type") != "TRIAL_DONE":
        continue
    t = (e.get("payload") or {}).get("trial") or {}
    cid = t.get("candidate_id")
    if cid:
        trials[cid].append(t)

# Spaces, to distinguish "the knob is gone" from "the knob exists but was never sampled at V".
space_params = {}
for e in evs:
    if e.get("type") != "SPACE_PUBLISHED":
        continue
    p = e.get("payload") or {}
    sp = p.get("space") or p
    cid = sp.get("candidate_id") or p.get("candidate_id")
    doms = sp.get("domains") or []
    if cid and doms:
        space_params[cid] = {d.get("name"): list(d.get("choices") or []) for d in doms if
                             isinstance(d, dict)}


def descendants(cid):
    out, stack = [], list(children.get(cid, []))
    while stack:
        x = stack.pop(0)
        out.append(x)
        stack.extend(children.get(x, []))
    return out


def best_shared(cid):
    """shared_bytes at the candidate's fastest COMPLETED trial -- its optimum, not its average."""
    best, val = None, None
    for t in trials.get(cid, []):
        if t.get("status") != "complete":
            continue
        lat = t.get("latency_ms") or {}
        med = lat.get("median")
        ms = med if isinstance(med, (int, float)) and med > 0 else lat.get("mean")
        if not isinstance(ms, (int, float)):
            continue
        if best is None or ms < best:
            best, val = ms, (t.get("profile") or {}).get("shared_bytes")
    return best, val


def verdict_for(child, knob, refused_value):
    space = space_params.get(child)
    if space is not None and knob not in space:
        return "DIMENSION GONE", "改写后空间里已无该 knob"
    saw_complete, saw_refused, saw_any = False, False, False
    for t in trials.get(child, []):
        params = (t.get("params") or {}).get("values") or t.get("params") or {}
        if not isinstance(params, dict) or knob not in params:
            continue
        try:
            same = float(params[knob]) == float(refused_value)
        except (TypeError, ValueError):
            same = params[knob] == refused_value
        if not same:
            continue
        saw_any = True
        if t.get("status") == "complete":
            saw_complete = True
        elif t.get("failure_kind") == "infeasible_shared_memory":
            saw_refused = True
    if saw_complete:
        return "FREED", f"{knob}={refused_value} 有 trial 跑通"
    if saw_refused:
        return "STILL WALLED", f"{knob}={refused_value} 仍被编译器拒绝"
    if saw_any:
        return "TRIED BUT FAILED OTHERWISE", "取到该值但因别的原因失败"
    if space is not None and refused_value not in [
            v for v in space.get(knob, [])] and str(refused_value) not in [
            str(v) for v in space.get(knob, [])]:
        return "VALUE NOT IN SPACE", f"{knob} 还在,但 {refused_value} 不在其 choices 里"
    return "NOT TRIED", "空间里有该值但 TPE 从未采样(非均匀采样,缺席不等于被拒)"


attributed = []
for e in evs:
    if e.get("type") != "RESOURCE_WALL_ATTRIBUTED":
        continue
    p = e.get("payload") or {}
    for w in (p.get("walls") or []):
        if w.get("verdict") == "attributed":
            attributed.append((p.get("candidate_id"), w))

print(f"=== A3 :: {os.path.basename(rd)}")
print(f"ATTRIBUTED walls: {len(attributed)}")
if not attributed:
    print("  no attributed wall -- A3 has no input from this run (that is a finding, not a defect:")
    print("  measured base rate is 3 probe-worthy walls per 24 candidates)")
    sys.exit(0)

for cid, w in attributed:
    knob, refused = w.get("param"), w.get("refused_value")
    base_ms, base_shared = best_shared(cid)
    print()
    print(f"--- {cid}  knob={knob}  refused={refused}  "
          f"shared_at_wall={w.get('max_shared')}/{w.get('limit')} "
          f"over={w.get('over_ratio')}  tail={w.get('tail_gain_pct')}%")
    print(f"    family {fam_of.get(cid)}   walled candidate's optimum: "
          f"{base_ms if base_ms is None else round(base_ms, 4)} ms, shared_bytes {base_shared}")
    kids = descendants(cid)
    if not kids:
        print("    NO DESCENDANT YET -- the rewrite this wall was meant to steer has not been")
        print("    registered, so A3 is not yet answerable for this wall (run still in flight?)")
        continue
    for k in kids:
        v, why = verdict_for(k, knob, refused)
        ms, shared = best_shared(k)
        delta = ""
        if isinstance(shared, (int, float)) and isinstance(base_shared, (int, float)):
            delta = f" ({shared - base_shared:+d} B vs parent)"
        print(f"    {k}  {v:26s} {why}")
        print(f"        optimum {ms if ms is None else round(ms, 4)} ms, "
              f"shared_bytes {shared}{delta}, trials {len(trials.get(k, []))}")
