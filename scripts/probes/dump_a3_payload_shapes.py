"""Dump the real shape of the payloads a3_did_the_rewrite_free_the_wall.py depends on.

Written before trusting that probe, because guessing field names has already cost two rounds this
session: `status == "ok"` (real value "complete") and `robust_ms` (a @property, never serialized).
Both failed loudly, which was luck -- a wrong guess about `params` or `domains` would instead have
produced a confident "NOT TRIED" for every child, which reads exactly like a real negative A3.
"""
import json
import os
import sys

rd = sys.argv[1]
want = {"CANDIDATE_REGISTERED", "SPACE_PUBLISHED", "TRIAL_DONE"}
seen = {}
with open(os.path.join(rd, "events.jsonl"), encoding="utf-8") as fh:
    for ln in fh:
        ln = ln.strip()
        if not ln:
            continue
        try:
            e = json.loads(ln)
        except Exception:
            continue
        t = e.get("type")
        if t in want and t not in seen:
            seen[t] = e.get("payload") or {}
        if len(seen) == len(want):
            break

for t, p in seen.items():
    print(f"=== {t}  payload keys: {sorted(p.keys())}")
    if t == "CANDIDATE_REGISTERED":
        c = p.get("candidate") or {}
        print("  candidate keys:", sorted(c.keys()))
        print("  parent_ids:", repr(c.get("parent_ids")), " family_id:", repr(c.get("family_id")))
    elif t == "SPACE_PUBLISHED":
        sp = p.get("space") or p
        print("  space keys:", sorted(sp.keys()))
        doms = sp.get("domains") or []
        print("  n_domains:", len(doms))
        if doms:
            print("  domain[0]:", json.dumps(doms[0], ensure_ascii=False)[:400])
    elif t == "TRIAL_DONE":
        tr = p.get("trial") or {}
        print("  params type:", type(tr.get("params")).__name__)
        print("  params:", json.dumps(tr.get("params"), ensure_ascii=False)[:400])
        print("  failure_kind:", repr(tr.get("failure_kind")))

# The refusal kind the probe filters on must actually appear, or every verdict silently becomes
# "NOT TRIED". Count it across the whole log.
kinds = {}
with open(os.path.join(rd, "events.jsonl"), encoding="utf-8") as fh:
    for ln in fh:
        if "TRIAL_DONE" not in ln:
            continue
        try:
            tr = (json.loads(ln).get("payload") or {}).get("trial") or {}
        except Exception:
            continue
        if tr.get("status") != "complete":
            kinds[tr.get("failure_kind")] = kinds.get(tr.get("failure_kind"), 0) + 1
print()
print("failure_kind values across the run:", kinds)
