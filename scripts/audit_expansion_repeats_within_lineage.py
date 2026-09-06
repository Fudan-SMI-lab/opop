"""Does improvement K re-buy the same widened range it already tested on a sibling?

`run-l3-43-20260906-091019` expanded `cand-1129b4d9` on `BLOCK_N/max`, adding `BLOCK_N=256`,
and measured it: 17.9 ms against the incumbent's 16.8 -- worse, 2 samples, not used by the
winner. One rewrite round later it expanded that candidate's CHILD, `cand-3df5fd86`, on the same
knob in the same direction, adding the same 256 to a structurally similar kernel whose ladder is
nearly identical:

    cand-1129b4d9  BLOCK_N  16:52.00  32:30.80  64:21.30  128:16.80   -> added 256 = 17.90
    cand-3df5fd86  BLOCK_N  16:52.60  32:31.50  64:20.70  128:17.00   -> adding 256 again

Nothing in the space-expansion path consults what a related candidate already learned about the
same knob, so the second expansion pays a parameterizer call plus a fresh 40-trial budget to
re-discover a value the run measured 20 minutes earlier.

This measures how often that happens across every run on disk, because "K repeats itself" is only
worth acting on if it is a pattern rather than one coincidence. Counted two ways:

  * LINEAGE repeats -- the same (knob, direction) expanded on an ancestor/descendant pair, where
    the kernels are genuinely related and the earlier measurement is most likely to transfer;
  * FAMILY repeats -- the same (knob, direction) expanded twice anywhere in one family.

For each repeat it reports whether the earlier expansion's added values turned out to HELP, since
a repeat that re-tests a value already shown to be worse is the wasteful case, while a repeat
after a helpful expansion is arguably rational.

Read-only over events.jsonl; safe against a live run.
"""
from __future__ import annotations

import glob
import json
import sys
from collections import defaultdict
from pathlib import Path

runs = sorted(glob.glob(sys.argv[1] if len(sys.argv) > 1 else "runs/run-l3-*"))

total_expansions = 0
lineage_repeats: list[dict] = []
family_repeats: list[dict] = []

for run in runs:
    ev = Path(run) / "events.jsonl"
    if not ev.exists():
        continue
    evs = [json.loads(l) for l in open(ev, encoding="utf-8")]

    cand_family: dict[str, str] = {}
    parents: dict[str, list[str]] = {}
    for e in evs:
        if e["type"] == "CANDIDATE_REGISTERED":
            c = e["payload"]["candidate"]
            cand_family[c["candidate_id"]] = c["family_id"]
            parents[c["candidate_id"]] = c.get("parent_ids") or []

    def ancestors(cid: str) -> set[str]:
        out, stack = set(), list(parents.get(cid, []))
        while stack:
            p = stack.pop()
            if p in out:
                continue
            out.add(p)
            stack.extend(parents.get(p, []))
        return out

    # space -> candidate, and the per-value best latency per space, so an added value's
    # usefulness can be scored.
    space_cand: dict[str, str] = {}
    space_choices: dict[str, dict[str, list]] = {}
    for e in evs:
        if e["type"] == "SPACE_PUBLISHED":
            s = e["payload"]["space"]
            space_cand[s["space_id"]] = s["candidate_id"]
            space_choices[s["space_id"]] = {d["name"]: d["choices"] for d in s["domains"]}

    per_value_best: dict[tuple[str, str], dict] = defaultdict(dict)
    space_best: dict[str, float] = {}
    for e in evs:
        if e["type"] != "TRIAL_DONE":
            continue
        t = e["payload"]["trial"]
        if t["status"] != "complete":
            continue
        sid, ms = t["space_id"], t["latency_ms"]["mean"]
        space_best[sid] = min(space_best.get(sid, 1e9), ms)
        for k, v in t["params"]["values"].items():
            d = per_value_best[(sid, k)]
            key = repr(v)
            if key not in d or ms < d[key]:
                d[key] = ms

    # expansions in order, each carrying the candidate and the knobs requested
    seen: list[dict] = []
    for e in evs:
        if e["type"] != "SPACE_EXPANDED":
            continue
        total_expansions += 1
        cid = e["payload"].get("candidate_id")
        knobs = [(k["name"], k["direction"]) for k in (e["payload"].get("knobs") or [])]
        # the space published for this candidate right after this event is the expanded one
        for prior in seen:
            shared = set(knobs) & set(prior["knobs"])
            if not shared:
                continue
            same_family = cand_family.get(cid) == cand_family.get(prior["cid"])
            related = prior["cid"] in ancestors(cid) or cid in ancestors(prior["cid"])
            if not same_family:
                continue
            rec = {
                "run": Path(run).name,
                "knobs": sorted(shared),
                "earlier": prior["cid"],
                "later": cid,
                "related": related,
            }
            (lineage_repeats if related else family_repeats).append(rec)
        seen.append({"cid": cid, "knobs": knobs})

print(f"runs scanned: {len(runs)}   expansions: {total_expansions}")
print(f"LINEAGE repeats (ancestor/descendant, same knob+direction): {len(lineage_repeats)}")
for r in lineage_repeats:
    print(f"   {r['run']}  {r['knobs']}  {r['earlier']} -> {r['later']}")
print(f"FAMILY repeats (same family, not directly related):         {len(family_repeats)}")
for r in family_repeats[:12]:
    print(f"   {r['run']}  {r['knobs']}  {r['earlier']} / {r['later']}")
