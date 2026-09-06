"""When improvement K repeats a knob within a lineage, was the first attempt already a dud?

`audit_expansion_repeats_within_lineage.py` counts the repeats: 16 lineage repeats and 11
same-family repeats out of 57 expansions on disk. A repeat is only waste if the earlier
expansion had already shown that direction does not pay. This scores that.

For each repeated (knob, direction) it asks of the EARLIER expansion:

  * did the widened range add values the tuner actually reached?
  * were any of them better than that space's incumbent best?
  * did the earlier expansion's winner use an added value at all?

If the answer is "reached, and worse, and unused", the later expansion is re-buying a
measurement the run already had -- a parameterizer call plus a full `trials_per_space` budget.

Read-only over events.jsonl; safe against a live run.
"""
from __future__ import annotations

import glob
import json
import sys
from collections import defaultdict
from pathlib import Path

runs = sorted(glob.glob(sys.argv[1] if len(sys.argv) > 1 else "runs/run-l3-*"))

verdicts: dict[str, int] = defaultdict(int)
detail: list[str] = []

for run in runs:
    ev = Path(run) / "events.jsonl"
    if not ev.exists():
        continue
    evs = [json.loads(l) for l in open(ev, encoding="utf-8")]

    cand_family, parents = {}, {}
    for e in evs:
        if e["type"] == "CANDIDATE_REGISTERED":
            c = e["payload"]["candidate"]
            cand_family[c["candidate_id"]] = c["family_id"]
            parents[c["candidate_id"]] = c.get("parent_ids") or []

    def ancestors(cid):
        out, stack = set(), list(parents.get(cid, []))
        while stack:
            p = stack.pop()
            if p not in out:
                out.add(p)
                stack.extend(parents.get(p, []))
        return out

    # spaces in publish order per candidate, so "the space before/after an expansion" is known
    spaces_of: dict[str, list[str]] = defaultdict(list)
    choices_of: dict[str, dict] = {}
    for e in evs:
        if e["type"] == "SPACE_PUBLISHED":
            s = e["payload"]["space"]
            spaces_of[s["candidate_id"]].append(s["space_id"])
            choices_of[s["space_id"]] = {d["name"]: d["choices"] for d in s["domains"]}

    per_value: dict[tuple[str, str], dict[str, float]] = defaultdict(dict)
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
            d = per_value[(sid, k)]
            if repr(v) not in d or ms < d[repr(v)]:
                d[repr(v)] = ms

    seen = []
    for e in evs:
        if e["type"] != "SPACE_EXPANDED":
            continue
        cid = e["payload"].get("candidate_id")
        knobs = [(k["name"], k["direction"]) for k in (e["payload"].get("knobs") or [])]
        for prior in seen:
            shared = set(knobs) & set(prior["knobs"])
            if not shared or cand_family.get(cid) != cand_family.get(prior["cid"]):
                continue
            related = prior["cid"] in ancestors(cid) or cid in ancestors(prior["cid"])
            # score the EARLIER expansion: its pre space is the one before its post space
            sp = spaces_of.get(prior["cid"]) or []
            if len(sp) < 2:
                continue
            pre, post = sp[-2], sp[-1]
            for name, _direction in sorted(shared):
                pre_ch = set(map(repr, choices_of.get(pre, {}).get(name, [])))
                post_ch = list(map(repr, choices_of.get(post, {}).get(name, [])))
                added = [c for c in post_ch if c not in pre_ch]
                if not added:
                    continue
                incumbent = space_best.get(pre)
                reached = {a: per_value[(post, name)].get(a) for a in added}
                got = {a: v for a, v in reached.items() if v is not None}
                if not got:
                    verdicts["added value never sampled"] += 1
                    kind = "NEVER SAMPLED"
                elif incumbent is not None and min(got.values()) < incumbent:
                    verdicts["earlier expansion's added value BEAT the incumbent"] += 1
                    kind = "HELPED"
                else:
                    verdicts["earlier added value reached and WORSE -> repeat re-buys it"] += 1
                    kind = "DUD"
                tag = "lineage" if related else "family"
                best_added = min(got.values()) if got else None
                detail.append(
                    f"  [{kind:13s}] {tag:7s} {Path(run).name} {name}: "
                    f"incumbent {incumbent:.1f} vs added best "
                    f"{best_added if best_added is None else round(best_added,1)}  "
                    f"({prior['cid']} -> {cid})"
                )
        seen.append({"cid": cid, "knobs": knobs})

print("Scoring the EARLIER expansion of each repeated (knob, direction):")
for k, v in sorted(verdicts.items(), key=lambda x: -x[1]):
    print(f"  {v:4d}  {k}")
print()
for line in detail:
    print(line)
