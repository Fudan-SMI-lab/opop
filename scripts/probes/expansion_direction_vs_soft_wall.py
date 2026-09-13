"""Does a K expansion widen the side, and the knob, that a wall actually points at?

WHY IT MATTERS. Two recorded findings collide here:
  * `a-k-expansion-is-s7-best-opportunity...` -- an expansion re-tunes the SAME candidate under a wider
    domain, so a wall survives the space closing and S7 gets a second look.
  * `the-soft-wall-always-points-high-so-half-are-unactionable` -- the soft criterion only ever proposes
    a HIGHER value, and 45.8% of soft walls had nothing left because the incumbent held the domain top.

The first is only an opportunity if the expansion adds values on the side the wall points at, for the
SAME knob, in the SAME space. The expansion's direction is not chosen with S7 in mind: it comes from
`boundary_direction`, which points wherever the best-latency argmin sat.

HOW THE SPACE TIE WORKS, and two ways I got it wrong before this version.
`RESOURCE_WALL_ATTRIBUTED` carries `space_id`. `RESOURCE_SOFT_WALL` carries only `candidate_id` -- no
space at all -- and is emitted immediately after its candidate's attribution event (seq 83 -> 84). So:
  * a first probe POOLED soft walls across the whole arm, printed `{'BLOCK_M': 1}`, and I matched it
    against a BLOCK_M expansion in a different space => a manufactured agreement;
  * a second grouped soft walls by a `space_id` the payload does not have, so every one fell into a "?"
    bucket and the intersection was empty 48/48 => a manufactured disagreement.
Both are the same fault: reading a key that is not there. This version takes the space from the
attribution event and carries it to the soft wall that follows it.

AND IT MUST LOOK AT HARD WALLS TOO. Restricting the comparison to soft walls is what produced the 0/48.
The control arm's `sp-5b5dd667` has a HARD wall naming NUM_WARPS on the high side, and that candidate's
expansion added NUM_WARPS=32 -- exactly the overlap the soft-only version reported as absent.

Run on box4:
  PYTHONPATH=/root/autodl-tmp/work/opop/src \
  /root/autodl-tmp/orch-venv/bin/python /root/probe-clean/expansion_direction_vs_soft_wall.py
"""

from __future__ import annotations

import json
from pathlib import Path

BASE = Path("/root/autodl-tmp/opop-workspace/opop-glm/runs-v3")


def _load(path: Path) -> list[dict]:
    out = []
    for line in path.open(encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except ValueError:
            pass
    return out


def _domains(payload: dict) -> dict[str, list]:
    sp = payload.get("space") or payload
    doms = sp.get("domains")
    out: dict[str, list] = {}
    if isinstance(doms, dict):
        for k, v in doms.items():
            out[str(k)] = list(v.get("choices") or []) if isinstance(v, dict) else list(v or [])
    elif isinstance(doms, list):
        for d in doms:
            if isinstance(d, dict) and d.get("name"):
                out[str(d["name"])] = list(d.get("choices") or [])
    return out


def main() -> int:
    rows: list[tuple[str, str, str, str, str, str, bool]] = []
    fields_shown = False

    for run in sorted(BASE.glob("*/run-l3-43-*")):
        ev = _load(run / "events.jsonl")
        label = f"{run.parent.name}"[-16:]

        # boundary_direction per (space, param), from STATS_DONE
        direction: dict[tuple[str, str], str] = {}
        for e in ev:
            if e.get("type") != "STATS_DONE":
                continue
            p = e.get("payload") or {}
            st = p.get("stats") or p
            sid = str(st.get("space_id") or p.get("space_id") or "?")
            ps = st.get("param_stats")
            if ps is None and not fields_shown:
                fields_shown = True
                print("STATS_DONE has no `param_stats`; payload keys: %s / stats keys: %s"
                      % (", ".join(sorted(p.keys())), ", ".join(sorted(st.keys()))))
            for s in (ps or []):
                if isinstance(s, dict) and s.get("name"):
                    direction[(sid, str(s["name"]))] = str(s.get("boundary_direction") or "-")

        # spaces per candidate, in publication order
        by_cand: dict[str, list[tuple[str, dict[str, list]]]] = {}
        for e in ev:
            if e.get("type") != "SPACE_PUBLISHED":
                continue
            p = e.get("payload") or {}
            sp = p.get("space") or p
            cid = str(sp.get("candidate_id") or p.get("candidate_id") or "?")
            by_cand.setdefault(cid, []).append((str(sp.get("space_id") or "?"), _domains(p)))

        # knobs a wall named, per SPACE. Hard walls carry space_id themselves; soft walls carry only
        # candidate_id and follow their candidate's attribution event, so the space comes from there.
        # `kind` and the wall's own `side`/`verdict` are kept: a wall that passed the slope filter can
        # still have failed attribution, and that distinction is the whole point below.
        walled: dict[str, list[dict]] = {}
        last_space_for_cand: dict[str, str] = {}
        for e in ev:
            t = e.get("type")
            p = e.get("payload") or {}
            if t == "RESOURCE_WALL_ATTRIBUTED":
                sid = str(p.get("space_id") or "?")
                cid = str(p.get("candidate_id") or "?")
                last_space_for_cand[cid] = sid
                for it in (p.get("walls") or []):
                    if isinstance(it, dict) and it.get("param"):
                        walled.setdefault(sid, []).append({
                            "kind": "hard", "param": str(it["param"]),
                            "side": str(it.get("side") or "?"),
                            "gain": it.get("tail_gain_pct"),
                            "monotone": it.get("monotone"),
                            "verdict": str(it.get("verdict") or "?")})
            elif t == "RESOURCE_SOFT_WALL":
                cid = str(p.get("candidate_id") or "?")
                sid = last_space_for_cand.get(cid, "?")
                for it in (p.get("walls") or []):
                    if isinstance(it, dict) and it.get("param"):
                        walled.setdefault(sid, []).append({
                            "kind": "soft", "param": str(it["param"]),
                            "side": "high",  # the soft criterion only ever proposes upward
                            "gain": it.get("tail_gain_pct"),
                            "monotone": it.get("monotone"),
                            "verdict": "soft"})

        for cid, versions in by_cand.items():
            for (sid_a, da), (sid_b, db) in zip(versions, versions[1:]):
                here = walled.get(sid_a, []) + walled.get(sid_b, [])
                for k in sorted(set(da) | set(db)):
                    old, new = da.get(k, []), db.get(k, [])
                    added = [v for v in new if v not in old]
                    if not added:
                        continue
                    try:
                        lo, hi = min(old), max(old)
                        side = ("high" if all(v > hi for v in added)
                                else "low" if all(v < lo for v in added) else "both/mid")
                    except (TypeError, ValueError):
                        side = "?"
                    match = [w for w in here if w["param"] == k]
                    rows.append((label, cid[:14], k, direction.get((sid_a, k), "-"),
                                 side, "%s -> %s" % (old, new), match))

    print("%-16s %-14s %-18s %-9s %-9s %s" % (
        "run", "candidate", "knob", "bnd_dir", "added_on", "wall in the same space / domain change"))
    for label, cid, k, d, side, change, match in rows:
        tag = "-"
        if match:
            tag = "; ".join(
                "%s %s side=%s gain=%s mono=%s verdict=%s"
                % (w["kind"], w["param"], w["side"],
                   "%.2f%%" % w["gain"] if isinstance(w["gain"], (int, float)) else "?",
                   w["monotone"], w["verdict"])
                for w in match)
        print("%-16s %-14s %-18s %-9s %-9s %-6s %s" % (label, cid, k, d, side, tag[:44], change))

    hits = [r for r in rows if r[6]]
    print()
    print("expansions whose widened knob is named by a wall IN THE SAME SPACE: %d of %d"
          % (len(hits), len(rows)))
    if not hits:
        print("None. An expansion is an opportunity for S7 only when it widens the knob a wall named,")
        print("and across this corpus that never coincided. Note this is about COINCIDENCE, not")
        print("direction: %d of %d expansions added values on the HIGH side, which is the side the soft"
              % (sum(1 for r in rows if r[4] == "high"), len(rows)))
        print("criterion proposes -- so the directions agree and the knobs simply do not.")
        return 0

    print()
    for label, cid, k, d, side, change, match in hits:
        for w in match:
            same_side = (w["side"] == side) or side == "both/mid"
            print("  %s %s %s: expansion added on %s, wall points %s => %s"
                  % (label, cid, k, side, w["side"], "ALIGNED" if same_side else "OPPOSITE"))
            print("      %s" % change)
            # A wall can pass the slope filter and still be useless: `verdict != attributed` means the
            # refusal was not traced to a resource limit at all, so there is nothing for a rewrite to
            # free. This is the distinction that made the control arm's `wallsfound=1` produce nothing.
            if w["kind"] == "hard" and w["verdict"] != "attributed":
                print("      BUT verdict=%s -- the wall passed monotone/gain yet was NOT attributed to a"
                      % w["verdict"])
                print("      resource limit, so it names no limit to free. A wall count of 1 here is not")
                print("      one usable wall.")
            if isinstance(w["gain"], (int, float)) and w["gain"] < 2.35:
                print("      AND gain %.2f%% is below the 2.35%% re-eval noise floor => not worth a point"
                      % w["gain"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
