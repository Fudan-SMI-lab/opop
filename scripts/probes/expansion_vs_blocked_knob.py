"""Did the treatment arm's FIRST K expansion widen the knob that was blocking S7?

WHY THIS IS THE MOMENT. Cause #4 of S7's zero enqueued points is that the soft criterion always points
HIGH and the incumbent already holds the top of the domain -- 11 of 24 soft walls (45.8%) had no value
left to propose. A K expansion is the one event that removes that condition: it re-tunes the SAME
candidate under a WIDER domain, so a knob whose top was the incumbent now has values above it. The
treatment arm just opened its first expanded space (`cand-5e033365` v1 -> v2, sp-8acfe096), which makes
the next ~16 trials the most likely window for a first enqueue in the whole pair.

WHAT WOULD MAKE IT A NON-EVENT. If the expansion widened some OTHER knob than the one S7 kept stalling
on, the condition is untouched and the window is worthless. So this reads three things and refuses to
guess any of them:
  * which knob(s) the expansion actually widened, old choices vs new,
  * which knob each SLOPE_GUIDE_STEP named as the wall it could not act on,
  * whether they intersect.

WHY NOT READ `at_boundary`. That is the trigger for the expansion, not evidence about S7. The question
here is specifically whether the widened knob is the one whose domain top blocked a proposal.

Run on box4:
  PYTHONPATH=/root/autodl-tmp/work/opop/src \
  /root/autodl-tmp/orch-venv/bin/python /root/probe-clean/expansion_vs_blocked_knob.py
"""

from __future__ import annotations

import json
from pathlib import Path

BASE = Path("/root/autodl-tmp/opop-workspace/opop-glm/runs-v3")
RUN = "run-l3-43-20260913-202332"


def _load(arm: str) -> list[dict]:
    out = []
    for line in (BASE / arm / RUN / "events.jsonl").open(encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except ValueError:
            pass
    return out


def _domains(payload: dict) -> dict[str, list]:
    """Knob -> choices, from a SPACE_PUBLISHED payload. The space may be nested one level down, and
    domains may be a list of objects or a dict -- read both rather than assuming, since a wrong shape
    would produce an empty dict and read as 'the expansion widened nothing'."""
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
    for arm in ("s7-treatment", "s7-control"):
        ev = _load(arm)
        print("=" * 78)
        print(arm)

        # ---- every published space, in order, with its domains
        spaces: list[tuple[str, str, dict[str, list]]] = []
        for e in ev:
            if e.get("type") != "SPACE_PUBLISHED":
                continue
            p = e.get("payload") or {}
            sp = p.get("space") or p
            doms = _domains(p)
            if not doms:
                print("  SPACE_PUBLISHED with NO readable domains -- payload keys: %s"
                      % ", ".join(sorted(p.keys())))
            spaces.append((str(sp.get("space_id") or "?"),
                           str(sp.get("candidate_id") or p.get("candidate_id") or "?"), doms))

        # ---- pair up the expansions: same candidate, later space
        by_cand: dict[str, list[tuple[str, dict[str, list]]]] = {}
        for sid, cid, doms in spaces:
            by_cand.setdefault(cid, []).append((sid, doms))

        # NOTE: `widened` is pooled across every expansion in the arm, so a knob widened twice keeps
        # only the last pair. That is fine for the one thing it is used for below -- "was anything
        # widened at all" -- and WRONG for anything per-space. Use `hits` for that.
        widened: dict[str, tuple[list, list]] = {}
        for cid, versions in by_cand.items():
            if len(versions) < 2:
                continue
            for (sid_a, da), (sid_b, db) in zip(versions, versions[1:]):
                print("  EXPANSION %s: %s -> %s" % (cid, sid_a, sid_b))
                for k in sorted(set(da) | set(db)):
                    old, new = da.get(k, []), db.get(k, [])
                    if old != new:
                        added = [v for v in new if v not in old]
                        print("    %-16s %s  ->  %s      added %s" % (k, old, new, added))
                        if added:
                            widened[k] = (old, new)
                if not any(da.get(k) != db.get(k) for k in set(da) | set(db)):
                    print("    (no domain changed -- the expansion re-published an identical space)")

        # ---- which knob did each slope-guide step name as the wall it could not act on?
        blocked: dict[str, int] = {}
        named_any = False
        for e in ev:
            if e.get("type") != "SLOPE_GUIDE_STEP":
                continue
            p = e.get("payload") or {}
            # The payload carries per-wall detail under a few possible spellings; print the keys once
            # so a rename shows up as a rename rather than as "no walls were blocked".
            if not named_any:
                print("  SLOPE_GUIDE_STEP payload keys: %s" % ", ".join(sorted(p.keys())))
                named_any = True
            for key in ("walls", "considered", "wall_details", "skipped"):
                items = p.get(key)
                if not isinstance(items, list):
                    continue
                for it in items:
                    if isinstance(it, dict) and it.get("param"):
                        blocked[str(it["param"])] = blocked.get(str(it["param"]), 0) + 1
        if blocked:
            print("  knobs named by slope-guide steps: %s" % blocked)
        else:
            print("  NO knob named in any SLOPE_GUIDE_STEP payload. Either the steps carry only counters")
            print("  (see the fields list above) or none reached a named wall. The soft-wall event is")
            print("  the other place a param appears -- checked next.")

        # RESOURCE_SOFT_WALL names the param the soft criterion pointed at -- KEYED BY SPACE.
        #
        # A first version pooled these across the whole arm and printed `{'BLOCK_M': 1}`, which I then
        # matched against an expansion of BLOCK_M in a DIFFERENT space and reported as "the expansion
        # widened the knob S7 named". It had not: per space the intersection is empty. Pooling across
        # spaces manufactures agreement -- the same defect that once had s7_no_wall_why.py scoring
        # candidate 2's trials against candidate 1's space.
        soft_by_space: dict[str, set[str]] = {}
        for e in ev:
            if e.get("type") != "RESOURCE_SOFT_WALL":
                continue
            p = e.get("payload") or {}
            sid = str(p.get("space_id") or "?")
            for it in (p.get("walls") or []):
                if isinstance(it, dict) and it.get("param"):
                    soft_by_space.setdefault(sid, set()).add(str(it["param"]))
        print("  knobs named by RESOURCE_SOFT_WALL, per space: %s"
              % ({k: sorted(v) for k, v in soft_by_space.items()} or "(none)"))

        # The comparison must be within ONE space: a wall found in sp-A says nothing about whether an
        # expansion of sp-B helped. `widened` is keyed by knob but recorded per expansion, so re-walk
        # the expansions and check each against its OWN pair of space ids.
        hits: list[tuple[str, str, str, list, list]] = []
        for cid, versions in by_cand.items():
            for (sid_a, da), (sid_b, db) in zip(versions, versions[1:]):
                named_here = soft_by_space.get(sid_a, set()) | soft_by_space.get(sid_b, set())
                for k in sorted(set(da) | set(db)):
                    old, new = da.get(k, []), db.get(k, [])
                    added = [v for v in new if v not in old]
                    if added and k in named_here:
                        hits.append((cid, sid_a, k, old, new))

        print()
        if not widened:
            print("  NO EXPANSION WITH A WIDENED DOMAIN YET -- the question is not yet answerable.")
        elif hits:
            print("  AN EXPANSION WIDENED A KNOB THE SOFT WALL NAMED IN THE SAME SPACE:")
            for cid, sid, k, old, new in hits:
                print("    %s %s  %s: %s -> %s" % (cid, sid, k, old, new))
            print("  => the next trials in that space are the pair's best chance of a first enqueue.")
        else:
            print("  THE EXPANSION(S) WIDENED %s. In the SAME space, the soft wall named %s."
                  % (", ".join(sorted(widened)),
                     ", ".join(sorted(set().union(*soft_by_space.values()))) if soft_by_space else "nothing"))
            print("  NO OVERLAP WITHIN A SPACE => cause #4 is untouched, so the coming trials are NOT")
            print("  a test of S7. (Pooling the soft walls across the arm would have shown a spurious")
            print("  match here -- see the comment above.)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
