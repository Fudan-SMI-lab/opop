"""Did the rewrite deliver what its hypothesis promised?

WHY THIS IS THE MEASUREMENT THAT MATTERS FOR LOOP C. A rewrite arrives with an explicit, quantified
claim -- H1 here promised "frees ~40% of operand registers -> occupancy 17%->~30%" and "expect 5-10%
end-to-end". The framework never checks it. The recorded consequence is
`the-framework-only-collects-confirmations`: rewrites almost never come out worse (12/3/1 and 4/2/1),
which is not survivorship bias since 92-100% reach measurement, so "which resource action works" cannot
be learned from the corpus as it stands. A reconciliation per candidate is the missing half.

WHAT IT COMPARES, and why per CANDIDATE rather than per round. `reconciliation must be per candidate,
not per round` -- a round holds two candidates by design, so a round-level tally makes the declarations
contradict each other. So: parent's best against the child's best, plus the resource dimensions the
hypothesis named (registers, occupancy, spills, shared bytes) read from the trial profiles at each
side's own best point.

WHY IT REFUSES TO SCORE AN UNFINISHED SPACE. A child with 9 of 40 trials has not had its chance: TPE's
winners arrive late by construction (4/4 champions in the back half, normalised 0.78 against a uniform
0.50). Scoring it now would report "the rewrite failed" for a candidate still climbing, so an open space
prints progress and explicitly declines a verdict.

Run on box4:
  PYTHONPATH=/root/autodl-tmp/work/opop/src \
  /root/autodl-tmp/orch-venv/bin/python /root/probe-clean/rewrite_vs_promise.py
"""

from __future__ import annotations

import json
from pathlib import Path

BASE = Path("/root/autodl-tmp/opop-workspace/opop-glm/runs-v3")
RUN = "run-l3-43-20260913-202332"

# The dimensions a rewrite hypothesis in this project actually names. Read from the trial's nested
# profile; `occupancy` is nested again and its limiter key is `limiter` (a flat read fakes "unmeasured").
_DIMS = ("n_regs", "n_spills", "shared_bytes", "num_warps")


def _robust_ms(lat: dict | None) -> float | None:
    if not isinstance(lat, dict):
        return None
    v = lat.get("median")
    if v is None:
        v = lat.get("mean")
    return float(v) if isinstance(v, (int, float)) else None


def _profile_at(trials: list[dict], best_id: str | None) -> dict:
    for tr in trials:
        if str(tr.get("trial_id")) == str(best_id):
            prof = tr.get("profile") or {}
            out = {k: prof.get(k) for k in _DIMS if prof.get(k) is not None}
            occ = prof.get("occupancy")
            if isinstance(occ, dict):
                for k in ("achieved", "value", "occupancy", "pct"):
                    if isinstance(occ.get(k), (int, float)):
                        out["occupancy"] = occ[k]
                        break
                if occ.get("limiter"):
                    out["occ_limiter"] = occ["limiter"]
            elif isinstance(occ, (int, float)):
                out["occupancy"] = occ
            return out
    return {}


def main() -> int:
    for arm in ("s7-treatment", "s7-control"):
        ev = []
        for line in (BASE / arm / RUN / "events.jsonl").open(encoding="utf-8"):
            line = line.strip()
            if line:
                try:
                    ev.append(json.loads(line))
                except ValueError:
                    pass
        print("=" * 78)
        print(arm)

        rewrites = [e for e in ev if e.get("type") == "REWRITE_PRODUCED"]
        if not rewrites:
            print("  no rewrites yet")
            continue

        # family -> candidates, and each candidate's trials
        fam_of: dict[str, str] = {}
        for e in ev:
            if e.get("type") != "CANDIDATE_REGISTERED":
                continue
            p = e.get("payload") or {}
            c = p.get("candidate") or p
            if c.get("candidate_id"):
                fam_of[str(c["candidate_id"])] = str(c.get("family_id") or "?")

        trials_of: dict[str, list[dict]] = {}
        closed: set[str] = set()
        # WHEN each candidate closed, by event seq. The parent must be a candidate that had already
        # closed when this rewrite was PRODUCED -- "the family's best closed candidate" read over
        # the whole run lets a LATER sibling become the parent of an EARLIER one. Live example:
        # cand-c56d8286 (H2, 2.9358 ms) overtook cand-7d02bbab (H1, 3.2620 ms), after which H1 was
        # scored against H2 and printed -11.11%, reading as "this rewrite made things 11% worse".
        # H1's real parent is cand-6f5cdc80 (3.3649 ms) and against it H1 improved. The same pair
        # then appears twice, once with each sign.
        closed_at: dict[str, int] = {}
        for e in ev:
            t = e.get("type")
            if t == "TRIAL_DONE":
                tr = (e.get("payload") or {}).get("trial") or {}
                if tr.get("status") == "complete":
                    trials_of.setdefault(str(tr.get("candidate_id")), []).append(tr)
            elif t == "TUNING_DONE":
                p = e.get("payload") or {}
                if p.get("candidate_id"):
                    cid = str(p["candidate_id"])
                    closed.add(cid)
                    closed_at.setdefault(cid, int(e.get("seq") or 0))

        def _best(cid: str) -> tuple[float | None, str | None, int]:
            rows = [(m, tr) for tr in trials_of.get(cid, [])
                    if (m := _robust_ms(tr.get("latency_ms"))) is not None]
            if not rows:
                return None, None, 0
            m, tr = min(rows, key=lambda r: r[0])
            return m, str(tr.get("trial_id")), len(rows)

        for e in rewrites:
            p = e.get("payload") or {}
            child = str(p.get("candidate_id") or "?")
            fid = str(p.get("family_id") or fam_of.get(child, "?"))
            # The parent is not in the payload, so take the family's best candidate that had ALREADY
            # CLOSED when this rewrite was produced -- and say that is what was done, rather than
            # implying the payload named it. The seq bound is what stops a later sibling becoming an
            # earlier one's parent (see the note where closed_at is built).
            born = int(e.get("seq") or 0)
            sibs = [c for c, f in fam_of.items()
                    if f == fid and c != child and c in closed and closed_at.get(c, 1 << 62) < born]
            parent_best = None
            parent = None
            for c in sibs:
                m, _, _ = _best(c)
                if m is not None and (parent_best is None or m < parent_best):
                    parent_best, parent = m, c
            cb, cbid, n = _best(child)
            print("  --- %s (%s) in family %s" % (child, p.get("hypothesis_id") or "?", fid))
            print("      parent (family's best candidate ALREADY CLOSED at seq %d, NOT named in "
                  "the payload): %s %s"
                  % (born, parent, "%.4f ms" % parent_best if parent_best else "-"))
            if cb is None:
                print("      child has no complete trial yet -- NO VERDICT")
                continue
            if child not in closed:
                print("      child at %d trials, space still OPEN -> best so far %.4f ms (%+.2f%%)"
                      % (n, cb, 100 * (parent_best - cb) / parent_best if parent_best else 0.0))
                print("      NO VERDICT: winners arrive late by construction (4/4 champions in the")
                print("      back half, normalised 0.78 vs a uniform 0.50), so an open space cannot")
                print("      disconfirm a rewrite.")
                continue
            print("      child closed at %d trials: %.4f ms (%+.2f%% vs parent)"
                  % (n, cb, 100 * (parent_best - cb) / parent_best if parent_best else 0.0))
            pp = _profile_at(trials_of.get(parent or "", []), _best(parent or "")[1])
            cp = _profile_at(trials_of.get(child, []), cbid)
            if pp or cp:
                print("      resource dims at each side's OWN best point:")
                print("      (NOT a mechanism reading. Each side sits at its own tile, so these")
                print("       move with the KNOBS too. On cand-7d02bbab this table shows regs")
                print("       210->254 and occupancy halved while the source edit is IDENTITY on")
                print("       all six dims at fixed knobs -- opposite directions. For source")
                print("       attribution use scripts/probes/rewrite_source_vs_knob.py.)")
                for k in sorted(set(pp) | set(cp)):
                    print("        %-14s parent %-12s child %-12s" % (k, pp.get(k, "-"), cp.get(k, "-")))
            else:
                print("      no profile on either best trial -- the promised register/occupancy")
                print("      change cannot be checked from this run's records")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
