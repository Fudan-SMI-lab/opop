"""Are the arms in the same PHASE, and if not, what does that do to the comparison?

WHY NOW. The control arm has finished tuning every candidate, decided convergence on two families, and
started a rewriter call -- loop C. The treatment arm is still tuning. From here the two arms are doing
DIFFERENT KINDS OF WORK, and three things follow that a trial count cannot express:

  * loop C trials are not loop B trials. A rewrite produces a new candidate whose space gets its own
    `trials_per_space`, so the totals keep diverging for a reason that has nothing to do with sampling.
  * P3 in `analyze_s7_pair.py` is family coverage of wall TEXT, which only exists once rewrites happen.
    An arm that never reached loop C scores 0 on P3 by construction, not by outcome -- so if the arms
    end in different phases, P3 compares "did not help" against "never ran".
  * `wall-clock is always the binding budget`: 8 of 8 finished runs ended on the clock, and loop C got
    only 7-8% of it. The arm that enters loop C first gets more of that share.

So this prints, per arm: which phase it is in, what convergence decided and why, which family a rewrite
targets, and whether the family that got the rewrite is the one holding that arm's best -- because a
rewrite aimed at a non-leading family cannot move the arm's headline number.

Run on box4:
  PYTHONPATH=/root/autodl-tmp/work/opop/src \
  /root/autodl-tmp/orch-venv/bin/python /root/probe-clean/phase_divergence.py
"""

from __future__ import annotations

import json
from pathlib import Path

BASE = Path("/root/autodl-tmp/opop-workspace/opop-glm/runs-v3")
RUN = "run-l3-43-20260913-202332"


def _robust_ms(lat: dict | None) -> float | None:
    if not isinstance(lat, dict):
        return None
    v = lat.get("median")
    if v is None:
        v = lat.get("mean")
    return float(v) if isinstance(v, (int, float)) else None


def main() -> int:
    for arm in ("s7-control", "s7-treatment"):
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

        counts = {}
        for e in ev:
            counts[e.get("type")] = counts.get(e.get("type"), 0) + 1
        phase_markers = ("FAMILY_SEEDED", "CONVERGENCE_DECIDED", "REWRITE_PRODUCED",
                         "FAMILY_ROUND_RECORDED", "NOVELTY_PRODUCED", "RUN_FINISHED")
        print("  phase markers: %s" % {k: counts.get(k, 0) for k in phase_markers})

        # What did convergence actually decide? The payload is the authority on verdict/stop_kind.
        for e in ev:
            if e.get("type") != "CONVERGENCE_DECIDED":
                continue
            p = e.get("payload") or {}
            dec = p.get("decision") or p
            print("  CONVERGENCE_DECIDED %s" % json.dumps(
                {k: dec[k] for k in sorted(dec) if k in
                 ("family_id", "verdict", "stop_kind", "rounds_used", "best_ms", "reason",
                  "improvement_pct", "scope")}, ensure_ascii=False)[:260])
            if not any(k in dec for k in ("verdict", "stop_kind")):
                print("    payload keys: %s" % ", ".join(sorted(p.keys())))

        # Which family holds this arm's best, and which family is being rewritten?
        fam_of_cand: dict[str, str] = {}
        for e in ev:
            if e.get("type") != "CANDIDATE_REGISTERED":
                continue
            p = e.get("payload") or {}
            c = p.get("candidate") or p
            cid, fid = c.get("candidate_id"), c.get("family_id")
            if cid:
                fam_of_cand[str(cid)] = str(fid or "?")

        best: dict[str, float] = {}
        for e in ev:
            if e.get("type") != "TRIAL_DONE":
                continue
            tr = (e.get("payload") or {}).get("trial") or {}
            if tr.get("status") != "complete":
                continue
            ms = _robust_ms(tr.get("latency_ms"))
            cid = str(tr.get("candidate_id") or "?")
            if ms is None:
                continue
            if cid not in best or ms < best[cid]:
                best[cid] = ms
        by_fam: dict[str, tuple[str, float]] = {}
        for cid, ms in best.items():
            fid = fam_of_cand.get(cid, "?")
            if fid not in by_fam or ms < by_fam[fid][1]:
                by_fam[fid] = (cid, ms)
        print("  best per family:")
        for fid, (cid, ms) in sorted(by_fam.items(), key=lambda kv: kv[1][1]):
            print("    %-16s %-16s %.4f ms" % (fid, cid, ms))
        leader = min(by_fam.items(), key=lambda kv: kv[1][1])[0] if by_fam else None

        # An in-flight rewriter call: which family is it for? The STARTED payload may not name it, in
        # which case say so rather than guessing -- a rewrite aimed at a non-leading family cannot move
        # the arm's headline number, and that is the whole reason to ask.
        for e in ev:
            if e.get("type") != "AGENT_CALL_STARTED":
                continue
            p = e.get("payload") or {}
            if p.get("module") != "rewriter":
                continue
            fid = p.get("family_id") or p.get("candidate_id")
            print("  rewriter call %s -> %s" % (p.get("call_id"), fid or "(family not in payload)"))
            if fid and leader:
                print("    targets the %s family (arm leader is %s)"
                      % ("LEADING" if str(fid) == leader else "NON-LEADING", leader))

    print()
    print("=" * 78)
    print("WHY A PHASE DIFFERENCE IS NOT A TRIAL-COUNT DIFFERENCE")
    print("  * A rewrite creates a NEW candidate with its OWN trials_per_space, so once one arm is in")
    print("    loop C the totals diverge for reasons unrelated to sampling. Do not price the gap in")
    print("    space-equivalents across a phase boundary without saying which phase each arm was in.")
    print("  * P3 (family coverage of wall text) is 0 BY CONSTRUCTION for an arm that never reached")
    print("    loop C. If the arms end in different phases, P3 compares 'did not help' against")
    print("    'never ran' -- the same conflation the instrument check exists to prevent.")
    print("  * 8 of 8 finished runs ended on the wall clock, with loop C taking 7-8% of it. The arm")
    print("    that enters loop C first gets more of that share, which is a head start, not an effect.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
