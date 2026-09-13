"""Per-space accounting for the two S7 arms: is the budget being applied the same way?

WHY. At the same wall clock the control arm had closed 2 spaces on ONE candidate at 80 trials while the
treatment arm was on its SECOND candidate. Two spaces on one candidate means a SPACE_EXPANDED (a K
expansion re-tunes the same candidate under a wider space); two candidates means the run moved on. Those
are different shapes of progress, and if the arms are spending their budget differently the pair's
comparison is confounded -- `check_arm_search_parity.py` answers the aggregate version of this question,
but it cannot say WHICH structural event produced the difference.

`budget is per space, not per candidate` is the recorded rule: 80 trials against `trials_per_space: 40`
is 2 spaces, not an overspend. This probe makes that visible per space rather than leaving the reader to
divide.

Run on box4:  /root/autodl-tmp/orch-venv/bin/python /root/probe-clean/space_accounting.py
"""

from __future__ import annotations

import json
from pathlib import Path

BASE = Path("/root/autodl-tmp/opop-workspace/opop-glm/runs-v3")


def report(arm: str) -> None:
    runs = sorted(BASE.glob(f"{arm}/run-l3-43-*"))
    if not runs:
        print(f"=== {arm}: no run")
        return
    run = runs[-1]
    spaces: dict[str, dict] = {}
    order: list[str] = []
    counts: dict[str, int] = {}
    for line in (run / "events.jsonl").open(encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        try:
            e = json.loads(line)
        except ValueError:
            continue
        t = str(e.get("type") or "")
        counts[t] = counts.get(t, 0) + 1
        p = e.get("payload") or {}
        if t == "SPACE_PUBLISHED":
            s = p.get("space") or {}
            key = str(s.get("space_id"))
            spaces[key] = {"cand": s.get("candidate_id"), "ver": s.get("version"),
                           "knobs": len(s.get("domains") or []), "trials": 0, "closed": False,
                           "best": None}
            order.append(key)
        elif t == "TRIAL_DONE":
            tr = p.get("trial") or {}
            key = str(tr.get("space_id"))
            if key in spaces:
                spaces[key]["trials"] += 1
        elif t == "TUNING_DONE":
            key = str(p.get("space_id"))
            if key in spaces:
                spaces[key]["closed"] = True
                spaces[key]["best"] = p.get("best_ms")

    print(f"=== {arm}  {run.name}")
    for key in order:
        d = spaces[key]
        best = f"{d['best']:.4f} ms" if isinstance(d["best"], (int, float)) else "-"
        print("    %-14s cand=%-16s v%-2s knobs=%-3d trials=%-4d closed=%-5s best=%s" % (
            key, str(d["cand"]), str(d["ver"]), d["knobs"], d["trials"],
            str(d["closed"]), best))
    cands = {d["cand"] for d in spaces.values()}
    print("    spaces %d over %d candidate(s)   closed %d" % (
        len(spaces), len(cands), sum(1 for d in spaces.values() if d["closed"])))
    for k in ("SPACE_EXPANDED", "REWRITE_PRODUCED", "CANDIDATE_REGISTERED",
              "FAMILY_ROUND_RECORDED", "CONVERGENCE_DECIDED", "AGENT_CALL_FAILED"):
        print("    %-22s %d" % (k, counts.get(k, 0)))


def main() -> int:
    report("s7-treatment")
    print()
    report("s7-control")
    print()
    print("HOW TO READ IT. Two spaces on the SAME candidate = a K expansion re-tuning it under a wider")
    print("space. Two candidates = the run moved on. `budget is per space, not per candidate`: 80 trials")
    print("against trials_per_space 40 is two spaces, not an overspend. A structural difference between")
    print("the arms here is what a rate difference in check_arm_search_parity.py would be MADE OF.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
