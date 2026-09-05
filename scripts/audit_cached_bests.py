"""Audit: which reported bests came from a CACHED measurement rather than a fresh one?

`_tune` caches measurements by parameter vector, so when a K expansion widens some knobs
and leaves others alone, the incumbent theta_best is still legal in the new space and is
replayed from cache instead of re-measured. A "flat" re-tune is therefore the DEFAULT
outcome, not evidence that the widened region is unproductive -- and a re-tune can never
reproduce or disconfirm a suspicious best, because the cache guarantees the same number.

See docs/finding-k-retune-cannot-disconfirm-its-incumbent.md

Usage:  python scripts/audit_cached_bests.py [runs/run-... ...]
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def main(argv: list[str]) -> int:
    runs = [Path(a) for a in argv] or sorted(REPO.glob("runs/run-l3-*"))
    total_bests = 0
    findings: list[dict] = []

    for run in runs:
        ev_path = run / "events.jsonl"
        if not ev_path.exists():
            continue
        events = [json.loads(line) for line in ev_path.read_text(encoding="utf-8").splitlines()
                  if line.strip()]

        trials: dict[tuple, list] = defaultdict(list)
        cached: set[str] = set()
        for e in events:
            if e["type"] != "TRIAL_DONE":
                continue
            t = e["payload"].get("trial") or e["payload"]
            trials[(t.get("candidate_id"), t.get("space_id"))].append(t)
            if e["payload"].get("reused_measurement"):
                cached.add(t.get("trial_id"))

        # Per candidate, remember each space's reported best in order, so a re-tune can be
        # compared against the space that preceded it.
        prev_best: dict[str, tuple[str, float | None]] = {}
        for e in events:
            if e["type"] != "TUNING_DONE":
                continue
            payload = e["payload"]
            cid, sid = payload.get("candidate_id"), payload.get("space_id")
            complete = [t for t in trials.get((cid, sid), [])
                        if t.get("status") == "complete" and t.get("latency_ms")]
            if not complete:
                continue
            total_bests += 1
            best = min(complete, key=lambda t: t["latency_ms"]["mean"])
            was_cached = best.get("trial_id") in cached
            prior = prev_best.get(cid)
            prev_best[cid] = (sid, payload.get("best_ms"))
            if not was_cached:
                continue
            findings.append({
                "run": run.name,
                "candidate_id": cid,
                "space_id": sid,
                "best_ms": payload.get("best_ms"),
                "trial_id": best.get("trial_id"),
                "prior_space": prior[0] if prior else None,
                "prior_best_ms": prior[1] if prior else None,
                # How many trials in this space used a value the prior space lacked is the
                # informative statistic; the equality of the two bests is not.
                "n_complete": len(complete),
            })

    if not findings:
        print(f"No reported best came from a cached measurement "
              f"({total_bests} bests checked).")
        return 0

    print(f"{len(findings)} of {total_bests} reported bests came from a CACHED measurement:\n")
    for f in findings:
        print(f"  {f['run']}  {f['candidate_id']}")
        print(f"      space {f['space_id']} best={f['best_ms']} ms  <- cached trial "
              f"{f['trial_id']} ({f['n_complete']} complete trials in this space)")
        if f["prior_space"]:
            same = f["prior_best_ms"] == f["best_ms"]
            print(f"      prior space {f['prior_space']} best={f['prior_best_ms']} ms"
                  f"{'  -- IDENTICAL, so the flatness was structural' if same else ''}")
    print("\nA flat re-tune whose best is cached is the expected outcome, not evidence about")
    print("the widened region. Judge an expansion by the trials that used NEW choices, and")
    print("treat final_reeval_ms as the only fresh-process re-measurement of a theta_best.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
