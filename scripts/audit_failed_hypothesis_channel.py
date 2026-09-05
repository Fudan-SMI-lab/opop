"""Audit: does the failed-hypothesis channel reach the rewriter, and does it help?

When a rewrite round fails to improve, the orchestrator appends the source candidate's
hypotheses to failed_hypotheses[family_id] (orchestrator.py:1077) and journals
HYPOTHESES_FAILED. The rewriter receives them as history/failed_hypotheses.json.

Two separate questions, and the script keeps them apart because only the first has a clean
answer:

1. DELIVERY -- does the context change what the rewriter proposes? Measured by whether the
   child's approach_summary contains explicitly contrastive language ("unlike", "failed",
   "instead of", ...). This is a keyword proxy, so read it as a floor: a rewrite can react
   without saying so.

2. OUTCOME -- did the child beat its family's incumbent? CONFOUNDED, and the script says so:
   a rewrite has this context precisely because its family already had a non-improving round,
   i.e. it is a round-2-or-later rewrite, and later rounds have an independent 0-for-13 record
   (finding-converged-stop-kind-is-unreachable.md). The comparison against first-round
   rewrites therefore mixes the context with the round number and cannot separate them.

Note the HYPOTHESES_FAILED journalling is recent, so earlier runs record zero events even
though the in-memory dict was populated. The trigger condition is reconstructed from
FAMILY_ROUND_RECORDED so those runs still count.

See docs/measurement-failed-hypothesis-channel.md

Usage:  python scripts/audit_failed_hypothesis_channel.py [runs/run-... ...]
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from statistics import median
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# Contrastive markers. A proxy for "the summary reacts to a prior attempt", so the count is
# a floor rather than an exact measure.
CONTRAST = ("unlike", "failed", "rather than", "instead of", "avoid", "previous", "earlier")


def main(argv: list[str]) -> int:
    runs = [Path(a) for a in argv] or sorted(REPO.glob("runs/run-l3-*"))

    with_ctx: list[tuple] = []
    without_ctx: list[float] = []
    journalled = 0

    for run in runs:
        ev_path = run / "events.jsonl"
        if not ev_path.exists():
            continue
        rounds: dict[str, list[tuple[float, float]]] = defaultdict(list)
        rewrites: list[tuple[float, dict]] = []
        best: dict[str, float] = {}
        for line in ev_path.open(encoding="utf-8"):
            e = json.loads(line)
            if e["type"] == "FAMILY_ROUND_RECORDED":
                rounds[e["payload"]["family_id"]].append((e["ts"], e["payload"]["best_ms"]))
            elif e["type"] == "HYPOTHESES_FAILED":
                journalled += 1
            elif e["type"] == "CANDIDATE_REGISTERED":
                c = e["payload"].get("candidate") or e["payload"]
                if c.get("origin") == "rewrite":
                    rewrites.append((e["ts"], c))
            elif e["type"] == "TUNING_DONE":
                cid, ms = e["payload"].get("candidate_id"), e["payload"].get("best_ms")
                if cid and ms is not None:
                    best[cid] = min(best.get(cid, float("inf")), ms)

        for ts, c in rewrites:
            seq = [m for t, m in rounds.get(c["family_id"], []) if t < ts]
            if not seq:
                continue
            incumbent = min(seq)
            child = best.get(c["candidate_id"])
            # The dict is non-empty iff some earlier round in this family failed to improve.
            has_ctx = len(seq) >= 2 and any(seq[i] >= seq[i - 1] for i in range(1, len(seq)))
            summary = c.get("approach_summary") or ""
            if has_ctx:
                cites = any(w in summary.lower() for w in CONTRAST)
                with_ctx.append((run.name[4:], c["candidate_id"], c["family_id"], cites,
                                 child, incumbent, summary[:95]))
            elif child is not None and incumbent > 0:
                without_ctx.append((incumbent - child) / incumbent * 100.0)

    print("=" * 92)
    print("DELIVERY -- rewrites issued while their family had failed hypotheses recorded")
    print("=" * 92)
    print(f"  HYPOTHESES_FAILED events journalled: {journalled}"
          f"   (the journalling is recent; earlier runs are reconstructed)")
    print(f"  rewrites with failed-hypothesis context: {len(with_ctx)}")
    cited = sum(1 for r in with_ctx if r[3])
    if with_ctx:
        print(f"  whose summary explicitly contrasts with a prior attempt: "
              f"{cited} ({100 * cited / len(with_ctx):.0f}%)  [keyword proxy -> a FLOOR]")
    print()
    print(f"  {'run':<24} {'candidate':<16} {'cites':<6} {'child':>8} {'fam_best':>9}  summary")
    for run, cid, fid, cites, child, inc, summary in with_ctx:
        print(f"  {run:<24} {cid:<16} {str(cites):<6} "
              f"{('%.1f' % child) if child else 'pending':>8} {inc:>9.1f}  {summary}")

    measurable = [r for r in with_ctx if r[4] is not None]
    beat = [r for r in measurable if r[4] < r[5]]
    print()
    print("=" * 92)
    print("OUTCOME -- CONFOUNDED, read the caveat below before quoting")
    print("=" * 92)
    if measurable:
        deltas = [(r[5] - r[4]) / r[5] * 100.0 for r in measurable]
        print(f"  WITH context     n={len(measurable):<3} median {median(deltas):+.1f}%"
              f"   beat incumbent {len(beat)}/{len(measurable)}")
    if without_ctx:
        print(f"  WITHOUT context  n={len(without_ctx):<3} median {median(without_ctx):+.1f}%"
              f"   beat incumbent {sum(1 for x in without_ctx if x > 0)}/{len(without_ctx)}")
    print()
    print("  CAVEAT: the two groups differ in ROUND NUMBER, not only in context. A rewrite has")
    print("  this context precisely because its family already had a non-improving round, so it")
    print("  is a round-2-or-later rewrite -- and later rounds have an independent 0-for-13")
    print("  record for reasons unrelated to what the rewriter was told. The effects are")
    print("  inseparable here; a round-2 rewrite issued WITHOUT the context would be the")
    print("  control, and the harness (correctly) never withholds known-failed information.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
