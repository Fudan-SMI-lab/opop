"""Audit: what does a K expansion (improvement K) actually do to the reported best?

Written after I asserted -- twice, wrongly -- that a flat re-tune is "the default outcome"
and that a particular 28.6 -> 28.0 was "the first expansion to improve a best". Both claims
came from generalizing over the three cases I happened to have investigated, which were
selected BECAUSE they were flat. This script prints the denominator so that mistake is a
command away from being caught.

It also flags which post-expansion bests were CACHE REPLAYS of the pre-expansion winner
(`reused_measurement`), because a flat outcome is uninformative about the widened region
whether or not it was replayed.

See docs/finding-k-retune-cannot-disconfirm-its-incumbent.md

Usage:  python scripts/audit_expansion_outcomes.py [runs/run-... ...]
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def main(argv: list[str]) -> int:
    runs = [Path(a) for a in argv] or sorted(REPO.glob("runs/run-l3-*"))
    outcomes: Counter[str] = Counter()
    cached_by_outcome: Counter[str] = Counter()
    rows: list[tuple] = []

    for run in runs:
        ev_path = run / "events.jsonl"
        if not ev_path.exists():
            continue
        events = [json.loads(line) for line in ev_path.read_text(encoding="utf-8").splitlines()
                  if line.strip()]

        seq: dict[str, list[tuple[str, float | None]]] = defaultdict(list)
        complete: dict[tuple[str, str], list[dict]] = defaultdict(list)
        cached: set[str] = set()
        for e in events:
            if e["type"] == "TRIAL_DONE":
                t = e["payload"].get("trial") or e["payload"]
                if e["payload"].get("reused_measurement"):
                    cached.add(t.get("trial_id"))
                if t.get("status") == "complete" and t.get("latency_ms"):
                    complete[(t.get("candidate_id"), t.get("space_id"))].append(t)
            elif e["type"] == "TUNING_DONE":
                seq[e["payload"].get("candidate_id")].append(
                    (e["payload"].get("space_id"), e["payload"].get("best_ms")))

        for cid, spaces in seq.items():
            for (prev_sid, prev_ms), (sid, ms) in zip(spaces, spaces[1:]):
                if prev_ms is None or ms is None:
                    continue
                trials = complete.get((cid, sid), [])
                best = min(trials, key=lambda t: t["latency_ms"]["mean"]) if trials else None
                was_cached = bool(best and best.get("trial_id") in cached)
                if ms < prev_ms:
                    kind = "improved"
                elif ms == prev_ms:
                    kind = "flat"
                else:
                    kind = "worse"
                outcomes[kind] += 1
                if was_cached:
                    cached_by_outcome[kind] += 1
                pct = (prev_ms - ms) / prev_ms * 100.0
                rows.append((run.name, cid, prev_sid, prev_ms, sid, ms, kind, was_cached, pct))

    total = sum(outcomes.values())
    if not total:
        print("No K expansion with a before/after TUNING_DONE pair found.")
        return 0

    print(f"{total} K expansions with a before/after pair:\n")
    for kind in ("improved", "flat", "worse"):
        n = outcomes[kind]
        print(f"  {kind:9s} {n:3d}  ({n / total * 100:4.1f}%)   "
              f"post-expansion best was a cache replay: {cached_by_outcome[kind]}")
    print()
    for run, cid, ps, pm, s, m, kind, wc, pct in sorted(rows, key=lambda r: (r[6], -r[8])):
        tag = "  [best is a CACHE REPLAY]" if wc else ""
        print(f"  {kind:9s} {pct:+6.1f}%  {run}  {cid}  {ps}={pm} -> {s}={m}{tag}")
    print()
    print("A flat outcome says nothing about the widened region -- judge an expansion by the")
    print("trials that used NEW choices. But flat is NOT the usual result: most expansions")
    print("improve the reported best, so do not predict flatness from a handful of examples.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
