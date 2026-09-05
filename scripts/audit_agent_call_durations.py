"""Audit: agent call durations and transport timeouts, per module.

Motivated by a live ReadTimeout on the repair agent. The question is not "did it recover"
(it always has: 10 of 10 non-final) but what the wall-clock cost is and whether the
timeout wall sits in a sensible place for each module.

The finding this reproduces: repair times out on 9.1% of calls while the parameterizer,
called three times as often, has never timed out once. And NO successful call on any
module has exceeded ~10 minutes while the wall sits at 20, so a timeout is a stuck
request rather than a slow one. See docs/measurement-repair-agent-transport-timeouts.md

Pairing note: reset the pending start on AGENT_CALL_FAILED as well as on
AGENT_CALL_FINISHED. Not doing so merges a 20-minute timeout with its retry into one
apparent 22-minute call, which is how an earlier pass of this analysis invented a
bimodal distribution and a 59-minute repair call that never happened.

Usage:  python scripts/audit_agent_call_durations.py [runs/run-... ...]
"""

from __future__ import annotations

import json
import statistics as st
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
TIMEOUT_MARKERS = ("ReadTimeout", "transport error", "timed out")
BUCKETS = ((2, "0-2"), (5, "2-5"), (10, "5-10"), (20, "10-20"), (float("inf"), "20+"))


def bucket(minutes: float) -> str:
    for upper, label in BUCKETS:
        if minutes < upper:
            return label
    return "20+"


def is_timeout(err: str) -> bool:
    return any(m in err for m in TIMEOUT_MARKERS)


def main(argv: list[str]) -> int:
    runs = [Path(a) for a in argv] or sorted(REPO.glob("runs/run-l3-*"))
    durations: dict[str, list[float]] = {}
    finished: Counter = Counter()
    timeouts: Counter = Counter()
    stalled_min = 0.0
    n_runs = 0

    for run in runs:
        ev_path = run / "events.jsonl"
        if not ev_path.exists():
            continue
        n_runs += 1
        events = [json.loads(line) for line in ev_path.read_text(encoding="utf-8").splitlines()
                  if line.strip()]
        # AGENT_CALL_STARTED carries no module, so pair each terminal event with the
        # nearest preceding start. Calls run sequentially, so this is exact today; a
        # concurrent call could mis-attribute one duration.
        last_start: float | None = None
        for e in events:
            payload = e.get("payload", {})
            if e["type"] == "AGENT_CALL_STARTED":
                last_start = e["ts"]
            elif e["type"] == "AGENT_CALL_FINISHED":
                module = payload.get("module") or "?"
                finished[module] += 1
                if last_start is not None:
                    durations.setdefault(module, []).append((e["ts"] - last_start) / 60)
                last_start = None
            elif e["type"] == "AGENT_CALL_FAILED":
                module = payload.get("module") or "?"
                if is_timeout(str(payload.get("error", ""))):
                    timeouts[module] += 1
                    if last_start is not None:
                        stalled_min += (e["ts"] - last_start) / 60
                last_start = None

    print(f"Across {n_runs} run(s).\n")
    print(f"{'module':16}{'finished':>9}{'timeouts':>10}{'rate':>8}"
          f"{'median':>9}{'p90':>7}{'max':>8}   (minutes)")
    for module in sorted(set(finished) | set(timeouts),
                         key=lambda m: -timeouts[m]):
        d = sorted(durations.get(module, []))
        med = f"{st.median(d):.1f}" if d else "-"
        p90 = f"{d[int(0.9 * (len(d) - 1))]:.1f}" if d else "-"
        mx = f"{d[-1]:.1f}" if d else "-"
        rate = 100 * timeouts[module] / max(finished[module] + timeouts[module], 1)
        print(f"  {module:16}{finished[module]:>9}{timeouts[module]:>10}{rate:>7.1f}%"
              f"{med:>9}{p90:>7}{mx:>8}")

    print(f"\nWall clock lost to timeouts: ~{stalled_min:.0f} min "
          f"({stalled_min / 60:.1f}h) across {sum(timeouts.values())} timeout(s)")

    # The bimodality is the actionable part: a gap below the wall means failing faster
    # would not cut off any call that was going to finish normally.
    for module in sorted(durations, key=lambda m: -timeouts[m]):
        d = durations[module]
        if timeouts[module] == 0 or len(d) < 20:
            continue
        hist = Counter(bucket(x) for x in d)
        print(f"\n{module} successful-call durations (n={len(d)}):")
        for _upper, label in BUCKETS:
            n = hist[label]
            print(f"   {label:6}{n:>4}  {'#' * n}")
        # No successful call has ever exceeded ~10 min on any module, while the wall is
        # at 20. So a timeout is not "a call that needed longer" -- it is a call that
        # produced nothing for twice the longest observed completion, which reads as a
        # stuck request. Report the headroom rather than claiming bimodality: an earlier
        # version of this analysis paired across AGENT_CALL_FAILED and invented a 59-min
        # call that never happened.
        print(f"   longest successful {module} call: {max(d):.1f} min "
              f"(wall is at request_timeout_s / 60)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
