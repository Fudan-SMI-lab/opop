"""How much of each arm's wall clock went to AGENT calls, and to WHICH module?

WHY. `check_arm_search_parity.py` reports `agent N% of wall clock` as one number, and at 3.5 h both arms
read 29%. But the arms are not spending it on the same thing: the treatment arm has entered loop A
(repair) on its 4th candidate after a witness rejection, and a repair call has an 1800 s deadline with
`repair_attempts = 2` -- so up to 3 parameterize attempts and 2 repair calls. The config's own comment
records an L3:43 repair that burned 0.99 h. Time in a repair produces NO trials, so a per-module split
is what says whether the trial-count gap is turning into an agent-time gap.

THIS IS NOT A DEFECT REPORT. The bound exists and is respected; the cost is real and known
(`agent-self-verification-burns-the-timeout`). The point is to price it for the comparison, because it
is a third asymmetry that the switch does not explain -- the witness gate judges default-config
correctness and S7 has enqueued nothing.

Pairing: AGENT_CALL_STARTED / FINISHED share `call_id`. An unmatched STARTED is IN FLIGHT and is priced
to "now" with its age shown separately, never folded into the finished total -- otherwise a call still
running would silently inflate or deflate the module's share depending on when the probe ran.

Run on box4:
  /root/autodl-tmp/orch-venv/bin/python /root/probe-clean/agent_time_by_module.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

BASE = Path("/root/autodl-tmp/opop-workspace/opop-glm/runs-v3")
RUN = "run-l3-43-20260913-202332"
ARMS = ("s7-control", "s7-treatment")


def main() -> int:
    now = time.time()
    for arm in ARMS:
        ev = []
        for line in (BASE / arm / RUN / "events.jsonl").open(encoding="utf-8"):
            line = line.strip()
            if line:
                try:
                    ev.append(json.loads(line))
                except ValueError:
                    pass
        if not ev:
            print("%s: no events" % arm)
            continue

        ts = [e["ts"] for e in ev if isinstance(e.get("ts"), (int, float))]
        span = max(ts) - min(ts) if ts else 0.0

        open_calls: dict[str, tuple[float, str]] = {}
        done: dict[str, list[float]] = {}
        inflight: list[tuple[str, float]] = []
        for e in ev:
            t = e.get("type") or ""
            if not t.startswith("AGENT_CALL_"):
                continue
            p = e.get("payload") or {}
            cid = str(p.get("call_id") or "")
            if t.endswith("STARTED"):
                open_calls[cid] = (float(e["ts"]), str(p.get("module") or "?"))
            elif cid in open_calls:
                t0, mod = open_calls.pop(cid)
                done.setdefault(mod, []).append(float(e["ts"]) - t0)
        for cid, (t0, mod) in open_calls.items():
            inflight.append((mod, now - t0))

        finished_s = sum(sum(v) for v in done.values())
        print("=" * 74)
        print("%s   span %.2f h   agent (finished) %.2f h = %.0f%% of wall clock"
              % (arm, span / 3600.0, finished_s / 3600.0,
                 100 * finished_s / span if span else 0.0))
        print("  %-12s %6s %10s %10s %10s" % ("module", "calls", "total_s", "median_s", "max_s"))
        for mod in sorted(done, key=lambda m: -sum(done[m])):
            v = sorted(done[mod])
            med = v[len(v) // 2] if len(v) % 2 else (v[len(v) // 2 - 1] + v[len(v) // 2]) / 2
            print("  %-12s %6d %10.0f %10.0f %10.0f" % (mod, len(v), sum(v), med, max(v)))
        for mod, age in inflight:
            print("  IN FLIGHT: %s, %.0f s so far (deadline 1800 s) -- NOT in the totals above"
                  % (mod, age))
            if mod == "repair":
                print("    repair_attempts=2 => at most 3 parameterize attempts / 2 repair calls.")
                print("    Time here yields NO trials; the config's own comment records an L3:43")
                print("    repair that burned 0.99 h.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
