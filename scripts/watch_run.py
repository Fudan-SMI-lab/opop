"""Watch one run's events.jsonl and print the lines worth waking a session for.

Used as a Monitor command: each stdout line becomes one notification. Exits 0 on
RUN_FINISHED, 1 if the run looks wedged or its process is gone -- silence must never be
the only signal, because a crashed run and a slow one look identical from the outside.

  python scripts/watch_run.py <run_dir> [--idle-min N]
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

# The failure modes that killed the previous GLM run, plus the real milestones. Anything
# outside this set is progress noise and would turn the monitor into a firehose.
WATCH = {
    "AGENT_CALL_FAILED", "AGENT_SESSION_RESET", "SPACE_REJECTED",
    "CANDIDATE_REGISTERED", "SPACE_PUBLISHED", "TUNING_DONE",
    "FAMILY_ROUND_RECORDED", "CONVERGENCE_DECIDED", "OUTER_LOOP_STUCK",
    "RUN_FINISHED", "NOVELTY_PRODUCED", "NOVELTY_REJECTED",
    "FAMILY_FROZEN_UNREWRITABLE", "REPAIR_PRODUCED", "OUTER_LOOP_EXHAUSTED",
}


def opencode_alive() -> bool:
    """True if any opencode.exe is running. Windows `tasklist`, since the orchestrator is
    a Windows process and a Bash `ps` cannot see it."""
    try:
        out = subprocess.run(["tasklist"], capture_output=True, timeout=30)
        return b"opencode.exe" in out.stdout
    except (OSError, subprocess.SubprocessError):
        return True      # can't tell -> assume alive rather than cry wolf


def main() -> int:
    args = sys.argv[1:]
    if not args:
        print("usage: watch_run.py <run_dir> [--idle-min N]")
        return 2
    events = Path(args[0]) / "events.jsonl"
    idle_limit = 60.0
    if "--idle-min" in args:
        idle_limit = float(args[args.index("--idle-min") + 1])

    seen = 0
    last_change = time.time()
    while True:
        try:
            ev = [json.loads(l) for l in events.read_text(encoding="utf-8").splitlines() if l.strip()]
        except (OSError, ValueError):
            ev = []
        for e in ev[seen:]:
            t = e.get("type")
            if t in WATCH:
                print(f"{t} {json.dumps(e.get('payload', {}), ensure_ascii=False)[:200]}",
                      flush=True)
            if t == "RUN_FINISHED":
                print(f"RUN_FINISHED -- {events.parent.name} complete "
                      f"({len(ev)} events)", flush=True)
                return 0
        if len(ev) > seen:
            last_change = time.time()
        seen = len(ev)

        idle_min = (time.time() - last_change) / 60.0
        if idle_min >= idle_limit:
            print(f"STALLED: no new event for {idle_min:.0f} min ({seen} events) -- "
                  f"may be wedged", flush=True)
            return 1
        if not opencode_alive():
            print(f"PROCESS GONE: no opencode.exe and no RUN_FINISHED ({seen} events) "
                  f"-- the run died", flush=True)
            return 1
        time.sleep(60)


if __name__ == "__main__":
    raise SystemExit(main())
