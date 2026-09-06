"""Refuse to start a run while another one on this machine is still active.

WHY THIS EXISTS, and why it is a script rather than a change to the locking code:

`GpuRwLock`'s lock file lives at `jobs_dir / "gpu.lock"` where
`jobs_dir = store.run_dir / "jobs"` (wiring.py:103, worker_client.py:116). It is therefore
**per-run, not machine-wide**: two concurrent runs create two different lock files, never see
each other, and both enter their "exclusive" timing lane simultaneously. Nothing errors --
the runs just contend for SMs and bandwidth, and every latency either one measures is wrong.

Not hypothetical: on 2026-09-07 a manual `tune-file` verification ended at 02:01 and the main
experiment began at 02:07. Six minutes apart, each with its own lock file. Five minutes'
difference and both would have been timing at once, silently.

Fixing the lock means moving it to a machine-wide path, which touches the GPU serialization
path -- get that wrong and every measurement is quietly corrupted, worse than today. That is
deferred. This script is the zero-risk part: it only READS state, and turns a silent
data-corruption bug into a loud refusal before any GPU time is spent.

WHAT IT DOES NOT DO, and why the obvious approach fails: it does not look for lock files.
`_release_file` unlinks the lock as soon as a job finishes (worker_client.py:67-68), so the
file exists only while a job is executing. Verified on a live run: six consecutive checks all
found no lock while the run was demonstrably active. Since a GLM run spends most of its wall
clock inside agent calls rather than GPU jobs, a lock scan would report "safe to start"
almost any time -- a guard that cannot see the thing it guards against.

The signal that actually tracks liveness is a run whose **event log is still growing**, which
is what this checks. Locks are reported too, as extra detail when one happens to be held.

Usage (exit 0 = safe to start, exit 1 = another run appears active):

    python scripts/preflight_gpu_free.py
    python scripts/preflight_gpu_free.py --runs-dir D:/... --runs-dir D:/...
    python scripts/preflight_gpu_free.py --idle-min 20    # how quiet counts as finished
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

DEFAULT_RUNS_DIRS = [
    "D:/Pyhon_projects/opop/v2/runs",
    "D:/Pyhon_projects/opop/v2/runs-glm",
    "D:/Pyhon_projects/opop/v2-glm/runs",
    "D:/Pyhon_projects/opop/v2-glm/runs-l2-37",
]


def last_event_age_min(events: Path) -> tuple[float, bool, int]:
    """(minutes since the last event, whether RUN_FINISHED is present, event count).

    Reads the file's tail rather than parsing every line: a spinning run once produced a
    991 MB log, and a preflight check must stay fast and must not choke on that.
    """
    finished = False
    count = 0
    last_ts = 0.0
    try:
        size = events.stat().st_size
        with events.open("rb") as fh:
            # Tail for the newest timestamp.
            fh.seek(max(0, size - 65536))
            tail = fh.read().decode("utf-8", "replace")
        for line in tail.splitlines():
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                ev = json.loads(line)
            except ValueError:
                continue
            ts = float(ev.get("ts") or 0)
            last_ts = max(last_ts, ts)
            if ev.get("type") == "RUN_FINISHED":
                finished = True
        # A cheap line count for context only.
        with events.open("rb") as fh:
            count = sum(1 for _ in fh)
    except OSError:
        return (1e9, False, 0)
    if last_ts <= 0:
        # No parseable timestamp in the tail; fall back to the file's own mtime.
        try:
            last_ts = events.stat().st_mtime
        except OSError:
            return (1e9, finished, count)
    return ((time.time() - last_ts) / 60.0, finished, count)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", action="append", default=None,
                    help="a runs directory to scan; repeatable. Defaults to the known ones.")
    ap.add_argument("--idle-min", type=float, default=20.0,
                    help="a run with no new event for this long is treated as finished. "
                         "Default 20: a single GLM agent call has been measured at 18m16s, "
                         "so anything shorter would call a healthy run dead.")
    args = ap.parse_args()

    roots = [Path(p) for p in (args.runs_dir or DEFAULT_RUNS_DIRS)]
    active: list[str] = []
    rows: list[str] = []
    scanned = 0
    for root in roots:
        if not root.is_dir():
            continue
        for events in sorted(root.glob("*/events.jsonl")):
            scanned += 1
            age, finished, count = last_event_age_min(events)
            run = events.parent.name
            lock = events.parent / "jobs" / "gpu.lock"
            held = " [GPU job running]" if lock.exists() else ""
            if finished:
                state = "finished"
            elif age <= args.idle_min:
                state = f"ACTIVE (last event {age:.1f} min ago)"
                active.append(run)
            else:
                state = f"abandoned? (quiet {age:.0f} min, no RUN_FINISHED)"
            rows.append(f"  [{state}] {run}  {count} events{held}")

    print(f"scanned {scanned} run(s) under {len([r for r in roots if r.is_dir()])} runs dir(s)")
    # Active runs are the answer, so always show those. Abandoned ones are only noise here --
    # 27 of them on this machine -- so summarise rather than list, except any still holding a
    # lock file, which is a real leftover worth naming (a killed run does not unlink it).
    for line in rows:
        if "[ACTIVE" in line:
            print(line)
    leftover = [l for l in rows if "abandoned?" in l and "[GPU job running]" in l]
    abandoned = [l for l in rows if "abandoned?" in l]
    if abandoned:
        print(f"  ({len(abandoned)} abandoned run(s): killed or crashed, holding nothing)")
    for line in leftover:
        print(f"  stale lock file left behind by:{line.split(']', 1)[1]}")

    if not active:
        print()
        print("OK: no active run found -- safe to start.")
        print("     Note: runs marked 'abandoned?' were killed or crashed; they hold nothing.")
        return 0

    print()
    print(f"REFUSING: {len(active)} run(s) still active: {', '.join(active)}")
    print("Starting another run would NOT queue behind them. The GPU lock is per-run")
    print("(wiring.py:103), so both would time simultaneously and both sets of latency")
    print("numbers would be wrong, with no error raised. Stop the other run first, or wait.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
