"""What a ReadTimeout on an agent call actually costs, measured over every run on disk.

WHY THIS EXISTS: `request_timeout_s` was raised 1200 -> 1800 on the belief that the 1200 s
kills were destroying real work ("each kill discards a candidate or a whole rewrite round").
This script re-derives that from the event logs and the belief does not survive: 14 of the 15
ReadTimeout kills FINISHED on a retry, and no successful call in 982 has ever exceeded 1200 s.
A timed-out call is hung, not slow, so the timeout is not a work budget -- it is the price of
noticing a hang, and raising it made every hang 50% more expensive.

THE PAIRING BUG THIS SCRIPT EXISTS TO AVOID: the obvious implementation keys attempts by
`call_id` and pops the start timestamp when it sees a terminal event. That silently reports
ZERO recoveries, because a retried call emits AGENT_CALL_FAILED and then AGENT_CALL_FINISHED
under the SAME call_id -- the pop on FAILED discards the start, so the FINISHED has nothing to
pair with and is dropped. The first version of this analysis reported "0 recovered, 15 lost"
for exactly that reason, which inverts the conclusion. Attempts are therefore accumulated per
call_id (a FAILED re-arms the clock for the next attempt) and never popped on failure.

Usage:
    python scripts/probe_agent_timeouts.py
    python scripts/probe_agent_timeouts.py --runs-dir D:/... --runs-dir D:/...
"""
from __future__ import annotations

import argparse
import collections
import json
import os
from glob import glob

DEFAULT_RUNS_DIRS = [
    "D:/Pyhon_projects/opop/v2/runs",
    "D:/Pyhon_projects/opop/v2/runs-glm",
    "D:/Pyhon_projects/opop/v2-glm/runs",
    "D:/Pyhon_projects/opop/v2-glm/runs-l2-37",
]


def collect(runs_dirs: list[str]) -> dict:
    """Per (run, call_id): the module, attempt durations, and terminal-event counts."""
    per: dict[tuple[str, str], dict] = collections.defaultdict(
        lambda: {"mod": None, "durs": [], "timeouts": 0, "finished": 0, "failed": 0}
    )
    for root in runs_dirs:
        for path in glob(os.path.join(root, "*", "events.jsonl")):
            run = os.path.basename(os.path.dirname(path))
            start: dict[tuple[str, str], float] = {}
            try:
                with open(path, encoding="utf-8", errors="replace") as fh:
                    for line in fh:
                        line = line.strip()
                        if not line.startswith("{"):
                            continue
                        try:
                            ev = json.loads(line)
                        except ValueError:
                            continue
                        payload = ev.get("payload") or {}
                        call_id = payload.get("call_id")
                        if not call_id:
                            continue
                        key = (run, call_id)
                        rec = per[key]
                        if payload.get("module"):
                            rec["mod"] = payload["module"]
                        ts = float(ev.get("ts") or 0)
                        kind = ev.get("type")
                        if kind == "AGENT_CALL_STARTED":
                            start[key] = ts
                        elif kind == "AGENT_CALL_FAILED":
                            rec["failed"] += 1
                            if "ReadTimeout" in str(payload.get("error", "")):
                                rec["timeouts"] += 1
                            if key in start:
                                rec["durs"].append(ts - start[key])
                                # Re-arm rather than pop: the retry runs under this same
                                # call_id, and popping here is what hid every recovery.
                                start[key] = ts
                        elif kind == "AGENT_CALL_FINISHED":
                            rec["finished"] += 1
                            if key in start:
                                rec["durs"].append(ts - start.pop(key))
            except OSError:
                continue
    return per


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", action="append", default=None)
    args = ap.parse_args()
    per = collect(args.runs_dir or DEFAULT_RUNS_DIRS)

    # A successful attempt is the last duration of a call that finished; every earlier
    # duration of that call was a failed attempt.
    succeeded: list[float] = []
    for rec in per.values():
        if rec["finished"] and rec["durs"]:
            succeeded.append(rec["durs"][-1])
    succeeded.sort()

    timed_out = [(k, v) for k, v in per.items() if v["timeouts"] > 0]
    recovered = [x for x in timed_out if x[1]["finished"] > 0]
    lost = [x for x in timed_out if x[1]["finished"] == 0]

    print(f"successful agent calls: n={len(succeeded)}")
    if succeeded:
        for q in (50, 90, 95, 99):
            v = succeeded[min(len(succeeded) - 1, int(len(succeeded) * q / 100))]
            print(f"   p{q:<3} {v:7.1f} s ({v / 60:.1f} min)")
        print(f"   max  {succeeded[-1]:7.1f} s ({succeeded[-1] / 60:.1f} min)")
        for wall in (1200, 1500, 1800):
            n = sum(1 for d in succeeded if d > wall)
            print(f"   exceeding {wall}s: {n}")
    print()
    print(f"calls hitting at least one ReadTimeout: {len(timed_out)}")
    print(f"   recovered on a retry: {len(recovered)}")
    print(f"   never finished      : {len(lost)}")
    if recovered:
        retries = sorted(round(v["durs"][-1] / 60, 1) for _, v in recovered)
        print(f"   successful retry durations (min): {retries}")
    for key, rec in sorted(lost):
        print(f"   LOST  {key[0][:34]:34} {rec['mod']}  timeouts={rec['timeouts']}")
    print()
    # The decision this supports: with hangs, not slow work, driving every timeout, the
    # value is a per-hang price. Report it against the observed hang count.
    n_hangs = sum(v["timeouts"] for v in per.values())
    print(f"observed hangs: {n_hangs}. Cost of noticing them, by timeout value:")
    for wall in (600, 900, 1200, 1500, 1800):
        killed = sum(1 for d in succeeded if d > wall)
        note = f"  <-- kills {killed} real call(s)" if killed else ""
        print(f"   {wall:5}s  {n_hangs * wall / 3600:5.2f} h{note}")
    if succeeded:
        print()
        print(f"Any timeout at or below {succeeded[-1]:.0f}s kills real work. Headroom:")
        for wall in (1200, 1500, 1800):
            print(f"   {wall}s: {wall - succeeded[-1]:6.0f} s "
                  f"({(wall / succeeded[-1] - 1) * 100:4.1f}%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
