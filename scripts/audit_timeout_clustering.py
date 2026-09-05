"""Do ReadTimeouts cluster in TIME rather than by module?

`measurement-agent-call-timeouts-cost-3.9h.md` noted that `repair` carried 8 of 12 timeouts and
flagged the module as where to look for a cause. That reading is now doubtful: `parameterizer`
timed out on run-l3-21-20260905-195615 after 306 clean attempts across every prior run, so
whatever causes this is not a property of the module's prompt.

The competing explanation is that the endpoint has bad windows, and whichever module happens to
call during one takes the hit. That predicts timeouts cluster in wall-clock time, and that the
module distribution should track each module's share of calls during those windows rather than
its overall share.

This reports, for every timeout on disk:
  - its wall-clock time, so clustering is visible;
  - the gap to the previous timeout;
  - what fraction of all calls each module made, against its share of timeouts.

A module-specific cause predicts the shares diverge. A time-clustered cause predicts they
roughly match once you condition on when the calls happened.

Usage:  python scripts/audit_timeout_clustering.py [--runs runs]
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from collections import Counter
from pathlib import Path


def per_attempt(events: list[dict]):
    """Yield (module, ts_start, ts_end, ok, error) for every ATTEMPT."""
    open_at: dict[str, tuple[str, float]] = {}
    for e in events:
        p = e.get("payload") or {}
        cid = p.get("call_id")
        mod = p.get("module")
        t = e.get("type")
        if t == "AGENT_CALL_STARTED" and cid:
            open_at[cid] = (mod, e["ts"])
        elif t in ("AGENT_CALL_FAILED", "AGENT_CALL_FINISHED") and cid in open_at:
            m0, t0 = open_at[cid]
            yield (m0 or mod, t0, e["ts"], t == "AGENT_CALL_FINISHED",
                   str(p.get("error") or ""))
            if t == "AGENT_CALL_FAILED":
                open_at[cid] = (m0, e["ts"])
            else:
                open_at.pop(cid, None)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--window-min", type=float, default=90.0,
                    help="calls within this many minutes of a timeout count as its window")
    args = ap.parse_args()

    calls = []
    for d in sorted(Path(args.runs).glob("run-*")):
        f = d / "events.jsonl"
        if not f.exists():
            continue
        events = []
        for line in f.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
        for mod, t0, t1, ok, err in per_attempt(events):
            calls.append({"run": d.name[-13:], "module": mod, "t0": t0, "t1": t1,
                          "ok": ok, "timeout": "ReadTimeout" in err})

    calls.sort(key=lambda c: c["t0"])
    timeouts = [c for c in calls if c["timeout"]]
    print(f"attempts: {len(calls)}   ReadTimeouts: {len(timeouts)}\n")

    print("timeouts in wall-clock order, with the gap to the previous one:")
    prev = None
    for c in timeouts:
        when = dt.datetime.fromtimestamp(c["t0"])
        gap = f"{(c['t0'] - prev) / 60:8.1f} min" if prev else "       --"
        print(f"  {when:%m-%d %H:%M}  {c['run']}  {c['module']:14s} gap {gap}")
        prev = c["t0"]

    # Cluster: consecutive timeouts less than window_min apart.
    clusters = []
    for c in timeouts:
        if clusters and (c["t0"] - clusters[-1][-1]["t0"]) / 60 <= args.window_min:
            clusters[-1].append(c)
        else:
            clusters.append([c])
    print(f"\nclusters (timeouts within {args.window_min:.0f} min of each other): "
          f"{len(clusters)}")
    for cl in clusters:
        span = (cl[-1]["t0"] - cl[0]["t0"]) / 60
        mods = ", ".join(sorted({c["module"] for c in cl}))
        print(f"  {dt.datetime.fromtimestamp(cl[0]['t0']):%m-%d %H:%M}  "
              f"n={len(cl)}  span {span:5.1f} min  modules: {mods}")

    # Share of calls vs share of timeouts, overall and inside the windows.
    overall = Counter(c["module"] for c in calls)
    to_share = Counter(c["module"] for c in timeouts)
    in_window = Counter()
    for c in calls:
        for t in timeouts:
            if abs(c["t0"] - t["t0"]) / 60 <= args.window_min:
                in_window[c["module"]] += 1
                break

    print(f"\n{'module':16s} {'all calls':>10s} {'share':>7s} "
          f"{'in-window':>10s} {'share':>7s} {'timeouts':>9s} {'share':>7s}")
    n_all = sum(overall.values())
    n_win = sum(in_window.values())
    n_to = sum(to_share.values())
    for m in sorted(overall):
        print(f"{m:16s} {overall[m]:10d} {overall[m] / n_all * 100:6.1f}% "
              f"{in_window[m]:10d} {in_window[m] / n_win * 100 if n_win else 0:6.1f}% "
              f"{to_share[m]:9d} {to_share[m] / n_to * 100 if n_to else 0:6.1f}%")

    print("\nRead the last two share columns against each other, not the first. If a")
    print("module's timeout share tracks its IN-WINDOW call share, the timeouts are")
    print("explained by when it called, not by what it sent.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
