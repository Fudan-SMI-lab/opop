"""Audit: what do agent-call timeouts actually cost, and would a shorter timeout help?

`opencode.request_timeout_s` is 1200 (20 min). A hung endpoint costs that in full, and the
retry gets a fresh 20 min, so one call can burn 40+ minutes and still fail.

Care is needed measuring this: an AGENT_CALL_FINISHED event carries the attempt number, so
timing it from the FIRST AGENT_CALL_STARTED includes any failed attempts before it. That
inflates the apparent duration of successful calls (one repair call reads as 59.5 min when
its successful attempt took 19.4). Per-attempt durations are what a timeout decision needs,
so the walk below splits each call at its intermediate FAILED events.

Usage:  python scripts/audit_agent_call_timeouts.py [--runs runs]
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path


def per_attempt(events: list[dict]) -> list[tuple[str, str, float, bool]]:
    """(module, call_id, minutes, succeeded) for every ATTEMPT, not every call."""
    out = []
    # call_id -> (module, timestamp the current attempt began)
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
            out.append((m0 or mod, cid, (e["ts"] - t0) / 60.0,
                        t == "AGENT_CALL_FINISHED"))
            if t == "AGENT_CALL_FAILED":
                # The retry begins now; a fresh timer runs for it.
                open_at[cid] = (m0, e["ts"])
            else:
                open_at.pop(cid, None)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    args = ap.parse_args()

    ok: dict[str, list[float]] = defaultdict(list)
    bad: list[tuple[str, str, float, str]] = []

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
        errs = {}
        for e in events:
            if e.get("type") == "AGENT_CALL_FAILED":
                p = e.get("payload") or {}
                errs.setdefault(p.get("call_id"), []).append(str(p.get("error") or ""))
        for mod, cid, mins, good in per_attempt(events):
            if good:
                ok[mod].append(mins)
            else:
                why = (errs.get(cid) or [""])[0][:44]
                bad.append((d.name[-13:], mod, mins, why))

    print("SUCCESSFUL attempts, per attempt (minutes):")
    print(f"{'module':16s} {'n':>4s} {'median':>8s} {'p90':>8s} {'p99':>8s} {'max':>8s}")
    every: list[float] = []
    for mod, vals in sorted(ok.items()):
        vals.sort()
        every += vals
        p90 = vals[min(len(vals) - 1, int(len(vals) * 0.90))]
        p99 = vals[min(len(vals) - 1, int(len(vals) * 0.99))]
        print(f"{mod:16s} {len(vals):4d} {statistics.median(vals):8.1f} "
              f"{p90:8.1f} {p99:8.1f} {max(vals):8.1f}")
    every.sort()
    print(f"\nall successful attempts: n={len(every)}  median {statistics.median(every):.1f}  "
          f"p99 {every[int(len(every) * 0.99)]:.1f}  max {max(every):.1f}")

    print("\nFAILED attempts:")
    lost = 0.0
    timeouts = 0
    for run, mod, mins, why in bad:
        lost += mins
        if "ReadTimeout" in why:
            timeouts += 1
        print(f"  {run}  {mod:14s} {mins:6.1f} min  {why}")
    print(f"\nfailed attempts: {len(bad)}  of which ReadTimeout: {timeouts}")
    print(f"wall lost to failed attempts: {lost:.1f} min ({lost / 60:.1f} h)")

    print("\nWhat a shorter timeout would trade:")
    for cand in (600, 720, 900):
        cm = cand / 60.0
        killed = sum(1 for x in every if x > cm)
        saved = sum(max(0.0, mins - cm) for _, _, mins, why in bad if "ReadTimeout" in why)
        print(f"  timeout {cand}s ({cm:.0f} min): would kill {killed} of {len(every)} "
              f"successful attempts, save {saved:.0f} min on the hangs")
    print("\nNote what this comparison cannot settle: a killed slow-but-working call is not")
    print("free -- it costs its own elapsed time AND a retry. The counts above are the")
    print("trade, not a recommendation.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
