"""Which task should the G9 re-run use? Pick the venue by measured headroom, not by convenience.

The first G9 attempt is uninformative for a reason that has nothing to do with its arms: it ran on
L3:48, which three independent runs have confirmed sits at a PHYSICAL plateau (93.7-95.5% of the
measured DRAM roof, 8 of 10 candidates across 4 different families within 2%). Asking "does more
evidence produce a better rewrite" on a candidate that is already against the memory wall is asking a
question the hardware has already answered: nothing produces a better rewrite there.

So the venue needs measured HEADROOM -- distance between the incumbent and the roof that a rewrite
could actually cross. This reports, per finished run, the facts that decide it:

  * best measured latency, and how it was reached
  * % of the measured DRAM roof the winner achieves  (high => plateau => bad venue)
  * whether the run ended by wall clock or by convergence
  * the spread of the top candidates (a tight spread across families is the plateau signature)

Reads through store/read.py, so a wrong field path raises instead of printing a clean empty table.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from kernel_optimizer.store.read import (  # noqa: E402
    EmptyResult,
    events_of_type,
    read_events,
    trials_with_latency,
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    args = ap.parse_args()

    print("%-34s %8s %9s %9s %8s %9s" % ("run", "trials", "best_ms", "p10_ms", "spread%", "cands"))
    rows = []
    for run in args.runs:
        try:
            trials = trials_with_latency(run)
        except (EmptyResult, Exception) as exc:  # noqa: BLE001
            print("%-34s  %s: %s" % (os.path.basename(run), type(exc).__name__, str(exc)[:60]))
            continue
        ms = sorted(t["latency_ms_value"] for t in trials)
        best = ms[0]
        # The plateau signature is a tight band at the TOP, not the whole distribution.
        p10 = ms[max(0, int(len(ms) * 0.10) - 1)]
        spread = (p10 - best) / best * 100.0
        cands = len({t.get("candidate_id") for t in trials})
        print("%-34s %8d %9.4f %9.4f %8.2f %9d"
              % (os.path.basename(run), len(trials), best, p10, spread, cands))
        rows.append({"run": run, "best_ms": best, "p10_ms": p10, "top_spread_pct": spread,
                     "n_trials": len(trials), "n_candidates": cands})

    # How each run ended, and the roof fraction, both of which live in other events.
    print()
    print("HOW EACH RUN ENDED, AND HOW CLOSE THE WINNER IS TO THE MEASURED ROOF")
    for r in rows:
        run = r["run"]
        name = os.path.basename(run)
        ended = "?"
        try:
            fin = events_of_type(run, "RUN_FINISHED", allow_empty=True)
            if fin:
                p = fin[-1].get("payload") or {}
                ended = str(p.get("stop_kind") or p.get("reason") or p.get("verdict") or "?")[:40]
        except Exception:  # noqa: BLE001
            pass
        # The roof fraction is reported in the bottleneck verdicts.
        roof = None
        basis = ""
        try:
            for e in events_of_type(run, "BOTTLENECK_REPORTED", allow_empty=True):
                ev = ((e.get("payload") or {}).get("evidence") or {})
                for key in ("pct_of_dram_peak", "dram_pct_of_peak", "pct_dram_roof"):
                    v = ev.get(key)
                    if isinstance(v, (int, float)):
                        roof = v if roof is None else max(roof, v)
                        basis = key
        except Exception:  # noqa: BLE001
            pass
        r["ended"] = ended
        r["roof_pct"] = roof
        print("  %-34s ended=%-22s roof=%s%s"
              % (name, ended,
                 ("%.1f%%" % roof) if isinstance(roof, (int, float)) else "not reported",
                 (" (%s)" % basis) if basis else ""))

    print()
    print("VENUE CHOICE")
    print("  A GOOD venue has a LOW roof fraction (room a rewrite could cross) and a WIDE top spread")
    print("  (candidates still differ, so a better brief can still matter).")
    print("  A BAD venue is the opposite: near the roof with the top candidates all within a few")
    print("  percent -- there, every arm is measuring the hardware, not the brief.")
    for r in sorted(rows, key=lambda r: r["top_spread_pct"], reverse=True):
        verdict = "GOOD" if r["top_spread_pct"] > 10 else ("marginal" if r["top_spread_pct"] > 4
                                                           else "PLATEAU - do not use")
        print("  %-34s top_spread=%6.2f%%  -> %s" % (os.path.basename(r["run"]),
                                                     r["top_spread_pct"], verdict))
    print(json.dumps(rows, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
