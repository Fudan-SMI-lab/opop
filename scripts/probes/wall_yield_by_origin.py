"""Where do actionable walls come from -- SEED candidates or REWRITTEN ones?

WHY THIS DECIDES SOMETHING. The S7 pair has 9.35 h of its 12 h budget left and both arms have finished
most of their seed tuning, so nearly all remaining work is loop C: rewrite a candidate, tune the result.
The intuition worth testing is "wait for the rewrites and new walls will appear" -- if true, S7's zero
enqueued points is a mid-run artefact; if false, the null is predictable now rather than merely likely.

WHAT COUNTS AS AN OPPORTUNITY. Not `walls_found`. A wall that fails the shipping monotone+gain filter is
one S7 cannot act on, so the quantity is `walls_found - walls_worthless` per attribution event, split by
the candidate's `origin` field. Counting raw walls would have reported rewrite-origin candidates as
producing walls at half the seed rate, when the actionable rate is zero -- a difference between "worse"
and "none at all".

WHY THE ANSWER IS MECHANISTIC, not incidental. A steered rewrite's PURPOSE is to free the wall it was
told about, so a rewritten candidate is by design less likely to hit that wall again. And it is freshly
written code whose domain has not been swept, so a refused value is more likely to sit INSIDE the
measured range (the truncation test in wall_attribution.py:265). Both effects push the yield down.

Run on box4:  /root/autodl-tmp/orch-venv/bin/python /root/probe-clean/wall_yield_by_origin.py
"""

from __future__ import annotations

import json
from pathlib import Path

BASE = Path("/root/autodl-tmp/opop-workspace/opop-glm/runs-v3")

# `seed` is the generator's own output; the rest are all downstream of an agent rewriting something.
_DERIVED = ("rewrite", "repair", "novelty")


def main() -> int:
    rows: dict[str, list[int]] = {"seed": [0, 0, 0], "derived": [0, 0, 0], "unknown": [0, 0, 0]}
    per_run: list[tuple[str, int, int]] = []
    excluded: list[tuple[str, int, int]] = []
    origin_runs: dict[str, set[str]] = {"seed": set(), "derived": set(), "unknown": set()}

    for run in sorted(BASE.glob("*/run-l3-43-*")):
        ev = []
        for line in (run / "events.jsonl").open(encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            try:
                ev.append(json.loads(line))
            except ValueError:
                pass

        # `origin` lives on CANDIDATE_REGISTERED, and the payload nests the candidate one level down
        # on this emitter -- read both shapes rather than assuming, since a missing origin would
        # silently land every candidate in "unknown" and read as "no data" instead of "wrong key".
        origin: dict[str, str] = {}
        for e in ev:
            if e.get("type") != "CANDIDATE_REGISTERED":
                continue
            p = (e.get("payload") or {}).get("candidate") or (e.get("payload") or {})
            cid = p.get("candidate_id")
            if cid:
                origin[str(cid)] = str(p.get("origin") or "?")

        run_worthy = run_events = 0
        for e in ev:
            if e.get("type") != "RESOURCE_WALL_ATTRIBUTED":
                continue
            p = e.get("payload") or {}
            o = origin.get(str(p.get("candidate_id")), "?")
            key = "seed" if o == "seed" else ("derived" if o in _DERIVED else "unknown")
            found = int(p.get("walls_found") or 0)
            worthless = int(p.get("walls_worthless") or 0)
            rows[key][0] += 1
            origin_runs[key].add(f"{run.parent.name}"[-18:])
            rows[key][1] += found
            rows[key][2] += worthless
            run_events += 1
            run_worthy += max(0, found - worthless)
        label = f"{run.parent.name}/{run.name}"[-42:]
        if run_events:
            per_run.append((label, run_events, run_worthy))
        else:
            # A run with refusals but no attribution had 2e OFF. Recorded separately so its silence
            # cannot be read as "this run yielded no walls".
            refusals = sum(
                1 for e in ev if e.get("type") == "TRIAL_DONE"
                and ((e.get("payload") or {}).get("trial") or {}).get("failure_kind")
                == "infeasible_shared_memory")
            if refusals or origin:
                excluded.append((label, refusals, len(origin)))

    print("%-10s %8s %12s %11s %9s %14s" % (
        "origin", "events", "walls_found", "worthless", "WORTHY", "worthy/event"))
    for key in ("seed", "derived", "unknown"):
        n, found, worthless = rows[key]
        worthy = max(0, found - worthless)
        print("%-10s %8d %12d %11d %9d %14.2f" % (
            key, n, found, worthless, worthy, worthy / n if n else 0.0))
    print()
    print("per run (events / worthy) -- runs with 0 events are EXCLUDED, not zero-yield:")
    for name, n, w in per_run:
        print("  %-42s %3d / %d" % (name, n, w))
    if excluded:
        print()
        print("  RUNS CONTRIBUTING NOTHING, and why it is not missing data:")
        for name, refusals, regs in excluded:
            print("    %-42s 0 attributions with %d refusals, %d candidates" % (
                name, refusals, regs))
        print("    Refusals present with zero attributions can only mean 2e was SWITCHED OFF (or")
        print("    crashed, which has its own event). These runs were never measuring walls, so their")
        print("    silence says nothing about wall yield -- the same confound that once made the P1-P5")
        print("    reader print a +7.99% 'improvement' against a control whose 2e was simply off.")
    print()
    # HOW THIN IS THE BASE, per ORIGIN and not just per run. The run count understates it: the two S7
    # arms have only seed candidates so far, so every rewrite-origin event can come from a single run
    # while three runs "contributed data". `three-same-cell-hits-are-not-a-model` is the recorded
    # version of this -- a rate whose evidence lives in one cell is a direction, not a rate.
    n_runs_with_data = len(per_run)
    print("BASE: %d run(s) contributed any attribution event, %d contributed none."
          % (n_runs_with_data, len(excluded)))
    for key, label in (("seed", "seed"), ("derived", "rewritten")):
        runs_for_key = sorted(origin_runs.get(key, set()))
        print("      %-9s events came from %d run(s): %s"
              % (label, len(runs_for_key), ", ".join(runs_for_key) or "none"))
        if len(runs_for_key) == 1:
            print("        ONE RUN ONLY => a DIRECTION, not a rate. That run's idiosyncrasy cannot be")
            print("        separated from a general property at this base.")
    print()

    sn, sf, sw = rows["seed"]
    dn, df, dw = rows["derived"]
    s_rate = (sf - sw) / sn if sn else 0.0
    d_rate = max(0, df - dw) / dn if dn else 0.0
    if rows["unknown"][0]:
        print("NOTE: %d events had an unreadable origin and are excluded from the comparison."
              % rows["unknown"][0])
    if dn and d_rate == 0.0 and s_rate > 0.0:
        print("REWRITTEN CANDIDATES YIELD NO ACTIONABLE WALL AT ALL (%d events, %d walls found, all"
              % (dn, df))
        print("dropped by the slope filter), against %.2f per event for seeds. So loop C is the WORST"
              % s_rate)
        print("venue for this mechanism, not the best -- 'wait for the rewrites' is disconfirmed, and")
        print("the lever for C2 coverage is the SEED stage (more seeds, wider initial domains), not")
        print("more rewrite rounds.")
    elif dn:
        print("seed %.2f vs rewritten %.2f actionable walls per event." % (s_rate, d_rate))
        print("Judge 'wait for the rewrites' against the ratio, not against the raw wall counts.")
    else:
        print("No rewrite-origin attribution events yet -- the question is NOT YET ANSWERABLE, which")
        print("is different from an answer of zero.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
