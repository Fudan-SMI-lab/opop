"""Where do actionable walls come from -- SEED candidates or REWRITTEN ones?

WHY THIS DECIDES SOMETHING. The S7 pair has 9.35 h of its 12 h budget left and both arms have finished
most of their seed tuning, so nearly all remaining work is loop C: rewrite a candidate, tune the result.
The intuition worth testing is "wait for the rewrites and new walls will appear" -- if true, S7's zero
enqueued points is a mid-run artefact; if false, the null is predictable now rather than merely likely.

WHAT COUNTS AS AN OPPORTUNITY. Not `walls_found`. A wall must clear TWO gates, and an earlier version of
this probe only counted the first:

  1. the slope filter -- `select_for_probing` keeps `monotone and tail_gain_pct > 0`. The emitter counts
     what it dropped as `walls_worthless` (orchestrator.py:1586), so `found - worthless` is the number
     that survived this gate. Counting raw walls would have reported rewrite-origin candidates as
     producing walls at half the seed rate, when the rate past this gate is zero -- a difference between
     "worse" and "none at all".

  2. ATTRIBUTION -- which runs AFTER the filter, so `walls_worthless` does not include its failures. A
     wall with `verdict != "attributed"` passed monotone and gain and was then found NOT to be caused by
     a resource limit at all: the control arm's NUM_WARPS=16 wall has `over_ratio 0.97`
     (max_shared 98304 against a limit of 101376 -- it never exceeded the limit), so nothing about it
     names a resource for a rewrite to free. `found - worthless` counts it as an opportunity; it is not.

So the quantity is walls that survived BOTH gates, per attribution event, split by the candidate's
`origin`. The two-gate figure is reported alongside the one-gate figure, because the one-gate figure is
what an earlier version of this probe published and it reads as an overcount, not as a different metric.

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
    # per key: [events, walls_found, worthless, attributed, probed_verdict_unknown]
    rows: dict[str, list[int]] = {"seed": [0, 0, 0, 0, 0], "derived": [0, 0, 0, 0, 0],
                                  "unknown": [0, 0, 0, 0, 0]}
    per_run: list[tuple[str, int, int]] = []
    excluded: list[tuple[str, int, int]] = []
    origin_runs: dict[str, set[str]] = {"seed": set(), "derived": set(), "unknown": set()}
    no_verdict_field = 0

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
            # Gate 2, and TWO wrong readings of it that this comment exists to stop.
            #
            # (a) `walls` is written on BOTH branches -- the no-probe one at orchestrator.py:1590 and
            #     the probe one at :1617 as `probe + skipped`, verdicts filled in. I first read it as
            #     no-probe-only and treated probed walls as verdict-unknown, which produced a bogus
            #     "lower bound 0.07, upper bound 0.33" spread over walls whose verdicts were right
            #     there in the record. There is no unknown class.
            # (b) An even earlier version added `walls_probed` into the attributed count to cover that
            #     imagined gap, which made the two-gate figure (0.33) exceed the one-gate figure (0.27)
            #     -- arithmetically impossible for a subset, and the tell that it was crediting
            #     probed-and-FAILED walls. The guard at the end now refuses to print such a result.
            attributed = 0
            for w in (p.get("walls") or []):
                if not isinstance(w, dict):
                    continue
                if not (w.get("monotone") and (w.get("tail_gain_pct") or 0) > 0):
                    continue  # already counted as worthless
                v = w.get("verdict")
                if v is None:
                    no_verdict_field += 1
                elif v == "attributed":
                    attributed += 1
            rows[key][0] += 1
            origin_runs[key].add(f"{run.parent.name}"[-18:])
            rows[key][1] += found
            rows[key][2] += worthless
            rows[key][3] += attributed
            rows[key][4] += int(p.get("walls_probed") or 0)
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

    print("%-10s %8s %12s %11s %11s %9s %9s %13s" % (
        "origin", "events", "walls_found", "worthless", "past_slope", "ATTRIB'D", "probed",
        "attrib/event"))
    for key in ("seed", "derived", "unknown"):
        n, found, worthless, attributed, probed = rows[key]
        worthy = max(0, found - worthless)
        print("%-10s %8d %12d %11d %11d %9d %9d %13.2f" % (
            key, n, found, worthless, worthy, attributed, probed,
            attributed / n if n else 0.0))
    print("  past_slope = found - worthless (gate 1 only; an OVERCOUNT of what S7 can use)")
    print("  ATTRIB'D   = also cleared attribution (gate 2), read from each wall's own `verdict`")
    print("  probed     = walls that were probed. Their verdicts ARE in the record (orchestrator.py")
    print("               :1617 writes probe+skipped), so this column is context, not uncertainty.")
    if no_verdict_field:
        print("  %d wall(s) carried NO `verdict` field (older runs) and are excluded from ATTRIB'D"
              % no_verdict_field)
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

    sn, sf, sw, sa, sp_ = rows["seed"]
    dn, df, dw, da, dp = rows["derived"]
    s_rate = sa / sn if sn else 0.0
    d_rate = da / dn if dn else 0.0
    s_gate1 = (sf - sw) / sn if sn else 0.0
    d_gate1 = max(0, df - dw) / dn if dn else 0.0
    if rows["unknown"][0]:
        print("NOTE: %d events had an unreadable origin and are excluded from the comparison."
              % rows["unknown"][0])
    print("one-gate (slope only):  seed %.2f  vs rewritten %.2f  per event" % (s_gate1, d_gate1))
    print("two-gate (+attributed): seed %.2f  vs rewritten %.2f  per event" % (s_rate, d_rate))
    print("                        (%d and %d wall(s) were probed; their verdicts are in the record)"
          % (sp_, dp))
    # A two-gate figure ABOVE the one-gate figure is arithmetically impossible for a subset, so say so
    # rather than printing it: that is exactly how the `+ walls_probed` bug showed itself (0.33 > 0.27).
    for label, g1, g2 in (("seed", s_gate1, s_rate), ("rewritten", d_gate1, d_rate)):
        if g2 > g1:
            print("  BUG: %s two-gate %.2f EXCEEDS one-gate %.2f -- gate 2 is a subset of gate 1, so"
                  % (label, g2, g1))
            print("  the attributed count is crediting walls that did not clear gate 1. Do not read on.")
            return 1
    if s_rate == 0.0 and d_rate == 0.0:
        print()
        print("NEITHER ORIGIN PRODUCED A CONFIRMED ATTRIBUTED WALL. The one-gate figures above are what")
        print("an earlier version of this probe published as 'actionable' and they are an OVERCOUNT: a")
        print("wall can clear monotone and gain and then fail attribution, which runs afterwards. So")
        print("seed-vs-rewrite is not the operative question -- the binding gate is ATTRIBUTION, and it")
        print("is unconfirmed or failing for BOTH origins. Price that before pricing either origin.")
    elif dn and d_rate == 0.0 and s_rate > 0.0:
        print()
        print("REWRITTEN CANDIDATES YIELD NO ATTRIBUTED WALL AT ALL (%d events), against %.2f per event"
              % (dn, s_rate))
        print("for seeds. So loop C is the WORST venue for this mechanism, not the best, and the lever")
        print("for C2 coverage is the SEED stage (more seeds, wider initial domains).")
    elif dn:
        print()
        print("Judge 'wait for the rewrites' against the TWO-GATE ratio, not the raw wall counts.")
    else:
        print()
        print("No rewrite-origin attribution events yet -- the question is NOT YET ANSWERABLE, which")
        print("is different from an answer of zero.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
