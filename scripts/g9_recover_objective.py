"""Recover the G9 A/B objective from the run's own worker results.

Needed because the A/B script read the latency under three names that do not exist
(`robust_ms`/`median_ms`/`mean_ms`; the stored keys are unsuffixed), so every candidate was recorded
as `ms: null`. The MEASUREMENTS THEMSELVES ARE INTACT -- the worker wrote each one to
`jobs/<tag>-eval-*.out.json` before the reporting line dropped it -- so the arms can be compared
without re-running a single agent call or re-timing a single kernel.

Joins on the job tag, which encodes arm/replicate/candidate: `g9-<arm>-r<rep>-c<idx>-eval-<hash>`.

Reports per arm, and says plainly what the sample size does and does not support: with one or two
calls per arm this is an ANECDOTE, and the noise floor on this task (per-trial std up to 16% of the
mean, near-tie bands spanning 5-9%) is wider than most differences it could show.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from kernel_optimizer.store.read import latency_ms_of  # noqa: E402

_TAG = re.compile(r"^g9-(?P<arm>[a-z]+)-r(?P<rep>\d+)-c(?P<cand>\d+)-eval-[0-9a-f]+\.out\.json$")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--results-json", default=None,
                    help="the A/B's own output, to merge agent-side facts (duration, cost)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    jobs = os.path.join(args.run_dir, "jobs")
    if not os.path.isdir(jobs):
        print("no jobs/ directory in %s" % args.run_dir)
        return 1

    rows: list[dict] = []
    for path in sorted(glob.glob(os.path.join(jobs, "g9-*-eval-*.out.json"))):
        m = _TAG.match(os.path.basename(path))
        if not m:
            continue
        with open(path, encoding="utf-8") as f:
            res = json.load(f)
        ms = latency_ms_of(res)
        rows.append({
            "arm": m.group("arm"),
            "replicate": int(m.group("rep")),
            "candidate": int(m.group("cand")),
            "ok": bool(res.get("ok")),
            "ms": ms,
            "failure_kind": res.get("failure_kind"),
            "n_samples": len((res.get("latency_ms") or {}).get("samples") or []),
            "std": (res.get("latency_ms") or {}).get("std"),
        })

    if not rows:
        # Loud, per G21: an empty read is not a finding.
        print("FAIL: no g9 eval results matched %s in %s -- the tag format may have changed."
              % (_TAG.pattern, jobs))
        return 1

    agent: dict[tuple[str, int], dict] = {}
    if args.results_json and os.path.isfile(args.results_json):
        for entry in json.load(open(args.results_json, encoding="utf-8")):
            agent[(entry["arm"], int(entry["replicate"]))] = entry

    print("PER-CANDIDATE (recovered from the worker's own result files)")
    print("%-8s %4s %4s %6s %10s %8s %6s" % ("arm", "rep", "cand", "ok", "median_ms", "std", "n"))
    for r in sorted(rows, key=lambda r: (r["arm"], r["replicate"], r["candidate"])):
        print("%-8s %4d %4d %6s %10s %8s %6d"
              % (r["arm"], r["replicate"], r["candidate"], r["ok"],
                 ("%.4f" % r["ms"]) if r["ms"] is not None else "-",
                 ("%.3f" % r["std"]) if isinstance(r["std"], (int, float)) else "-",
                 r["n_samples"]))

    print()
    print("PER ARM")
    print("%-8s %6s %8s %8s %11s %11s %10s %9s"
          % ("arm", "calls", "produced", "correct", "best_ms", "mean_best", "agent_s", "cost"))
    summary: dict[str, dict] = {}
    for arm in ("none", "verdict", "rich"):
        arm_rows = [r for r in rows if r["arm"] == arm]
        if not arm_rows:
            continue
        reps = sorted({r["replicate"] for r in arm_rows})
        timed = [r["ms"] for r in arm_rows if r["ms"] is not None]
        # Per-call best, then averaged: a call producing two kernels must not count twice.
        per_call_best = []
        for rep in reps:
            got = [r["ms"] for r in arm_rows if r["replicate"] == rep and r["ms"] is not None]
            if got:
                per_call_best.append(min(got))
        ags = [agent.get((arm, rep), {}).get("agent_s") for rep in reps]
        ags = [a for a in ags if isinstance(a, (int, float))]
        costs = [agent.get((arm, rep), {}).get("cost") for rep in reps]
        costs = [c for c in costs if isinstance(c, (int, float))]
        summary[arm] = {
            "calls": len(reps),
            "produced": len(arm_rows),
            "correct": sum(1 for r in arm_rows if r["ok"]),
            "best_ms": min(timed) if timed else None,
            "mean_of_per_call_best": statistics.fmean(per_call_best) if per_call_best else None,
            "per_call_best": per_call_best,
            "agent_s_mean": statistics.fmean(ags) if ags else None,
            "cost_total": sum(costs) if costs else None,
        }
        s = summary[arm]
        print("%-8s %6d %8d %8d %11s %11s %10s %9s"
              % (arm, s["calls"], s["produced"], s["correct"],
                 ("%.4f" % s["best_ms"]) if s["best_ms"] else "-",
                 ("%.4f" % s["mean_of_per_call_best"]) if s["mean_of_per_call_best"] else "-",
                 ("%.0f" % s["agent_s_mean"]) if s["agent_s_mean"] else "-",
                 ("%.4f" % s["cost_total"]) if s["cost_total"] is not None else "-"))

    print()
    n_calls = sum(s["calls"] for s in summary.values())
    print("READING THIS HONESTLY")
    print("  %d agent calls across %d arms. This is an ANECDOTE, not a measurement of the arms."
          % (n_calls, len(summary)))
    spread = [s["best_ms"] for s in summary.values() if s["best_ms"]]
    if len(spread) >= 2:
        rel = (max(spread) - min(spread)) / min(spread) * 100.0
        print("  best-of-arm spread: %.2f%%. The recorded near-tie band on this task is 5-9%% and"
              " per-trial std reaches 16%% of the mean," % rel)
        print("  so a spread below that is INDISTINGUISHABLE FROM NOISE and must not be read as an"
              " arm difference.")
        if rel < 5.0:
            print("  VERDICT: no arm separated. On this evidence the information content of the"
                  " rewriter's brief did not change the best kernel it produced.")
    print()
    print("  Also note: 3 of the completed calls ended in a transport ReadTimeout at the 1500 s"
          " ceiling (G20) and their")
    print("  candidates were recovered from the sandbox, so the arms differ in how much of the"
          " agent's own summary survived,")
    print("  not only in what they were told. That is a confound in the ARM COMPARISON, not a"
          " defect in the kernels measured.")

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump({"per_candidate": rows, "per_arm": summary}, f, indent=2)
        print()
        print("wrote %s" % args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
