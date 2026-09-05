"""Audit: `latency_by_value` reports a MEDIAN, and it disagrees with the minimum 45% of the time.

`_param_stat` reports the median latency per choice. The analyst reads that table to decide
which knobs are worth pushing, so what it ranks best matters.

The case that prompted this, on run-l3-21-20260905-195615 sp-274540fa:

    GEMM_BLOCK_N=256   n=4    median 20.55   min 14.70   <- the run's best trial
    GEMM_BLOCK_N=128   n=19   median 17.20   min 15.60

The median ranks 256 WORST of five values while its minimum is the best result the run had
produced. The tuner found it; the statistic then tells the analyst it did not work.

MY FIRST HYPOTHESIS WAS WRONG, and this script is what refuted it. I proposed that the median
systematically penalises newly-ADDED values, since a value the TPE has just started sampling
has mostly exploratory trials. Measured over 1035 knobs, the median's winner has FEWER trials
in 336 cases and MORE in only 106 -- the opposite direction. A small-n value gets a lucky
median about as readily as a lucky minimum, so there is no systematic low-n penalty.

What survives is narrower and still worth knowing: median and min disagree on the best value
for **45%** of knobs. GEMM_BLOCK_N=256 is a real instance of the shape I described, just not
the dominant pattern. The script reports both so the claim cannot drift back.

Usage:  python scripts/audit_latency_by_value_median_bias.py [--runs runs]
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--min-values", type=int, default=3,
                    help="only consider knobs with at least this many measured values")
    args = ap.parse_args()

    disagreements = []
    checked = 0
    n_of_median_winner = []
    n_of_min_winner = []

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

        # space_id -> knob -> value -> [latencies]
        per_space: dict[str, dict[str, dict[str, list[float]]]] = defaultdict(
            lambda: defaultdict(lambda: defaultdict(list)))
        for e in events:
            if e.get("type") != "TRIAL_DONE":
                continue
            t = (e.get("payload") or {}).get("trial") or {}
            if t.get("status") != "complete":
                continue
            lm = t.get("latency_ms")
            ms = lm.get("mean") if isinstance(lm, dict) else lm
            sid = t.get("space_id")
            if not ms or not sid:
                continue
            for k, v in ((t.get("params") or {}).get("values") or {}).items():
                per_space[sid][k][repr(v)].append(float(ms))

        for sid, knobs in per_space.items():
            for knob, byval in knobs.items():
                if len(byval) < args.min_values:
                    continue
                checked += 1
                med = {v: statistics.median(x) for v, x in byval.items()}
                mn = {v: min(x) for v, x in byval.items()}
                med_winner = min(med, key=lambda v: med[v])
                min_winner = min(mn, key=lambda v: mn[v])
                n_of_median_winner.append(len(byval[med_winner]))
                n_of_min_winner.append(len(byval[min_winner]))
                if med_winner != min_winner:
                    disagreements.append((
                        d.name[-13:], sid, knob,
                        med_winner, len(byval[med_winner]), med[med_winner], mn[med_winner],
                        min_winner, len(byval[min_winner]), med[min_winner], mn[min_winner],
                    ))

    print(f"knobs examined (>= {args.min_values} measured values): {checked}")
    print(f"where the MEDIAN's best value differs from the MINIMUM's: "
          f"{len(disagreements)}  ({len(disagreements) / checked * 100:.0f}%)")

    if n_of_median_winner:
        print(f"\ntrials behind the median's winner : median {statistics.median(n_of_median_winner):.0f}")
        print(f"trials behind the minimum's winner: median {statistics.median(n_of_min_winner):.0f}")

    # The bias claim: is the minimum's winner systematically LESS sampled?
    fewer = sum(1 for a, b in zip(n_of_min_winner, n_of_median_winner) if a < b)
    more = sum(1 for a, b in zip(n_of_min_winner, n_of_median_winner) if a > b)
    print(f"\nthe minimum's winner had FEWER trials than the median's in {fewer} cases, "
          f"MORE in {more}")
    if fewer > more:
        print("'fewer' dominates: the median favours well-sampled (older) values, which")
        print("would be a bias against every value improvement K adds.")
    else:
        print("'more' dominates, so there is NO systematic low-n penalty -- the median does")
        print("not favour well-sampled values. That refutes the hypothesis this script was")
        print("written to test; what stands is only the disagreement RATE above.")

    print(f"\nworst disagreements (median's pick vs minimum's pick):")
    disagreements.sort(key=lambda r: r[10] - r[6])
    for r in disagreements[:12]:
        run, sid, knob = r[0], r[1], r[2]
        print(f"  {run}  {sid}  {knob}")
        print(f"     median picks {r[3]:>6s} (n={r[4]:2d}, med {r[5]:6.2f}, min {r[6]:6.2f})")
        print(f"     minimum picks {r[7]:>6s} (n={r[8]:2d}, med {r[9]:6.2f}, min {r[10]:6.2f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
