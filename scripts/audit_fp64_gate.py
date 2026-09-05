"""Audit: what did the fp64 relative correctness gate actually do?

Reads BOTH sources, because they cover different runs:

- `runs/*/jobs/*.out.json` -- the worker's own result files, which carry
  `fp64_gate_enabled` / `fp64_rescued_trials` / `fp64_rescue_metrics` from the moment
  the worker-side change landed;
- `TRIAL_DONE.trial.fp64_rescued_trials` in `events.jsonl`, which only exists for runs
  started after the driver-side journalling landed.

A run started between the two carries the data only in the job files, so the audit
prefers the event log and falls back to the job files rather than reporting nothing.

The question the gate has to answer to justify staying on:

  Does it admit candidates whose error is comparable to the REFERENCE's own error
  (ratio near 1.0), and do those candidates then survive the final re-eval?

A rescued candidate that fails `final_reeval_ok` means the multiplier is too loose and
the flag should go back off.

Usage:  python scripts/audit_fp64_gate.py [--runs runs]
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

PRECISION_KNOB_NAMES = (
    "COMPUTE_DTYPE", "DOT_PRECISION", "PRECISION", "DTYPE", "ACC_DTYPE",
    "INPUT_PRECISION", "MATMUL_PRECISION", "COMPUTE_PRECISION",
)


def load_events(run: Path) -> list[dict]:
    f = run / "events.jsonl"
    if not f.exists():
        return []
    out = []
    for line in f.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return out


def gate_config(run: Path) -> dict:
    m = run / "manifest.json"
    if not m.exists():
        return {}
    try:
        cfg = json.load(open(m, encoding="utf-8")).get("config") or {}
    except (OSError, json.JSONDecodeError):
        return {}
    return cfg.get("evaluation") or {}


def from_job_files(run: Path) -> tuple[int, int, list[dict]]:
    """(results with the gate on, trials rescued, rescue metric dicts)."""
    on = 0
    rescued = 0
    metrics = []
    for f in sorted((run / "jobs").glob("*.out.json")):
        try:
            r = json.load(open(f, encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not r.get("fp64_gate_enabled"):
            continue
        on += 1
        n = r.get("fp64_rescued_trials") or 0
        rescued += n
        if n and r.get("fp64_rescue_metrics"):
            metrics.append(dict(r["fp64_rescue_metrics"], _n=n))
    return on, rescued, metrics


def from_events(events: list[dict]) -> tuple[int, int]:
    """(trials carrying the field, trials rescued) from the event log."""
    carried = 0
    rescued = 0
    for e in events:
        if e.get("type") != "TRIAL_DONE":
            continue
        t = (e.get("payload") or {}).get("trial") or {}
        v = t.get("fp64_rescued_trials")
        if v is None:
            continue
        carried += 1
        rescued += v
    return carried, rescued


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    args = ap.parse_args()

    runs = [d for d in sorted(Path(args.runs).glob("run-*")) if (d / "events.jsonl").exists()]
    any_on = False
    all_ratios: list[float] = []
    verdicts: list[tuple[str, str, bool | None, int]] = []

    for run in runs:
        cfg = gate_config(run)
        if not cfg.get("fp64_relative_gate"):
            continue
        any_on = True
        events = load_events(run)
        job_on, job_rescued, metrics = from_job_files(run)
        ev_carried, ev_rescued = from_events(events)

        # Opportunities: only a trial that FAILED the absolute gate can be rescued, and
        # only a low-precision candidate produces those failures in practice.
        spaces = [e["payload"]["space"] for e in events if e.get("type") == "SPACE_PUBLISHED"]
        with_knob = sum(1 for s in spaces
                        if any(d["name"] in PRECISION_KNOB_NAMES for d in s.get("domains", [])))
        trials = [(e.get("payload") or {}).get("trial") or {}
                  for e in events if e.get("type") == "TRIAL_DONE"]
        mismatches = sum(1 for t in trials if t.get("failure_kind") == "correctness_mismatch")

        rescued = ev_rescued if ev_carried else job_rescued
        source = "event log" if ev_carried else "job files"

        print(f"\n=== {run.name} ===")
        print(f"  gate: on, multiplier {cfg.get('fp64_rel_multiplier')} "
              f"({cfg.get('fp64_rel_multiplier_lowp')} for low precision)")
        print(f"  spaces {len(spaces)} ({with_knob} with a precision knob), "
              f"trials {len(trials)}, correctness_mismatch {mismatches}")
        print(f"  results with the gate armed: {job_on}")
        print(f"  trials rescued by the relative arm: {rescued}   (source: {source})")
        if not mismatches and not rescued:
            print("  -> no opportunity yet: nothing failed the absolute gate on "
                  "correctness, so the arm had no occasion to fire")

        for m in metrics:
            try:
                all_ratios.append(float(m["ratio_to_reference"]))
            except (KeyError, TypeError, ValueError):
                pass
        if metrics:
            print("  rescue metrics (ratio = candidate error / the REFERENCE's own error):")
            for m in metrics[:12]:
                print(f"     x{m.get('_n')}  multiplier {m.get('multiplier')}  "
                      f"ratio {m.get('ratio_to_reference')}  "
                      f"ref {m.get('reference_rmse_vs_fp64')} "
                      f"cand {m.get('candidate_rmse_vs_fp64')}")

        # The falsification test: did a rescued candidate survive the final re-eval?
        fin = [e for e in events if e.get("type") == "RUN_FINISHED"]
        if fin and rescued:
            best = ((fin[0].get("payload") or {}).get("summary") or {}).get("best") or {}
            verdicts.append((run.name, str(best.get("candidate_id")),
                             best.get("final_reeval_ok"), rescued))

    if not any_on:
        print("no run has the fp64 relative gate enabled")
        return 0

    print("\n" + "=" * 74)
    if all_ratios:
        print(f"rescued-candidate error ratios: n={len(all_ratios)}  "
              f"median {statistics.median(all_ratios):.3f}  "
              f"min {min(all_ratios):.3f}  max {max(all_ratios):.3f}")
        print("  A ratio near 1.0 means the candidate is about as accurate as the")
        print("  reference itself -- the case the gate exists to admit. Ratios crowding")
        print("  the multiplier (2.0 / 3.0) are the ones to be suspicious of.")
    else:
        print("no trial has been rescued yet, so there is nothing to judge")

    if verdicts:
        print("\nfinal re-eval of runs whose search used a rescue:")
        bad = 0
        for name, cid, ok, n in verdicts:
            flag = "" if ok else "   <-- FAILED the final re-eval"
            if not ok:
                bad += 1
            print(f"  {name[-13:]}  best {cid}  final_reeval_ok={ok}  rescues {n}{flag}")
        if bad:
            print(f"\n  {bad} run(s) rescued candidates and then failed the final "
                  f"re-eval: the multiplier is too loose, turn fp64_relative_gate off.")
        else:
            print("\n  every run that used a rescue still passed its final re-eval.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
