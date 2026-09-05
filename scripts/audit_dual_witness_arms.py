"""Audit: is the dual-precision witness's tf32 arm ever the more permissive comparison?

Improvement A accepts a candidate when it matches EITHER the ieee reference or the tf32
reference. This walks every recorded correctness rejection and asks which arm was the
looser test -- i.e. whether the second witness has ever mattered.

Generic over runs and tasks: it reads only the dual-witness metric block that
`worker_main._relaxed_metrics` writes into `failure_detail`, so it works on any task
whose candidates declare any precision knob (or none).

Usage:  python scripts/audit_dual_witness_arms.py [--runs runs]
"""
from __future__ import annotations

import argparse
import ast
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

# Any name a parameterizer has used for a precision knob. Only used to LABEL rows;
# the audit's verdict never depends on it.
PRECISION_KNOB_NAMES = (
    "COMPUTE_DTYPE", "DOT_PRECISION", "PRECISION", "DTYPE", "ACC_DTYPE",
    "INPUT_PRECISION", "MATMUL_PRECISION",
)

WITNESS_TAGS = (("vs ieee ref:", "ieee"), ("vs tf32 ref:", "tf32"),
                ("floor, NOT a bug):", "floor"))


def parse_witness_metrics(detail: str) -> dict[str, dict] | None:
    """Pull the three metric dicts out of a rich dual-witness failure message."""
    out: dict[str, dict] = {}
    for tag, key in WITNESS_TAGS:
        i = detail.find(tag)
        if i < 0:
            return None
        j = detail.find("{", i)
        k = detail.find("}", j)
        if j < 0 or k < 0:
            return None
        try:
            out[key] = ast.literal_eval(detail[j:k + 1])
        except (ValueError, SyntaxError):
            return None
    return out


def task_of(run_name: str) -> str:
    """Best-effort task label from a run directory name; falls back to the name."""
    parts = run_name.split("-")
    for i, p in enumerate(parts):
        if p.startswith("l") and i + 1 < len(parts) and parts[i + 1].isdigit():
            return f"{p}:{parts[i + 1]}"
    return run_name


def collect(runs_dir: Path) -> list[tuple[str, str, str | None, dict]]:
    rows = []
    for d in sorted(runs_dir.glob("run-*")):
        f = d / "events.jsonl"
        if not f.exists():
            continue
        task = task_of(d.name)
        for line in f.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                continue
            if e.get("type") != "TRIAL_DONE":
                continue
            t = (e.get("payload") or {}).get("trial") or {}
            if t.get("failure_kind") != "correctness_mismatch":
                continue
            m = parse_witness_metrics(str(t.get("failure_detail") or ""))
            if not m:
                continue
            values = (t.get("params") or {}).get("values") or {}
            knob = next((p for p in PRECISION_KNOB_NAMES if p in values), None)
            rows.append((task, t.get("candidate_id", "?"),
                         values.get(knob) if knob else None, m))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    args = ap.parse_args()

    rows = collect(Path(args.runs))
    if not rows:
        print("no rejections with dual-witness metrics found")
        return 0

    print(f"rejections carrying full dual-witness metrics: {len(rows)}\n")

    print("=== which arm was the MORE PERMISSIVE comparison? ===")
    verdict_flipped = 0
    for crit in ("frac_within_tol", "cosine"):
        tf = sum(1 for *_, m in rows if m["tf32"][crit] > m["ieee"][crit])
        ie = sum(1 for *_, m in rows if m["tf32"][crit] < m["ieee"][crit])
        print(f"  {crit:16s}  tf32 better {tf:5d}   ieee better {ie:5d}   "
              f"tie {len(rows) - tf - ie}")
        verdict_flipped += tf

    print("\n=== per task / candidate dtype ===")
    per: dict[tuple, list[int]] = defaultdict(lambda: [0, 0])
    for task, _cid, dt, m in rows:
        k = (task, str(dt))
        per[k][0] += 1
        if m["tf32"]["frac_within_tol"] > m["ieee"]["frac_within_tol"]:
            per[k][1] += 1
    for k in sorted(per):
        n, better = per[k]
        print(f"  {k[0]:8s} dtype={k[1]:6s}  n={n:5d}  tf32-arm-better={better}")

    print("\n=== would either arm ALONE have passed the gate? ===")
    # Thresholds are the job defaults; a run that overrode them is still ranked
    # correctly because both arms are scored against the same numbers.
    tf_pass = sum(1 for *_, m in rows
                  if m["tf32"]["frac_within_tol"] > 0.99
                  and m["tf32"]["cosine"] >= 0.99985)
    ie_pass = sum(1 for *_, m in rows
                  if m["ieee"]["frac_within_tol"] > 0.99
                  and m["ieee"]["cosine"] >= 0.99985)
    print(f"  tf32 arm alone: {tf_pass}      ieee arm alone: {ie_pass}")

    ratios = [float(m["tf32"]["median_rel_err"]) / float(m["ieee"]["median_rel_err"])
              for *_, m in rows if float(m["ieee"]["median_rel_err"]) > 0]
    if ratios:
        print("\n=== deviation ratio (vs-tf32 / vs-ieee) ===")
        print(f"  median {statistics.median(ratios):.3f}   "
              f"independent-rounding prediction sqrt(2)={2 ** 0.5:.3f}   n={len(ratios)}")
        gt = sum(1 for r in ratios if r > 1.0)
        print(f"  ratio > 1 (tf32 arm stricter) in {gt}/{len(ratios)} "
              f"= {gt / len(ratios):.1%}")

    print()
    if verdict_flipped == 0:
        print(f"CONCLUSION: the tf32 arm was never the permissive one in {len(rows)} "
              f"rejections.")
        print("The second witness has not changed a verdict; see "
              "docs/finding-tf32-witness-is-never-the-permissive-one.md")
    else:
        print(f"CONCLUSION: the tf32 arm was the permissive one in {verdict_flipped} "
              f"case(s) -- the finding no longer holds universally.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
