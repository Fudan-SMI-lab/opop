"""Is `precision: "unknown"` ever hiding a tensor-core candidate?

`_honest_verdict` compares a candidate against the SAME-PRECISION baseline. When
`_detect_candidate_precision` returns "unknown" it falls back to the ieee baseline
(`torch_compile`), which is the slower bar on a matmul-bound task -- L3:43's ieee
baseline is 35.40 ms against 18.40 ms at tf32, so a misclassified tensor-core
candidate would have its speedup reported against a strawman.

The fallback is correct only if "unknown" really means "no tensor-core arithmetic".
This script checks that directly: every candidate classified unknown is scanned for
tl.dot / float16 / bfloat16 / tf32. Any hit is a misclassification and a real
reporting defect.

It also reports, per finished run, whether the reported best was classified unknown
and how wide that task's ieee-vs-tf32 baseline gap is -- a wide gap is where the
fallback's choice actually moves the headline.

Usage: python scripts/audit_precision_classification.py [runs_dir]
"""
from __future__ import annotations

import ast
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from kernel_optimizer.control.orchestrator import (  # noqa: E402
    _detect_candidate_precision,
)
from kernel_optimizer.models.core import ParamSet  # noqa: E402

TENSOR_CORE_TOKENS = ("tl.dot", "float16", "bfloat16", "tf32")


def params_of(source: str) -> dict:
    m = re.search(r"^PARAMS\s*=\s*(\{.*?^\})", source, re.M | re.S)
    if not m:
        return {}
    try:
        got = ast.literal_eval(m.group(1))
        return got if isinstance(got, dict) else {}
    except (ValueError, SyntaxError):
        return {}


def main() -> int:
    runs_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "runs")
    unknown: list[tuple[str, str, list[str]]] = []
    n_total = 0

    for run in sorted(runs_dir.iterdir()):
        cand_dir = run / "candidates"
        if not cand_dir.is_dir():
            continue
        for f in sorted(cand_dir.glob("cand-*/source.py")):
            n_total += 1
            src = f.read_text(encoding="utf-8", errors="replace")
            prec = _detect_candidate_precision(src, ParamSet(values=params_of(src)))
            if prec != "unknown":
                continue
            hits = [t for t in TENSOR_CORE_TOKENS if t in src]
            unknown.append((run.name, f.parent.name, hits))

    print(f"candidates scanned: {n_total}")
    print(f"classified 'unknown': {len(unknown)}")
    bad = [u for u in unknown if u[2]]
    print(f"  of those containing {list(TENSOR_CORE_TOKENS)}: {len(bad)}\n")
    if bad:
        print("*** MISCLASSIFIED — timed against the ieee bar while using tensor cores: ***")
        for run, cid, hits in bad:
            print(f"      {run}  {cid}  contains {hits}")
    else:
        print("no misclassification: every 'unknown' candidate is genuinely dot-free.")

    print("\nfinished runs whose reported best was classified 'unknown':")
    any_row = False
    for run in sorted(runs_dir.iterdir()):
        ev_path = run / "events.jsonl"
        if not ev_path.exists():
            continue
        best = None
        base: dict[str, float] = {}
        for line in ev_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            e = json.loads(line)
            if e["type"] == "BASELINE_DONE":
                b = e["payload"].get("baseline") or e["payload"]
                lat = b.get("latency_ms") or {}
                if b.get("kind") and lat.get("mean"):
                    base[b["kind"]] = lat["mean"]
            elif e["type"] == "RUN_FINISHED":
                best = (e["payload"].get("summary") or {}).get("best") or {}
        if not best:
            continue
        hv = best.get("honest_verdict") or {}
        if hv.get("candidate_precision") != "unknown":
            continue
        any_row = True
        ieee, tf32 = base.get("torch_compile"), base.get("torch_compile_tf32")
        gap = (f"{ieee / tf32:.2f}x" if ieee and tf32 else "n/a")
        print(f"  {run.name}  best={best.get('candidate_id')} "
              f"reeval={best.get('final_reeval_ms')} "
              f"speedup={hv.get('same_precision_speedup')} "
              f"| ieee/tf32 baseline gap {gap}")
        if ieee and tf32 and ieee / tf32 > 1.25:
            print("      ^ WIDE baseline gap: the fallback's choice of bar materially "
                  "moves this headline. Verify the candidate by hand.")
    if not any_row:
        print("  (none)")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
