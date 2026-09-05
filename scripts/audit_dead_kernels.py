"""Audit: did any candidate's tuning budget measure a kernel that never ran?

A `@triton.jit` kernel defined in a candidate but launched by NO trial is dead code, and
every latency number for that candidate measures a different path. The failure is silent
by construction — correctness passes on the live branch, timings are produced, the
candidate is published — so nothing in the event log flags it. This finds it after the
fact, and `KERNELS_NEVER_LAUNCHED` (improvement M) reports it live from the next run on.

Two exclusions keep the result factual:
  - a candidate whose trials carry NO kernel names at all is skipped (CUDA backend, or
    profiling unavailable): absence of data is not absence of launches;
  - `@triton.jit` DEVICE HELPERS — called by name from inside another kernel, hence
    inlined by Triton — never appear in kernel_names even though they run every trial.

Usage:  python scripts/audit_dead_kernels.py [runs/run-... ...]
        (no args = every run under runs/)
See docs/finding-optimization-behind-a-dead-mode-branch.md
"""

from __future__ import annotations

import ast
import json
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from kernel_optimizer.paramspace.triton_lint import (  # noqa: E402
    device_helper_names,
    jit_kernel_names,
    lint_triton_source,
)


def host_path(p: str) -> Path:
    """Translate a WSL /mnt/d/... path from a job file back to a Windows path."""
    return Path(p[5].upper() + ":/" + p[7:]) if p.startswith("/mnt/") else Path(p)


def candidate_source(run: Path, cid: str) -> str | None:
    for job in sorted(run.glob(f"jobs/{cid}-wit-default-eval-*.json")):
        if job.name.endswith(".out.json"):
            continue
        try:
            src = host_path(json.loads(job.read_text(encoding="utf-8"))["kernel_src_path"])
        except (OSError, ValueError, KeyError):
            continue
        if src.exists():
            return src.read_text(encoding="utf-8")
    return None


def audit_run(run: Path) -> list[dict]:
    events = run / "events.jsonl"
    if not events.exists():
        return []
    trials = []
    for line in events.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        ev = json.loads(line)
        if ev["type"] == "TRIAL_DONE":
            trials.append(ev["payload"].get("trial") or ev["payload"])
    findings = []
    for cid in sorted({t.get("candidate_id") for t in trials if t.get("candidate_id")}):
        source = candidate_source(run, cid)
        if source is None:
            continue
        try:
            tree = ast.parse(source)
        except SyntaxError:
            continue
        defined = jit_kernel_names(tree)
        if not defined:
            continue  # CUDA backend or no jit kernels: nothing to check
        mine = [t for t in trials if t.get("candidate_id") == cid]
        launched: set[str] = set()
        has_names = False
        for t in mine:
            names = (t.get("profile") or {}).get("kernel_names") or []
            if names:
                has_names = True
                launched |= set(names)
        if not has_names:
            continue  # no metadata: cannot conclude anything
        dead = defined - launched - device_helper_names(tree, defined)
        if not dead:
            continue
        complete = [t for t in mine if t.get("status") == "complete"]
        best = min((t["latency_ms"]["mean"] for t in complete if t.get("latency_ms")),
                   default=None)
        _hard, warns = lint_triton_source(source)
        findings.append({
            "run": run.name,
            "candidate_id": cid,
            "never_launched": sorted(dead),
            "launched": sorted(launched),
            "n_trials": len(mine),
            "best_ms": best,
            # Whether the STATIC lint would also have caught it. It cannot when the
            # launch hides inside a host wrapper, which is why both checks exist.
            "static_lint_fires": any("if ...training:" in w for w in warns),
        })
    return findings


def main(argv: list[str]) -> int:
    runs = [Path(a) for a in argv] or sorted(REPO.glob("runs/run-*"))
    all_findings: list[dict] = []
    checked = 0
    for run in runs:
        if not (run / "events.jsonl").exists():
            continue
        found = audit_run(run)
        checked += 1
        all_findings.extend(found)

    if not all_findings:
        print(f"No never-launched kernels across {checked} run(s).")
        return 0

    print(f"{len(all_findings)} candidate(s) across {checked} run(s) spent a tuning "
          f"budget with a kernel that never ran:\n")
    for f in all_findings:
        static = "static lint fires" if f["static_lint_fires"] else "STATIC LINT BLIND"
        print(f"  {f['run']}  {f['candidate_id']}")
        print(f"      never launched: {f['never_launched']}")
        print(f"      launched:       {f['launched']}")
        print(f"      {f['n_trials']} trials, best {f['best_ms']} ms  ({static})")
    blind = sum(1 for f in all_findings if not f["static_lint_fires"])
    print(f"\n{blind} of {len(all_findings)} would be missed by the static lint alone "
          f"(launch hidden in a host wrapper): the runtime check is not redundant.")
    print("Trials wasted: " + str(sum(f["n_trials"] for f in all_findings)))
    print("By run: " + str(dict(Counter(f["run"] for f in all_findings))))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
