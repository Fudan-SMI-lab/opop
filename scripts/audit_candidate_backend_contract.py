"""Does any candidate on disk violate the Backend contract?

Two structural violations, both of which reached a published space and one of
which became a run's best candidate before being caught
(docs/finding-candidate-delegates-to-baseline-compiler.md):

  1. NO KERNEL      -- no @triton.jit and no inline CUDA extension.
  2. DELEGATES      -- calls torch.compile / torch.jit.script / torch.jit.trace,
                       i.e. hands the work to the compiler that IS the baseline.

Both are now hard errors in every code-producing agent's check_output, so new
candidates cannot carry them. This script is the retrospective half: it sweeps
every candidate already on disk, reports which runs are affected, and -- crucially
-- says whether a violating candidate was ever a run's BEST, since that is the
difference between wasted budget and a number that must not be reported.

It also cross-checks the dynamic evidence where available: a candidate whose
trials only ever launched a copy-shaped kernel is the signature of the observed
case, and `profile.kernel_names` records it.

Usage: python scripts/audit_candidate_backend_contract.py [runs_dir]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from kernel_optimizer.paramspace.triton_lint import (  # noqa: E402
    declares_no_custom_kernel,
    delegates_to_baseline_compiler,
)


def main() -> int:
    runs_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "runs")
    rows: list[dict] = []
    n_scanned = 0

    for run in sorted(runs_dir.iterdir()):
        if not run.is_dir():
            continue
        cand_dir = run / "candidates"
        if not cand_dir.is_dir():
            continue

        # best latency per candidate, and the run's overall best
        best: dict[str, float] = {}
        kernels: dict[str, set[str]] = {}
        ev_path = run / "events.jsonl"
        if ev_path.exists():
            for line in ev_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                e = json.loads(line)
                if e["type"] != "TRIAL_DONE":
                    continue
                t = e["payload"].get("trial") or e["payload"]
                if t.get("status") != "complete" or not t.get("latency_ms"):
                    continue
                cid = t["candidate_id"]
                best[cid] = min(best.get(cid, float("inf")), t["latency_ms"]["mean"])
                names = ((t.get("profile") or {}).get("kernel_names")) or []
                kernels.setdefault(cid, set()).update(names)
        run_best = min(best.values()) if best else None

        for f in sorted(cand_dir.glob("cand-*/source.py")):
            n_scanned += 1
            src = f.read_text(encoding="utf-8", errors="replace")
            no_kernel = declares_no_custom_kernel(src)
            delegates = delegates_to_baseline_compiler(src)
            if not (no_kernel or delegates):
                continue
            cid = f.parent.name
            rows.append({
                "run": run.name,
                "cand": cid,
                "no_kernel": bool(no_kernel),
                "delegates": bool(delegates),
                "best": best.get(cid),
                "was_run_best": (run_best is not None and cid in best
                                 and abs(best[cid] - run_best) < 1e-9),
                "kernels": sorted(kernels.get(cid, ())),
                "abandoned": (run / "ABANDONED.md").exists(),
            })

    print(f"candidates scanned: {n_scanned}")
    print(f"violating the Backend contract: {len(rows)}\n")
    if not rows:
        print("clean.")
        return 0

    hdr = (f"{'run':34s} {'candidate':16s} {'no-kernel':>9s} {'delegates':>9s} "
           f"{'best_ms':>8s} {'was run best':>12s}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['run']:34s} {r['cand']:16s} "
              f"{('yes' if r['no_kernel'] else '-'):>9s} "
              f"{('yes' if r['delegates'] else '-'):>9s} "
              f"{(f'{r["best"]:.2f}' if r['best'] else '-'):>8s} "
              f"{('**YES**' if r['was_run_best'] else 'no'):>12s}")

    print()
    for r in rows:
        if r["kernels"]:
            print(f"  {r['cand']} launched kernels: {r['kernels']}")

    tainted = [r for r in rows if r["was_run_best"] and not r["abandoned"]]
    if tainted:
        print("\n*** A violating candidate is the BEST of a run that is NOT marked "
              "ABANDONED. Its headline number must not be reported: ***")
        for r in tainted:
            print(f"      {r['run']}  {r['cand']}  {r['best']:.2f} ms")
        return 1
    marked = [r for r in rows if r["abandoned"]]
    if marked:
        print(f"\nall {len(marked)} violation(s) are in runs marked ABANDONED.md — "
              f"no reportable result is affected.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
