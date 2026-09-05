"""Audit: does a rewrite's KERNEL-COUNT direction predict whether it beats its parent?

`TrialRecord.profile.kernel_names` comes from Triton's compile metadata, so the number of
kernels a candidate launches is observed rather than claimed by the agent. Comparing each
child's tuned best against its parent's, grouped by whether the child FUSED (fewer kernels),
SPLIT (more), or kept the same count.

Measured: fusing improved 6/6 (+14.9% median), splitting 2/11 (-21.5%). But the sign REVERSES
by task -- L3:21 splits improved 2/3 (+16.6%), including the project's best L3:21 kernel -- so
the per-task table is the one to read.

The separator is the PARENT's REGISTER occupancy, not its shared memory (I had that wrong at
first): splits from parents at >=200 of 255 registers are 0 for 6 with a -37% median, while the
shared-memory cut leaves 7 failures on its low side. A register-saturated kernel holds its live
state in registers, so splitting forces that state out to HBM and back with no budget to pay
with; the cost shows up as child-side spills. Low occupancy makes a split possible, not good --
the low-register group is only 2 improved of 5.

Two guards on the measurement, both printed:
  - kernel count must be stable per candidate (a union across trials would inflate it)
  - zero-kernel candidates are excluded: a triton candidate launching no kernel is the
    dead-mode-branch bug, not a fusion

See docs/result-kernel-count-direction-predicts-outcome.md

Usage:  python scripts/audit_kernel_count_moves.py [runs/run-... ...]
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from statistics import median
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def main(argv: list[str]) -> int:
    runs = [Path(a) for a in argv] or sorted(REPO.glob("runs/run-l3-*"))

    groups: dict[tuple[str, str], list[float]] = defaultdict(list)
    profiles: list[tuple[str, str, str, str, float, float, dict, dict, int, int]] = []
    unstable = candidates_seen = 0
    zero_kernel = 0

    for run in runs:
        ev_path = run / "events.jsonl"
        if not ev_path.exists():
            continue
        parts = run.name.split("-")
        task = f"{parts[1]}:{parts[2]}" if len(parts) > 2 else run.name

        best: dict[str, float] = {}
        parent_of: dict[str, str | None] = {}
        names: dict[str, set[str]] = defaultdict(set)
        counts: dict[str, list[int]] = defaultdict(list)
        best_trial: dict[str, tuple[float, dict]] = {}

        for line in ev_path.open(encoding="utf-8"):
            e = json.loads(line)
            if e["type"] == "TUNING_DONE":
                cid, ms = e["payload"].get("candidate_id"), e["payload"].get("best_ms")
                if cid and ms is not None:
                    best[cid] = min(best.get(cid, float("inf")), ms)
            elif e["type"] == "CANDIDATE_REGISTERED":
                c = e["payload"].get("candidate") or e["payload"]
                parent_of[c["candidate_id"]] = (c.get("parent_ids") or [None])[0]
            elif e["type"] == "TRIAL_DONE":
                t = e["payload"]["trial"]
                prof = t.get("profile") or {}
                kn = prof.get("kernel_names")
                if kn:
                    names[t["candidate_id"]] |= set(kn)
                    counts[t["candidate_id"]].append(len(kn))
                if t.get("status") == "complete" and prof and t.get("latency_ms"):
                    lat = t["latency_ms"]["mean"]
                    cur = best_trial.get(t["candidate_id"])
                    if cur is None or lat < cur[0]:
                        best_trial[t["candidate_id"]] = (lat, prof)

        for cid, seen in counts.items():
            candidates_seen += 1
            if len(set(seen)) > 1:
                unstable += 1

        for cid, par in parent_of.items():
            if not par or cid not in best or par not in best:
                continue
            n_child, n_parent = len(names.get(cid, ())), len(names.get(par, ()))
            if n_child == 0 or n_parent == 0:
                zero_kernel += 1
                continue
            delta = (best[par] - best[cid]) / best[par] * 100.0
            label = ("FUSED" if n_child < n_parent
                     else "SPLIT" if n_child > n_parent else "same")
            groups[(task, label)].append(delta)
            groups[("ALL", label)].append(delta)
            if label != "same" and cid in best_trial and par in best_trial:
                lp, pp = best_trial[par]
                lc, pc = best_trial[cid]
                profiles.append((task, label, par, cid, lp, lc, pp, pc, n_parent, n_child))

    print("=" * 78)
    print("KERNEL-COUNT DIRECTION vs whether the child beat its parent")
    print("=" * 78)
    print(f"  {'task':<8} {'change':<6} {'n':>4} {'median':>9} {'improved':>10}")
    for key in sorted(groups, key=lambda k: (k[0] != "ALL", k[0], k[1])):
        arr = groups[key]
        print(f"  {key[0]:<8} {key[1]:<6} {len(arr):>4} {median(arr):>+8.1f}% "
              f"{sum(1 for x in arr if x > 0):>7}/{len(arr)}")
    print("\n  The ALL rows hide a sign reversal -- read the per-task rows. Splitting pays on a")
    print("  task whose parents have resource headroom and costs on one whose parents are")
    print("  saturated, which is the mechanism the profiles below show.")

    print()
    print("=" * 78)
    print("GUARDS on the measurement")
    print("=" * 78)
    print(f"  candidates whose per-trial kernel COUNT varies: {unstable} of {candidates_seen}"
          f"   (a varying count would inflate the union)")
    print(f"  parent/child pairs excluded for a ZERO-kernel side: {zero_kernel}"
          f"   (dead-mode-branch bug, not a fusion)")

    if profiles:
        print()
        print("=" * 78)
        print("MECHANISM -- profile of the PARENT's best trial, sorted by register occupancy")
        print("=" * 78)
        print(f"  {'task':<7} {'move':<6} {'delta':>8} {'p_regs':>7} {'p_shared':>9} "
              f"{'p_spill':>8}  parent")
        for task, label, par, cid, lp, lc, pp, pc, nkp, nkc in sorted(
                profiles, key=lambda r: (r[1], r[6].get("n_regs") or 0)):
            delta = (lp - lc) / lp * 100.0 if lp else 0.0
            print(f"  {task:<7} {label:<6} {delta:>+7.1f}% {str(pp.get('n_regs')):>7} "
                  f"{str(pp.get('shared_bytes')):>9} {str(pp.get('n_spills')):>8}  {par}")

        # Which parent resource actually separates a winning split from a losing one?
        splits = [(r, (r[4] - r[5]) / r[4] * 100.0 if r[4] else 0.0)
                  for r in profiles if r[1] == "SPLIT"]
        if splits:
            print()
            print("  SEPARATOR TEST on splits -- which parent resource predicts the outcome?")
            for field, limit, thresholds in (("n_regs", 255, (200,)),
                                             ("shared_bytes", 101376, (70000,))):
                for thr in thresholds:
                    hi = [d for r, d in splits if (r[6].get(field) or 0) >= thr]
                    lo = [d for r, d in splits if (r[6].get(field) or 0) < thr]
                    print(f"    parent {field} >= {thr:<7} n={len(hi):<2} "
                          f"improved {sum(1 for x in hi if x > 0)}  "
                          f"median {median(hi) if hi else 0:+.1f}%")
                    print(f"    parent {field} <  {thr:<7} n={len(lo):<2} "
                          f"improved {sum(1 for x in lo if x > 0)}  "
                          f"median {median(lo) if lo else 0:+.1f}%")
            print("    -> registers separate it (0 of 6 above 200 improved); shared memory does")
            print("       not (7 failures sit BELOW its threshold). Every high-register parent is")
            print("       at 248-255, so the data cannot distinguish a 200 cut from a 240 one.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
