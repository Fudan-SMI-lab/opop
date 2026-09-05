"""Audit: do the spaces' shared-memory constraints actually describe what fails?

Three questions, all answered from events.jsonl alone:

1. Of every trial that died with `OutOfResources: shared memory`, how many violated a
   constraint its own space declared? (A violation would mean the guard let an illegal
   config through; satisfaction means the constraint set does not describe the failure.)
2. For multi-kernel candidates, which kernel groups have NO shared-memory constraint?
3. Every shared-memory constraint hard-codes `* 4` bytes. On spaces that also expose a
   precision knob, how much of the fp16/bf16 (2-byte) region does that wrongly exclude?

Usage:  python scripts/audit_shared_memory_constraints.py [runs_dir]
"""

from __future__ import annotations

import collections
import itertools
import json
import random
import re
import sys
from pathlib import Path

# Device limits for this machine; the constraint expressions reference these by name.
LIMITS = {
    "MAX_THREADS_PER_BLOCK": 1024,
    "MAX_THREADS": 1024,
    "MAX_REGS_PER_THREAD": 255,
    "MAX_REGS": 255,
    "MAX_SHARED_BYTES_OPTIN": 101376,
    "MAX_SHARED_BYTES_STATIC": 49152,
    "MAX_SHARED_BYTES": 101376,
}

OOM_MARK = "out of resource: shared memory"


def satisfies(exprs: list[str], env: dict) -> bool | None:
    """True/False, or None if any expression could not be evaluated."""
    for expr in exprs:
        try:
            if not eval(expr, {"__builtins__": {}}, env):  # noqa: S307 - audit script
                return False
        except Exception:
            return None
    return True


def kernel_groups(names: list[str]) -> set[str]:
    """Knob-name prefixes that carry their own stage count = one launched kernel each."""
    groups = set()
    for name in names:
        m = re.match(r"^(.*?)_?(NUM_STAGES|PIPE_STAGES|STAGES)$", name)
        if m:
            groups.add(m.group(1))
    return groups


def dtype_knobs(names: list[str]) -> list[str]:
    return [n for n in names if "DTYPE" in n or "PRECISION" in n]


def load(run: Path) -> list[dict]:
    return [json.loads(l) for l in (run / "events.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]


def q1_oom_vs_constraints(runs: list[Path]) -> None:
    satisfied = violated = unevaluable = 0
    per_run: collections.Counter[str] = collections.Counter()
    per_cand: dict[str, list[int]] = collections.defaultdict(lambda: [0, 0])

    for run in runs:
        spaces: dict[str, dict] = {}
        for ev in load(run):
            if ev["type"] == "SPACE_PUBLISHED":
                sp = ev["payload"]["space"]
                spaces[sp["space_id"]] = sp
            elif ev["type"] == "TRIAL_DONE":
                t = ev["payload"]["trial"]
                per_cand[t["candidate_id"]][0] += 1
                if OOM_MARK not in (t.get("failure_detail") or ""):
                    continue
                per_cand[t["candidate_id"]][1] += 1
                sp = spaces.get(t["space_id"])
                if sp is None:
                    unevaluable += 1
                    continue
                env = dict(t["params"]["values"]) | LIMITS
                verdict = satisfies([c["expr"] for c in sp.get("constraints", [])], env)
                if verdict is None:
                    unevaluable += 1
                elif verdict:
                    satisfied += 1
                    per_run[run.name] += 1
                else:
                    violated += 1

    total = satisfied + violated + unevaluable
    print("=" * 78)
    print("Q1  shared-memory OOM trials vs the constraints their space declared")
    print("=" * 78)
    print(f"  OOM trials found:                                  {total}")
    print(f"  VIOLATED a declared constraint (guard let it slip): {violated}")
    print(f"  SATISFIED every declared constraint:                {satisfied}")
    print(f"  not evaluable:                                      {unevaluable}")
    if total:
        print(f"\n  -> {100 * satisfied / total:.0f}% of shared-memory OOMs were legal by their space's own rules.")
    print("\n  by run:")
    for name, n in per_run.most_common():
        print(f"    {name[4:]:<32} {n}")
    print("\n  worst candidates (trials / OOM / rate):")
    ranked = sorted(per_cand.items(), key=lambda kv: -kv[1][1])[:10]
    for cand, (n, oom) in ranked:
        if oom:
            print(f"    {cand:<16} {n:>4} {oom:>5}  {100 * oom / n:>3.0f}%")


def q2_kernel_coverage(runs: list[Path]) -> None:
    rows = []
    multi = 0
    for run in runs:
        for ev in load(run):
            if ev["type"] != "SPACE_PUBLISHED":
                continue
            sp = ev["payload"]["space"]
            names = [d["name"] for d in sp["domains"]]
            groups = kernel_groups(names)
            if len(groups) < 2:
                continue
            multi += 1
            shared = [c["expr"] for c in sp.get("constraints", []) if "SHARED" in c["expr"]]
            missing = sorted(g for g in groups if not any(f"{g}_" in c for c in shared))
            if missing:
                rows.append((run.name[4:], sp["space_id"], sp["candidate_id"],
                             sorted(groups), missing, len(shared)))

    print()
    print("=" * 78)
    print("Q2  multi-kernel spaces with an uncovered kernel group")
    print("=" * 78)
    print(f"  multi-kernel spaces published:                     {multi}")
    print(f"  with >=1 group having NO shared-memory constraint:  {len(rows)}")
    print(f"  with ZERO shared-memory constraints at all:         {sum(1 for r in rows if r[5] == 0)}")
    print()
    for run, sid, cand, groups, missing, n in rows:
        print(f"  {run:<30} {sid:<13} {cand:<15} groups={','.join(groups):<22} "
              f"MISSING={','.join(missing):<22} n_shared={n}")


def q3_two_byte_region(runs: list[Path], cap: int = 400_000) -> None:
    rows = []
    for run in runs:
        for ev in load(run):
            if ev["type"] != "SPACE_PUBLISHED":
                continue
            sp = ev["payload"]["space"]
            names = [d["name"] for d in sp["domains"]]
            dks = dtype_knobs(names)
            exprs = [c["expr"] for c in sp.get("constraints", [])]
            shared = [c for c in exprs if "SHARED" in c and ("* 4" in c or "*4" in c)]
            if not dks or not shared:
                continue
            if any(any(k in c for k in dks) for c in shared):
                continue  # already dtype-aware
            domains = {d["name"]: d["choices"] for d in sp["domains"]}
            keys = list(domains)
            values = [domains[k] for k in keys]
            size = 1
            for v in values:
                size *= len(v)
            if size > cap:
                continue
            relaxed = [c.replace("* 4", "* 2").replace("*4", "*2") if "SHARED" in c else c for c in exprs]
            lost = total16 = 0
            for combo in itertools.product(*values):
                env = dict(zip(keys, combo))
                if str(env.get(dks[0])) not in ("fp16", "bf16"):
                    continue
                env |= LIMITS
                total16 += 1
                if not satisfies(exprs, env) and satisfies(relaxed, env):
                    lost += 1
            if total16:
                rows.append((run.name[4:], sp["space_id"], sp["candidate_id"], lost, total16))

    print()
    print("=" * 78)
    print("Q3  fp16/bf16 grid points excluded by the hard-coded 4-byte width")
    print("=" * 78)
    rows.sort(key=lambda r: -(r[3] / r[4]))
    print(f"  {'run':<30} {'space':<13} {'candidate':<15} {'excluded':>9} {'2B pts':>8} {'%':>5}")
    for run, sid, cand, lost, tot in rows:
        print(f"  {run:<30} {sid:<13} {cand:<15} {lost:>9} {tot:>8} {100 * lost / tot:>4.0f}%")
    if rows:
        lost = sum(r[3] for r in rows)
        tot = sum(r[4] for r in rows)
        print(f"\n  TOTAL: {lost} of {tot} 2-byte grid points excluded ({100 * lost / tot:.0f}%)")


def q4_feasibility_headroom(runs: list[Path], gate: float = 0.25) -> None:
    """Tighter constraints shrink the feasible fraction; validation.py rejects below `gate`."""
    fracs = []
    for run in runs:
        for ev in load(run):
            if ev["type"] != "SPACE_PUBLISHED":
                continue
            sp = ev["payload"]["space"]
            exprs = [c["expr"] for c in sp.get("constraints", [])]
            domains = {d["name"]: d["choices"] for d in sp["domains"]}
            keys = list(domains)
            values = [domains[k] for k in keys]
            size = 1
            for v in values:
                size *= len(v)
            rng = random.Random(0)
            samples = (list(itertools.product(*values)) if size <= 512
                       else [tuple(rng.choice(g) for g in values) for _ in range(512)])
            good = sum(1 for s in samples if satisfies(exprs, dict(zip(keys, s)) | LIMITS))
            fracs.append((good / len(samples), run.name[4:], sp["space_id"]))

    fracs.sort()
    print()
    print("=" * 78)
    print(f"Q4  feasibility headroom before the {gate:.0%} gate")
    print("=" * 78)
    if not fracs:
        return
    print(f"  spaces: {len(fracs)}   min {100 * fracs[0][0]:.0f}%   "
          f"p10 {100 * fracs[len(fracs) // 10][0]:.0f}%   median {100 * fracs[len(fracs) // 2][0]:.0f}%")
    print(f"  below 40% (little room for a tighter constraint): {sum(1 for f in fracs if f[0] < 0.40)}")
    for frac, run, sid in fracs[:6]:
        print(f"    {100 * frac:>3.0f}%  {run:<30} {sid}")


def main() -> None:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "runs")
    runs = sorted(p.parent for p in root.glob("run-*/events.jsonl"))
    if not runs:
        print(f"no runs under {root}")
        return
    q1_oom_vs_constraints(runs)
    q2_kernel_coverage(runs)
    q3_two_byte_region(runs)
    q4_feasibility_headroom(runs)


if __name__ == "__main__":
    main()
