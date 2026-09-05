"""How many published constraints can never reject anything?

Reports TWO numbers, because the obvious one is misleading:

  1. GRID-VACUOUS      -- rejects nothing over the space's own choice grid. This
                          is mostly benign: `NUM_WARPS * 32 <= MAX_THREADS_PER_BLOCK`
                          is correct and binds at 32 warps, and no agent has offered
                          more than 16. It measures the DOMAIN, not the constraint.
  2. STRUCTURALLY VACUOUS -- rejects nothing even with the numeric domains widened to
                          powers of two up to 65536. This is the real signal: an
                          expression that cannot constrain anything at any scale.

The gap between them is large (55% vs 1.3% at the time of writing), which is why a
publish-time "reject vacuous constraints" gate would be a mistake -- it could only
afford the grid test, and would throw away correct bounds that
finding-expansion-drops-inherited-constraints.md exists to preserve.

The row worth acting on is a structurally-vacuous MAX_SHARED_BYTES bound on a kernel
that actually stages operands (uses `tl.dot` or `num_stages`), so the report flags
that combination explicitly. A zero footprint is truthful for an elementwise kernel.

Usage: python scripts/audit_vacuous_constraints.py [runs_dir]
"""
from __future__ import annotations

import itertools
import json
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from kernel_optimizer.config import DeviceLimits  # noqa: E402
from kernel_optimizer.models.core import (  # noqa: E402
    Constraint,
    ParamDomain,
    ParameterSpace,
    ParamSet,
)
from kernel_optimizer.paramspace.guard import check_config  # noqa: E402

WIDE = [1, 2, 4, 8, 16, 32, 64, 128, 256, 1024, 4096, 16384, 65536]


def _rejects_any(space: ParameterSpace, device: DeviceLimits,
                 rnd: random.Random, cap: int = 3000) -> bool:
    ch = {d.name: list(d.choices) for d in space.domains}
    names = list(ch)
    total = 1
    for v in ch.values():
        total *= len(v)
    grid = (list(itertools.product(*[ch[k] for k in names])) if total <= cap
            else [tuple(rnd.choice(ch[k]) for k in names) for _ in range(cap)])
    for combo in grid:
        if check_config(space, ParamSet(values=dict(zip(names, combo))), device) is not None:
            return True
    return False


def _single(sp: dict, cons: Constraint, domains: list[ParamDomain]) -> ParameterSpace:
    return ParameterSpace(space_id=sp["space_id"], candidate_id=sp["candidate_id"],
                          source_sha=sp["source_sha"], version=sp.get("version", 1),
                          domains=domains, constraints=[cons])


def main() -> int:
    runs_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "runs")
    device = DeviceLimits()
    rnd = random.Random(0)

    total = 0
    grid_vac: list[tuple[str, str, str]] = []
    struct_vac: list[tuple[str, str, str, bool]] = []

    for run in sorted(runs_dir.iterdir()):
        if not run.is_dir():
            continue
        ev_path = run / "events.jsonl"
        if not ev_path.exists():
            continue
        label = run.name.replace("run-", "").replace("20260", "0")
        for line in ev_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            e = json.loads(line)
            if e["type"] != "SPACE_PUBLISHED":
                continue
            sp = e["payload"]["space"]
            own = [ParamDomain.model_validate(d) for d in sp["domains"]]
            wide = [
                ParamDomain(name=d.name, kind=d.kind,
                            choices=(list(WIDE) if d.kind == "int" else list(d.choices)))
                for d in own
            ]
            # does the candidate stage operands at all?
            src = run / "candidates" / sp["candidate_id"] / "source.py"
            stages = False
            if src.exists():
                text = src.read_text(encoding="utf-8", errors="replace")
                stages = ("tl.dot" in text) or ("num_stages" in text)

            for cd in sp["constraints"]:
                total += 1
                cons = Constraint(expr=cd["expr"], rationale=cd.get("rationale", ""))
                try:
                    if _rejects_any(_single(sp, cons, own), device, rnd):
                        continue
                except Exception:
                    continue
                grid_vac.append((label, sp["candidate_id"], cons.expr))
                try:
                    if not _rejects_any(_single(sp, cons, wide), device, rnd):
                        struct_vac.append((label, sp["candidate_id"], cons.expr, stages))
                except Exception:
                    pass

    if not total:
        print("no published spaces found")
        return 0

    print(f"constraints across all published spaces: {total}")
    print(f"  GRID-vacuous (reject nothing over their own grid):   "
          f"{len(grid_vac):4d}  ({100 * len(grid_vac) / total:.0f}%)")
    print(f"  STRUCTURALLY vacuous (nothing even when widened):    "
          f"{len(struct_vac):4d}  ({100 * len(struct_vac) / total:.1f}%)")
    print("\nThe first number measures the DOMAIN, not the constraint — a correct bound "
          "\nthat the offered choices never reach counts there. Watch the second.\n")

    print("most common grid-vacuous expressions (mostly benign):")
    for expr, n in Counter(x[2] for x in grid_vac).most_common(8):
        print(f"  {n:3d}x  {expr}")

    if struct_vac:
        print("\nstructurally vacuous — cannot constrain anything at any scale:")
        for label, cid, expr, stages in struct_vac:
            note = "  <-- ON A STAGING KERNEL, INVESTIGATE" if (
                stages and "SHARED" in expr.upper()) else ""
            print(f"  {label:22s} {cid:16s} {expr}{note}")
        bad = [x for x in struct_vac if x[3] and "SHARED" in x[2].upper()]
        if bad:
            print(f"\n*** {len(bad)} shared-memory bound(s) that cannot bind, on kernels that "
                  f"DO stage operands (tl.dot/num_stages). That is a real gap. ***")
            return 1
        print("\nNone is on a staging kernel: a zero shared-memory footprint is truthful "
              "for an elementwise kernel, and the rest restate a domain minimum.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
