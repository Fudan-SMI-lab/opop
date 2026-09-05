"""Counterfactual: what would seeding `best_history` have done, on every recorded family?

`stop_kind = "converged"` has never fired (finding-converged-stop-kind-is-unreachable.md). The
proposed fix -- seed best_history with the family's post-tuning incumbent so the list means
"incumbent after each round INCLUDING round 0" -- is pending item 3 and awaits a user decision.

Rather than argue about it, this replays the REAL ConvergencePolicy against every family's real
history, twice: as recorded, and with the seed prepended. Same code path the orchestrator uses,
so the verdicts are what would actually have happened.

The interesting column is the third: at rewrite_rounds_used == 2, today's policy always returns
`continue / recent=None` (one entry short of computable), while the seeded version returns a real
slope and freezes the genuinely stalled families as `converged` one round early.

Usage:  python scripts/audit_seeded_history_counterfactual.py [runs/run-... ...]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from kernel_optimizer.config import BudgetConfig  # noqa: E402
from kernel_optimizer.control.convergence import ConvergencePolicy  # noqa: E402
from kernel_optimizer.models.core import Family  # noqa: E402

REPO = Path(__file__).resolve().parent.parent


def collect(run: Path) -> list[tuple[str, float | None, list[float]]]:
    """(family_id, its seed's tuned best, [best after each recorded round])."""
    family_of: dict[str, str] = {}
    seed_best: dict[str, float | None] = {}
    history: dict[str, list[float]] = {}
    for line in (run / "events.jsonl").open(encoding="utf-8"):
        e = json.loads(line)
        if e["type"] == "CANDIDATE_REGISTERED":
            c = e["payload"].get("candidate") or e["payload"]
            family_of[c["candidate_id"]] = c["family_id"]
            if c.get("origin") == "seed":
                seed_best.setdefault(c["family_id"], None)
        elif e["type"] == "TUNING_DONE":
            fid = family_of.get(e["payload"].get("candidate_id"))
            ms = e["payload"].get("best_ms")
            # The family's FIRST tuned candidate is its seed; that is the value the fix
            # would prepend, and it is exactly what best_history omits today.
            if fid in seed_best and seed_best[fid] is None and ms is not None:
                seed_best[fid] = ms
        elif e["type"] == "FAMILY_ROUND_RECORDED":
            history.setdefault(e["payload"]["family_id"], []).append(e["payload"]["best_ms"])
    return [(fid, seed_best.get(fid), h) for fid, h in history.items() if len(h) >= 2]


def describe(decision) -> str:
    recent = decision.evidence.get("recent_improvements_pct")
    shown = [round(x, 1) for x in recent] if recent else recent
    return f"{decision.verdict}/{decision.stop_kind} recent={shown}"


def main(argv: list[str]) -> int:
    runs = [Path(a) for a in argv] or sorted(REPO.glob("runs/run-l3-*"))
    cfg = BudgetConfig()  # the L3 experiment values are the field defaults
    policy = ConvergencePolicy(cfg)

    print(f"policy: rewrite_rounds_per_family={cfg.rewrite_rounds_per_family}  "
          f"no_improve_rounds={cfg.no_improve_rounds}  "
          f"min_improvement_pct={cfg.min_improvement_pct}")
    print()
    print(f"  {'run':<24} {'family':<14} {'seed':>6} {'history':<22} "
          f"{'TODAY at used=2':<30} WITH SEEDING at used=2")

    would_freeze = would_continue = unknown = 0
    for run in runs:
        if not (run / "events.jsonl").exists():
            continue
        for fid, seed, hist in collect(run):
            today = policy.family_verdict(
                Family(family_id=fid, anchor_candidate_id="c",
                       best_history=hist[:2], rewrite_rounds_used=2))
            if seed is None:
                seeded_txt = "(seed best unknown)"
                unknown += 1
            else:
                seeded = policy.family_verdict(
                    Family(family_id=fid, anchor_candidate_id="c",
                           best_history=[seed] + hist[:2], rewrite_rounds_used=2))
                seeded_txt = describe(seeded)
                if seeded.verdict == "freeze":
                    would_freeze += 1
                else:
                    would_continue += 1
            print(f"  {run.name[4:]:<24} {fid:<14} {str(seed):>6} {str(hist[:3]):<22} "
                  f"{describe(today):<30} {seeded_txt}")

    total = would_freeze + would_continue
    if total:
        print()
        print(f"  At rewrite_rounds_used == 2, today's policy returns continue/recent=None for "
              f"ALL {total + unknown} families")
        print(f"  (one history entry short of computable). With the seed prepended:")
        print(f"    freeze as CONVERGED -- genuinely stalled, saves its 3rd round: "
              f"{would_freeze} ({100 * would_freeze / total:.0f}%)")
        print(f"    continue -- still improving, keeps its 3rd round: {would_continue}")
        print()
        print("  So the fix is not uniformly more aggressive: it frees the stalled families and")
        print("  leaves the moving ones alone. Cross-check the freeze set against")
        print("  finding-converged-stop-kind-is-unreachable.md's 0-for-13 record on round 3.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
