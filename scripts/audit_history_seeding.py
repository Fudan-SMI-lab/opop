"""Audit: replay the best_history seeding fix through the REAL convergence policy.

Answers two questions with one walk:

1. Does `stop_kind="converged"` become reachable? It never fired on any run, because
   `family_verdict` checks the round budget BEFORE the convergence test and the test
   needs `no_improve_rounds + 1` history entries -- with (3 rounds, 2 no-improve) the
   budget freeze fires at round 4 while the history only reaches 3 entries then.
2. Does it freeze a STALLED family earlier without cutting short an IMPROVING one?
   That is the risk: a premature freeze would delete structural search, which is the
   failure mode this project exists to avoid.

The verdict must be evaluated BEFORE each round, exactly as `_rewrite_round` does.
Evaluating it once at `used == len(history)` -- i.e. after the last round -- shows no
change at all and is the wrong reading; that mistake is why this script exists rather
than a one-off snippet.

Usage:  python scripts/audit_history_seeding.py [--runs runs] [--config configs/experiments_l3.yaml]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, "src")

from kernel_optimizer.config import load_config  # noqa: E402
from kernel_optimizer.control.convergence import ConvergencePolicy  # noqa: E402
from kernel_optimizer.models.core import BestRecord, Family, ParamSet  # noqa: E402


def simulate(policy: ConvergencePolicy, budgets, seed_ms: float,
             rounds: list[float], seeded: bool) -> tuple[int | None, str | None]:
    """Walk the rounds the way the outer loop does and report where it freezes."""
    for r in range(1, budgets.rewrite_rounds_per_family + 2):
        used = r - 1
        history = ([seed_ms] if seeded else []) + rounds[:used]
        if not history:
            continue
        fam = Family(
            family_id="f", anchor_candidate_id="c", member_ids=["c"],
            best=BestRecord(candidate_id="c", params=ParamSet(values={}),
                            latency_ms=history[-1]),
            best_history=list(history), rewrite_rounds_used=used, status="active")
        v = policy.family_verdict(fam)
        if v.verdict == "freeze":
            return r, v.stop_kind
    return None, "never froze within budget"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--config", default="configs/experiments_l3.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)
    policy = ConvergencePolicy(cfg.budgets)
    print(f"config: rewrite_rounds_per_family={cfg.budgets.rewrite_rounds_per_family}  "
          f"no_improve_rounds={cfg.budgets.no_improve_rounds}  "
          f"min_improvement_pct={cfg.budgets.min_improvement_pct}\n")

    print(f"{'run':14s} {'family':14s} {'seed':>6s} {'rounds':22s} "
          f"{'unseeded':>26s} {'seeded':>26s}")

    newly_converged = 0
    froze_earlier = 0
    improving_cut_short = []
    examined = 0

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
        fin = [e for e in events if e.get("type") == "RUN_FINISHED"]
        if not fin:
            continue
        families = ((fin[0].get("payload") or {}).get("summary") or {}).get("families")
        if not isinstance(families, dict):
            continue

        for fid, info in families.items():
            history = info.get("history") or []
            if len(history) < cfg.budgets.no_improve_rounds + 1:
                continue  # never ran enough rounds for the test to differ
            anchor = info.get("anchor")
            seeds = [e["payload"]["best_ms"] for e in events
                     if e.get("type") == "TUNING_DONE"
                     and (e.get("payload") or {}).get("candidate_id") == anchor
                     and (e.get("payload") or {}).get("best_ms")]
            if not seeds:
                continue
            seed_ms = seeds[0]
            examined += 1

            un = simulate(policy, cfg.budgets, seed_ms, history, seeded=False)
            se = simulate(policy, cfg.budgets, seed_ms, history, seeded=True)
            if se[1] == "converged" and un[1] != "converged":
                newly_converged += 1
            if se[0] and un[0] and se[0] < un[0]:
                froze_earlier += 1
                # Was this family still improving when the seeded run froze it?
                # Compare the last two entries it had actually seen by then.
                seen = ([seed_ms] + history)[:se[0]]
                if len(seen) >= 2 and seen[-2] > 0:
                    gain = (seen[-2] - seen[-1]) / seen[-2] * 100.0
                    if gain >= cfg.budgets.min_improvement_pct:
                        improving_cut_short.append((d.name[-13:], fid[:13], gain))

            print(f"{d.name[-13:]:14s} {fid[:13]:14s} {seed_ms:6.1f} "
                  f"{str([round(x, 1) for x in history]):22s} "
                  f"{f'round {un[0]}: {un[1]}':>26s} {f'round {se[0]}: {se[1]}':>26s}")

    print(f"\nfamilies examined: {examined}")
    print(f"newly reach stop_kind='converged': {newly_converged}")
    print(f"freeze one round EARLIER (saved GPU time): {froze_earlier}")

    if improving_cut_short:
        print(f"\nWARNING: {len(improving_cut_short)} family/families were still "
              f"improving by >= {cfg.budgets.min_improvement_pct}% when the seeded "
              f"policy froze them -- that is structural search being deleted:")
        for run, fid, gain in improving_cut_short:
            print(f"   {run}  {fid}  last gain {gain:.1f}%")
        return 1

    print("\nNo family that was still improving was frozen earlier: every family the "
          "seeding freezes sooner had already gone flat.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
