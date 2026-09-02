"""Harness-owned convergence policy. Agent suggestions are recorded, never decisive."""

from __future__ import annotations

from kernel_optimizer.config import BudgetConfig
from kernel_optimizer.models.core import Family
from kernel_optimizer.models.reports import ConvergenceDecision


class ConvergencePolicy:
    def __init__(self, cfg: BudgetConfig):
        self.cfg = cfg

    def family_verdict(self, family: Family,
                       agent_suggestion: str | None = None) -> ConvergenceDecision:
        evidence: dict = {
            "best_history": family.best_history,
            "rewrite_rounds_used": family.rewrite_rounds_used,
            "agent_suggestion": agent_suggestion,
        }
        if family.rewrite_rounds_used >= self.cfg.rewrite_rounds_per_family:
            return ConvergenceDecision(scope="family", verdict="freeze",
                                       stop_kind="budget_exhausted", evidence=evidence)

        history = family.best_history
        needed = self.cfg.no_improve_rounds
        if len(history) >= needed + 1:
            window = history[-(needed + 1):]
            improvements = [
                (window[i] - window[i + 1]) / window[i] * 100.0 if window[i] > 0 else 0.0
                for i in range(len(window) - 1)
            ]
            evidence["recent_improvements_pct"] = [round(x, 3) for x in improvements]
            if all(imp < self.cfg.min_improvement_pct for imp in improvements):
                return ConvergenceDecision(scope="family", verdict="freeze",
                                           stop_kind="converged", evidence=evidence)
        return ConvergenceDecision(scope="family", verdict="continue", evidence=evidence)

    def global_verdict(self, families: list[Family],
                       elapsed_hours: float) -> ConvergenceDecision:
        evidence: dict = {
            "elapsed_hours": round(elapsed_hours, 3),
            "families": {f.family_id: f.status for f in families},
        }
        if elapsed_hours >= self.cfg.wall_clock_hours:
            return ConvergenceDecision(scope="global", verdict="freeze",
                                       stop_kind="budget_exhausted", evidence=evidence)
        if families and all(f.status != "active" for f in families):
            frozen_kinds = {f.status for f in families}
            stop_kind = ("converged" if frozen_kinds == {"frozen_converged"}
                         else "budget_exhausted")
            return ConvergenceDecision(scope="global", verdict="freeze",
                                       stop_kind=stop_kind, evidence=evidence)
        return ConvergenceDecision(scope="global", verdict="continue", evidence=evidence)
