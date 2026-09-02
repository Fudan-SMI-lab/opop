"""Convergence policy tests: freeze rules must be harness-owned and deterministic."""

from kernel_optimizer.config import BudgetConfig
from kernel_optimizer.control.convergence import ConvergencePolicy
from kernel_optimizer.models.core import Family


def fam(history, rounds_used=0, status="active"):
    return Family(family_id="f", anchor_candidate_id="a", member_ids=["a"],
                  best_history=history, rewrite_rounds_used=rounds_used, status=status)


CFG = BudgetConfig(rewrite_rounds_per_family=3, no_improve_rounds=2,
                   min_improvement_pct=2.0, wall_clock_hours=12)


def test_continue_when_improving():
    policy = ConvergencePolicy(CFG)
    decision = policy.family_verdict(fam([10.0, 9.0, 8.0]))
    assert decision.verdict == "continue"


def test_freeze_converged_after_no_improve_rounds():
    policy = ConvergencePolicy(CFG)
    # Two consecutive rounds with <2% improvement.
    decision = policy.family_verdict(fam([10.0, 9.99, 9.98]))
    assert decision.verdict == "freeze"
    assert decision.stop_kind == "converged"


def test_freeze_budget_exhausted():
    policy = ConvergencePolicy(CFG)
    decision = policy.family_verdict(fam([10.0, 9.0], rounds_used=3))
    assert decision.verdict == "freeze"
    assert decision.stop_kind == "budget_exhausted"


def test_agent_suggestion_is_not_decisive():
    policy = ConvergencePolicy(CFG)
    decision = policy.family_verdict(fam([10.0, 9.0, 8.0]), agent_suggestion="stop")
    assert decision.verdict == "continue"  # improving; agent's "stop" is only evidence
    assert decision.evidence["agent_suggestion"] == "stop"


def test_short_history_continues():
    policy = ConvergencePolicy(CFG)
    assert policy.family_verdict(fam([10.0])).verdict == "continue"
    assert policy.family_verdict(fam([])).verdict == "continue"


def test_global_wall_clock():
    policy = ConvergencePolicy(CFG)
    decision = policy.global_verdict([fam([10.0])], elapsed_hours=13.0)
    assert decision.verdict == "freeze" and decision.stop_kind == "budget_exhausted"


def test_global_all_frozen_converged():
    policy = ConvergencePolicy(CFG)
    families = [fam([10.0], status="frozen_converged"),
                fam([9.0], status="frozen_converged")]
    decision = policy.global_verdict(families, elapsed_hours=1.0)
    assert decision.verdict == "freeze" and decision.stop_kind == "converged"


def test_global_mixed_frozen_is_budget():
    policy = ConvergencePolicy(CFG)
    families = [fam([10.0], status="frozen_converged"),
                fam([9.0], status="frozen_budget")]
    decision = policy.global_verdict(families, elapsed_hours=1.0)
    assert decision.verdict == "freeze" and decision.stop_kind == "budget_exhausted"


def test_global_continue_when_any_active():
    policy = ConvergencePolicy(CFG)
    families = [fam([10.0], status="frozen_budget"), fam([9.0], status="active")]
    assert policy.global_verdict(families, elapsed_hours=1.0).verdict == "continue"
