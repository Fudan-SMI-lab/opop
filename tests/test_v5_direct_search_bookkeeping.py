"""Direct-search admission and measured-history regressions with real CPU trials."""

from collections.abc import Iterator
from contextlib import closing
from pathlib import Path
from typing import Literal, assert_never

import pytest

from kernel_optimizer.agents.base import AgentOutcome
from kernel_optimizer.agents.runtime import AgentCallError, OpencodeClient
from kernel_optimizer.agents.sandbox import Sandbox, SandboxFactory
from kernel_optimizer.agents.task_rewriter import TaskRewriteInputs, TaskRewriteResult, TaskRewriterAgent
from kernel_optimizer.config import AgentModuleConfig, BudgetConfig
from kernel_optimizer.control.direct_task import TaskSearch, TaskSpace
from kernel_optimizer.control.task_rewrite import RewriteRun, run_rewrites
from kernel_optimizer.evaluation.task_eval import TaskEvaluator
from kernel_optimizer.models.core import Constraint, ParamDomain
from kernel_optimizer.store.run_store import RunStore
from kernel_optimizer.tuning.objective import Objective


@pytest.fixture
def search() -> Iterator[TaskSearch]:
    project = Path(__file__).resolve().parents[1] / "examples/direct_task"
    with TaskEvaluator(project / "eval.py", "assess") as evaluator:
        search = TaskSearch(evaluator, Objective(direction="minimize"), BudgetConfig(
            trials_per_space=1, rewrite_rounds_per_family=2, no_improve_rounds=1,
        ))
        space = TaskSpace(params=[ParamDomain(name="width", kind="int", choices=[2])])
        search.evaluate_candidate(project / "candidate.py", space, {"data": [2, 1]})
        yield search


@pytest.fixture
def agent(tmp_path: Path) -> Iterator[TaskRewriterAgent]:
    with closing(OpencodeClient("http://127.0.0.1:0")) as client:
        yield TaskRewriterAgent(client, SandboxFactory(tmp_path / "sandboxes"),
                                RunStore(tmp_path), AgentModuleConfig(max_retries=0))


@pytest.mark.parametrize("wall_expired", [False, True])
def test_history_stays_measured_when_child_evaluates_nothing(search: TaskSearch, wall_expired: bool) -> None:
    # Given: an incumbent, and either no legal child parameters or an expired clock.
    parent = search.trials[0]
    family = search.families.families[search.family_ids[parent.candidate_id]]
    space = search.spaces[parent.candidate_id]
    if wall_expired:
        search.started -= search.budgets.wall_clock_hours * 3600 + 1
    else:
        space = space.model_copy(update={"constraints": [Constraint(expr="width < 0", rationale="empty")]})
    # When: child tuning returns without a single actual evaluation.
    result = search.evaluate_candidate(search.paths[parent.candidate_id], space,
                                       {"data": [2, 1]}, parent_id=parent.candidate_id)
    # Then: a spent but unmeasured round is not plateau evidence.
    assert result is None and search.trials == [parent]
    assert family.best_history == [0.0]
    assert (family.rewrite_rounds_used, family.rounds_not_evaluated) == (1, 1)
    assert family.status == "active"


@pytest.mark.parametrize("other_active", [False, True])
@pytest.mark.parametrize("blocked_by", ["frozen", "cap", "plateau"])
def test_child_admission_uses_own_family_when_blocked(
    search: TaskSearch, other_active: bool, blocked_by: Literal["frozen", "cap", "plateau"],
) -> None:
    # Given: own-family freeze/cap/convergence, optionally alongside another active family.
    parent = search.trials[0]
    path, space = search.paths[parent.candidate_id], search.spaces[parent.candidate_id]
    if other_active:
        search.evaluate_candidate(path, space, {"data": [2, 1]})
    family = search.families.families[search.family_ids[parent.candidate_id]]
    match blocked_by:
        case "frozen":
            family.status = "frozen_converged"
        case "cap":
            family.rewrite_rounds_used = search.budgets.rewrite_rounds_per_family
        case "plateau":
            family.best_history.append(0.0)
        case unreachable:
            assert_never(unreachable)
    before = (dict(search.paths), list(search.trials), list(search.stats),
              list(family.member_ids), list(family.best_history), family.rewrite_rounds_used)
    # When: a direct API caller tries to admit a child.
    result = search.evaluate_candidate(path, space, {"data": [2, 1]}, parent_id=parent.candidate_id)
    # Then: denial registers nothing and spends no additional round, regardless of other families.
    assert result is None
    assert (search.paths, search.trials, search.stats, family.member_ids,
            family.best_history, family.rewrite_rounds_used) == before
    assert family.status != "active"


@pytest.mark.parametrize("invalid", [False, True])
def test_evaluated_child_retains_plateau_policy_when_trials_run(search: TaskSearch, invalid: bool) -> None:
    # Given: a child with one real valid or invalid evaluation and an existing incumbent.
    parent = search.trials[0]
    space = TaskSpace(params=[ParamDomain(name="width", kind="int", choices=[9 if invalid else 2])])
    # When: real candidate execution completes a tuning round.
    result = search.evaluate_candidate(search.paths[parent.candidate_id], space,
                                       {"data": [2, 1]}, parent_id=parent.candidate_id)
    # Then: invalid-only actual evaluations keep their existing plateau policy.
    family = search.families.families[search.family_ids[parent.candidate_id]]
    assert len(search.trials) == 2
    assert (result is None) == invalid
    assert family.best_history == [0.0, 0.0]
    assert (family.rewrite_rounds_used, family.rounds_not_evaluated) == (1, 0)
    assert family.status == "frozen_converged"


def test_failed_attempts_freeze_immediately_when_cap_reached(
    search: TaskSearch, agent: TaskRewriterAgent, monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given: one eligible family with a single remaining attempt.
    family = next(iter(search.families.families.values()))
    family.rewrite_rounds_used = 1

    def fail(inputs: TaskRewriteInputs) -> AgentOutcome[TaskRewriteResult]:
        raise AgentCallError(inputs.candidate_id)

    monkeypatch.setattr(agent, "invoke", fail)
    # When: the last allowed agent attempt fails.
    run_rewrites(search, agent, RewriteRun(rounds=1, project_root=Path.cwd(), goal="sort", context={"data": [2, 1]}))
    # Then: status is correct even without a subsequent loop iteration.
    assert family.status == "frozen_budget"
    assert (family.rewrite_rounds_used, family.rounds_not_evaluated) == (2, 1)
    assert family.best_history == [0.0]


def test_eligible_family_runs_when_first_family_is_exhausted(
    search: TaskSearch, agent: TaskRewriterAgent, monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given: a stale active exhausted family ranked ahead of another eligible family.
    parent = search.trials[0]
    path, space = search.paths[parent.candidate_id], search.spaces[parent.candidate_id]
    second = search.evaluate_candidate(path, space, {"data": [2, 1]})
    assert second is not None
    first_family, second_family = search.families.families.values()
    first_family.rewrite_rounds_used = 2
    second_family.rewrite_rounds_used = 1
    search.families.max_families_active = 1
    calls: list[str] = []

    def fail(inputs: TaskRewriteInputs) -> AgentOutcome[TaskRewriteResult]:
        calls.append(inputs.candidate_id)
        raise AgentCallError(inputs.candidate_id)

    monkeypatch.setattr(agent, "invoke", fail)
    # When: only one attempt remains globally; skipping cannot consume that attempt.
    run_rewrites(search, agent, RewriteRun(rounds=1, project_root=path.parent, goal="sort", context={"data": [2, 1]}))
    # Then: the eligible family receives it, and both exhausted statuses are current.
    assert calls == [second.candidate_id]
    assert first_family.status == second_family.status == "frozen_budget"


def test_rewrite_handles_denied_admission_when_agent_returns(
    search: TaskSearch, agent: TaskRewriterAgent, monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given: family eligibility changes at the agent boundary before child admission.
    parent = search.trials[0]
    path = search.paths[parent.candidate_id]
    family = next(iter(search.families.families.values()))

    def freeze(inputs: TaskRewriteInputs) -> AgentOutcome[TaskRewriteResult]:
        family.status = "frozen_budget"
        return AgentOutcome(output=TaskRewriteResult(candidate_file=path.name),
                            sandbox=Sandbox(path.parent), session_id="local", attempts=1, tokens={}, cost=0.0)

    monkeypatch.setattr(agent, "invoke", freeze)
    # When: the successful agent result reaches the admission seam.
    run_rewrites(search, agent, RewriteRun(rounds=1, project_root=path.parent, goal="sort", context={"data": [2, 1]}))
    # Then: no child is fabricated and the caller records the denied admission without crashing.
    assert list(search.paths) == [parent.candidate_id]
    assert search.rewrites[-1]["error"] == "child_not_admitted"
