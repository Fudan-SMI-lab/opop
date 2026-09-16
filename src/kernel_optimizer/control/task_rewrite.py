"""Bounded response -> structural agent -> same-evaluator child tuning."""

from pathlib import Path
from collections.abc import Mapping
from typing import ClassVar

from pydantic import BaseModel, ConfigDict, Field, JsonValue

from kernel_optimizer.agents.runtime import AgentCallError
from kernel_optimizer.agents.task_rewriter import TaskRewriteInputs, TaskRewriterAgent
from kernel_optimizer.conditional.task_response import TaskResponse, complete_response
from kernel_optimizer.control.direct_task import TaskSearch
from kernel_optimizer.evaluation.task_eval import TaskEvaluation
from kernel_optimizer.models.core import ParamSet, TrialRecord
from kernel_optimizer.paramspace.guard import ConstraintError, eval_constraint
from kernel_optimizer.tuning.objective import objective_value, rank_record


class RewriteRun(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)
    rounds: int = Field(ge=0)
    project_root: Path
    goal: str
    context: dict[str, JsonValue]
    source_paths: tuple[Path, ...] = ()
    seed: int = 0


def probe_responses[T](search: TaskSearch, parent: TrialRecord,
                       context: Mapping[str, T]) -> list[TaskResponse]:
    responses: list[TaskResponse] = []
    space = search.spaces[parent.candidate_id]
    for domain in space.params:
        legal: list[ParamSet] = []
        for choice in domain.choices:
            params = ParamSet(values={**parent.params.values, domain.name: choice})
            try:
                admitted = all(eval_constraint(c.expr, params.values) for c in space.constraints)
            except ConstraintError:
                admitted = False
            if admitted and params not in legal:
                legal.append(params)
        response = TaskResponse(candidate_id=parent.candidate_id, axis=domain.name,
                                a_params=parent.params, unknown_resources=list(search.resource_metrics))
        if len(legal) < 2:
            responses.append(response.model_copy(update={"reason": "fewer_than_two_legal_endpoints"}))
            continue
        a_params, b_params = legal[0], legal[-1]
        response = response.model_copy(update={"a_params": a_params, "b_params": b_params})
        if search.probe_budget - search.probe_calls < 2:
            responses.append(response.model_copy(update={"reason": "probe_budget"}))
            continue
        observations: list[TaskEvaluation] = []
        for params in (a_params, b_params):
            if not search.within_wall_budget():
                break
            search.probe_calls += 1
            values: dict[str, JsonValue] = dict(params.values)
            observations.append(search.evaluator.evaluate(search.paths[parent.candidate_id], values, context))
        response = response.model_copy(update={
            "a": observations[0] if observations else None,
            "b": observations[1] if len(observations) == 2 else None,
        })
        responses.append(complete_response(response, search.objective, domain.kind != "str"))
    search.responses.extend(responses)
    return responses


def run_rewrites(search: TaskSearch, agent: TaskRewriterAgent, run: RewriteRun) -> None:
    for _ in range(run.rounds):
        for family in search.families.families.values():
            search.refresh_family_status(family)
        active = search.families.active_families()
        if not active or not search.within_wall_budget():
            break
        family = active[0]
        parent = min(
            (t for t in search.trials if search.family_ids[t.candidate_id] == family.family_id
             and objective_value(t, search.objective) is not None),
            key=lambda t: rank_record(t, search.objective),
        )
        responses = probe_responses(search, parent, run.context)
        if not search.within_wall_budget():
            break
        resources = (parent.task_evaluation.metrics if parent.task_evaluation else {})
        inputs = TaskRewriteInputs(
            project_root=run.project_root, source_paths=run.source_paths,
            candidate_id=parent.candidate_id, candidate_path=search.paths[parent.candidate_id],
            goal=run.goal, context=run.context, objective=search.objective,
            params=parent.params, space=search.spaces[parent.candidate_id],
            resource_metrics=search.resource_metrics,
            resources={name: resources[name] for name in search.resource_metrics if name in resources},
            responses=responses,
        )
        try:
            outcome = agent.invoke(inputs)
        except AgentCallError as exc:
            family.rewrite_rounds_used += 1
            search.families.record_round_not_evaluated(family.family_id)
            search.refresh_family_status(family)
            search.rewrites.append({"parent_id": parent.candidate_id, "error": str(exc)})
            continue
        path = (outcome.sandbox.root / outcome.output.candidate_file).resolve()
        if not search.within_wall_budget():
            search.rewrites.append({"parent_id": parent.candidate_id, "child_path": str(path),
                                    "error": "wall_budget_before_child"})
            break
        before = set(search.paths)
        child = search.evaluate_candidate(
            path, outcome.output.space or inputs.space, run.context,
            parent_id=parent.candidate_id, seed=run.seed,
        )
        child_id = next((cid for cid in search.paths if cid not in before), None)
        if child_id is None:
            search.rewrites.append({"parent_id": parent.candidate_id, "child_path": str(path),
                                    "error": "child_not_admitted"})
            continue
        search.rewrites.append({
            "parent_id": parent.candidate_id, "parent_path": str(inputs.candidate_path),
            "parent_score": objective_value(parent, search.objective),
            "parent_params": parent.params.model_dump(mode="json"),
            "child_id": child_id, "child_path": str(path),
            "child_score": objective_value(child, search.objective) if child else None,
            "child_params": child.params.model_dump(mode="json") if child else None,
        })
