"""Reuse the existing TaskSearch/TPE loop through a project evaluator adapter."""
from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from kernel_optimizer.control.direct_task import TaskSearch, TaskSpace
from kernel_optimizer.conditional.task_response import TaskResponse, complete_response
from kernel_optimizer.evaluation.task_eval import TaskEvaluation
from kernel_optimizer.models.core import ParamSet
from kernel_optimizer.project.measurement import metric, valid_run
from kernel_optimizer.tuning.objective import Objective


class ProjectTaskEvaluator:
    def __init__(self, coordinator, template):
        self.coordinator, self.template = coordinator, template
        self.observations = []

    def evaluate(self, candidate_path, params, context):
        bundle = self.template.model_copy(update={"id": self.template.id+"-param-"+uuid4().hex[:8], "parameters": dict(params)})
        revision = self.coordinator.revisions.stage(bundle)
        run = self.coordinator.execute(revision, bundle.id+"-screen", "evaluate")
        request = self.coordinator.request
        objective = request.objective
        valid = valid_run(run)
        reducer = objective.get("rank_aggregation", "max" if objective["direction"] == "minimize" else "min")
        metrics = {name: metric(run, name, reducer) for name in run["ranks"][0]["result"]["metrics"]} if valid else {}
        result = TaskEvaluation(valid=valid, score=metrics.get(objective["name"]), metrics=metrics,
                                detail=None if valid else "project quality/coverage/worker failure; see raw observation")
        self.observations.append({"bundle": bundle, "revision": revision, "run": run, "result": result})
        return result


def tune(coordinator, template, path: Path, space: dict, trials: int):
    evaluator = ProjectTaskEvaluator(coordinator, template)
    objective = Objective(direction=coordinator.request.objective["direction"],
                          label=coordinator.request.objective["name"], unit=coordinator.request.objective["unit"])
    # This is an explicit GPU parameter experiment. It makes no provider calls and
    # adds no wall-clock or API quota. Legacy callers retain their original budgets.
    budgets = coordinator.config.budgets.model_copy(update={"trials_per_space": trials, "wall_clock_hours": float("inf")})
    search = TaskSearch(evaluator, objective, budgets, device=coordinator.config.device)
    winner = search.evaluate_candidate(path, TaskSpace.model_validate(space), {}, seed=coordinator.config.run.seed)
    selected = None
    if winner is not None:
        selected = next(item for item in evaluator.observations if item["bundle"].parameters == winner.params.values
                        and item["result"].valid and item["result"].score == winner.task_evaluation.score)
    responses = []
    for left, right in zip(evaluator.observations, evaluator.observations[1:]):
        a, b = left["bundle"].parameters, right["bundle"].parameters
        axes = [name for name in a if a[name] != b.get(name)]
        if len(axes) == 1:
            axis = axes[0]
            response = complete_response(TaskResponse(candidate_id=template.id, axis=axis,
                a_params=ParamSet(values=a), b_params=ParamSet(values=b), a=left["result"], b=right["result"],
                unknown_resources=sorted(set(left["result"].metrics)|set(right["result"].metrics))), objective,
                isinstance(a[axis], (float, int)) and isinstance(b[axis], (float, int)))
            responses.append(response.model_dump(mode="json"))
    for item in evaluator.observations:
        if item is not selected:
            coordinator.revisions.abort(item["revision"].id, "parameter screening only; not the selected validation candidate")
    receipt = {"engine": "existing TaskSearch + OptunaTPETuner", "measurement_scope": "complete project",
        "parameter_trials": trials, "space": space, "selected": selected["bundle"].id if selected else None,
        "stats": [s.model_dump(mode="json") for s in search.stats], "responses": responses,
        "observations": [{"bundle": x["bundle"].model_dump(mode="json"), "revision": x["revision"].id,
                          "result": x["result"].model_dump(mode="json"), "run": x["run"]} for x in evaluator.observations]}
    return selected, receipt
