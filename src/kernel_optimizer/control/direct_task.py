"""Direct task search; evaluate_candidate also accepts Task5 rewrite children."""

from collections.abc import Callable, Mapping
from pathlib import Path
from time import monotonic
from typing import TYPE_CHECKING, ClassVar, Protocol, final
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, JsonValue

from kernel_optimizer.config import BudgetConfig
from kernel_optimizer.control.convergence import ConvergencePolicy
from kernel_optimizer.control.families import FamilyManager
from kernel_optimizer.evaluation.task_eval import TaskEvaluation
from kernel_optimizer.models.core import (
    BestRecord, Constraint, DeviceLimits, Family, ParamDomain, ParameterSpace, ParamSet, TrialRecord,
)
from kernel_optimizer.models.reports import TuningStats
from kernel_optimizer.paramspace.guard import eval_constraint
from kernel_optimizer.tuning.objective import Objective, objective_value, rank_record
from kernel_optimizer.tuning.stats import TuningStatsAnalyzer
from kernel_optimizer.tuning.tpe import OptunaTPETuner

if TYPE_CHECKING:
    from kernel_optimizer.conditional.task_response import TaskResponse


class TaskSpace(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid")
    params: list[ParamDomain] = Field(default_factory=list)
    constraints: list[Constraint] = Field(default_factory=list)


class SearchEvaluator(Protocol):
    def evaluate[T](self, candidate_path: Path, params: Mapping[str, JsonValue],
                    context: Mapping[str, T]) -> TaskEvaluation: ...


@final
class TaskSearch:
    """Own run history; borrow one open evaluator across candidates and trials."""

    def __init__(self, evaluator: SearchEvaluator, objective: Objective,
                 budgets: BudgetConfig | None = None, *, device: DeviceLimits | None = None) -> None:
        self.evaluator = evaluator
        self.objective = objective
        self.budgets = budgets or BudgetConfig()
        self.device = device or DeviceLimits()
        self.families = FamilyManager(objective=objective)
        self.convergence = ConvergencePolicy(self.budgets, objective)
        self.trials: list[TrialRecord] = []
        self.stats: list[TuningStats] = []
        self.paths: dict[str, Path] = {}
        self.family_ids: dict[str, str] = {}
        self.spaces: dict[str, TaskSpace] = {}
        self.parents: dict[str, str | None] = {}
        self.resource_metrics: tuple[str, ...] = ()
        self.probe_budget: int = 0
        self.probe_calls: int = 0
        self.responses: list[TaskResponse] = []
        self.rewrites: list[dict[str, JsonValue]] = []
        self.final_execution: TrialRecord | None = None
        self.started = monotonic()

    def within_wall_budget(self) -> bool:
        return (monotonic() - self.started) / 3600 < self.budgets.wall_clock_hours

    def refresh_family_status(self, family: Family) -> None:
        """Apply the existing stop verdict without reopening an already frozen family."""
        if family.status != "active":
            return
        verdict = self.convergence.family_verdict(family)
        if verdict.verdict == "freeze":
            family.status = "frozen_converged" if verdict.stop_kind == "converged" else "frozen_budget"

    def evaluate_candidate[T](
        self, candidate: Path, space: TaskSpace, context: Mapping[str, T], *,
        parent_id: str | None = None, seed: int = 0,
        anchors: tuple[ParamSet, ...] = (), startup_trials: int = 10,
        can_evaluate: Callable[[], bool] | None = None, stop_on_invalid: bool = False,
    ) -> TrialRecord | None:
        """Return the child's best valid trial, or None for denial/no valid trial.

        Family-ineligible children are not registered. Admitted children remain in
        paths even when no evaluation runs or all actual evaluations are invalid.
        """
        candidate = candidate.resolve()
        cid = f"cand-{uuid4().hex[:8]}"
        fid = self.family_ids[parent_id] if parent_id else f"fam-{uuid4().hex[:8]}"
        if parent_id:
            family = self.families.families[fid]
            self.refresh_family_status(family)
            if family.status != "active":
                return None
        if fid not in self.families.families:
            self.families.families[fid] = Family(family_id=fid, anchor_candidate_id=cid)
        self.families.families[fid].member_ids.append(cid)
        self.paths[cid] = candidate
        self.family_ids[cid] = fid
        self.spaces[cid] = space
        self.parents[cid] = parent_id
        parameter_space = ParameterSpace(
            space_id=f"space-{cid}", candidate_id=cid, source_sha="",
            domains=space.params, constraints=space.constraints,
        )
        tuner = OptunaTPETuner(
            parameter_space,
            lambda p: all(eval_constraint(c.expr, {**self.device.as_env(), **p.values})
                          for c in space.constraints),
            budget=min(self.budgets.trials_per_space, 1) if not space.params else self.budgets.trials_per_space,
            seed=seed, objective=self.objective, anchors=anchors, n_startup_trials=startup_trials,
        )
        records: list[TrialRecord] = []
        while self.convergence.global_verdict(
            list(self.families.families.values()), (monotonic() - self.started) / 3600,
        ).verdict == "continue":
            if can_evaluate is not None and not can_evaluate():
                break
            asked = tuner.ask()
            if asked is None:
                break
            trial_id, params = asked
            values: dict[str, JsonValue] = dict(params.values)
            result = self.evaluator.evaluate(candidate, values, context)
            record = TrialRecord(
                trial_id=trial_id, candidate_id=cid, space_id=parameter_space.space_id,
                params=params, status="complete" if result.valid else "fail",
                task_evaluation=result, failure_detail=result.detail or "",
            )
            tuner.tell(trial_id, record)
            records.append(record)
            self.trials.append(record)
            if objective_value(record, self.objective) is not None:
                _ = self.families.update_record(fid, BestRecord(
                    candidate_id=cid, params=params, task_evaluation=result,
                ))
            if stop_on_invalid and not result.valid:
                break
        family = self.families.families[fid]
        if records and family.best is not None:
            value = objective_value(family.best, self.objective)
            if value is not None:
                self.families.record_round(fid, value)
        if parent_id:
            family.rewrite_rounds_used += 1
            if not records:
                self.families.record_round_not_evaluated(fid)
        self.refresh_family_status(family)
        self.stats.append(TuningStatsAnalyzer(self.device, self.objective).analyze(parameter_space, records))
        return tuner.best()

    def best(self) -> TrialRecord | None:
        valid = [t for t in self.trials if objective_value(t, self.objective) is not None]
        return min(valid, key=lambda t: rank_record(t, self.objective)) if valid else None

    def execute_best[T](self, context: Mapping[str, T]) -> TrialRecord | None:
        """Execute the selected winner once more, without reranking search observations."""
        best = self.best()
        if best is None:
            self.final_execution = None
            return None
        values: dict[str, JsonValue] = dict(best.params.values)
        result = self.evaluator.evaluate(self.paths[best.candidate_id], values, context)
        record = TrialRecord(
            trial_id=f"final-{uuid4().hex[:8]}", candidate_id=best.candidate_id,
            space_id=best.space_id, params=best.params,
            status="complete" if result.valid else "fail",
            task_evaluation=result, failure_detail=result.detail or "",
        )
        if objective_value(record, self.objective) is None:
            record = record.model_copy(update={"status": "fail"})
        self.final_execution = record
        return record
