"""Derive every constraint from raw observations, never reported feasibility."""

from typing import assert_never

from kernel_optimizer.evaluation.measurement_values import reduce_values, select_values
from kernel_optimizer.models.evaluation_bundle import ConstraintResult, EvalResult, FeasibilityStatus
from kernel_optimizer.models.objective import MetricValue
from kernel_optimizer.models.task_bundle import ConstraintSpec, TaskDefinition


def _satisfies(value: float, spec: ConstraintSpec) -> bool:
    match spec.operator:
        case "lt":
            return value < spec.threshold
        case "le":
            return value <= spec.threshold
        case "eq":
            return value == spec.threshold
        case "ge":
            return value >= spec.threshold
        case "gt":
            return value > spec.threshold
        case _:
            assert_never(spec.operator)


def evaluate_constraints(task: TaskDefinition, raw: EvalResult) -> tuple[ConstraintResult, ...]:
    """Null resource entities require both scope and entity_type to be canonical 'task'."""
    results: list[ConstraintResult] = []
    for spec in task.constraints:
        match spec.source_kind:
            case "metric":
                records = tuple(record for record in raw.case_metrics + raw.metrics
                                if record.name == spec.name and spec.entity_id is None)
            case "resource":
                records = tuple(MetricValue(name=record.name, value=record.value,
                                            case_id=record.case_id, unit=record.unit)
                                for record in raw.resources
                                if record.name == spec.name and record.entity_id == spec.entity_id
                                and (spec.entity_id is not None
                                     or (record.scope == "task" and record.entity_type == "task")))
            case _:
                assert_never(spec.source_kind)
        selected = select_values(records, spec.case_ids, spec.unit)
        observed = {record.case_id: record.value for record in selected.values}
        reasons = selected.errors + selected.missing
        aggregate: float | None = None
        status = "unknown"
        match spec.aggregation:
            case "all":
                if any(not _satisfies(value, spec) for value in observed.values()):
                    status = "violated"
                elif not reasons:
                    status = "satisfied"
            case "sum" | "max" | "weighted_mean":
                if not reasons:
                    weights = tuple(task.objective.aggregation.case_weights[case] for case in spec.case_ids
                                    if case in task.objective.aggregation.case_weights)
                    aggregate = reduce_values(tuple(observed.values()), spec.aggregation, weights)
                    if aggregate is None:
                        reasons += ("aggregation_overflow",)
                    else:
                        status = "satisfied" if _satisfies(aggregate, spec) else "violated"
            case _:
                assert_never(spec.aggregation)
        if status == "violated":
            reasons += ("threshold_violated",)
        results.append(ConstraintResult(id=spec.id, status=status, observed_values=observed,
                                        aggregate_value=aggregate, unit=spec.unit, reasons=reasons))
    return tuple(results)


def derive_feasibility(results: tuple[ConstraintResult, ...]) -> FeasibilityStatus:
    """Violation dominates unknown; the empty set is feasible."""
    if any(result.status == "violated" for result in results):
        return "infeasible"
    if any(result.status == "unknown" for result in results):
        return "unknown"
    return "feasible"
