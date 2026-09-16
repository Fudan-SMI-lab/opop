"""Authoritative native J from raw measurements; independent quality is never inferred."""

from math import isfinite
from typing import Final, assert_never

from pydantic import TypeAdapter, ValidationError

from kernel_optimizer.evaluation.constraints import derive_feasibility, evaluate_constraints
from kernel_optimizer.evaluation.measurement_values import reduce_values, select_values
from kernel_optimizer.evaluation.objective_types import (
    Comparison, ComparisonResult, ImprovementResult, Incompatibility, ScoringContext,
)
from kernel_optimizer.models.evaluation_bundle import EvalResult, MeasurementValidity
from kernel_optimizer.models.contract_common import FiniteNumber
from kernel_optimizer.models.objective import MetricValue

_FINITE_J: Final[TypeAdapter[float]] = TypeAdapter(FiniteNumber)


def aggregate_metrics(context: ScoringContext, raw: EvalResult) -> EvalResult:
    """Raw quality must come from the independent oracle, never candidate claims.

    Recompute J/constraints even for apparently valid supplied results. The caller
    owns oracle execution (Task 12); this function only preserves its verdict.
    """
    task = context.task
    objective = task.objective
    constraints = evaluate_constraints(task, raw)
    errors: list[str] = []
    missing: list[str] = []
    values: list[MetricValue] = []
    if (raw.protocol_id != context.protocol_id or raw.objective_unit != objective.unit
            or raw.direction != objective.direction):
        errors.append("incompatible_objective_context")
    if raw.measurement_validity == "invalid":
        errors.append("upstream_invalid_measurement")
    if not objective.required_metrics:
        missing.append("empty_required_metrics")
    for required in objective.required_metrics:
        records = tuple(record for record in raw.case_metrics + raw.metrics if record.name == required.name)
        selected = select_values(records, task.workload.case_ids, required.unit)
        errors.extend(f"{required.name}:{reason}" for reason in selected.errors)
        missing.extend(f"{required.name}:{reason}" for reason in selected.missing)
        values.extend(selected.values)
        if any(record.case_id not in task.workload.case_ids for record in records):
            errors.append(f"{required.name}:undeclared_case")
    native_j: float | None = None
    if not errors and not missing:
        match objective.aggregation.kind:
            case "sum" | "weighted_mean":
                if len(objective.required_metrics) != 1:
                    errors.append("ambiguous_builtin_metrics")
                elif objective.required_metrics[0].unit != objective.unit:
                    errors.append("objective_unit_mismatch")
                else:
                    weights = tuple(objective.aggregation.case_weights[case] for case in task.workload.case_ids
                                    if case in objective.aggregation.case_weights)
                    native_j = reduce_values(tuple(record.value for record in values),
                                             objective.aggregation.kind, weights)
                    if native_j is None:
                        errors.append("aggregation_overflow")
            case "bundle_callable":
                if context.aggregator is None:
                    missing.append("aggregation_callable_unresolved")
                else:
                    try:
                        native_j = _FINITE_J.validate_python(context.aggregator(objective, tuple(values)))
                    except ArithmeticError:
                        errors.append("aggregation_arithmetic_error")
                    except ValidationError:
                        errors.append("aggregation_invalid_number")
            case _:
                assert_never(objective.aggregation.kind)
    validity: MeasurementValidity = "invalid" if errors else "unknown" if missing else "valid"
    return EvalResult(measurement_validity=validity, quality=raw.quality,
                      feasibility=derive_feasibility(constraints), constraint_results=constraints,
                      case_metrics=raw.case_metrics, metrics=raw.metrics, native_j=native_j,
                      objective_unit=raw.objective_unit, direction=raw.direction, protocol_id=raw.protocol_id,
                      resources=raw.resources, reasons=tuple(dict.fromkeys((*raw.reasons, *errors, *missing))))


def _eligible(result: EvalResult) -> bool:
    return (result.measurement_validity == "valid" and result.quality == "pass"
            and result.feasibility == "feasible" and result.native_j is not None)


def is_eligible(context: ScoringContext, result: EvalResult) -> bool:
    """Recompute evidence; self-reported J and feasibility cannot establish eligibility."""
    return _eligible(aggregate_metrics(context, result))


def directional_improvement(context: ScoringContext, candidate: EvalResult,
                            reference: EvalResult) -> ImprovementResult:
    """Return q*(candidate-reference) in native units, or a typed refusal."""
    candidate_result = aggregate_metrics(context, candidate)
    reference_result = aggregate_metrics(context, reference)
    if not _eligible(candidate_result) or not _eligible(reference_result):
        return Incompatibility(("ineligible_or_incompatible_endpoint",))
    candidate_j, reference_j = candidate_result.native_j, reference_result.native_j
    # The explicit narrowing is needed because Python cannot infer _eligible's postcondition.
    if candidate_j is None or reference_j is None:
        return Incompatibility(("missing_objective",))
    match context.task.objective.direction:
        case "minimize":
            difference = reference_j - candidate_j
        case "maximize":
            difference = candidate_j - reference_j
        case _:
            assert_never(context.task.objective.direction)
    if not isfinite(difference):
        return Incompatibility(("improvement_overflow",))
    return difference


def compare_results(context: ScoringContext, candidate: EvalResult, incumbent: EvalResult) -> ComparisonResult:
    """Only a strict improvement above the frozen absolute threshold replaces incumbent."""
    result = directional_improvement(context, candidate, incumbent)
    match result:
        case Incompatibility() as refused:
            return refused
        case float() | int() as improvement:
            winner = "candidate" if improvement > context.task.objective.improvement_threshold else "incumbent"
            return Comparison(winner, improvement)
        case _:
            assert_never(result)
