"""Native J truth tables, independent of runtime and search."""

from pathlib import Path
from typing import Literal

import pytest

from kernel_optimizer.models.evaluation_bundle import EvalResult, QualityStatus
from kernel_optimizer.models.objective import AggregationSpec, Direction, MetricValue, ObjectiveSpec, RequiredMetric
from kernel_optimizer.models.task_bundle import ConstraintSpec, TaskDefinition


def task_for(direction: Direction = "minimize", kind: Literal["sum", "weighted_mean"] = "sum") -> TaskDefinition:
    task = TaskDefinition.model_validate_json(Path("tests/fixtures/v5/contracts/task.json").read_text())
    objective = ObjectiveSpec(name="harvest", unit="fruit", direction=direction,
                              required_metrics=(RequiredMetric(name="ripe_fruits", unit="fruit"),),
                              aggregation=AggregationSpec(kind=kind, case_weights={"a": 1.0, "b": 3.0}))
    return task.model_copy(update={"objective": objective})


def raw(values: tuple[float, ...], quality: QualityStatus = "pass") -> EvalResult:
    return EvalResult(objective_unit="fruit", direction="minimize", protocol_id="scientific:p",
                      quality=quality, feasibility="feasible", native_j=999.0,
                      case_metrics=tuple(MetricValue(name="ripe_fruits", unit="fruit", value=value, case_id=case)
                                         for case, value in zip(("a", "b"), values, strict=False)))


@pytest.mark.parametrize("direction", ["minimize", "maximize"])
@pytest.mark.parametrize("case", [
    ("sum", (2.0, 4.0), 6.0), ("weighted_mean", (2.0, 4.0), 3.5),
    ("sum", (-2.0, -4.0), -6.0), ("sum", (-0.0, 0.0), 0.0),
])
def test_min_max_zero_signed_ties_and_aggregation(
    direction: Direction, case: tuple[Literal["sum", "weighted_mean"], tuple[float, ...], float],
) -> None:
    from kernel_optimizer.evaluation.objective import aggregate_metrics, compare_results, is_eligible
    from kernel_optimizer.evaluation.objective_types import ScoringContext

    # Given
    kind, values, expected = case
    context = ScoringContext(task_for(direction, kind), "scientific:p")
    evidence = raw(values).model_copy(update={"direction": direction})
    # When
    result = aggregate_metrics(context, evidence)
    comparison = compare_results(context, result, result)
    # Then
    assert result.native_j == expected
    assert is_eligible(context, result)
    assert comparison.kind == "comparable"
    assert comparison.winner == "incumbent"
    assert comparison.improvement == 0.0


@pytest.mark.parametrize("quality", ["fail", "unknown"])
def test_unknown_infeasible_and_incompatible_refused(quality: QualityStatus) -> None:
    from kernel_optimizer.evaluation.objective import aggregate_metrics, compare_results, is_eligible
    from kernel_optimizer.evaluation.objective_types import ScoringContext

    # Given
    context = ScoringContext(task_for(), "scientific:p")
    evidence = raw((1.0, 2.0), quality)
    # When
    result = aggregate_metrics(context, evidence)
    comparison = compare_results(context, result, raw((4.0, 5.0)))
    # Then
    assert result.native_j == 3.0
    assert result.quality == quality
    assert not is_eligible(context, result)
    assert comparison.kind == "incompatible"


@pytest.mark.parametrize("aggregation", ["all", "weighted_mean", "max", "sum"])
def test_min_max_zero_signed_ties_and_aggregation_nonresource(
    aggregation: Literal["all", "weighted_mean", "max", "sum"],
) -> None:
    from kernel_optimizer.evaluation.objective import aggregate_metrics, is_eligible
    from kernel_optimizer.evaluation.objective_types import ScoringContext

    # Given
    constraints = (
        ConstraintSpec(id="completed", source_kind="metric", name="completed_requests", unit="request",
                       operator="ge", threshold=2.0, case_ids=("a", "b"), aggregation=aggregation),
        ConstraintSpec(id="errors", source_kind="metric", name="error_rate", unit="ratio",
                       operator="le", threshold=0.1, case_ids=("a", "b"), aggregation=aggregation),
    )
    evidence = raw((2.0, 4.0)).model_copy(update={"metrics": tuple(
        MetricValue(name=name, unit=unit, case_id=case, value=value)
        for name, unit, value in (("completed_requests", "request", 2.0), ("error_rate", "ratio", 0.04))
        for case in ("a", "b")
    )})
    context = ScoringContext(task_for().model_copy(update={"constraints": constraints}), "scientific:p")
    # When
    result = aggregate_metrics(context, evidence)
    # Then
    assert tuple(item.status for item in result.constraint_results) == ("satisfied", "satisfied")
    assert is_eligible(context, result)


@pytest.mark.parametrize("violation", [True, False])
def test_unknown_infeasible_and_incompatible_refused_constraints(violation: bool) -> None:
    from kernel_optimizer.evaluation.objective import aggregate_metrics, is_eligible
    from kernel_optimizer.evaluation.objective_types import ScoringContext

    # Given
    constraints = (ConstraintSpec(id="known", source_kind="metric", name="ripe_fruits", unit="fruit",
                                  operator="le", threshold=1.0 if violation else 10.0,
                                  case_ids=("a", "b"), aggregation="all"),
                   ConstraintSpec(id="missing", source_kind="metric", name="absent", unit="fruit",
                                  operator="le", threshold=1.0, case_ids=("a",), aggregation="all"))
    context = ScoringContext(task_for().model_copy(update={"constraints": constraints}), "scientific:p")
    # When
    result = aggregate_metrics(context, raw((2.0, 4.0)))
    # Then
    assert result.feasibility == ("infeasible" if violation else "unknown")
    assert len(result.constraint_results) == 2
    assert result.native_j == 6.0
    assert not is_eligible(context, result)


@pytest.mark.parametrize("field", ["protocol_id", "objective_unit", "direction"])
def test_unknown_infeasible_and_incompatible_refused_identity(field: str) -> None:
    from kernel_optimizer.evaluation.objective import compare_results
    from kernel_optimizer.evaluation.objective_types import ScoringContext

    # Given
    changed = {"protocol_id": "other", "objective_unit": "ms", "direction": "maximize"}
    candidate = raw((1.0, 2.0)).model_copy(update={field: changed[field]})
    # When
    result = compare_results(ScoringContext(task_for(), "scientific:p"), candidate, raw((4.0, 5.0)))
    # Then
    assert result.kind == "incompatible"


@pytest.mark.parametrize("values", [(), (1.0,)])
def test_unknown_infeasible_and_incompatible_refused_missing(values: tuple[float, ...]) -> None:
    from kernel_optimizer.evaluation.objective import aggregate_metrics, is_eligible
    from kernel_optimizer.evaluation.objective_types import ScoringContext

    # Given
    context = ScoringContext(task_for(), "scientific:p")
    # When
    result = aggregate_metrics(context, raw(values))
    # Then
    assert result.measurement_validity == "unknown"
    assert result.native_j is None
    assert not is_eligible(context, result)
