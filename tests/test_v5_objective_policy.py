"""Frozen policy completeness, serialization, and numerical edge contracts."""

import json

import pytest
from pydantic import ValidationError

from kernel_optimizer.evaluation.constraints import evaluate_constraints
from kernel_optimizer.evaluation.objective import aggregate_metrics, compare_results, is_eligible
from kernel_optimizer.evaluation.objective_types import ScoringContext
from kernel_optimizer.models.contract_common import CallableRef
from kernel_optimizer.models.evaluation_bundle import EvalResult
from kernel_optimizer.models.objective import AggregationSpec, MetricValue, ObjectiveSpec
from kernel_optimizer.models.task_bundle import ConstraintSpec, TaskDefinition
from tests.test_v5_objective import raw, task_for


@pytest.mark.parametrize("payload", [
    '{"threshold":true,"case_ids":["a"]}', '{"threshold":NaN,"case_ids":["a"]}',
    '{"threshold":Infinity,"case_ids":["a"]}', '{"threshold":1,"case_ids":[]}',
])
def test_unknown_infeasible_and_incompatible_refused_malformed_constraint(payload: str) -> None:
    # Given
    base = '{"id":"c","source_kind":"metric","name":"x","unit":"u","operator":"le","aggregation":"all",'
    # When / Then
    with pytest.raises(ValidationError):
        _ = ConstraintSpec.model_validate_json(base + payload[1:])


def test_weighted_constraint_normalizes_selected_subset() -> None:
    # Given
    task = task_for("minimize", "weighted_mean")
    constraint = ConstraintSpec(id="subset", source_kind="metric", name="ripe_fruits", unit="fruit",
                                operator="eq", threshold=4.0, case_ids=("b",), aggregation="weighted_mean")
    task = task.model_copy(update={"constraints": (constraint,)})
    # When
    results = evaluate_constraints(task, raw((2.0, 4.0)))
    # Then
    assert results[0].aggregate_value == 4.0
    assert results[0].status == "satisfied"


@pytest.mark.parametrize("weights", [{"a": 0.0, "b": 3.0}, {"a": 1e308, "b": 1e308}])
def test_weighted_zero_and_overflowing_weight_sum(weights: dict[str, float]) -> None:
    # Given
    task = task_for("minimize", "weighted_mean")
    task = task.model_copy(update={"objective": task.objective.model_copy(update={"aggregation":
        AggregationSpec(kind="weighted_mean", case_weights=weights)})})
    # When
    result = aggregate_metrics(ScoringContext(task, "scientific:p"), raw((2.0, 4.0)))
    # Then
    assert result.native_j == (4.0 if weights["a"] == 0.0 else 3.0)


def test_sum_cancellation_survives_intermediate_overflow() -> None:
    # Given
    task = task_for()
    task = task.model_copy(update={"workload": task.workload.model_copy(update={"case_ids": ("a", "b", "c")})})
    evidence = raw((1e308, 1e308))
    evidence = evidence.model_copy(update={"metrics": (MetricValue(name="ripe_fruits", unit="fruit", value=-1e308, case_id="c"),)})
    # When
    result = aggregate_metrics(ScoringContext(task, "scientific:p"), evidence)
    # Then
    assert result.native_j == 1e308


@pytest.mark.parametrize("threshold", [6.0, 7.0, 8.0])
def test_improvement_threshold_is_strict_and_absolute(threshold: float) -> None:
    # Given
    task = task_for()
    task = task.model_copy(update={"objective": task.objective.model_copy(update={"improvement_threshold": threshold})})
    # When
    result = compare_results(ScoringContext(task, "scientific:p"), raw((-2.0, 0.0)), raw((2.0, 3.0)))
    # Then
    assert result.kind == "comparable"
    assert result.winner == ("candidate" if threshold < 7.0 else "incumbent")


def test_real_json_models_to_score_to_serialized_result() -> None:
    # Given
    task = TaskDefinition.model_validate_json(task_for().model_dump_json())
    evidence = EvalResult.model_validate_json(raw((-2.0, 5.0)).model_dump_json())
    context = ScoringContext(task, "scientific:p")
    # When
    result = aggregate_metrics(context, evidence)
    serialized = result.model_dump_json()
    # Then
    assert json.loads(serialized)["native_j"] == 3.0
    assert is_eligible(context, EvalResult.model_validate_json(serialized))


def test_complete_metrics_do_not_manufacture_quality_pass() -> None:
    # Given
    evidence = raw((2.0, 4.0), "unknown")
    context = ScoringContext(task_for(), "scientific:p")
    # When
    result = aggregate_metrics(context, evidence)
    # Then
    assert result.quality == "unknown"
    assert result.native_j == 6.0
    assert not is_eligible(context, result)


@pytest.mark.parametrize("resolved", [True, False])
def test_bundle_unresolved_or_arithmetic_failure_is_unavailable(resolved: bool) -> None:
    # Given
    task = task_for()
    task = task.model_copy(update={"objective": task.objective.model_copy(update={"aggregation":
        AggregationSpec(kind="bundle_callable", callable=CallableRef(module="frozen", callable="ratio"))})})

    def ratio(_spec: ObjectiveSpec, values: tuple[MetricValue, ...]) -> float:
        return values[0].value / values[1].value

    # When
    result = aggregate_metrics(ScoringContext(task, "scientific:p", ratio if resolved else None), raw((1.0, 0.0)))
    # Then
    assert result.native_j is None
    assert result.measurement_validity == ("invalid" if resolved else "unknown")


def test_constraint_conflicting_duplicates_do_not_legitimize_pass() -> None:
    # Given
    task = task_for()
    constraint = ConstraintSpec(id="limit", source_kind="metric", name="ripe_fruits", unit="fruit",
                                operator="le", threshold=3.0, case_ids=("a",), aggregation="all")
    evidence = raw((2.0, 4.0)).model_copy(update={"metrics": (
        MetricValue(name="ripe_fruits", unit="fruit", value=99.0, case_id="a"),)})
    context = ScoringContext(task.model_copy(update={"constraints": (constraint,)}), "scientific:p")
    # When
    result = aggregate_metrics(context, evidence)
    # Then
    assert result.constraint_results[0].status == "unknown"
    assert result.constraint_results[0].reasons
    assert not is_eligible(context, result)


def test_callable_unrepresentable_integer_returns_typed_invalid() -> None:
    # Given
    task = task_for()
    task = task.model_copy(update={"objective": task.objective.model_copy(update={"aggregation":
        AggregationSpec(kind="bundle_callable", callable=CallableRef(module="frozen", callable="score"))})})

    def score(_spec: ObjectiveSpec, _values: tuple[MetricValue, ...]) -> float:
        return 10 ** 1000

    # When
    result = aggregate_metrics(ScoringContext(task, "scientific:p", score), raw((1.0, 2.0)))
    # Then
    assert result.measurement_validity == "invalid"
    assert result.native_j is None
