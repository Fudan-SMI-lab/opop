"""Constraint truth tables exercise raw evidence rather than reported feasibility."""

from typing import Literal

import pytest

from kernel_optimizer.models.objective import MetricValue
from kernel_optimizer.models.resource_axis import ResourceObservation
from kernel_optimizer.models.task_bundle import ConstraintSpec
from tests.test_v5_objective import raw, task_for


@pytest.mark.parametrize(("aggregation", "threshold", "expected"), [
    ("all", 2.0, "satisfied"), ("sum", 6.0, "satisfied"),
    ("weighted_mean", 3.5, "satisfied"), ("max", 4.0, "satisfied"),
    ("all", 3.0, "violated"), ("sum", 7.0, "violated"),
    ("weighted_mean", 3.6, "violated"), ("max", 5.0, "violated"),
])
def test_nonresource_constraint_aggregation(
    aggregation: Literal["all", "sum", "weighted_mean", "max"], threshold: float, expected: str,
) -> None:
    from kernel_optimizer.evaluation.constraints import evaluate_constraints

    # Given
    constraints = (ConstraintSpec(id="completed", source_kind="metric", name="completed_requests", unit="request",
                                  operator="ge", threshold=threshold, case_ids=("a", "b"), aggregation=aggregation),
                   ConstraintSpec(id="errors", source_kind="metric", name="error_rate", unit="ratio",
                                  operator="le", threshold=0.1, case_ids=("a", "b"), aggregation="all"))
    task = task_for().model_copy(update={"constraints": constraints})
    evidence = raw((2.0, 4.0)).model_copy(update={"metrics": (
        MetricValue(name="completed_requests", unit="request", value=2.0, case_id="a"),
        MetricValue(name="completed_requests", unit="request", value=4.0, case_id="b"),
        MetricValue(name="error_rate", unit="ratio", value=0.1, case_id="a"),
        MetricValue(name="error_rate", unit="ratio", value=0.0, case_id="b"))})
    # When
    results = evaluate_constraints(task, evidence)
    # Then
    assert tuple(item.status for item in results) == (expected, "satisfied")


@pytest.mark.parametrize(("operator", "expected"), [("lt", "violated"), ("le", "satisfied"),
                                                    ("eq", "satisfied"), ("ge", "satisfied"), ("gt", "violated")])
def test_constraint_operator_boundary(operator: Literal["lt", "le", "eq", "ge", "gt"], expected: str) -> None:
    from kernel_optimizer.evaluation.constraints import evaluate_constraints

    # Given
    constraint = ConstraintSpec(id="limit", source_kind="metric", name="ripe_fruits", unit="fruit",
                                operator=operator, threshold=2.0, case_ids=("a",), aggregation="all")
    # When
    results = evaluate_constraints(task_for().model_copy(update={"constraints": (constraint,)}), raw((2.0, 4.0)))
    # Then
    assert results[0].status == expected


@pytest.mark.parametrize(("aggregation", "expected"), [("all", "infeasible"), ("sum", "unknown"),
                                                       ("max", "unknown"), ("weighted_mean", "unknown")])
def test_incomplete_constraints_keep_known_violation_only_for_all(
    aggregation: Literal["all", "sum", "weighted_mean", "max"], expected: str,
) -> None:
    from kernel_optimizer.evaluation.objective import aggregate_metrics, is_eligible
    from kernel_optimizer.evaluation.objective_types import ScoringContext

    # Given
    violation = ConstraintSpec(id="limit", source_kind="metric", name="ripe_fruits", unit="fruit",
                               operator="le", threshold=1.0, case_ids=("a", "b"), aggregation=aggregation)
    missing = violation.model_copy(update={"id": "missing", "name": "missing"})
    context = ScoringContext(task_for().model_copy(update={"constraints": (violation, missing)}), "scientific:p")
    # When
    result = aggregate_metrics(context, raw((2.0,)))
    # Then
    assert result.feasibility == expected
    assert len(result.constraint_results) == 2
    assert result.constraint_results[1].status == "unknown"
    assert not is_eligible(context, result)


@pytest.mark.parametrize("entity", [None, "kernel:a", "kernel:b"])
def test_resource_entity_is_exact(entity: str | None) -> None:
    from kernel_optimizer.evaluation.constraints import evaluate_constraints

    # Given
    constraint = ConstraintSpec(id="memory", source_kind="resource", name="memory", unit="byte", entity_id=entity,
                                operator="le", threshold=10.0, case_ids=("a",), aggregation="max")
    observation = ResourceObservation(name="memory", entity_id="kernel:a", unit="byte", scope="kernel",
                                      provenance="probe", value=5.0, case_id="a", entity_type="kernel", context_ref="p")
    # When
    result = evaluate_constraints(task_for().model_copy(update={"constraints": (constraint,)}),
                                  raw((2.0, 4.0)).model_copy(update={"resources": (observation,)}))
    # Then
    assert result[0].status == ("satisfied" if entity == "kernel:a" else "unknown")


@pytest.mark.parametrize("conflict", [True, False])
def test_resource_conflict_or_wrong_unit_remains_unknown(conflict: bool) -> None:
    from kernel_optimizer.evaluation.constraints import evaluate_constraints

    # Given
    constraint = ConstraintSpec(id="memory", source_kind="resource", name="memory", unit="byte",
                                operator="le", threshold=10.0, case_ids=("a",), aggregation="max")
    observation = ResourceObservation(name="memory", entity_id=None, unit="byte", scope="task",
                                      provenance="probe", value=5.0, case_id="a", entity_type="task", context_ref="p")
    other = observation.model_copy(update={"value": 99.0} if conflict else {"unit": "KiB"})
    # When
    result = evaluate_constraints(task_for().model_copy(update={"constraints": (constraint,)}),
                                  raw((2.0, 4.0)).model_copy(update={"resources": (observation, other)}))
    # Then
    assert result[0].status == "unknown"
    assert result[0].aggregate_value is None
    assert result[0].reasons


def test_constraint_aggregate_overflow_remains_unknown() -> None:
    from kernel_optimizer.evaluation.constraints import evaluate_constraints

    # Given
    constraint = ConstraintSpec(id="limit", source_kind="metric", name="ripe_fruits", unit="fruit",
                                operator="ge", threshold=0.0, case_ids=("a", "b"), aggregation="sum")
    # When
    result = evaluate_constraints(task_for().model_copy(update={"constraints": (constraint,)}), raw((1e308, 1e308)))
    # Then
    assert result[0].status == "unknown"
    assert result[0].aggregate_value is None
