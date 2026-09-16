"""Adversarial measurement boundaries and arbitrary callable objectives."""

from pathlib import Path
from typing import Literal, assert_never

import pytest
from pydantic import ValidationError

from kernel_optimizer.evaluation.objective import aggregate_metrics, compare_results, directional_improvement, is_eligible
from kernel_optimizer.evaluation.objective_types import Incompatibility, ScoringContext
from kernel_optimizer.models.contract_common import CallableRef
from kernel_optimizer.models.evaluation_bundle import EvalResult
from kernel_optimizer.models.objective import AggregationSpec, MetricValue, ObjectiveSpec, RequiredMetric
from kernel_optimizer.models.task_bundle import ConstraintSpec
from tests.test_v5_objective import raw, task_for


@pytest.mark.parametrize("change", ["unit", "conflict", "outside", "empty", "ambiguous", "conversion"])
def test_malformed_or_ambiguous_evidence_cannot_be_eligible(
    change: Literal["unit", "conflict", "outside", "empty", "ambiguous", "conversion"],
) -> None:
    # Given
    task = task_for()
    evidence = raw((2.0, 4.0))
    match change:
        case "unit":
            evidence = evidence.model_copy(update={"metrics": (MetricValue(name="ripe_fruits", unit="ms", value=2.0, case_id="a"),)})
        case "conflict":
            evidence = evidence.model_copy(update={"metrics": (MetricValue(name="ripe_fruits", unit="fruit", value=9.0, case_id="a"),)})
        case "outside":
            evidence = evidence.model_copy(update={"metrics": (MetricValue(name="ripe_fruits", unit="fruit", value=2.0, case_id="c"),)})
        case "empty":
            task = task.model_copy(update={"objective": task.objective.model_copy(update={"required_metrics": ()})})
        case "ambiguous":
            task = task.model_copy(update={"objective": task.objective.model_copy(update={"required_metrics": (
                *task.objective.required_metrics, RequiredMetric(name="other", unit="fruit"))})})
            evidence = evidence.model_copy(update={"metrics": tuple(m.model_copy(update={"name": "other"}) for m in evidence.case_metrics)})
        case "conversion":
            task = task.model_copy(update={"objective": task.objective.model_copy(update={"unit": "basket"})})
            evidence = evidence.model_copy(update={"objective_unit": "basket"})
        case _:
            assert_never(change)
    context = ScoringContext(task, "scientific:p")
    # When
    result = aggregate_metrics(context, evidence)
    # Then
    assert result.native_j is None
    assert result.measurement_validity == ("unknown" if change == "empty" else "invalid")
    assert not is_eligible(context, result)


def test_identical_duplicates_are_idempotent_across_raw_channels() -> None:
    # Given
    evidence = raw((2.0, 4.0))
    evidence = evidence.model_copy(update={"metrics": evidence.case_metrics})
    # When
    result = aggregate_metrics(ScoringContext(task_for(), "scientific:p"), evidence)
    # Then
    assert result.native_j == 6.0


def test_arbitrary_callable_outputs_declared_unit_without_implicit_conversion() -> None:
    # Given
    task = task_for()
    objective = ObjectiveSpec(name="arbitrary", unit="basket", direction="minimize",
                              required_metrics=(*task.objective.required_metrics, RequiredMetric(name="cost", unit="coin")),
                              aggregation=AggregationSpec(kind="bundle_callable", callable=CallableRef(module="frozen.math", callable="score")))
    task = task.model_copy(update={"objective": objective})
    evidence = raw((2.0, 4.0)).model_copy(update={"objective_unit": "basket", "metrics": (
        MetricValue(name="cost", unit="coin", case_id="a", value=3.0),
        MetricValue(name="cost", unit="coin", case_id="b", value=5.0))})

    def score(spec: ObjectiveSpec, values: tuple[MetricValue, ...]) -> float:
        assert spec == objective
        return sum(value.value ** 2 for value in values) - 60.0

    # When
    result = aggregate_metrics(ScoringContext(task, "scientific:p", score), evidence)
    # Then
    assert result.native_j == -6.0
    assert result.objective_unit == "basket"


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf"), True])
def test_callable_output_is_a_numeric_boundary(value: float) -> None:
    # Given
    task = task_for()
    task = task.model_copy(update={"objective": task.objective.model_copy(update={"aggregation":
        AggregationSpec(kind="bundle_callable", callable=CallableRef(module="score", callable="main"))})})

    def score(_spec: ObjectiveSpec, values: tuple[MetricValue, ...]) -> float:
        return value * sum(item.value for item in values) if value is not True else value

    # When
    result = aggregate_metrics(ScoringContext(task, "scientific:p", score), raw((2.0, 4.0)))
    # Then
    assert result.measurement_validity == "invalid"
    assert result.native_j is None


@pytest.mark.parametrize("missing", [True, False])
def test_callable_is_not_called_before_complete_units(missing: bool) -> None:
    # Given
    task = task_for()
    task = task.model_copy(update={"objective": task.objective.model_copy(update={"aggregation":
        AggregationSpec(kind="bundle_callable", callable=CallableRef(module="score", callable="main"))})})
    evidence = raw((2.0,) if missing else (2.0, 4.0))
    if not missing:
        evidence = evidence.model_copy(update={"case_metrics": tuple(m.model_copy(update={"unit": "ms"}) for m in evidence.case_metrics)})

    def score(_spec: ObjectiveSpec, _values: tuple[MetricValue, ...]) -> float:
        pytest.fail("incomplete measurements reached aggregation")

    # When
    result = aggregate_metrics(ScoringContext(task, "scientific:p", score), evidence)
    # Then
    assert result.native_j is None


@pytest.mark.parametrize(("kind", "expected"), [("sum", None), ("weighted_mean", 1e308)])
def test_finite_inputs_do_not_leak_overflow(kind: str, expected: float | None) -> None:
    # Given
    task = task_for("minimize", "weighted_mean" if kind == "weighted_mean" else "sum")
    context = ScoringContext(task, "scientific:p")
    # When
    result = aggregate_metrics(context, raw((1e308, 1e308)))
    # Then
    assert result.native_j == expected
    assert result.measurement_validity == ("valid" if expected is not None else "invalid")


def test_directional_difference_overflow_is_incompatible() -> None:
    # Given
    context = ScoringContext(task_for(), "scientific:p")
    # When
    result = directional_improvement(context, raw((-1e308, 0.0)), raw((1e308, 0.0)))
    # Then
    assert isinstance(result, Incompatibility)
    assert result.kind == "incompatible"


@pytest.mark.parametrize(("direction", "expected"), [("minimize", 7.0), ("maximize", -7.0)])
def test_comparison_recomputes_raw_j(direction: str, expected: float) -> None:
    # Given
    task = task_for("minimize" if direction == "minimize" else "maximize")
    candidate = raw((-2.0, 0.0)).model_copy(update={"direction": direction})
    reference = raw((2.0, 3.0)).model_copy(update={"direction": direction})
    # When
    result = compare_results(ScoringContext(task, "scientific:p"), candidate, reference)
    # Then
    assert result.kind == "comparable"
    assert result.improvement == expected
    assert result.winner == ("candidate" if expected > 0 else "incumbent")


def test_json_supplied_score_is_not_authoritative() -> None:
    # Given
    evidence = EvalResult.model_validate_json(Path("tests/fixtures/v5/contracts/result.json").read_text())
    context = ScoringContext(task_for(), evidence.protocol_id)
    # When
    result = aggregate_metrics(context, evidence)
    roundtrip = EvalResult.model_validate_json(result.model_dump_json())
    # Then
    assert roundtrip.native_j is None
    assert not is_eligible(context, roundtrip)


@pytest.mark.parametrize("invalid", ["true", "NaN", "Infinity", "-Infinity"])
def test_nonfinite_and_bool_json_are_rejected(invalid: str) -> None:
    # Given
    payload = '{"name":"arbitrary","unit":"u","case_id":"a","value":' + invalid + '}'
    # When / Then
    with pytest.raises(ValidationError):
        _ = MetricValue.model_validate_json(payload)


def test_infeasible_better_j_never_becomes_incumbent() -> None:
    # Given
    constraint = ConstraintSpec(id="floor", source_kind="metric", name="ripe_fruits", unit="fruit",
                                operator="ge", threshold=1.0, case_ids=("a", "b"), aggregation="all")
    context = ScoringContext(task_for().model_copy(update={"constraints": (constraint,)}), "scientific:p")
    # When
    result = aggregate_metrics(context, raw((-10.0, 2.0)))
    comparison = compare_results(context, result, raw((2.0, 3.0)))
    # Then
    assert result.native_j == -8.0
    assert result.measurement_validity == "valid"
    assert result.feasibility == "infeasible"
    assert comparison.kind == "incompatible"
