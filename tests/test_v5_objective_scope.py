"""JSON boundary regressions for the independently reproduced null-entity scope bug."""

import json
from pathlib import Path
from typing import Final

import pytest

from kernel_optimizer.evaluation.objective import aggregate_metrics, compare_results, is_eligible
from kernel_optimizer.evaluation.objective_types import ScoringContext
from kernel_optimizer.models.evaluation_bundle import EvalResult
from kernel_optimizer.models.task_bundle import TaskDefinition

type ResourceRow = tuple[str, str, str | None, float]

TASK_ROW: Final[ResourceRow] = ("task", "task", None, 5.0)
KERNEL_ROW: Final[ResourceRow] = ("kernel", "kernel", None, 5.0)
KERNEL_HIGH_ROW: Final[ResourceRow] = ("kernel", "kernel", None, 50.0)


def scope_context(entity_id: str | None = None) -> ScoringContext:
    document = Path("tests/fixtures/v5/objective_scope/task.json").read_text(encoding="utf-8")
    if entity_id is not None:
        document = document.replace('"entity_id": null', '"entity_id": ' + json.dumps(entity_id))
    return ScoringContext(TaskDefinition.model_validate_json(document), "independent:p")


def scope_evidence(rows: tuple[ResourceRow, ...], values: tuple[float, float]) -> EvalResult:
    document = {
        "protocol_id": "independent:p", "objective_unit": "quanta", "direction": "minimize",
        "quality": "pass", "native_j": 999,
        "case_metrics": [{"name": "novel_flux", "unit": "quanta", "case_id": case, "value": value}
                         for case, value in zip(("a", "b"), values, strict=True)],
        "resources": [{"name": "novel_capacity", "entity_id": entity, "unit": "cells",
                       "scope": scope, "entity_type": entity_type, "case_id": "a",
                       "context_ref": "resource:acquisition", "provenance": "sensor", "value": value}
                      for scope, entity_type, entity, value in rows],
    }
    return EvalResult.model_validate_json(json.dumps(document))


@pytest.mark.parametrize(("rows", "eligible"), [
    pytest.param((TASK_ROW,), True, id="explicit-task-5"),
    pytest.param((KERNEL_ROW,), False, id="kernel-null-5"),
    pytest.param((TASK_ROW, KERNEL_HIGH_ROW), True, id="task-5-plus-kernel-50"),
    pytest.param((KERNEL_HIGH_ROW, TASK_ROW), True, id="kernel-50-before-task-5"),
    pytest.param((KERNEL_HIGH_ROW,), False, id="kernel-null-50"),
    pytest.param((("task", "kernel", None, 5.0),), False, id="contradictory-task-kernel"),
    pytest.param((("kernel", "task", None, 5.0),), False, id="contradictory-kernel-task"),
    pytest.param((("unknown", "task", None, 5.0),), False, id="unknown-scope"),
    pytest.param((("task", "unknown", None, 5.0),), False, id="unknown-entity-type"),
    pytest.param((("kernel", "kernel", "k1", 5.0),), False, id="nonnull-kernel-not-task"),
])
def test_null_entity_consumes_only_explicit_task_scope(rows: tuple[ResourceRow, ...], eligible: bool) -> None:
    # Given
    context = scope_context()
    candidate = scope_evidence(rows, (-7.0, 2.0))
    # When
    result = aggregate_metrics(context, candidate)
    # Then
    assert result.native_j == -5.0
    assert result.feasibility == ("feasible" if eligible else "unknown")
    assert result.constraint_results[0].observed_values == ({"a": 5.0} if eligible else {})
    assert is_eligible(context, result) is eligible


@pytest.mark.parametrize(("rows", "eligible"), [
    pytest.param((TASK_ROW,), True, id="task-candidate-selected"),
    pytest.param((TASK_ROW, KERNEL_HIGH_ROW), True, id="mixed-candidate-selected"),
    pytest.param((KERNEL_ROW,), False, id="kernel-candidate-refused"),
    pytest.param((KERNEL_HIGH_ROW,), False, id="high-kernel-candidate-refused"),
])
def test_selection_uses_task_scope_not_null_entity_alone(rows: tuple[ResourceRow, ...], eligible: bool) -> None:
    # Given
    context = scope_context()
    candidate = scope_evidence(rows, (-7.0, 2.0))
    incumbent = scope_evidence((TASK_ROW,), (2.0, 6.0))
    # When
    comparison = compare_results(context, candidate, incumbent)
    # Then
    if eligible:
        assert comparison.kind == "comparable"
        assert comparison.winner == "candidate"
        assert comparison.improvement == 13.0
    else:
        assert comparison.kind == "incompatible"


@pytest.mark.parametrize("entity_id", ["k1", "k2"])
def test_nonnull_kernel_entity_keeps_exact_matching(entity_id: str) -> None:
    # Given
    context = scope_context(entity_id)
    candidate = scope_evidence((("kernel", "kernel", "k1", 5.0),), (-7.0, 2.0))
    # When
    result = aggregate_metrics(context, candidate)
    # Then
    assert result.feasibility == ("feasible" if entity_id == "k1" else "unknown")
    assert is_eligible(context, result) is (entity_id == "k1")
