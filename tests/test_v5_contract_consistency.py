"""Cross-field contracts that guard against misleading valid-looking records."""

import json
from pathlib import Path

import pytest
from pydantic import JsonValue, TypeAdapter, ValidationError

from kernel_optimizer.models.candidate_artifact import GenericCandidate
from kernel_optimizer.models.evaluation_bundle import ConstraintResult, CostRecord, EvalResult
from kernel_optimizer.models.task_bundle import TaskDefinition


@pytest.mark.parametrize("seconds", [0, 7])
def test_cost_wall_duration_must_match_monotonic_endpoints(seconds: float) -> None:
    # Given/When: recorded duration contradicts available start/end measurements.
    with pytest.raises(ValidationError):
        _ = CostRecord(invocation_id="i", record_id="r", phase="search", wall_start=1, wall_end=4, wall_seconds=seconds)
    # Then: a contradictory cost record is rejected, not silently recomputed.


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_task_semantic_payload_is_finite_json(bad: float) -> None:
    # Given/When: nested task parameters must still be valid finite JSON.
    with pytest.raises(ValidationError):
        _ = GenericCandidate.model_validate_json(json.dumps({"candidate_id": "c", "family_id": "f", "artifact_ref": "a", "params": {"custom": {"nested": bad}}}))
    # Then: JSON extension points do not leak NaN through serialization.


@pytest.mark.parametrize("field", ["observed_values", "aggregate_value"])
@pytest.mark.parametrize("bad", [True, float("nan"), float("inf")])
def test_constraint_results_reject_non_numeric_observations(field: str, bad: JsonValue) -> None:
    # Given: a malformed raw observation or aggregate.
    payload: dict[str, JsonValue] = {"id": "cap", "unit": "fruit", field: {"a": bad} if field == "observed_values" else bad}
    # When/Then: status labels cannot legitimize invalid measurements.
    with pytest.raises(ValidationError):
        _ = ConstraintResult.model_validate_json(json.dumps(payload))


def test_invalid_measurement_cannot_carry_native_j() -> None:
    # Given/When: invalid evidence must not carry an actionable native objective.
    with pytest.raises(ValidationError):
        _ = EvalResult(measurement_validity="invalid", native_j=0, objective_unit="u", direction="minimize", protocol_id="p")
    # Then: the invalid response boundary requires native_j=null.


@pytest.mark.parametrize("weights", [{}, {"a": 0, "b": 1}, {"b": 1}])
def test_weighted_constraint_requires_positive_selected_subset(weights: dict[str, JsonValue]) -> None:
    # Given: sum objective, but a weighted constraint on case a alone.
    payload = TypeAdapter(dict[str, JsonValue]).validate_json(Path("tests/fixtures/v5/contracts/task.json").read_bytes())
    objective = payload["objective"]
    assert isinstance(objective, dict)
    objective["aggregation"] = {"kind": "sum", "case_weights": weights}
    payload["constraints"] = [{"id": "cap", "source_kind": "resource", "name": "twigs", "unit": "twig", "operator": "le", "threshold": 10, "case_ids": ["a"], "entity_id": None, "aggregation": "weighted_mean"}]
    # When/Then: global positive weight cannot hide an empty/zero selected subset.
    with pytest.raises(ValidationError):
        _ = TaskDefinition.model_validate_json(json.dumps(payload))


def test_weighted_objective_accepts_zero_individual_weight() -> None:
    # Given: nonnegative weights with a positive total.
    payload = TypeAdapter(dict[str, JsonValue]).validate_json(Path("tests/fixtures/v5/contracts/task.json").read_bytes())
    objective = payload["objective"]
    assert isinstance(objective, dict)
    objective["aggregation"] = {"kind": "weighted_mean", "case_weights": {"a": 0, "b": 3}}
    # When: the structural contract is parsed without computing J.
    parsed = TaskDefinition.model_validate_json(json.dumps(payload))
    # Then: zero individual weight is legal and the declared mapping survives.
    assert parsed.objective.aggregation.case_weights == {"a": 0, "b": 3}
