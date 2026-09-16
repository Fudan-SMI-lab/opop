"""Open-domain boundary contracts, independent of runtime/scoring algorithms."""

from pathlib import Path
import importlib.util
import json
from typing import Literal

import pytest
from pydantic import JsonValue, TypeAdapter, ValidationError

from kernel_optimizer.models.core import Candidate, TaskSpec


def test_open_contract_modules_exist() -> None:
    # Given/When: discover the planned boundary modules without importing missing code.
    names = ("task_bundle", "evaluation_bundle", "candidate_artifact", "objective", "resource_axis", "routing")
    missing = [name for name in names if importlib.util.find_spec(f"kernel_optimizer.models.{name}") is None]
    # Then: all planned contracts are available for downstream tasks.
    assert missing == [], f"Missing planned domain behavior: {missing}"


def test_legacy_constructors_and_serialization() -> None:
    # Given: original constructor arguments, without a kind discriminator.
    task = TaskSpec(level=2, problem_id=7, name="old", ref_path=Path("ref.py"), ref_src_sha="r")
    candidate = Candidate(candidate_id="c", family_id="f", origin="seed", backend="cuda",
                          source_sha="s", structural_signature="sig")
    # When: the real JSON boundary reloads the models.
    restored_task = TaskSpec.model_validate_json(task.model_dump_json())
    restored_candidate = Candidate.model_validate_json(candidate.model_dump_json())
    # Then: the original values/defaults survive, including mutable legacy status.
    assert restored_task == task
    assert restored_candidate == candidate
    assert restored_candidate.parent_ids == []
    restored_candidate.status = "tuned"
    assert restored_candidate.status == "tuned"


def test_legacy_backend_remains_restricted() -> None:
    # Given/When: an unsupported backend enters the legacy JSON boundary.
    with pytest.raises(ValidationError):
        _ = Candidate.model_validate_json(
            '{"candidate_id":"c","family_id":"f","origin":"seed",'
            + '"backend":"novel_cpu","source_sha":"s","structural_signature":"x"}'
        )
    # Then: it is rejected, independently of the open generic backend names.


@pytest.mark.parametrize("native_j", [0.0, -2.0, 3.5])
@pytest.mark.parametrize("direction", ["minimize", "maximize"])
def test_novel_names_and_signed_finite_j(native_j: float, direction: Literal["minimize", "maximize"]) -> None:
    # Given: a real JSON task with novel names and no GPU/LLM fields.
    from kernel_optimizer.models.task_bundle import GenericTaskSpec, TaskDefinition
    from kernel_optimizer.models.candidate_artifact import GenericCandidate
    from kernel_optimizer.models.evaluation_bundle import EvalResult
    from kernel_optimizer.models.core import CandidateLike, TaskLike

    task = TaskDefinition.model_validate_json(Path("tests/fixtures/v5/contracts/task.json").read_bytes())
    candidate = GenericCandidate(candidate_id="c", family_id="f", artifact_ref="snapshot:c", params={})
    generic = GenericTaskSpec(task_id=task.task_id, task_definition_ref="task:1", bundle_ref="bundle:1", protocol_id="p", input_binding_ref="binding:1")
    result = EvalResult(native_j=native_j, objective_unit="basket", direction=direction, protocol_id="p")
    # When: strict tagged unions and result JSON make a real roundtrip.
    restored_task = TypeAdapter[TaskLike](TaskLike).validate_json(generic.model_dump_json())
    restored_candidate = TypeAdapter[CandidateLike](CandidateLike).validate_json(candidate.model_dump_json())
    restored_result = EvalResult.model_validate_json(result.model_dump_json())
    # Then: no legacy identifiers or source file is fabricated; missing evidence stays unknown.
    assert restored_task == generic
    assert restored_candidate == candidate
    assert restored_result.native_j == native_j
    assert restored_result.quality == "unknown"
    assert restored_result.measurement_validity == "unknown"
    assert restored_result.feasibility == "unknown"
    assert task.constraints == ()
    test_legacy_constructors_and_serialization()


@pytest.mark.parametrize("bad", [True, float("nan"), float("inf"), -float("inf"), "1"])
@pytest.mark.parametrize("target", ["metric", "native_j", "threshold", "resource"])
def test_malformed_or_nonfinite_result_rejected(bad: JsonValue, target: str) -> None:
    # Given: untrusted numeric data, including bool and JSON's nonstandard NaN/Infinity.
    from kernel_optimizer.models.objective import MetricValue
    from kernel_optimizer.models.evaluation_bundle import EvalResult
    from kernel_optimizer.models.resource_axis import ResourceObservation
    from kernel_optimizer.models.task_bundle import ConstraintSpec

    models = {"metric": MetricValue, "native_j": EvalResult, "threshold": ConstraintSpec, "resource": ResourceObservation}
    payloads: dict[str, dict[str, JsonValue]] = {
        "metric": {"name": "fruit", "value": bad, "unit": "fruit", "case_id": "a"},
        "native_j": {"native_j": bad, "objective_unit": "basket", "direction": "minimize", "protocol_id": "p"},
        "threshold": {"id": "cap", "source_kind": "metric", "name": "fruit", "unit": "fruit", "operator": "le", "threshold": bad, "case_ids": ["a"], "aggregation": "all"},
        "resource": {"name": "twigs", "entity_id": None, "unit": "twig", "scope": "task", "provenance": "sensor", "value": bad, "context_ref": "ctx", "case_id": "a", "entity_type": "task"},
    }
    # When/Then: actual JSON validation rejects the malformed measurement.
    with pytest.raises(ValidationError):
        _ = models[target].model_validate_json(json.dumps(payloads[target]))


@pytest.mark.parametrize("field", ["objective_unit", "direction"])
def test_malformed_or_nonfinite_result_rejected_missing_contract(field: str) -> None:
    from kernel_optimizer.models.evaluation_bundle import EvalResult

    # Given: a result missing an essential interpretation field.
    payload = {"objective_unit": "basket", "direction": "minimize", "protocol_id": "p"}
    del payload[field]
    # When/Then: omission is rejected rather than silently assuming latency/minimize.
    with pytest.raises(ValidationError):
        _ = EvalResult.model_validate_json(json.dumps(payload))


@pytest.mark.parametrize(("section", "field", "value"), [
    ("workload", "case_ids", []), ("workload", "case_ids", ["a", "a"]),
    ("constraint", "case_ids", []), ("constraint", "case_ids", ["a", "a"]),
    ("constraint", "case_ids", ["absent"]), ("constraint", "operator", "approximately"),
    ("constraint", "source_kind", "compiled"), ("constraint", "threshold", True),
    ("constraint", "unit", ""), ("constraint", "aggregation", "median"),
    ("objective", "direction", "up"), ("objective", "unit", ""),
    ("objective", "improvement_threshold", -1),
    ("objective", "improvement_threshold", True),
    ("protocol", "repeat_count", 0), ("protocol", "repeat_count", True),
    ("protocol", "paired_repeat_count", 1), ("protocol", "c2_pair_roles", []),
    ("protocol", "c2_threshold", -1), ("state", "kind", "unknown-mode"),
])
def test_malformed_or_nonfinite_result_rejected_structure(section: str, field: str, value: JsonValue) -> None:
    from kernel_optimizer.models.task_bundle import TaskDefinition

    # Given: a valid fixture with one adversarial structural change.
    payload = TypeAdapter(dict[str, JsonValue]).validate_json(Path("tests/fixtures/v5/contracts/task.json").read_bytes())
    constraint: dict[str, JsonValue] = {"id": "limit", "source_kind": "metric", "name": "fruit", "unit": "fruit", "operator": "le", "threshold": 3, "case_ids": ["a"], "aggregation": "all"}
    payload["constraints"] = [constraint]
    selected = constraint if section == "constraint" else payload[section]
    assert isinstance(selected, dict)
    selected[field] = value
    # When/Then: structural invalidity is rejected before any scorer runs.
    with pytest.raises(ValidationError):
        _ = TaskDefinition.model_validate_json(json.dumps(payload))


@pytest.mark.parametrize("weights", [{}, {"a": 0, "b": 0}, {"a": 1}, {"a": -1, "b": 2}, {"a": True, "b": 2}, {"a": float("inf"), "b": 1}, {"a": 1, "b": 2, "absent": 1}])
def test_malformed_or_nonfinite_result_rejected_weights(weights: dict[str, JsonValue]) -> None:
    from kernel_optimizer.models.task_bundle import TaskDefinition

    # Given: objective weights must cover the workload only when used.
    payload = TypeAdapter(dict[str, JsonValue]).validate_json(Path("tests/fixtures/v5/contracts/task.json").read_bytes())
    objective = payload["objective"]
    assert isinstance(objective, dict)
    objective["aggregation"] = {"kind": "weighted_mean", "case_weights": weights}
    # When/Then: missing, negative, boolean, nonfinite and foreign weights fail.
    with pytest.raises(ValidationError):
        _ = TaskDefinition.model_validate_json(json.dumps(payload))


def test_empty_required_metrics_preserve_unknown_boundary() -> None:
    from kernel_optimizer.models.objective import ObjectiveSpec
    from kernel_optimizer.models.evaluation_bundle import EvalResult

    # Given: an explicitly empty list is representable, not an eligibility proof.
    objective = ObjectiveSpec.model_validate_json('{"name":"signed","unit":"fruit","direction":"minimize","required_metrics":[],"aggregation":{"kind":"sum"}}')
    # When: a self-reported J crosses the result boundary without evidence.
    result = EvalResult(native_j=-1, objective_unit=objective.unit, direction=objective.direction, protocol_id="p")
    # Then: no state is promoted; required-metric eligibility is Task 5's job.
    assert objective.required_metrics == ()
    assert (result.measurement_validity, result.quality, result.feasibility) == ("unknown", "unknown", "unknown")


@pytest.mark.parametrize("extra", ["compiled", "eligible", "passed"])
def test_candidate_success_flags_do_not_create_quality(extra: str) -> None:
    from kernel_optimizer.models.evaluation_bundle import EvalResult

    # Given/When: a foreign success flag cannot replace quality evidence.
    with pytest.raises(ValidationError):
        _ = EvalResult.model_validate_json(json.dumps({"objective_unit": "fruit", "direction": "minimize", "protocol_id": "p", extra: True}))
    # Then: strict extras prevent silently interpreting compiled as pass.


@pytest.mark.parametrize("field", ["unit", "direction"])
def test_malformed_or_nonfinite_result_rejected_objective_omission(field: str) -> None:
    from kernel_optimizer.models.objective import ObjectiveSpec

    # Given: even an unrecognized objective must declare its interpretation.
    payload: dict[str, JsonValue] = {"name": "orchard", "unit": "fruit", "direction": "maximize", "required_metrics": [], "aggregation": {"kind": "sum"}}
    del payload[field]
    # When/Then: missing objective units or direction cannot acquire defaults.
    with pytest.raises(ValidationError):
        _ = ObjectiveSpec.model_validate_json(json.dumps(payload))


@pytest.mark.parametrize(("field", "bad"), [("measurement_validity", "compiled"), ("quality", "valid"), ("feasibility", "pass")])
def test_malformed_or_nonfinite_result_rejected_status(field: str, bad: str) -> None:
    from kernel_optimizer.models.evaluation_bundle import EvalResult

    # Given/When: measurement, quality and feasibility vocabularies are distinct.
    with pytest.raises(ValidationError):
        _ = EvalResult.model_validate_json(json.dumps({"objective_unit": "u", "direction": "minimize", "protocol_id": "p", field: bad}))
    # Then: one status cannot stand in for another.
