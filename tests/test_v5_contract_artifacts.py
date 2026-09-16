"""Artifact, resource and evaluation-message serialization contracts."""

import json
from typing import Literal

import pytest
from pydantic import JsonValue, TypeAdapter, ValidationError

from kernel_optimizer.models.candidate_artifact import CandidateArtifact, ParameterApplicationRequest
from kernel_optimizer.models.contract_common import CallableRef, FileRef
from kernel_optimizer.models.evaluation_bundle import (
    ControlExpected, ControlSpec, CostRecord, EvalBundleManifest, EvalResult,
    EvaluationRequest, EvaluationResponse, ResourceProbeResult,
)
from kernel_optimizer.models.resource_axis import ParameterAxis


@pytest.mark.parametrize("kind", ["source_file", "project_snapshot"])
def test_complete_snapshot_and_transform_manifest(kind: Literal["source_file", "project_snapshot"]) -> None:
    # Given: backend name and artifact kind are independent, with a full replacement snapshot.
    data = {"artifact_kind": kind, "backend_type": "new-cpu-engine", "snapshot_ref": "snapshot:new", "parent_ref": "snapshot:old", "files": [{"path": "pipeline.py", "content_ref": "file:new", "sha256": "a" * 64}, {"path": "support.py", "content_ref": "file:support", "sha256": "b" * 64}], "changes": [{"operation": "replace", "path": "pipeline.py"}, {"operation": "add", "path": "support.py"}, {"operation": "delete", "path": "obsolete.py"}], "entrypoint": {"module": "pipeline", "callable": "run"}, "parameter_application": {"module": "support", "callable": "apply_parameters"}, "accepted_transforms": [{"id": "first", "status": "superseded", "superseded_by": "second", "validation_ref": "validation:second"}, {"id": "second", "status": "active", "validation_ref": "validation:second"}, {"id": "independent", "status": "active", "validation_ref": "validation:both"}]}
    # When: an actual snapshot manifest roundtrips through JSON.
    artifact = CandidateArtifact.model_validate_json(json.dumps(data))
    restored = CandidateArtifact.model_validate_json(artifact.model_dump_json())
    # Then: full files, deletion record, and independent accepted transforms survive.
    assert restored == artifact
    assert restored.backend_type == "new-cpu-engine"
    assert tuple(t.id for t in restored.accepted_transforms if t.status == "active") == ("second", "independent")


@pytest.mark.parametrize("change", [{"operation": "delete", "path": "present.py"}, {"operation": "add", "path": "absent.py"}, {"operation": "replace", "path": "absent.py"}, {"operation": "patch", "path": "present.py"}])
def test_inconsistent_snapshot_rejected(change: dict[str, str]) -> None:
    # Given: a change list inconsistent with the complete snapshot.
    data = {"artifact_kind": "project_snapshot", "backend_type": "cpu", "snapshot_ref": "s", "files": [{"path": "present.py", "content_ref": "c", "sha256": "a" * 64}], "changes": [change], "entrypoint": {"module": "present", "callable": "run"}, "parameter_application": {"module": "present", "callable": "apply"}}
    # When/Then: reject an invalid manifest without touching the filesystem.
    with pytest.raises(ValidationError):
        _ = CandidateArtifact.model_validate_json(json.dumps(data))


def test_bundle_manifest_has_arbitrary_file_and_callable_names() -> None:
    # Given: task-specific files and controls, no conventional evaluator.py or controls directory.
    entry = CallableRef(module="my.package.judge", callable="compute_fruit")
    control = ControlSpec(id="spoiled", role="fruit-quality-negative", request_binding_ref="cases:spoiled", expected=ControlExpected(quality="fail"), required=True)
    manifest = EvalBundleManifest(schema_version="1", task_definition_ref="t", files=(FileRef(path="my/package/judge.py", content_ref="program", sha256="a" * 64),), entrypoint=entry, request_schema_ref="schema", input_binding_ref="binding", control_specs=(control,), protocol_id="p", bundle_id="b")
    # When: the manifest crosses its real JSON boundary.
    restored = EvalBundleManifest.model_validate_json(manifest.model_dump_json())
    # Then: no resource provider or fixed template filename is required.
    assert restored.entrypoint == entry
    assert restored.resource_provider is None
    assert restored.files[0].path == "my/package/judge.py"
    assert restored.control_specs[0].expected.quality == "fail"


def test_request_response_and_resource_probe_cost_roundtrip() -> None:
    # Given: task-specific payload and measured/unknown costs remain distinct.
    request = EvaluationRequest(schema_version="1", task_id="fruit", bundle_id="b", protocol_id="p", candidate_ref="c", input_binding_ref="i", execution_binding_ref="worker:local", role="custom-control", repeat_index=0, payload={"basket": [1, 2]})
    cost = CostRecord(invocation_id="inv", record_id="record", phase="search", wall_start=1, wall_end=2, wall_seconds=1, gpu_seconds=0, gpu_measurement_scope="per_device_busy_union")
    response = EvaluationResponse(result=EvalResult(objective_unit="fruit", direction="maximize", protocol_id="p"), cost=cost)
    probe = ResourceProbeResult(cost=cost)
    parameters = ParameterApplicationRequest(candidate_ref="c", params={"orchard": "west"})
    # When: complete messages cross JSON boundaries.
    parsed_request = EvaluationRequest.model_validate_json(request.model_dump_json())
    parsed_response = EvaluationResponse.model_validate_json(response.model_dump_json())
    parsed_probe = ResourceProbeResult.model_validate_json(probe.model_dump_json())
    # Then: wire identity, nullable collection and unknown resource fit are preserved.
    assert parsed_request == request
    assert parsed_response == response
    assert parsed_response.cost.build_seconds is None
    assert parsed_response.cost.gpu_seconds == 0
    assert parsed_probe.fit_status == "unknown"
    assert ParameterApplicationRequest.model_validate_json(parameters.model_dump_json()) == parameters


@pytest.mark.parametrize("field", ["wall_start", "wall_end", "wall_seconds", "build_seconds", "load_seconds", "evaluation_seconds", "failure_seconds", "gpu_seconds", "llm_cost"])
@pytest.mark.parametrize("bad", [True, -1, float("nan"), float("inf")])
def test_cost_rejects_malformed_measured_values(field: str, bad: JsonValue) -> None:
    # Given/When: cost numbers cannot be fabricated from booleans or invalid values.
    with pytest.raises(ValidationError):
        _ = CostRecord.model_validate_json(json.dumps({"invocation_id": "i", "record_id": "r", "phase": "prepare", field: bad}))
    # Then: only finite nonnegative measurements or null cross the boundary.


def test_measured_gpu_cost_requires_scope() -> None:
    # Given/When: a GPU measurement without scope is ambiguous.
    with pytest.raises(ValidationError):
        _ = CostRecord(invocation_id="i", record_id="r", phase="final", gpu_seconds=1)
    # Then: scope is required only for an actual measurement.


@pytest.mark.parametrize("kind", ["numeric", "ordinal", "nominal"])
def test_new_axis_names_roundtrip(kind: str) -> None:
    # Given: axis semantics, not the name, determine whether spacing is available.
    choices: list[JsonValue] = [1, 3] if kind == "numeric" else ["west", "east"]
    value_type = "int" if kind == "numeric" else "str"
    data = {"name": "orchard_lane", "kind": kind, "value_type": value_type, "choices": choices, "semantics": "orchard traversal", "unit": "lane"}
    # When: a new named axis is parsed.
    axis = ParameterAxis.model_validate_json(json.dumps(data))
    # Then: ordinal/nominal labels acquire no invented numeric positions.
    assert axis.numeric_positions is None
    assert ParameterAxis.model_validate_json(axis.model_dump_json()) == axis


@pytest.mark.parametrize("bad", [True, float("nan"), float("inf")])
def test_numeric_axis_rejects_bool_and_nonfinite(bad: JsonValue) -> None:
    # Given/When: numeric spacing requires actual finite numbers.
    with pytest.raises(ValidationError):
        _ = ParameterAxis.model_validate_json(json.dumps({"name": "x", "kind": "numeric", "value_type": "float", "choices": [bad], "semantics": "x", "unit": "u"}))
    # Then: invalid numeric geometry cannot enter downstream response calculations.


def test_frozen_outer_boundary_and_unknown_fields() -> None:
    # Given: the boundary model is frozen and structurally closed.
    result = EvalResult(objective_unit="u", direction="minimize", protocol_id="p")
    # When/Then: both reassignment and unexpected structural fields are rejected.
    with pytest.raises(ValidationError):
        setattr(result, "native_j", 1.0)
    payload = TypeAdapter(dict[str, JsonValue]).validate_json(result.model_dump_json())
    payload["surprise"] = 1
    with pytest.raises(ValidationError):
        _ = EvalResult.model_validate_json(json.dumps(payload))


@pytest.mark.parametrize("positions", [[1, 1], [2, 1], [1], [True, 2]])
def test_ordinal_spacing_rejects_invalid_geometry(positions: list[JsonValue]) -> None:
    # Given/When: declared ordinal distances must be finite, ordered and aligned.
    with pytest.raises(ValidationError):
        _ = ParameterAxis.model_validate_json(json.dumps({"name": "lane", "kind": "ordinal", "value_type": "str", "choices": ["west", "east"], "semantics": "traversal", "unit": "step", "numeric_positions": positions}))
    # Then: ordinal labels cannot silently use invalid numeric geometry.


def test_resource_probe_preserves_entity_context_and_bound_provenance() -> None:
    # Given: task-scope and entity-specific measurements must not be conflated.
    raw = {"observations": [{"name": "fruit_pressure", "entity_id": "branch:west", "entity_type": "branch", "unit": "twig", "scope": "entity", "provenance": "sensor", "value": -1, "case_id": "a", "context_ref": "ctx"}], "bounds": [{"name": "fruit_pressure", "entity_id": "branch:west", "entity_type": "branch", "unit": "twig", "scope": "entity", "provenance": "experiment", "value": 3, "context_ref": "ctx", "kind": "empirical_turn"}], "cost": {"invocation_id": "i", "record_id": "r", "phase": "prepare"}}
    # When: a populated resource response roundtrips.
    probe = ResourceProbeResult.model_validate_json(json.dumps(raw))
    # Then: scope/provenance survive; empirical bounds are not physical guarantees.
    assert ResourceProbeResult.model_validate_json(probe.model_dump_json()) == probe
    assert probe.observations[0].entity_id == "branch:west"
    assert probe.bounds[0].kind == "empirical_turn"
