"""State and legacy/generic boundary integration tests."""

import json
from pathlib import Path

import pytest
from pydantic import JsonValue, TypeAdapter, ValidationError

from kernel_optimizer.models.candidate_artifact import GenericCandidate
from kernel_optimizer.models.contract_common import ContractError
from kernel_optimizer.models.core import Candidate, CandidateLike, LegacyCandidate, LegacyTaskSpec, TaskLike, TaskSpec
from kernel_optimizer.models.evaluation_bundle import ExecutionBinding, ScientificExecutionContext
from kernel_optimizer.models.input_binding import InputBinding
from kernel_optimizer.models.routing import (
    candidate_identity, execution_binding, read_legacy_candidate_json, read_legacy_task_json, task_identity,
)
from kernel_optimizer.models.task_bundle import GenericTaskSpec, StateSpec, TaskDefinition


def test_stateful_cpu_with_custom_capability_roundtrips() -> None:
    # Given: custom CPU state, no checkpoint/CUDA/KV or model-brand obligations.
    data = {"kind": "stateful", "schema_ref": "schema:counter", "initial_binding_ref": "state:zero", "reset_policy": "restore-counter", "adapter_hooks": {name: {"module": "cpu.counter", "callable": name} for name in ("load", "bind", "reset", "observe_state")}, "capability_bindings": [{"name": "garden-counter", "applicability": "applicable", "binding_ref": "state:garden"}]}
    # When: both the union and containing task cross JSON boundaries.
    state = TypeAdapter[StateSpec](StateSpec).validate_json(json.dumps(data))
    task = TypeAdapter(dict[str, JsonValue]).validate_json(Path("tests/fixtures/v5/contracts/task.json").read_bytes())
    task["state"] = TypeAdapter(JsonValue).validate_json(state.model_dump_json())
    parsed = TaskDefinition.model_validate_json(json.dumps(task))
    # Then: custom bindings survive and stateless-only assumptions were not imposed.
    assert parsed.state == state
    assert parsed.state.capability_bindings[0].name == "garden-counter"


@pytest.mark.parametrize("missing", ["schema_ref", "initial_binding_ref", "reset_policy", "adapter_hooks"])
def test_stateful_missing_binding_rejected(missing: str) -> None:
    # Given: a stateful declaration must bind its whole lifecycle.
    data = {"kind": "stateful", "schema_ref": "s", "initial_binding_ref": "b", "reset_policy": "restore", "adapter_hooks": {name: {"module": "cpu.counter", "callable": name} for name in ("load", "bind", "reset", "observe_state")}}
    del data[missing]
    # When/Then: incomplete state is rejected without runtime execution.
    with pytest.raises(ValidationError):
        _ = TypeAdapter[StateSpec](StateSpec).validate_json(json.dumps(data))


@pytest.mark.parametrize("transport", ["local", "wsl"])
def test_execution_routes_are_outside_scientific_context(transport: str) -> None:
    # Given: independently declared worker paths and scientific fingerprints.
    context = ScientificExecutionContext(hardware_identity="cpu:one", software_fingerprints=())
    # When: a real route roundtrips through JSON.
    binding = ExecutionBinding.model_validate_json(json.dumps({"transport": transport, "target_os": "linux", "python": "/env/bin/python", "project_root": "/workspace", "path_mappings": [{"source": "D:/project", "target": "/workspace"}], "asset_locators": [{"asset_ref": "asset:data", "locator": "/data", "provenance": "operator"}]}))
    # Then: routes retain paths, while context rejects route-field contamination.
    assert binding.transport == transport
    assert context.model_dump() == {"hardware_identity": "cpu:one", "software_fingerprints": (), "execution_policies": {}}
    with pytest.raises(ValidationError):
        _ = ScientificExecutionContext.model_validate_json(json.dumps({"hardware_identity": "cpu:one", "software_fingerprints": [], "python": binding.python}))


def test_legacy_and_generic_unions_and_routing() -> None:
    # Given: real legacy and generic instances with deliberately distinct identities.
    old_task = TaskSpec(level=3, problem_id=8, name="old", ref_path=Path("truth.py"), ref_src_sha="truth")
    old_candidate = Candidate(candidate_id="old-c", family_id="old-f", origin="seed", backend="triton", source_sha="old-source", structural_signature="sig")
    task = GenericTaskSpec(task_id="garden", task_definition_ref="task:1", bundle_ref="bundle:2", protocol_id="p3", input_binding_ref="binding:4")
    candidate = GenericCandidate(candidate_id="new-c", family_id="new-f", artifact_ref="snapshot:5", params={"style": "gentle"})
    # When: all four variants are read through their discriminated wire unions.
    tasks = [TypeAdapter[TaskLike](TaskLike).validate_json(value.model_dump_json()) for value in (old_task, task)]
    candidates = [TypeAdapter[CandidateLike](CandidateLike).validate_json(value.model_dump_json()) for value in (old_candidate, candidate)]
    # Then: accessors preserve the right values rather than inventing legacy IDs.
    assert LegacyTaskSpec is TaskSpec and LegacyCandidate is Candidate
    assert task_identity(tasks[0]).model_dump() == {"kind": "legacy", "level": 3, "problem_id": 8}
    assert task_identity(tasks[1]).model_dump() == {"kind": "generic", "task_id": "garden"}
    assert candidate_identity(candidates[1]).candidate_id == "new-c"
    assert execution_binding(task, candidate).model_dump()["artifact_ref"] == "snapshot:5"
    assert execution_binding(old_task, old_candidate).model_dump()["source_sha"] == "old-source"
    with pytest.raises(ContractError):
        _ = execution_binding(task, old_candidate)
    with pytest.raises(ContractError):
        _ = execution_binding(old_task, candidate)


@pytest.mark.parametrize("kind", ["task", "candidate"])
def test_only_explicit_legacy_reader_normalizes_missing_kind(kind: str) -> None:
    # Given: pre-discriminator legacy storage.
    task_json = '{"level":1,"problem_id":2,"name":"old","ref_path":"r.py","ref_src_sha":"r"}'
    candidate_json = '{"candidate_id":"c","family_id":"f","origin":"seed","backend":"cuda","source_sha":"s","structural_signature":"sig"}'
    raw = task_json if kind == "task" else candidate_json
    reader = read_legacy_task_json if kind == "task" else read_legacy_candidate_json
    adapter = TypeAdapter[TaskLike | CandidateLike](TaskLike | CandidateLike)
    # When: the explicit reader parses an old wire document.
    restored = reader(raw)
    # Then: direct union rejects the missing tag, and the reader never overwrites unknown tags.
    assert restored.kind == "legacy"
    with pytest.raises(ValidationError):
        _ = adapter.validate_json(raw)
    payload = TypeAdapter(dict[str, JsonValue]).validate_json(raw)
    payload["kind"] = "future"
    with pytest.raises(ValidationError):
        _ = reader(json.dumps(payload))


def test_candidate_relative_prototype_uses_runtime_slots() -> None:
    # Given: independent candidate/reference/quality callables and argument slots.
    binding = {"candidate_call": {"callable": {"module": "pipeline", "callable": "run"}, "args": [{"kind": "runtime_ref", "slot": "candidate.parameters"}], "kwargs": {}}, "reference_call": {"callable": {"module": "independent", "callable": "truth"}, "args": [{"kind": "artifact", "artifact_ref": "data:frozen"}]}, "quality_call": {"callable": {"module": "oracle", "callable": "compare"}, "args": [{"kind": "literal", "value": "fruit"}]}}
    # When: the frozen prototype is parsed and serialized.
    parsed = InputBinding.model_validate_json(json.dumps(binding))
    # Then: the candidate argument stays a slot, not candidate-specific parameter values.
    assert InputBinding.model_validate_json(parsed.model_dump_json()) == parsed
    assert parsed.candidate_call.args[0].kind == "runtime_ref"
