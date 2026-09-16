"""Typed legacy/generic projections; routing does not materialize or execute code."""

from pathlib import Path
from typing import Annotated, Literal, assert_never

from pydantic import Field, JsonValue, TypeAdapter

from kernel_optimizer.models.contract_common import ContractError, ContractModel
from kernel_optimizer.models.core import CandidateLike, TaskLike


class LegacyTaskIdentity(ContractModel):
    kind: Literal["legacy"] = "legacy"
    level: int
    problem_id: int


class GenericTaskIdentity(ContractModel):
    kind: Literal["generic"] = "generic"
    task_id: str


TaskIdentity = Annotated[LegacyTaskIdentity | GenericTaskIdentity, Field(discriminator="kind")]


class CandidateIdentity(ContractModel):
    kind: Literal["legacy", "generic"]
    candidate_id: str
    family_id: str


class LegacyExecutionInputs(ContractModel):
    kind: Literal["legacy"] = "legacy"
    task: LegacyTaskIdentity
    ref_path: Path
    ref_src_sha: str
    source_sha: str
    backend: Literal["triton", "cuda"]


class GenericExecutionInputs(ContractModel):
    kind: Literal["generic"] = "generic"
    task_id: str
    task_definition_ref: str
    bundle_ref: str
    protocol_id: str
    input_binding_ref: str
    artifact_ref: str
    params: dict[str, JsonValue]


ExecutionInputs = Annotated[LegacyExecutionInputs | GenericExecutionInputs, Field(discriminator="kind")]


def task_identity(task: TaskLike) -> TaskIdentity:
    match task.kind:
        case "legacy":
            return LegacyTaskIdentity(level=task.level, problem_id=task.problem_id)
        case "generic":
            return GenericTaskIdentity(task_id=task.task_id)
        case _:
            assert_never(task.kind)


def candidate_identity(candidate: CandidateLike) -> CandidateIdentity:
    match candidate.kind:
        case "legacy" | "generic":
            return CandidateIdentity(kind=candidate.kind, candidate_id=candidate.candidate_id, family_id=candidate.family_id)
        case _:
            assert_never(candidate.kind)


def execution_binding(task: TaskLike, candidate: CandidateLike) -> ExecutionInputs:
    """Project scientific inputs; supply the separate ExecutionBinding for local/WSL paths."""
    match task.kind:
        case "legacy":
            match candidate.kind:
                case "legacy":
                    return LegacyExecutionInputs(task=LegacyTaskIdentity(level=task.level, problem_id=task.problem_id), ref_path=task.ref_path, ref_src_sha=task.ref_src_sha, source_sha=candidate.source_sha, backend=candidate.backend)
                case "generic":
                    raise ContractError("candidate.kind", "must match task.kind")
                case _:
                    assert_never(candidate.kind)
        case "generic":
            match candidate.kind:
                case "generic":
                    return GenericExecutionInputs(task_id=task.task_id, task_definition_ref=task.task_definition_ref, bundle_ref=task.bundle_ref, protocol_id=task.protocol_id, input_binding_ref=task.input_binding_ref, artifact_ref=candidate.artifact_ref, params=candidate.params)
                case "legacy":
                    raise ContractError("candidate.kind", "must match task.kind")
                case _:
                    assert_never(candidate.kind)
        case _:
            assert_never(task.kind)


def read_legacy_task_json(data: str | bytes) -> TaskLike:
    """Explicit old-storage boundary: only a missing tag is normalized, never an unknown tag."""
    payload = TypeAdapter(dict[str, JsonValue]).validate_json(data)
    return TypeAdapter[TaskLike](TaskLike).validate_python({"kind": "legacy", **payload})


def read_legacy_candidate_json(data: str | bytes) -> CandidateLike:
    payload = TypeAdapter(dict[str, JsonValue]).validate_json(data)
    return TypeAdapter[CandidateLike](CandidateLike).validate_python({"kind": "legacy", **payload})
