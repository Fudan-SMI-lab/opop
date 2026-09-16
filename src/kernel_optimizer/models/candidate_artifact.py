"""Complete candidate snapshot manifests, independent of legacy kernel models."""

from typing import Annotated, Literal, Self, assert_never

from pydantic import Field, model_validator

from kernel_optimizer.models.contract_common import CallableRef, ContractError, ContractModel, FileRef, FiniteJsonValue, Name


class FileChange(ContractModel):
    operation: Literal["add", "replace", "delete"]
    path: Name


class AcceptedTransform(ContractModel):
    id: Name
    status: Literal["active", "superseded"]
    superseded_by: Name | None = None
    validation_ref: Name

    @model_validator(mode="after")
    def consistent_status(self) -> Self:
        match self.status:
            case "active":
                if self.superseded_by is not None:
                    raise ContractError("superseded_by", "active transform has no replacement")
            case "superseded":
                if self.superseded_by is None or self.superseded_by == self.id:
                    raise ContractError("superseded_by", "requires a distinct replacement")
            case _:
                assert_never(self.status)
        return self


class CandidateArtifact(ContractModel):
    artifact_kind: Literal["source_file", "project_snapshot"]
    backend_type: Name
    snapshot_ref: Name
    files: Annotated[tuple[FileRef, ...], Field(min_length=1)]
    parent_ref: Name | None = None
    changes: tuple[FileChange, ...] = ()
    entrypoint: CallableRef
    parameter_application: CallableRef
    accepted_transforms: tuple[AcceptedTransform, ...] = ()

    @model_validator(mode="after")
    def complete_manifest(self) -> Self:
        paths = [file.path for file in self.files]
        changed = [change.path for change in self.changes]
        if len(paths) != len(set(paths)) or len(changed) != len(set(changed)):
            raise ContractError("files", "snapshot and change paths must be unique")
        for change in self.changes:
            match change.operation:
                case "add" | "replace":
                    if change.path not in paths:
                        raise ContractError("changes", "added/replaced file must be in full snapshot")
                case "delete":
                    if change.path in paths:
                        raise ContractError("changes", "deleted file cannot be in full snapshot")
                case _:
                    assert_never(change.operation)
        ids = {transform.id for transform in self.accepted_transforms}
        if len(ids) != len(self.accepted_transforms):
            raise ContractError("accepted_transforms", "IDs must be unique")
        replacements = {t.id: t.superseded_by for t in self.accepted_transforms}
        for transform in self.accepted_transforms:
            seen = {transform.id}
            successor = transform.superseded_by
            while successor is not None:
                if successor not in ids or successor in seen:
                    raise ContractError("superseded_by", "must resolve to an acyclic accepted replacement")
                seen.add(successor)
                successor = replacements[successor]
        return self


class ParameterApplicationRequest(ContractModel):
    candidate_ref: Name
    params: dict[Name, FiniteJsonValue]


class GenericCandidate(ContractModel):
    kind: Literal["generic"] = "generic"
    candidate_id: Name
    family_id: Name
    artifact_ref: Name
    parent_ref: Name | None = None
    params: dict[Name, FiniteJsonValue]
