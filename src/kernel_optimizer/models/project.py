"""Versioned project contracts. No CUDA, profiler, or provider imports."""
from __future__ import annotations

from typing import Literal, Self
from pydantic import Field, model_validator

from kernel_optimizer.models.contract_common import ContractModel, FileRef, FiniteJsonValue, Name
from kernel_optimizer.models.objective import ObjectiveSpec
from kernel_optimizer.models.task_bundle import TaskDefinition

Scope = Literal["group", "joint", "region", "execution_plan"]


class ProjectModel(ContractModel):
    schema_version: Literal["project.v1"] = "project.v1"


class Observation(ProjectModel):
    name: Name
    status: Literal["observed", "unavailable", "not_applicable", "failed", "unknown"]
    value: float | None = None
    unit: Name
    entity: str | None = None
    reason: str | None = None
    evidence: tuple[str, ...] = ()

    @model_validator(mode="after")
    def measured_or_missing(self) -> Self:
        if (self.status == "observed") != (self.value is not None):
            raise ValueError("only observed measurements have a numeric value")
        if self.status != "observed" and not self.reason:
            raise ValueError("missing measurements need a reason")
        return self


class CapabilityManifest(ProjectModel):
    id: Name
    devices: tuple[dict[str, FiniteJsonValue], ...] = ()
    supported_scopes: tuple[Scope, ...] = ("group",)
    fields: tuple[Observation, ...] = ()
    tools: dict[str, FiniteJsonValue] = Field(default_factory=dict)


class CallSite(ProjectModel):
    id: Name
    phase: Name
    kind: Name
    target: Name
    signature: dict[str, FiniteJsonValue] = Field(default_factory=dict)
    evidence: tuple[str, ...] = ()


class Dependency(ProjectModel):
    producer: Name
    consumer: Name
    kind: Literal["data", "control", "alias", "state", "stream", "collective", "unknown"]
    evidence: tuple[str, ...] = ()


class ExecutionPlan(ProjectModel):
    id: Name
    calls: tuple[CallSite, ...] = ()
    dependencies: tuple[Dependency, ...] = ()
    policies: dict[str, FiniteJsonValue] = Field(default_factory=dict)
    unknown_dependencies: tuple[str, ...] = ()

    @model_validator(mode="after")
    def references_exist(self) -> Self:
        names = {c.id for c in self.calls}
        if len(names) != len(self.calls):
            raise ValueError("duplicate call identity")
        if any(e.producer not in names or e.consumer not in names for e in self.dependencies):
            raise ValueError("dependency refers to absent call")
        return self


class GroupSpec(ProjectModel):
    id: Name
    calls: tuple[Name, ...]
    semantics: Name
    supported_shapes: tuple[dict[str, FiniteJsonValue], ...] = ()
    resource_refs: tuple[str, ...] = ()
    lineage: tuple[str, ...] = ()


class GroupRegistry(ProjectModel):
    id: Name
    plan_id: Name
    groups: tuple[GroupSpec, ...] = ()
    parent_id: str | None = None
    retired: tuple[str, ...] = ()
    invalidated_evidence: tuple[str, ...] = ()

    @model_validator(mode="after")
    def unique_groups(self) -> Self:
        if len({g.id for g in self.groups}) != len(self.groups):
            raise ValueError("duplicate group identity")
        return self


class ProjectContract(ProjectModel):
    id: Name
    goal: Name
    objective: ObjectiveSpec
    workload: dict[str, FiniteJsonValue]
    quality: dict[str, FiniteJsonValue]
    completion: Name
    invariants: tuple[Name, ...]
    task_definition: TaskDefinition | None = None
    state_protocol: Name
    world_size: int = Field(default=1, ge=1)
    allowed_scopes: tuple[Scope, ...] = ("group", "joint", "region", "execution_plan")


class OptimizationTask(ProjectModel):
    id: Name
    parent_revision: Name
    contract_ref: Name
    plan_ref: Name
    registry_ref: Name
    scope: Scope
    groups: tuple[Name, ...]
    calls: tuple[Name, ...]
    read_set: tuple[Name, ...]
    write_set: tuple[Name, ...]
    rationale: Name
    feedback_refs: tuple[str, ...] = ()


class FileEdit(ProjectModel):
    path: Name
    before_sha256: str | None
    content_ref: str | None  # None deletes an existing file


class CandidateBundle(ProjectModel):
    id: Name
    parent_revision: Name
    contract_ref: Name
    scope: Scope
    implementation: Literal["kernel", "library", "composite", "distributed", "strategy"]
    groups: tuple[Name, ...]
    calls: tuple[Name, ...]
    edits: tuple[FileEdit, ...]
    read_set: tuple[Name, ...]
    write_set: tuple[Name, ...]
    requires: tuple[Name, ...] = ()
    replaces: tuple[Name, ...] = ()
    external_invariants: tuple[Name, ...]
    plan_ref: str | None = None
    registry_ref: str | None = None
    installer: str | None = None
    parameters: dict[str, FiniteJsonValue] = Field(default_factory=dict)

    @model_validator(mode="after")
    def coherent_edit_set(self) -> Self:
        if len({e.path for e in self.edits}) != len(self.edits):
            raise ValueError("overlapping file edits must be composed before staging")
        if self.scope == "group" and len(self.groups) != 1:
            raise ValueError("group scope requires one group")
        if set(self.requires) & set(self.replaces):
            raise ValueError("cannot both require and replace a bundle")
        return self


class ProjectRevision(ProjectModel):
    id: Name
    parent_id: str | None
    anchor_id: Name
    contract_ref: Name
    plan_ref: Name
    registry_ref: Name
    files: tuple[FileRef, ...]
    bundles: tuple[Name, ...] = ()
    retired_bundles: tuple[Name, ...] = ()


class ProjectEvaluation(ProjectModel):
    id: Name
    revision_id: Name
    parent_id: Name
    contract_ref: Name
    bundle_id: Name
    quality: Literal["pass", "fail", "unknown"]
    feasible: bool
    objective: float | None
    reference_objective: float | None
    metrics: dict[str, FiniteJsonValue] = Field(default_factory=dict)
    required_ranks: tuple[int, ...]
    rank_bundle_hashes: dict[str, str]
    required_calls: tuple[str, ...]
    executed_calls: tuple[str, ...]
    evidence: tuple[str, ...] = ()
    reasons: tuple[str, ...] = ()


class GroupFeedback(ProjectModel):
    revision_id: Name
    bundle_id: Name
    groups: tuple[Name, ...]
    scope: Scope
    project_metrics: dict[str, FiniteJsonValue]
    observations: tuple[Observation, ...] = ()
    neighbors: tuple[str, ...] = ()
    failures: tuple[str, ...] = ()
    interaction: dict[str, FiniteJsonValue] = Field(default_factory=dict)
    attribution_status: Literal["measured", "joint_only", "unknown"] = "unknown"
    evidence: tuple[str, ...] = ()
