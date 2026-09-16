"""Evaluation wire contracts; collection, hashing and eligibility live elsewhere."""

from math import isclose
from typing import Literal, Protocol, Self, assert_never

from pydantic import Field, model_validator

from kernel_optimizer.models.contract_common import (
    CallableRef, ContractError, ContractModel, FileRef, FiniteJsonValue, FiniteNumber, Name,
    NonnegativeInt, NonnegativeNumber,
)
from kernel_optimizer.models.objective import Direction, MetricValue
from kernel_optimizer.models.resource_axis import ResourceBound, ResourceObservation

MeasurementValidity = Literal["valid", "invalid", "unknown"]
QualityStatus = Literal["pass", "fail", "unknown"]
FeasibilityStatus = Literal["feasible", "infeasible", "unknown"]


class SoftwareFingerprint(ContractModel):
    name: Name
    version: Name
    build: Name
    capabilities: tuple[Name, ...] = ()


class ScientificExecutionContext(ContractModel):
    hardware_identity: Name
    software_fingerprints: tuple[SoftwareFingerprint, ...]
    execution_policies: dict[Name, FiniteJsonValue] = Field(default_factory=dict)


class PathMapping(ContractModel):
    source: Name
    target: Name


class AssetLocator(ContractModel):
    asset_ref: Name
    locator: Name
    provenance: Name


class ExecutionBinding(ContractModel):
    transport: Literal["local", "wsl"]
    target_os: Name
    python: Name
    project_root: Name
    path_mappings: tuple[PathMapping, ...] = ()
    asset_locators: tuple[AssetLocator, ...] = ()


class TaskObservation(ContractModel):
    output: FiniteJsonValue
    final_state: FiniteJsonValue = None


class ConstraintResult(ContractModel):
    id: Name
    status: Literal["satisfied", "violated", "unknown"] = "unknown"
    observed_values: dict[Name, FiniteNumber] = Field(default_factory=dict)
    aggregate_value: FiniteNumber | None = None
    unit: Name
    reasons: tuple[str, ...] = ()


class EvalResult(ContractModel):
    measurement_validity: MeasurementValidity = "unknown"
    quality: QualityStatus = "unknown"
    feasibility: FeasibilityStatus = "unknown"
    constraint_results: tuple[ConstraintResult, ...] = ()
    case_metrics: tuple[MetricValue, ...] = ()
    metrics: tuple[MetricValue, ...] = ()
    native_j: FiniteNumber | None = None
    objective_unit: Name
    direction: Direction
    protocol_id: Name
    resources: tuple[ResourceObservation, ...] = ()
    reasons: tuple[str, ...] = ()

    @model_validator(mode="after")
    def invalid_has_no_objective(self) -> Self:
        match self.measurement_validity:
            case "invalid":
                if self.native_j is not None:
                    raise ContractError("native_j", "invalid measurement must have null objective")
            case "valid" | "unknown":
                pass
            case _:
                assert_never(self.measurement_validity)
        return self


class CostRecord(ContractModel):
    invocation_id: Name
    record_id: Name
    wall_start: NonnegativeNumber | None = None
    wall_end: NonnegativeNumber | None = None
    wall_seconds: NonnegativeNumber | None = None
    phase: Literal["prepare", "search", "final"]
    build_seconds: NonnegativeNumber | None = None
    load_seconds: NonnegativeNumber | None = None
    evaluation_seconds: NonnegativeNumber | None = None
    failure_seconds: NonnegativeNumber | None = None
    gpu_seconds: NonnegativeNumber | None = None
    gpu_measurement_scope: Literal["per_device_busy_union", "cuda_event_aggregate"] | None = None
    llm_tokens: NonnegativeInt | None = None
    llm_cost: NonnegativeNumber | None = None

    @model_validator(mode="after")
    def validate_measurement_scope(self) -> Self:
        if self.gpu_seconds is not None and self.gpu_measurement_scope is None:
            raise ContractError("gpu_measurement_scope", "measured GPU seconds require a scope")
        if self.wall_start is not None and self.wall_end is not None:
            if self.wall_end < self.wall_start:
                raise ContractError("wall_end", "cannot precede wall_start")
            if self.wall_seconds is not None and not isclose(
                self.wall_seconds, self.wall_end - self.wall_start, rel_tol=1e-9, abs_tol=1e-9,
            ):
                raise ContractError("wall_seconds", "must equal wall_end minus wall_start")
        return self


class EvaluationRequest(ContractModel):
    schema_version: Name
    task_id: Name
    bundle_id: Name
    protocol_id: Name
    candidate_ref: Name
    input_binding_ref: Name
    execution_binding_ref: Name
    role: Name
    repeat_index: NonnegativeInt
    payload: dict[str, FiniteJsonValue]


class EvaluationResponse(ContractModel):
    result: EvalResult
    output_ref: Name | None = None
    cost: CostRecord


class EvaluationCallable(Protocol):
    def __call__(self, request: EvaluationRequest) -> EvaluationResponse: ...


class ControlExpected(ContractModel):
    measurement_validity: MeasurementValidity = "unknown"
    quality: QualityStatus = "unknown"
    feasibility: FeasibilityStatus = "unknown"
    metrics: tuple[MetricValue, ...] = ()
    native_j: FiniteNumber | None = None
    objective_unit: Name | None = None
    direction: Direction | None = None
    output_ref: Name | None = None


class ControlSpec(ContractModel):
    id: Name
    role: Name
    request_binding_ref: Name
    expected: ControlExpected
    required: bool


class EvalBundleManifest(ContractModel):
    schema_version: Name
    task_definition_ref: Name
    files: tuple[FileRef, ...]
    entrypoint: CallableRef
    request_schema_ref: Name
    input_binding_ref: Name
    control_specs: tuple[ControlSpec, ...]
    resource_provider: CallableRef | None = None
    protocol_id: Name
    bundle_id: Name

    @model_validator(mode="after")
    def unique_entries(self) -> Self:
        paths = [file.path for file in self.files]
        ids = [control.id for control in self.control_specs]
        if len(paths) != len(set(paths)) or len(ids) != len(set(ids)):
            raise ContractError("manifest", "file paths and control IDs must be unique")
        return self


class ResourceProbeResult(ContractModel):
    observations: tuple[ResourceObservation, ...] = ()
    bounds: tuple[ResourceBound, ...] = ()
    fit_status: Literal["fit", "refused", "unknown"] = "unknown"
    reasons: tuple[str, ...] = ()
    cost: CostRecord
