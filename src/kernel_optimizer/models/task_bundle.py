"""Frozen task declarations and structural consistency, without evaluation policy."""

from typing import Annotated, Literal, Self, assert_never

from pydantic import Field, model_validator

from kernel_optimizer.models.contract_common import (
    CallableRef, ContractError, ContractModel, FiniteJsonValue, FiniteNumber, Name,
    NonnegativeInt, NonnegativeNumber, PositiveInt,
)
from kernel_optimizer.models.evaluation_bundle import ScientificExecutionContext
from kernel_optimizer.models.objective import ObjectiveSpec
from kernel_optimizer.models.resource_axis import ParameterAxis, ResourceDeclaration


class ReferenceSpec(ContractModel):
    artifact_ref: Name
    callable: CallableRef
    input_binding_ref: Name
    expected_outputs_ref: Name | None = None


class WorkloadSpec(ContractModel):
    cases_ref: Name
    case_ids: Annotated[tuple[Name, ...], Field(min_length=1)]
    completion: Name
    execution_parameters: dict[str, FiniteJsonValue] | None = None

    @model_validator(mode="after")
    def unique_cases(self) -> Self:
        if len(self.case_ids) != len(set(self.case_ids)):
            raise ContractError("case_ids", "must be unique")
        return self


class CapabilityBinding(ContractModel):
    name: Name
    applicability: Literal["applicable", "not_applicable", "unknown"]
    binding_ref: Name


class AdapterHooks(ContractModel):
    load: CallableRef
    bind: CallableRef
    reset: CallableRef
    observe_state: CallableRef


class StatelessSpec(ContractModel):
    kind: Literal["stateless"] = "stateless"
    capability_bindings: tuple[CapabilityBinding, ...] = ()


class StatefulSpec(ContractModel):
    kind: Literal["stateful"] = "stateful"
    schema_ref: Name
    initial_binding_ref: Name
    reset_policy: Name
    adapter_hooks: AdapterHooks
    capability_bindings: tuple[CapabilityBinding, ...] = ()


StateSpec = Annotated[StatelessSpec | StatefulSpec, Field(discriminator="kind")]


class QualitySpec(ContractModel):
    oracle_ref: Name
    callable: CallableRef
    output_schema_ref: Name
    tolerances: dict[Name, NonnegativeNumber]
    negative_controls_ref: Name


class ConstraintSpec(ContractModel):
    id: Name
    source_kind: Literal["metric", "resource"]
    name: Name
    unit: Name
    operator: Literal["lt", "le", "eq", "ge", "gt"]
    threshold: FiniteNumber
    case_ids: Annotated[tuple[Name, ...], Field(min_length=1)]
    entity_id: Name | None = None
    aggregation: Literal["all", "weighted_mean", "max", "sum"]

    @model_validator(mode="after")
    def unique_cases(self) -> Self:
        if len(self.case_ids) != len(set(self.case_ids)):
            raise ContractError("case_ids", "must be unique")
        return self


class PairRoles(ContractModel):
    repeat_index: NonnegativeInt
    phase: Literal["discovery", "validation"]
    f_role: Name
    n_role: Name


class ProtocolSpec(ContractModel):
    version: Name
    execution_context: ScientificExecutionContext
    warmup_count: NonnegativeInt
    repeat_count: PositiveInt
    paired_repeat_count: Annotated[int, Field(strict=True, ge=2)]
    c2_pair_roles: tuple[PairRoles, ...]
    c2_order_seed: int
    c2_mode: Literal["empirical_difference_envelope", "legacy_log4percent"]
    c2_threshold: NonnegativeNumber = 0.0
    measurement_scope: Name
    cost_scope: Name

    @model_validator(mode="after")
    def consistent_roles(self) -> Self:
        if tuple(role.repeat_index for role in self.c2_pair_roles) != tuple(range(self.paired_repeat_count)):
            raise ContractError("c2_pair_roles", "must cover each declared pair in order")
        names = [name for pair in self.c2_pair_roles for name in (pair.f_role, pair.n_role)]
        if len(names) != len(set(names)):
            raise ContractError("c2_pair_roles", "endpoint roles must be unique")
        for pair in self.c2_pair_roles:
            if pair.phase != ("discovery" if pair.repeat_index == 0 else "validation"):
                raise ContractError("c2_pair_roles", "only pair zero is discovery")
        return self


class TaskDefinition(ContractModel):
    schema_version: Name
    task_id: Name
    project_ref: Name
    reference: ReferenceSpec
    workload: WorkloadSpec
    state: StateSpec
    quality: QualitySpec
    objective: ObjectiveSpec
    constraints: tuple[ConstraintSpec, ...]
    resources: tuple[ResourceDeclaration, ...]
    parameter_domains: tuple[ParameterAxis, ...]
    protocol: ProtocolSpec

    @model_validator(mode="after")
    def consistent_cases(self) -> Self:
        cases = set(self.workload.case_ids)
        weights = self.objective.aggregation.case_weights
        if set(weights) - cases:
            raise ContractError("case_weights", "must belong to workload")
        weighted_groups: list[tuple[str, ...]] = []
        match self.objective.aggregation.kind:
            case "weighted_mean":
                weighted_groups.append(self.workload.case_ids)
            case "sum" | "bundle_callable":
                pass
            case _:
                assert_never(self.objective.aggregation.kind)
        for constraint in self.constraints:
            if set(constraint.case_ids) - cases:
                raise ContractError("constraint.case_ids", "must belong to workload")
            match constraint.aggregation:
                case "weighted_mean":
                    weighted_groups.append(constraint.case_ids)
                case "all" | "max" | "sum":
                    pass
                case _:
                    assert_never(constraint.aggregation)
        for group in weighted_groups:
            if set(group) - set(weights) or sum(weights[case] for case in group) <= 0:
                raise ContractError("case_weights", "weighted cases need defined weights with positive sum")
        for field, names in (("constraints", [c.id for c in self.constraints]),
                             ("parameter_domains", [a.name for a in self.parameter_domains])):
            if len(names) != len(set(names)):
                raise ContractError(field, "identifiers must be unique")
        return self


class GenericTaskSpec(ContractModel):
    kind: Literal["generic"] = "generic"
    task_id: Name
    task_definition_ref: Name
    bundle_ref: Name
    protocol_id: Name
    input_binding_ref: Name
