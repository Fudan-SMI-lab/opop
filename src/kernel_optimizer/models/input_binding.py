"""Frozen candidate-relative call prototypes, never live runtime objects."""

from typing import Annotated, Literal

from pydantic import Field

from kernel_optimizer.models.contract_common import CallableRef, ContractModel, FiniteJsonValue, Name


class LiteralBinding(ContractModel):
    kind: Literal["literal"] = "literal"
    value: FiniteJsonValue


class ArtifactBinding(ContractModel):
    kind: Literal["artifact"] = "artifact"
    artifact_ref: Name


class RuntimeBinding(ContractModel):
    kind: Literal["runtime_ref"] = "runtime_ref"
    slot: Name


BindingValue = Annotated[LiteralBinding | ArtifactBinding | RuntimeBinding, Field(discriminator="kind")]


class CallBinding(ContractModel):
    callable: CallableRef
    args: tuple[BindingValue, ...] = ()
    kwargs: dict[Name, BindingValue] = Field(default_factory=dict)


class InputBinding(ContractModel):
    candidate_call: CallBinding
    reference_call: CallBinding
    quality_call: CallBinding
    state_binding: BindingValue | None = None
