"""Native objective declarations and finite raw measurements, without scoring."""

from typing import Literal, Self, assert_never

from pydantic import Field, model_validator

from kernel_optimizer.models.contract_common import (
    CallableRef, ContractError, ContractModel, FiniteNumber, Name, NonnegativeNumber,
)

Direction = Literal["minimize", "maximize"]


class RequiredMetric(ContractModel):
    name: Name
    unit: Name


class MetricValue(RequiredMetric):
    value: FiniteNumber
    case_id: Name


class AggregationSpec(ContractModel):
    kind: Literal["weighted_mean", "sum", "bundle_callable"]
    case_weights: dict[Name, NonnegativeNumber] = Field(default_factory=dict)
    callable: CallableRef | None = None

    @model_validator(mode="after")
    def validate_requirements(self) -> Self:
        match self.kind:
            case "weighted_mean":
                if not self.case_weights or sum(self.case_weights.values()) <= 0:
                    raise ContractError("case_weights", "weighted mean requires positive total weight")
            case "bundle_callable":
                if self.callable is None:
                    raise ContractError("callable", "bundle aggregation requires a callable")
            case "sum":
                pass
            case _:
                assert_never(self.kind)
        return self


class ObjectiveSpec(ContractModel):
    name: Name
    unit: Name
    direction: Direction
    required_metrics: tuple[RequiredMetric, ...]
    aggregation: AggregationSpec
    improvement_threshold: NonnegativeNumber = 0.0

    @model_validator(mode="after")
    def unique_metrics(self) -> Self:
        names = [metric.name for metric in self.required_metrics]
        if len(names) != len(set(names)):
            raise ContractError("required_metrics", "names must be unique")
        return self
