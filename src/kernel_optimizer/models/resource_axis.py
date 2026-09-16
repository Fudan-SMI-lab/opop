"""Open resource observations and parameter semantics; no evaluator dependency."""

from typing import Literal, Self, assert_never

from pydantic import model_validator

from kernel_optimizer.models.contract_common import ContractError, ContractModel, FiniteNumber, Name


class ResourceDeclaration(ContractModel):
    name: Name
    entity_id: Name | None = None
    unit: Name
    scope: Name
    provenance: Name


class ResourceObservation(ResourceDeclaration):
    value: FiniteNumber
    case_id: Name
    entity_type: Name
    context_ref: Name


class ResourceBound(ResourceDeclaration):
    value: FiniteNumber
    entity_type: Name
    context_ref: Name
    kind: Literal["configured_capacity", "device_capacity", "measured_throughput", "empirical_turn"]


class ParameterAxis(ContractModel):
    name: Name
    kind: Literal["numeric", "ordinal", "nominal"]
    value_type: Literal["int", "float", "str", "bool"]
    choices: tuple[int | FiniteNumber | str | bool, ...]
    semantics: Name
    unit: Name | None = None
    numeric_positions: tuple[FiniteNumber, ...] | None = None

    @model_validator(mode="after")
    def validate_domain(self) -> Self:
        if not self.choices or len(set(self.choices)) != len(self.choices):
            raise ContractError("choices", "must be nonempty and unique")
        expected_types = {"int": int, "float": float, "str": str, "bool": bool}
        if any(type(value) is not expected_types[self.value_type] for value in self.choices):
            raise ContractError("choices", "must match value_type without bool/numeric coercion")
        match self.kind:
            case "numeric":
                if self.value_type not in ("int", "float") or self.unit is None:
                    raise ContractError("numeric", "requires numeric values and a unit")
            case "ordinal":
                if self.numeric_positions is not None:
                    if len(self.numeric_positions) != len(self.choices):
                        raise ContractError("numeric_positions", "must match choices")
                    if any(b <= a for a, b in zip(self.numeric_positions, self.numeric_positions[1:])):
                        raise ContractError("numeric_positions", "must strictly increase")
                    if self.unit is None:
                        raise ContractError("unit", "numeric spacing requires a unit")
            case "nominal":
                if self.numeric_positions is not None:
                    raise ContractError("numeric_positions", "nominal axes have no numeric spacing")
            case _:
                assert_never(self.kind)
        return self
