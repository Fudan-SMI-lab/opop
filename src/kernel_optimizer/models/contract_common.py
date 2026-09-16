"""Shared strict boundary primitives; no legacy or execution dependencies."""

import json
from typing import Annotated, ClassVar

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, JsonValue

Name = Annotated[str, Field(min_length=1, pattern=r"\S")]
FiniteNumber = Annotated[float, Field(strict=True, allow_inf_nan=False)]
NonnegativeNumber = Annotated[FiniteNumber, Field(ge=0)]
NonnegativeInt = Annotated[int, Field(strict=True, ge=0)]
PositiveInt = Annotated[int, Field(strict=True, ge=1)]


def finite_json(value: JsonValue) -> JsonValue:
    """JsonValue bypasses model float policy; enforce finite JSON at extension boundaries."""
    _ = json.dumps(value, allow_nan=False)
    return value


FiniteJsonValue = Annotated[JsonValue, AfterValidator(finite_json)]


class ContractModel(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, strict=True, extra="forbid", allow_inf_nan=False)


class ContractError(ValueError):
    def __init__(self, field: str, rule: str) -> None:
        self.field: str = field
        self.rule: str = rule
        super().__init__(f"{field}: {rule}")


class CallableRef(ContractModel):
    module: Name
    callable: Name


class FileRef(ContractModel):
    path: Name
    content_ref: Name
    sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
