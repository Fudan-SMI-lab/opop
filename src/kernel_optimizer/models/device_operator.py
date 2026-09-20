"""Authored-kernel metadata only; execution/output participation needs runner proof."""

from typing import Annotated, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field


class DeviceKernelDeclaration(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid")

    backend: Literal["triton", "cuda"]
    source_file: str = Field(min_length=1)
    entry: str = Field(min_length=1)
    output_arg_indices: tuple[Annotated[int, Field(ge=0)], ...] = ()
