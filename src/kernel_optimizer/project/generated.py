"""Machine-readable contracts for generated artifacts, before GPU execution."""
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class Selection(BaseModel):
    model_config = ConfigDict(extra="allow")
    scope: Literal["group", "joint", "region", "execution_plan"]
    groups: list[str] = Field(min_length=1)
    affected_calls: list[str] = Field(min_length=1)
    rationale: str
    hypothesis: str


class Proposal(BaseModel):
    model_config = ConfigDict(extra="allow")
    implementation: Literal["kernel", "library", "composite", "distributed", "strategy"]
    replacement_calls: list[str] = Field(min_length=1)
    rationale: str
    setup_work: str | dict | list
    quality_risks: str | list[str]
    eliminated_calls: list[str] = Field(default_factory=list)
    requires: list[str] = Field(default_factory=list)
    replaces: list[str] = Field(default_factory=list)
    parameter_space: dict[str, Any] = Field(default_factory=dict)


class CollectionPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")
    collectors: list[Literal["torch", "nsys"]]
    requested_fields: list[str]
    unavailable_fields: dict[str, str]
    rationale: str


def validate_proposal(directory: Path):
    from kernel_optimizer.control.direct_task import TaskSpace
    proposal = Proposal.model_validate_json((directory/"proposal.json").read_bytes())
    TaskSpace.model_validate(proposal.parameter_space)
    compile((directory/"candidate.py").read_bytes(), "candidate.py", "exec")
