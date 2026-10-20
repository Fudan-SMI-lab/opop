"""User-supplied project, workload and goal; no predefined operator groups."""
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field
from kernel_optimizer.models.contract_common import Name


class ProjectUnderstanding(BaseModel):
    """Agent output types are checked before leaving the existing repair loop."""
    goal: Name
    entrypoint: Name
    completion: Name
    state_protocol: Name
    invariants: list[Name] = Field(min_length=1)
    objective: dict[str, Any]
    workload: dict[str, Any]
    quality: dict[str, Any]
    missing_information: list[Name]
    assumptions: list[Name]


class ProjectRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: Literal["project-request.v1"] = "project-request.v1"
    source_root: Path
    entry: str
    files: list[str]
    goal: str
    objective: dict[str, Any]
    workload: dict[str, Any]
    quality: dict[str, Any]
    assets: dict[str, str] = Field(default_factory=dict)
    devices: list[str]
    distributed_backend: str | None = None
    communication: Literal["auto", "off", "required"] = "auto"
    required_observations: list[str] = Field(default_factory=list)
    tools: dict[str, str] = Field(default_factory=dict)
    profiling: dict[str, Any] = Field(default_factory=dict)
    binding_backend: Literal["legacy", "scoped"] = "legacy"
    warmup: int = Field(default=2, ge=0)
    repeats: int = Field(default=5, ge=3)
    continuous_steps: int = Field(default=2, ge=2)
    worker_timeout_s: float | None = 600.0
    lease_dir: Path
    notes: list[str] = Field(default_factory=list)

    def portable(self) -> dict:
        return self.model_dump(mode="json")
