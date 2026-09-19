"""CPU-only structural boundary for the frozen task-local runner interface."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, JsonValue

from kernel_optimizer.evaluation.task_eval import TaskEvaluation

if TYPE_CHECKING:
    from examples.c3_qwen3.model_binding import BundleSpec
    from examples.c3_qwen3.runner_records import GoalSpec, Prompt

type GoalId = Literal["ttft", "single", "multi"]
type Split = Literal["search", "heldout"]


class Receipt(Protocol):
    @property
    def bundle_sha256(self) -> str: ...
    @property
    def source_hashes(self) -> Mapping[str, str]: ...
    @property
    def params_sha256(self) -> str: ...
    @property
    def site_ids(self) -> tuple[str, ...]: ...
    @property
    def baseline_restored(self) -> bool: ...


class QualityReport(Protocol):
    @property
    def valid(self) -> bool: ...
    @property
    def metrics(self) -> Mapping[str, float]: ...
    @property
    def raw_path(self) -> Path: ...
    @property
    def detail(self) -> str | None: ...


class MeasurementReport(Protocol):
    @property
    def evaluation(self) -> TaskEvaluation: ...
    @property
    def raw_path(self) -> Path: ...


@runtime_checkable
class ResidentRunner(Protocol):
    def bind(self, bundle_spec: BundleSpec) -> Receipt: ...
    def reset(self, request_specs: tuple[Prompt, ...]) -> None: ...
    def quality(self, prompt_ids: tuple[str, ...], oracle_refs: Path) -> QualityReport: ...
    def measure(self, goal_spec: GoalSpec, prompt_ids: tuple[str, ...]) -> MeasurementReport: ...


@runtime_checkable
class CallBudget(Protocol):
    def admit(self) -> bool: ...


class Record(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)


class Context(Record):
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, arbitrary_types_allowed=True)
    assets_manifest: Path
    contract: Path
    bundle_path: Path
    output_dir: Path
    goal_id: GoalId
    split: Split
    prompt_ids: tuple[str, ...]
    resident_runner: ResidentRunner
    oracle_refs: Path
    call_budget: CallBudget
    frozen_files: dict[Path, str] = Field(default_factory=dict)
