"""Typed inputs, fixed cells, and observable opportunity results for the method program."""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from time import monotonic
from typing import Final, Literal, Self

from pydantic import Field, model_validator

from kernel_optimizer.models.core import Backend, ParamSet, TrialRecord
from kernel_optimizer.control.direct_task import TaskSpace
from scripts.experiments.c2_local_inputs import InputError, Shared, Strict
from scripts.experiments.c2_retune import RetuneResult

Slot = Literal["A0", "A1", "B0", "B1"]
Arm = Literal["G0", "L", "P", "H"]
Phase = Literal["core", "A", "B", "C"]
Status = Literal["complete", "invalid", "failed", "dependency_failed", "censored"]


@dataclass(frozen=True, slots=True)
class Cell:
    task: str
    rep: int
    order: tuple[Arm, ...]


SLOTS: Final[dict[Slot, Cell]] = {
    "A0": Cell("level3:43", 0, ("G0", "L", "P", "H")),
    "A1": Cell("level3:21", 0, ("L", "H", "G0", "P")),
    "B0": Cell("level3:21", 1, ("P", "G0", "H", "L")),
    "B1": Cell("level3:43", 1, ("H", "P", "L", "G0")),
}


def final_orders(slot: Slot) -> tuple[tuple[str, ...], ...]:
    first = ("parent", *SLOTS[slot].order)
    return first, first[::-1], first[1:] + first[:1]


class MethodInputs(Strict):
    slot: Slot
    shared: Path
    reference: Path
    helpers: tuple[Path, ...] = ()
    core_results: tuple[Path, ...] = ()

    @model_validator(mode="after")
    def absolute_paths(self) -> Self:
        if not all(p.is_absolute() for p in (self.shared, self.reference, *self.helpers, *self.core_results)):
            raise InputError("input wrapper paths must be absolute")
        return self

    def load_parent(self) -> Shared:
        shared = Shared.model_validate_json(self.shared.read_text(encoding="utf-8"))
        if shared.task != SLOTS[self.slot].task or shared.state != "off":
            raise InputError("slot requires its original task P1 off-state")
        if shared.parent.status != "complete" or shared.parent.latency_ms is None:
            raise InputError("original parent requires a valid native tuning record")
        if self.reference.read_text(encoding="utf-8") != shared.reference_source:
            raise InputError("reference differs from original Shared")
        for helper in self.helpers:
            if not helper.is_file():
                raise InputError(f"missing helper: {helper}")
            if not helper.resolve().is_relative_to(self.reference.resolve().parent):
                raise InputError("helpers must reside beneath reference's directory")
        return shared


@dataclass(frozen=True, slots=True)
class Deadline:
    started: float
    clock: Callable[[], float] = monotonic
    seconds: float = 4 * 3600

    def remaining(self) -> float:
        return max(0.0, self.started + self.seconds - self.clock())

    def check(self) -> None:
        if self.remaining() <= 0:
            raise SchedulingExpired(self.started + self.seconds)


class SchedulingExpired(RuntimeError):
    def __init__(self, deadline: float) -> None:
        self.deadline = deadline
        super().__init__(f"scheduling deadline reached: {deadline}")


class Opportunity(Strict):
    arm: Arm
    status: Status = "censored"
    error: str | None = None
    asked: int = 0
    selected: Literal["parent", "child"] = "parent"
    artifact: Path
    params: ParamSet
    backend: Backend
    proposal: Path | None = None
    space: Path | None = None
    selected_space: TaskSpace | None = None
    retune: RetuneResult | None = None
    finals: list[TrialRecord] = Field(default_factory=list)
    final_status: Literal["complete", "failed", "censored"] = "censored"
    native_expansion_eligible: bool = False


class EventCosts(Strict):
    store: Path
    worker_attempts: int | None
    worker_wall_s: float | None
    agent_calls: int
    agent_attempts: int
    agent_wall_s: float
    provider_cost: float | None = None
    provider_tokens: dict[str, float] | None = None


class CellResult(Strict):
    slot: Slot
    task: str
    rep: int
    shared_id: str
    opportunities: list[Opportunity]
    parent_finals: list[TrialRecord] = Field(default_factory=list)
    acquisition_calls: int | None = 0
    acquisition_error: str | None = None
    elapsed_s: float = 0
    drain_s: float = 0
    provider_cost: float | None = None
    provider_tokens: dict[str, float] | None = None
    costs: tuple[EventCosts, ...] = ()


class BranchPlan(Strict):
    phase: Literal["A", "B", "C"] | None
    eligible_slots: tuple[Slot, ...] = ()
    study_opportunities: int = Field(default=0, ge=0, le=8)
    rounds: int = 0
    ready: Literal[False] = False
    reason: str
    not_applicable: dict[Slot, str] = Field(default_factory=dict)


class ConditionalReadiness(Strict):
    planner_requested_slots: tuple[Slot, ...] = ()
    bridge_verified: bool = False
    worker_budget_verified: bool = False
