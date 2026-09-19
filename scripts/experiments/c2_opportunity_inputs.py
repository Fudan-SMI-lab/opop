"""Fixed opportunity assignments and original-P1 input boundaries; no tunable program knobs."""

from collections.abc import Callable
from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from time import monotonic, time
from typing import Final, Literal, Self

from pydantic import Field, model_validator

from kernel_optimizer.conditional.task_response import TaskResponse
from kernel_optimizer.config import AppConfig
from kernel_optimizer.paramspace.materializer import extract_defaults
from scripts.experiments.c2_local_costs import Costs
from scripts.experiments.c2_local_inputs import InputError, Responses, Shared, Strict
from scripts.experiments.c2_method_protocol import Deadline

Wave = Literal[1, 2]
Slot = Literal["A0", "A1", "B0", "B1"]
Arm = Literal["G0", "C2"]


@dataclass(frozen=True, slots=True)
class Cell:
    task: str
    arm: Arm
    rep: int
    seed: int


CELLS: Final[dict[tuple[Wave, Slot], Cell]] = {
    (1, "A0"): Cell("level3:43", "G0", 0, 0), (1, "A1"): Cell("level3:43", "C2", 0, 0),
    (1, "B0"): Cell("level3:21", "G0", 0, 0), (1, "B1"): Cell("level3:21", "C2", 0, 0),
    (2, "A0"): Cell("level3:21", "C2", 1, 2), (2, "A1"): Cell("level3:21", "G0", 1, 2),
    (2, "B0"): Cell("level3:43", "C2", 1, 2), (2, "B1"): Cell("level3:43", "G0", 1, 2),
}
PARENTS: Final = {"level3:43": ("cand-b67a1cb4", "tr-6fb84baa"),
                 "level3:21": ("cand-4c96b8c4", "tr-6afd5e56")}


class OpportunityInputs(Strict):
    wave: Wave
    slot: Slot
    shared: Path
    reference: Path
    helpers: tuple[Path, ...] = ()
    probe_strategy: Literal["provided", "targeted"] = "provided"

    @model_validator(mode="after")
    def absolute_paths(self) -> Self:
        if not all(p.is_absolute() for p in (self.shared, self.reference, *self.helpers)):
            raise InputError("opportunity paths must be absolute")
        return self

    def load_parent(self, cfg: AppConfig) -> Shared:
        shared = Shared.model_validate_json(self.shared.read_text(encoding="utf-8"))
        task = CELLS[self.wave, self.slot].task
        if shared.task != task or shared.state != "off" or (
                shared.parent.candidate_id, shared.parent.trial_id) != PARENTS[task]:
            raise InputError("slot requires its exact original P1 candidate/trial and off-state")
        if shared.parent.status != "complete" or shared.parent.latency_ms is None:
            raise InputError("original parent requires a valid native record")
        if not any(t == shared.parent for t in shared.trials):
            raise InputError("parent record differs from ordinary Shared history")
        defaults = extract_defaults(shared.source)
        domains = {d.name: d.choices for d in shared.space.params}
        if set(defaults) != set(domains) or set(shared.parent.params.values) != set(domains):
            raise InputError("original source/selected parameters differ from declared space keys")
        if any(defaults[k] not in v or shared.parent.params.values[k] not in v for k, v in domains.items()):
            raise InputError("original source/selected parameters leave the declared domain")
        if self.reference.read_text(encoding="utf-8") != shared.reference_source:
            raise InputError("reference differs from original Shared")
        if shared.evaluation != cfg.evaluation.model_dump(mode="json") or shared.device != cfg.device:
            raise InputError("config evaluation/device differs from original Shared")
        if cfg.evaluation.perf_trials != 100 or cfg.evaluation.correctness_trials != 5:
            raise InputError("program full measurements require 100 samples/5 correctness")
        for helper in self.helpers:
            if not helper.is_file() or not helper.resolve().is_relative_to(self.reference.resolve().parent):
                raise InputError("helpers must exist beneath the reference directory")
        return shared


class G0Responses(Responses):
    """Zero-observation subtype for G0; ordinary Shared evidence is not removed."""

    responses: list[TaskResponse] = Field(default_factory=list, max_length=0)
    probe_calls: Literal[0] = 0
    costs: Costs = Field(default_factory=Costs)

    def validate_for(self, shared: Shared) -> None:
        if self.shared_id != shared.identity() or self.costs.worker_attempts != 0:
            raise InputError("empty G0 envelope must match its parent and contain zero acquisition attempts")


def validate_deadline(deadline_unix_s: float, now: float) -> None:
    if not isfinite(deadline_unix_s) or deadline_unix_s <= 0 or deadline_unix_s - now > 14400:
        raise InputError("deadline must be a finite positive Unix timestamp no more than four hours ahead")


class PilotClock(Strict):
    campaign_started_unix_s: float = Field(gt=0, allow_inf_nan=False)
    work_cutoff_unix_s: float = Field(gt=0, allow_inf_nan=False)
    final_deadline_unix_s: float = Field(gt=0, allow_inf_nan=False)

    @model_validator(mode="after")
    def fixed_reserve(self) -> Self:
        if (self.work_cutoff_unix_s != self.campaign_started_unix_s + 16800
                or self.final_deadline_unix_s != self.campaign_started_unix_s + 18000):
            raise InputError("pilot clock requires fixed S, S+16800 work cutoff and S+18000 final deadline")
        return self


@dataclass(frozen=True, slots=True)
class CampaignRun:
    cfg: AppConfig
    output: Path
    deadline_unix_s: float
    now: Callable[[], float] = time
    clock: Callable[[], float] = monotonic
    pilot_clock: PilotClock | None = None

    @property
    def campaign_started_unix_s(self) -> float:
        return self.pilot_clock.campaign_started_unix_s if self.pilot_clock else self.deadline_unix_s - 14400

    def admission_deadline(self, *, heldout: bool = False) -> float:
        return self.pilot_clock.work_cutoff_unix_s if self.pilot_clock and not heldout else self.deadline_unix_s

    def start(self, *, heldout: bool = False) -> tuple[float, Deadline]:
        started = self.now()
        if self.pilot_clock is None:
            validate_deadline(self.deadline_unix_s, started)
        elif self.deadline_unix_s != self.pilot_clock.final_deadline_unix_s or started < self.campaign_started_unix_s:
            raise InputError("pilot final identity mismatch or launch before declared start")
        return started, Deadline(self.clock(), self.clock, max(0.0, self.admission_deadline(heldout=heldout) - started))
