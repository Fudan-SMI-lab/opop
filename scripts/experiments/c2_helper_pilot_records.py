"""Pilot subprocess boundaries and honest terminal records, including absent artifacts."""

from pathlib import Path
from typing import Literal, Self

from pydantic import Field, model_validator

from kernel_optimizer.config import load_config
from kernel_optimizer.models.core import sha256_text
from scripts.experiments.c2_local_inputs import InputError, Shared, Strict
from scripts.experiments.c2_opportunity_inputs import CELLS, OpportunityInputs, PilotClock, Slot, Wave
from scripts.experiments.c2_opportunity_records import HeldoutResult, OpportunityResult


class PilotParent(Strict):
    shared: Path
    reference: Path
    helpers: tuple[Path, ...] = ()


class HostPilot(Strict):
    host: Literal["A", "B"]
    clock: PilotClock
    output: Path
    gpu0_config: Path
    gpu1_config: Path
    task21: PilotParent
    task43: PilotParent

    @model_validator(mode="after")
    def absolute_paths(self) -> Self:
        paths = [self.output, self.gpu0_config, self.gpu1_config]
        for parent in (self.task21, self.task43):
            paths.extend((parent.shared, parent.reference, *parent.helpers))
        if not all(p.is_absolute() for p in paths):
            raise InputError("pilot paths must be absolute")
        return self

    def opportunity(self, wave: Wave, slot: Slot) -> OpportunityInputs:
        cell = CELLS[wave, slot]
        parent = self.task21 if cell.task == "level3:21" else self.task43
        return OpportunityInputs(wave=wave, slot=slot, **parent.model_dump(),
                                 probe_strategy="targeted" if cell.arm == "C2" else "provided")


class WorkerJob(Strict):
    mode: Literal["opportunity", "heldout"]
    config: Path
    inputs: Path
    output: Path
    gpu: Literal[0, 1]
    clock: PilotClock


class ProcessExit(Strict):
    exit_code: int | None = None
    launched_unix_s: float | None = None
    ended_unix_s: float
    error: str | None = None


class DispatchRecord(Strict):
    wave: Wave
    slot: Slot
    status: Literal["complete", "failed", "censored"] = "censored"
    exit_code: int | None = None
    launched_unix_s: float | None = None
    ended_unix_s: float | None = None
    result: Path | None = None
    error: str | None = "not_started"
    drain_s: float = 0
    final_drain_s: float = 0


class PairDispatch(DispatchRecord):
    intended_blocks: Literal[9] = 9
    pair_ready_unix_s: float | None = None
    heldout_started_unix_s: float | None = None
    ready_to_heldout_start_s: float | None = None


class HostReceipt(Strict):
    host: Literal["A", "B"]
    clock: PilotClock
    opportunities: list[DispatchRecord] = Field(default_factory=list)
    pairs: list[PairDispatch] = Field(default_factory=list)


def terminal_opportunity(job: WorkerJob, exited: ProcessExit) -> OpportunityResult:
    inputs = OpportunityInputs.model_validate_json(job.inputs.read_text(encoding="utf-8"))
    cell = CELLS[inputs.wave, inputs.slot]
    shared: Shared | None = None
    try:
        shared = inputs.load_parent(load_config(job.config))
        result = OpportunityResult.model_validate_json((job.output / "result.json").read_text(encoding="utf-8"))
        if (result.wave, result.slot, result.task, result.arm, result.rep, result.sampler_seed,
                result.shared_id, result.parent_source_sha256, result.deadline_unix_s, result.pilot_clock) != (
                inputs.wave, inputs.slot, cell.task, cell.arm, cell.rep, cell.seed, shared.identity(),
                sha256_text(shared.source), job.clock.final_deadline_unix_s, job.clock):
            raise InputError("opportunity result identity differs from scheduled pilot slot")
        return result
    except (OSError, ValueError, RuntimeError) as exc:
        return OpportunityResult(wave=inputs.wave, slot=inputs.slot, task=cell.task, arm=cell.arm,
            rep=cell.rep, sampler_seed=cell.seed, shared_id=shared.identity() if shared else "",
            parent_source_sha256=sha256_text(shared.source) if shared else "", status="censored",
            started_unix_s=exited.launched_unix_s or exited.ended_unix_s,
            deadline_unix_s=job.clock.final_deadline_unix_s, pilot_clock=job.clock,
            admission_deadline_unix_s=job.clock.work_cutoff_unix_s,
            elapsed_s=max(0, exited.ended_unix_s - (exited.launched_unix_s or exited.ended_unix_s)),
            drain_s=max(0, exited.ended_unix_s - job.clock.work_cutoff_unix_s) if exited.launched_unix_s else 0,
            error=exited.error or f"{type(exc).__name__}: {exc}")


def terminal_heldout(job: WorkerJob, exited: ProcessExit) -> tuple[HeldoutResult, bool]:
    from scripts.experiments.c2_opportunity_heldout import HeldoutInputs
    inputs = HeldoutInputs.model_validate_json(job.inputs.read_text(encoding="utf-8"))
    shared: Shared | None = None
    try:
        shared = inputs.load_parent(load_config(job.config))
        result = HeldoutResult.model_validate_json((job.output / "result.json").read_text(encoding="utf-8"))
        if (result.wave, result.slot, result.task, result.shared_id, result.deadline_unix_s,
                result.pilot_clock, result.g0_result, result.c2_result) != (
                inputs.wave, inputs.slot, shared.task, shared.identity(), job.clock.final_deadline_unix_s,
                job.clock, inputs.g0, inputs.c2):
            raise InputError("heldout result identity differs from scheduled pilot pair")
        return result, True
    except (OSError, ValueError, RuntimeError) as exc:
        return HeldoutResult(wave=inputs.wave, slot=inputs.slot, task=CELLS[inputs.wave, inputs.slot].task,
            shared_id=shared.identity() if shared else "", g0_result=inputs.g0, c2_result=inputs.c2, status="censored",
            started_unix_s=exited.launched_unix_s or exited.ended_unix_s,
            deadline_unix_s=job.clock.final_deadline_unix_s, pilot_clock=job.clock,
            admission_deadline_unix_s=job.clock.final_deadline_unix_s,
            elapsed_s=max(0, exited.ended_unix_s - (exited.launched_unix_s or exited.ended_unix_s)), transfer_wait_s=0,
            drain_s=max(0, exited.ended_unix_s - job.clock.final_deadline_unix_s) if exited.launched_unix_s else 0,
            error=exited.error or f"{type(exc).__name__}: {exc}"), False
