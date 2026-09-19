"""Persisted opportunity and independent-heldout records, separate from selection scores."""

from pathlib import Path
from typing import Literal

from pydantic import Field

from kernel_optimizer.models.core import Backend, ParameterSpace, ParamSet, TrialRecord
from scripts.experiments.c2_information_inputs import InformationResult
from scripts.experiments.c2_local_inputs import Strict
from scripts.experiments.c2_opportunity_inputs import Arm, PilotClock, Slot, Wave


class Incumbent(Strict):
    selected: Literal["parent", "child"]
    source: Path
    source_sha256: str
    params: ParamSet
    backend: Backend
    space: ParameterSpace
    helper_root: Path
    helpers: tuple[Path, ...]
    helper_sha256: dict[str, str]


class OpportunityResult(Strict):
    wave: Wave
    slot: Slot
    task: str
    rep: int
    arm: Arm
    sampler_seed: int
    shared_id: str
    parent_source_sha256: str
    status: Literal["complete", "failed", "censored"]
    information: InformationResult | None = None
    incumbent: Incumbent | None = None
    responses: Path | None = None
    acquisition_calls: int | None = None
    generation_events: Path | None = None
    proposal_source: Path | None = None
    native_events: Path | None = None
    mechanism_status: Literal["not_run"] = "not_run"
    model_calls_started: int = 0
    model_calls_finished: int = 0
    model_attempts: int = 0
    provider_cost: float | None = None
    started_unix_s: float
    deadline_unix_s: float
    elapsed_s: float
    late_start_s: float = 0
    drain_s: float
    error: str | None = None
    pilot_clock: PilotClock | None = None
    admission_deadline_unix_s: float | None = None
    final_drain_s: float = 0


class HeldoutResult(Strict):
    wave: Wave
    slot: Literal["A0", "B0"]
    task: str
    shared_id: str
    g0_result: Path
    c2_result: Path
    status: Literal["complete", "failed", "censored"]
    parent_finals: list[TrialRecord] = Field(default_factory=list)
    g0_finals: list[TrialRecord] = Field(default_factory=list)
    c2_finals: list[TrialRecord] = Field(default_factory=list)
    started_unix_s: float
    deadline_unix_s: float
    elapsed_s: float
    transfer_wait_s: float
    late_start_s: float = 0
    drain_s: float
    error: str | None = None
    pilot_clock: PilotClock | None = None
    admission_deadline_unix_s: float | None = None
