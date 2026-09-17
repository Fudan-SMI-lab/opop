"""Backward-compatible controls and records for the existing two-round loop."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from kernel_optimizer.config import AppConfig
from kernel_optimizer.models.core import TrialRecord
from scripts.experiments.c2_local_inputs import Strict

if TYPE_CHECKING:
    from scripts.experiments.c2_method_protocol import Deadline

LoopGroup = Literal["G0", "G2", "H"]


@dataclass(frozen=True, slots=True)
class LoopRun:
    cfg: AppConfig
    group: LoopGroup
    run_dir: Path
    sampler_seeds: tuple[int, int] = (0, 0)
    deadline: Deadline | None = None
    helpers: tuple[Path, ...] = ()
    helper_root: Path | None = None
    require_b40: bool = False


class RoundState(Strict):
    number: int
    elapsed_started_s: float
    elapsed_ready_s: float | None
    information_result: str | None
    responses: str | None
    artifact: str | None
    incumbent: TrialRecord | None
    fresh_finals: list[TrialRecord]
    error: str | None
    status: Literal["valid", "failed", "censored"] = "valid"
    selected: Literal["parent", "child"] = "parent"
    sampler_seed: int = 0
    asked: int | None = None
    acquisition_calls: int | None = None


class LoopResult(Strict):
    task: str
    group: LoopGroup
    initial_parent_artifact: str
    initial_parent_finals: list[TrialRecord]
    rounds: list[RoundState]
    elapsed_total_s: float
    terminal_error: str | None
    terminal_shared: str = "current.json"
    terminal_helper_root: Path | None = None
    terminal_helpers: tuple[Path, ...] = ()
