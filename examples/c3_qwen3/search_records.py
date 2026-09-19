"""Task-local search identities, explicit clocks and shared admission accounting."""

from collections.abc import Callable
from pathlib import Path
from threading import Lock
from time import monotonic, time
from typing import Literal, Self

from pydantic import Field, model_validator

from kernel_optimizer.evaluation.task_eval import TaskEvaluation
from kernel_optimizer.models.core import ParamSet, TrialRecord
from .runner_records import FrozenRecord, GoalId, RunnerError


class SearchClock(FrozenRecord):
    campaign_started_unix_s: float = Field(gt=0)
    search_cutoff_unix_s: float = Field(gt=0)
    final_deadline_unix_s: float = Field(gt=0)

    @model_validator(mode="after")
    def fixed_windows(self) -> Self:
        if (self.search_cutoff_unix_s != self.campaign_started_unix_s + 9000
                or self.final_deadline_unix_s != self.campaign_started_unix_s + 10800):
            raise RunnerError("C3 clock requires S, S+9000 search and S+10800 final")
        return self


class SlotBudget:
    """Mutable admission counter shared by native evaluations, repairs and agent self-tests."""

    def __init__(self, limit: int, deadline_unix_s: float, *, now: Callable[[], float] = time,
                 clock: Callable[[], float] = monotonic) -> None:
        self.limit, self.deadline_unix_s = limit, deadline_unix_s
        self.now, self.clock = now, clock
        self.stop = clock() + max(0, deadline_unix_s - now())
        self.used = 0
        self.lock = Lock()

    def available(self) -> bool:
        return self.used < self.limit and self.clock() < self.stop

    def admit(self) -> bool:
        with self.lock:
            if not self.available():
                return False
            self.used += 1
            return True


class AdmittedSlot:
    """The manual callback consumes this already-charged ticket once, never a second slot."""

    def __init__(self) -> None:
        self.used = False

    def admit(self) -> bool:
        if self.used:
            return False
        self.used = True
        return True


class EvaluationRequest(FrozenRecord):
    bundle: Path
    params: ParamSet
    site_groups: dict[str, tuple[str, ...]] = Field(default_factory=dict)


class Selection(FrozenRecord):
    bundle: Path
    bundle_sha256: str
    source_hashes: dict[str, str]
    params: ParamSet
    params_sha256: str
    site_groups: dict[str, tuple[str, ...]]
    evaluation: TaskEvaluation
    contract_sha256: str
    model_revision: str
    baseline_fallback: bool = False
    goal_id: GoalId | None = None
    clock: SearchClock | None = None


class Opportunity(FrozenRecord):
    number: int
    parent_bundle_sha256: str
    status: Literal["accepted", "retained", "failed", "censored"]
    slots_used: int
    model_calls: int
    repair_calls: int
    trials: list[TrialRecord] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)


class SearchResult(FrozenRecord):
    goal_id: GoalId
    goal: str
    clock: SearchClock
    started_unix_s: float
    deadline_unix_s: float
    finished_unix_s: float
    drain_s: float
    baseline: Selection
    selected: Selection
    opportunities: list[Opportunity]
    protocol_violations: list[str] = Field(default_factory=list)
    private_gpu_audit: Literal["pending"] = "pending"
    slot_counts: dict[str, int] = Field(default_factory=dict)
    slots_reconciled: bool = False


class SearchInputs(FrozenRecord):
    contract: Path
    assets_manifest: Path
    oracle_refs: Path
    profile: Path
    goal_id: GoalId
    goal: str = Field(min_length=1)
    output: Path
    clock: SearchClock
    eval_file: Path = Path(__file__).with_name("manual_task.py")

    @model_validator(mode="after")
    def absolute_paths(self) -> Self:
        if not all(p.is_absolute() for p in (self.contract, self.assets_manifest, self.oracle_refs,
                                             self.profile, self.output, self.eval_file)):
            raise RunnerError("C3 search requires absolute artifact paths")
        return self
