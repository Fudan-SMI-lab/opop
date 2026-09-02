"""Replaceability seam: every module is consumed through these Protocols.

`wiring.py` is the only composition root; `orchestrator` depends on ports, not concretes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from kernel_optimizer.models.core import (
    Baseline,
    Candidate,
    Family,
    ParameterSpace,
    ParamSet,
    ProfileRecord,
    TaskSpec,
    TrialRecord,
)
from kernel_optimizer.models.reports import (
    BottleneckReport,
    ConvergenceDecision,
    GenerationResult,
    NoveltyResult,
    ParameterizationResult,
    RepairResult,
    RewriteResult,
    TuningStats,
)


@runtime_checkable
class TaskAdapterPort(Protocol):
    def load(self, level: int, problem_id: int) -> TaskSpec: ...
    def ref_source(self, task: TaskSpec) -> str: ...


@runtime_checkable
class GpuWorkerPort(Protocol):
    def run_job(self, job: dict[str, Any], timeout_s: float, tag: str,
                lock_mode: str = "exclusive") -> dict[str, Any]: ...


@runtime_checkable
class CorrectnessPort(Protocol):
    def quick_test(self, task: TaskSpec, kernel_src_path: Path, tag: str) -> dict[str, Any]: ...
    def full_eval(self, task: TaskSpec, kernel_src_path: Path, tag: str) -> dict[str, Any]: ...


@runtime_checkable
class BenchmarkPort(Protocol):
    def measure_baseline(self, task: TaskSpec) -> list[Baseline]: ...
    def final_reeval(self, task: TaskSpec, kernel_src_path: Path) -> dict[str, Any]: ...


@runtime_checkable
class ProfilerPort(Protocol):
    def extract(self, worker_result: dict[str, Any]) -> ProfileRecord: ...


@runtime_checkable
class MaterializerPort(Protocol):
    def extract_defaults(self, source: str) -> dict[str, Any]: ...
    def materialize(self, source: str, params: ParamSet) -> str: ...


@runtime_checkable
class TunerPort(Protocol):
    def ask(self) -> tuple[str, ParamSet] | None: ...
    def tell(self, trial_id: str, record: TrialRecord) -> None: ...
    def best(self) -> TrialRecord | None: ...


@runtime_checkable
class StatsPort(Protocol):
    def analyze(self, space: ParameterSpace, trials: list[TrialRecord]) -> TuningStats: ...


@runtime_checkable
class ConvergencePort(Protocol):
    def family_verdict(self, family: Family,
                       agent_suggestion: str | None = None) -> ConvergenceDecision: ...
    def global_verdict(self, families: list[Family],
                       elapsed_hours: float) -> ConvergenceDecision: ...


@runtime_checkable
class FamilyManagerPort(Protocol):
    def register_candidate(self, source: str, origin: str, parent_ids: list[str],
                           backend: str, approach: str) -> Candidate | None: ...
    def accept_novel_seed(self, source: str, backend: str, approach: str,
                          claim: str) -> Candidate | None: ...
    def active_families(self) -> list[Family]: ...


@runtime_checkable
class CandidateGeneratorPort(Protocol):
    def invoke(self, inputs: Any) -> Any: ...  # AgentOutcome[GenerationResult]


@runtime_checkable
class ParameterizerPort(Protocol):
    def invoke(self, inputs: Any) -> Any: ...  # AgentOutcome[ParameterizationResult]


@runtime_checkable
class AnalystPort(Protocol):
    def invoke(self, inputs: Any) -> Any: ...  # AgentOutcome[BottleneckReport]


@runtime_checkable
class RewriterPort(Protocol):
    def invoke(self, inputs: Any) -> Any: ...  # AgentOutcome[RewriteResult]


@runtime_checkable
class NoveltyPort(Protocol):
    def invoke(self, inputs: Any) -> Any: ...  # AgentOutcome[NoveltyResult]


@runtime_checkable
class RepairPort(Protocol):
    def invoke(self, inputs: Any) -> Any: ...  # AgentOutcome[RepairResult]


__all__ = [
    "TaskAdapterPort", "GpuWorkerPort", "CorrectnessPort", "BenchmarkPort",
    "ProfilerPort", "MaterializerPort", "TunerPort", "StatsPort",
    "ConvergencePort", "FamilyManagerPort", "CandidateGeneratorPort",
    "ParameterizerPort", "AnalystPort", "RewriterPort", "NoveltyPort", "RepairPort",
    "GenerationResult", "ParameterizationResult", "BottleneckReport",
    "RewriteResult", "NoveltyResult", "RepairResult",
]
