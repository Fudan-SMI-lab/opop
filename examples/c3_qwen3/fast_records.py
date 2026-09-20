"""Fixed fast-workflow artifacts, stage results and explicit development review decisions."""

from pathlib import Path
from typing import Literal

from pydantic import Field

from kernel_optimizer.agents.model_operator_rewriter import OperatorBrief
from kernel_optimizer.evaluation.task_eval import TaskEvaluation
from kernel_optimizer.models.core import ParamSet
from kernel_optimizer.models.device_operator import DeviceKernelDeclaration
from .device_records import LocalReport
from .runner_records import FrozenRecord, GoalId
from .search_records import SearchClock


class TargetSpec(FrozenRecord):
    goal: GoalId
    brief: OperatorBrief
    profile: Path
    profile_sha256: str
    family_sha256: str
    cuda_kernel_names: tuple[str, ...]
    associated_cuda_us: float = Field(gt=0)
    exclusive_fraction: float | None = Field(default=None, ge=0, lt=1)
    headroom_assumptions: tuple[str, ...] = Field(min_length=1)
    reference_noise_us: float | None = Field(default=None, ge=0)
    noise_evidence: Path | None = None
    noise_evidence_sha256: str | None = None
    representatives: tuple[tuple[str, int], ...] = Field(min_length=1, max_length=2)


class Framework(FrozenRecord):
    revision: str
    implementation_sha256: str
    schema_sha256: str
    prompt_sha256: str
    agent_config_sha256: str
    contract_sha256: str
    profile: Literal["model_operator"] = "model_operator"
    task_kind: Literal["model_project_operator"] = "model_project_operator"


class Artifact(FrozenRecord):
    author: Literal["agent", "control"]
    version: int
    session_id: str | None
    bundle: Path
    bundle_sha256: str
    source_hashes: dict[str, str]
    params: ParamSet
    params_sha256: str
    kernels: tuple[DeviceKernelDeclaration, ...]
    alternative: ParamSet | None = None
    repair_of: str | None = None


class CandidateResult(FrozenRecord):
    version: int
    status: Literal["draft_ready", "source_rejected", "local_failed", "model_failed", "valid", "transport_failed", "censored", "framework_invalid"]
    artifact: Artifact | None = None
    locals: tuple[LocalReport, ...] = ()
    model: TaskEvaluation | None = None
    confirmation: TaskEvaluation | None = None
    error: str | None = None
    failed_sandbox: Path | None = None


class ProbeResult(FrozenRecord):
    phase: Literal["development", "formal_search"]
    epoch: int
    framework: Framework
    target: TargetSpec
    original_baseline_sha256: str
    started_unix_s: float
    finished_unix_s: float
    deadline_unix_s: float
    drain_s: float
    status: Literal["DEV_GO", "PARENT_TRIAGE", "STOP", "accepted", "baseline", "censored", "framework_invalid"]
    candidates: tuple[CandidateResult, ...]
    baseline: TaskEvaluation | None = None
    selected: Artifact | None = None
    budget_path: Path
    clock: SearchClock | None = None


class DevelopmentState(FrozenRecord):
    D: float
    deadline_unix_s: float
    target_key: str
    original_baseline_sha256: str
    budget_path: Path
    results: tuple[Path, ...] = ()
    active_epoch: int | None = None
    pending_framework: str | None = None
    status: Literal["ready", "running", "PARENT_TRIAGE", "DEV_GO", "STOP"] = "ready"


class FrameworkReview(FrozenRecord):
    result: Path
    decision: Literal["framework_defect", "ordinary_failure", "freeze"]
    started_unix_s: float
    ended_unix_s: float
    lines: tuple[str, ...] = Field(min_length=1, max_length=10)
    regression_evidence: Path | None = None
    c2_compatibility_evidence: Path | None = None


class FrozenFramework(FrozenRecord):
    framework: Framework
    development_result: Path
    development_result_sha256: str
    targets: dict[GoalId, TargetSpec]
    compatibility_evidence: Path
    compatibility_sha256: str
