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
    # `valid` means every gate passed AND the candidate beat the baseline on a confirmed repeat.
    # `not_improved` / `not_confirmed` are the two ways a candidate can be fully CORRECT and still not
    # win: it did not beat the baseline, or its first win did not survive the repeat measurement. They
    # are distinct from `model_failed` on purpose -- a reader (and the paper) must be able to tell "the
    # kernel was wrong" from "the kernel was right but not faster", and the lane keeps sampling after
    # either rather than stopping on a correct-but-slower draw.
    status: Literal["draft_ready", "source_rejected", "local_failed", "model_failed", "valid",
                    "not_improved", "not_confirmed", "transport_failed", "censored", "framework_invalid"]
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
    # The equivalence floor this lane calibrated on its own site group before judging anything, or the
    # one it inherited. None means the contract's declared absolute limit was in force throughout.
    # Recorded so a published result states which limit produced its verdicts, rather than leaving that
    # to be reconstructed from the evaluation receipts.
    floor_evidence: Path | None = None


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
    # Optional. A C2 regression receipt is recorded when one is supplied, and its hash is then
    # re-checked at every formal lane; its ABSENCE no longer blocks a freeze, because the presence
    # of a file was never evidence that C2 still behaves the same. C2 isolation is structural
    # instead: every C3 relaxation is reached only through an explicit opt-in switch.
    compatibility_evidence: Path | None = None
    compatibility_sha256: str | None = None
