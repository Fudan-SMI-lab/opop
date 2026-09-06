"""Core domain models.

Paper terminology mapping: structure S = Candidate, structural route = Family,
parameter space Theta(S) = ParameterSpace, tuning records D_S = list[TrialRecord].
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

ParamValue = int | float | str

Origin = Literal["seed", "repair", "rewrite", "novelty"]
Backend = Literal["triton", "cuda"]
FailureKind = Literal[
    "compile_error",
    "runtime_error",
    "correctness_mismatch",
    "oom",
    "timeout",
    "worker_crash",
    "static_check_failed",
    "guard_rejected",
    "materialize_error",
    # >=10x faster than the reference: treated as not doing the reference's work
    # (anti-reward-hacking), not as a legitimate result.
    "excessive_speedup",
]


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class TaskSpec(BaseModel):
    model_config = ConfigDict(frozen=True)

    level: int
    problem_id: int
    name: str
    ref_path: Path
    ref_src_sha: str


class Candidate(BaseModel):
    candidate_id: str
    family_id: str
    parent_ids: list[str] = Field(default_factory=list)
    origin: Origin
    backend: Backend
    source_sha: str
    structural_signature: str
    approach_summary: str = ""
    status: Literal["registered", "parameterized", "tuned", "dropped"] = "registered"


class ParamDomain(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    kind: Literal["int", "float", "str"]
    # Ordered cheapest -> most expensive (resource-wise); used for the "minimal" witness.
    choices: list[ParamValue]
    description: str = ""


class Constraint(BaseModel):
    model_config = ConfigDict(frozen=True)

    expr: str
    rationale: str = ""


class ParameterSpace(BaseModel):
    space_id: str
    candidate_id: str
    source_sha: str
    version: int = 1
    domains: list[ParamDomain]
    constraints: list[Constraint] = Field(default_factory=list)

    def param_names(self) -> list[str]:
        return [d.name for d in self.domains]

    def domain(self, name: str) -> ParamDomain:
        for d in self.domains:
            if d.name == name:
                return d
        raise KeyError(name)


class ParamSet(BaseModel):
    model_config = ConfigDict(frozen=True)

    values: dict[str, ParamValue]

    def key(self) -> str:
        return sha256_text(repr(sorted(self.values.items())))[:16]


class LatencyStats(BaseModel):
    model_config = ConfigDict(frozen=True)

    mean: float
    std: float
    min: float
    max: float
    n_samples: int
    # The robust location estimate, and the one the tuner optimizes. Optional because runs
    # recorded before this field existed replay without it, and because two timing paths
    # (the baseline helper and KernelBench's own runtime_stats) return summary statistics
    # with no samples to compute it from. `robust_ms` is what callers should read: it falls
    # back to the mean so an older record or a summary-only path still yields a number.
    median: float | None = None
    # The raw per-sample timings, when the timing path had them. Retained so an estimator
    # choice can be re-examined from the log instead of by re-running the GPU: verifying that
    # a median beats the mean as an objective required a fresh 2000-sample probe precisely
    # because the log held only mean/std/min/max. Optional for the same reasons as `median`,
    # and never used for a decision -- decisions read `robust_ms`; this is the evidence.
    samples: tuple[float, ...] | None = None

    @property
    def robust_ms(self) -> float:
        """Median when available, else the mean.

        Prefer this over `.mean` for any comparison or objective. At the 20 samples a tuning
        trial uses, the mean's coefficient of variation is 24-37% on level2:37 while the
        median's is 3-8%, and on configurations whose true costs differ by 7.6% the mean
        picks the faster one only 64.8% of the time against the median's 93.2%
        (scripts/probe_robust_objective.py). Deliberately NOT `min`: at n=20 min is biased
        +9.8% to +156% and ranks pairs backwards more often than not, because it reports the
        luckiest sample rather than the cost.
        """
        return self.median if self.median is not None and self.median > 0 else self.mean


def latency_cell(value: float | None, places: int = 4) -> str | float:
    """Format a latency for a CSV cell: rounded, or empty when there is no value.

    Lives here, beside `LatencyStats`, because both trials.csv writers need it (the
    orchestrator's per-candidate file that the analyst agent reads, and the report's
    deliverable one) and neither of those modules imports the other.

    Why rounding matters rather than being cosmetic: every field the worker summarizes is
    already rounded to 4 decimals, but `median` is computed from the raw samples and so
    arrives at full float precision. Emitted as-is it printed 17 significant digits beside
    a 4-digit mean in the same row -- presenting a 20-sample estimate whose coefficient of
    variation is 3-8% as though it were exact, and inviting a comparison of two configs on
    digits orders of magnitude below the noise floor.
    """
    if value is None or value == "":
        return ""
    try:
        return round(float(value), places)
    except (TypeError, ValueError):
        return ""


class ProfileRecord(BaseModel):
    model_config = ConfigDict(frozen=True)

    n_regs: int | None = None
    n_spills: int | None = None
    shared_bytes: int | None = None
    num_warps: int | None = None
    num_stages: int | None = None
    compile_s: float | None = None
    kernel_names: list[str] = Field(default_factory=list)


class TrialRecord(BaseModel):
    trial_id: str
    candidate_id: str
    space_id: str
    params: ParamSet
    status: Literal["complete", "fail"]
    failure_kind: FailureKind | None = None
    failure_detail: str = ""
    latency_ms: LatencyStats | None = None
    profile: ProfileRecord | None = None
    # How many of this trial's correctness trials were accepted ONLY by the fp64
    # relative arm (0 when the gate is on and did not change the verdict, None when the
    # gate is off). Journalled so the gate's effect is measurable from the event log --
    # the fp64 metrics otherwise appear only on FAILURES, i.e. never on the cases the
    # gate was added to admit.
    fp64_rescued_trials: int | None = None


class BestRecord(BaseModel):
    candidate_id: str
    params: ParamSet
    latency_ms: float


class Family(BaseModel):
    family_id: str
    anchor_candidate_id: str
    member_ids: list[str] = Field(default_factory=list)
    best: BestRecord | None = None
    # Best latency after each completed tuning round (per rewrite round), for convergence.
    best_history: list[float] = Field(default_factory=list)
    rewrite_rounds_used: int = 0
    status: Literal["active", "frozen_converged", "frozen_budget"] = "active"


class Baseline(BaseModel):
    # eager / torch_compile, optionally suffixed with a matmul-precision tag
    # (e.g. eager_tf32) when the dual-witness mode records both precisions.
    kind: str
    latency_ms: LatencyStats
    note: str = ""


class DeviceLimits(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str = "unknown"
    vram_gb: float = 16.0
    max_regs_per_thread: int = 255
    max_shared_bytes_static: int = 49152
    max_shared_bytes_optin: int = 101376
    max_threads_per_block: int = 1024

    def as_env(self) -> dict[str, ParamValue]:
        """Constants available to constraint expressions."""
        return {
            "MAX_REGS_PER_THREAD": self.max_regs_per_thread,
            "MAX_SHARED_BYTES": self.max_shared_bytes_static,
            "MAX_SHARED_BYTES_OPTIN": self.max_shared_bytes_optin,
            "MAX_THREADS_PER_BLOCK": self.max_threads_per_block,
        }
