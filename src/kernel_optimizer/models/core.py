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
