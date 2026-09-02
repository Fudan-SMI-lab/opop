"""Analysis / decision models: tuning stats, bottleneck reports, convergence."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from kernel_optimizer.models.core import (
    FailureKind,
    ParamValue,
    TrialRecord,
)


class ParamStat(BaseModel):
    name: str
    best_value: ParamValue
    at_boundary: bool = False
    boundary_direction: Literal["min", "max"] | None = None
    # Relative latency spread across this param's choices: (worst - best) / best.
    effect_pct: float = 0.0
    # Median latency per choice (stringified choice -> ms), complete trials only.
    latency_by_value: dict[str, float] = Field(default_factory=dict)
    failure_rate_by_value: dict[str, float] = Field(default_factory=dict)


class ResourceSnapshot(BaseModel):
    n_regs: int | None = None
    regs_frac_of_limit: float | None = None
    shared_bytes: int | None = None
    shared_frac_of_limit: float | None = None
    n_spills: int | None = None


class FailureCluster(BaseModel):
    param: str
    value: str
    failure_rate: float
    dominant_kind: FailureKind | None = None


class TuningStats(BaseModel):
    candidate_id: str
    space_id: str
    n_complete: int
    n_fail: int
    best: TrialRecord | None = None
    param_stats: list[ParamStat] = Field(default_factory=list)
    resource_at_best: ResourceSnapshot | None = None
    failure_clusters: list[FailureCluster] = Field(default_factory=list)


class ParamLimit(BaseModel):
    param: str
    headroom_direction: Literal["increase", "decrease"]
    blocked_by: str  # e.g. "registers", "shared_memory", "oom", "compile_failure"
    predicted_gain_pct: float | None = None
    evidence: str = ""


class Hypothesis(BaseModel):
    id: str
    change: str
    expected_effect: str
    risk: str = ""


class BottleneckReport(BaseModel):
    """Agent-produced analysis of one candidate's tuning results. Advisory only."""

    summary: str
    parameter_limits: list[ParamLimit] = Field(default_factory=list)
    hypotheses: list[Hypothesis] = Field(default_factory=list)
    suggested_action: Literal["tune_more", "rewrite", "stop"] = "rewrite"


class ConvergenceDecision(BaseModel):
    scope: Literal["family", "global"]
    verdict: Literal["continue", "freeze"]
    stop_kind: Literal["converged", "budget_exhausted"] | None = None
    evidence: dict = Field(default_factory=dict)


# --- Agent structured-output envelopes -------------------------------------


class GeneratedCandidate(BaseModel):
    file: str  # sandbox-relative path to the candidate .py
    backend: Literal["triton", "cuda"] = "triton"
    approach_summary: str
    structural_axes: list[str] = Field(default_factory=list)


class GenerationResult(BaseModel):
    candidates: list[GeneratedCandidate]


class ProposedParam(BaseModel):
    name: str
    kind: Literal["int", "float", "str"]
    choices: list[int | float | str]
    description: str = ""


class ProposedConstraint(BaseModel):
    expr: str
    rationale: str = ""


class ProposedSpace(BaseModel):
    params: list[ProposedParam]
    constraints: list[ProposedConstraint] = Field(default_factory=list)


class ParameterizationResult(BaseModel):
    file: str  # sandbox-relative path to the rewritten (PARAMS-routed) candidate
    space: ProposedSpace


class RewriteCandidate(BaseModel):
    file: str
    hypothesis_id: str = ""
    change_summary: str


class RewriteResult(BaseModel):
    candidates: list[RewriteCandidate]


class NoveltyCandidate(BaseModel):
    file: str
    backend: Literal["triton", "cuda"] = "triton"
    approach_summary: str
    difference_claim: str


class NoveltyResult(BaseModel):
    candidates: list[NoveltyCandidate]


class RepairResult(BaseModel):
    file: str
    diagnosis: str
    change_summary: str
