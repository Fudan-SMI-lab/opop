"""Analysis / decision models: tuning stats, bottleneck reports, convergence."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from kernel_optimizer.models.core import (
    FailureKind,
    ParamValue,
    TrialRecord,
)


class ParamStat(BaseModel):
    name: str
    # The median table's pick: argmin over per-choice MEDIAN latency. A trend statistic,
    # robust to a single quiet moment on the card.
    best_value: ParamValue
    # The value this knob took in the single fastest trial -- the objective's own winner,
    # since best_ms is min(trials). Differs from best_value on 44.7% of knobs measured, so
    # `at_boundary` / `boundary_direction` are anchored to THIS rather than to the median
    # (see TuningStatsAnalyzer._param_stat). None when no trial completed, or when the
    # fastest trial's value is not among this domain's measured choices.
    best_trial_value: ParamValue | None = None
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


class ResourceExpectation(BaseModel):
    """S2d(a): which way the agent expects one resource dimension to move, and why.

    A DIRECTION, never a rate. Measured three ways that the rate is not derivable in advance: no
    closed form for shared memory (0 of 96 configurations matched exactly), the cost map is not
    separable (0 of 10 one-step deltas agreed), and even the SIGN is unreliable across a wide sweep
    (13 non-monotone slices, BK 16->32 dropping 58 registers while 32->64 added 87 and hit the 255
    cap). So asking for a number would be asking to be lied to.

    Why ask at all, given that an earlier prediction requirement was withdrawn (D-2)? Two differences.
    That one asked for the MAGNITUDE OF A LATENCY GAIN and used it to allocate budget. This asks for
    the SIGN OF A RESOURCE CHANGE -- a far more structural inference ("a bigger tile needs more
    shared memory" does not require knowing how many bytes) which is verifiable AT COMPILE TIME -- and
    it allocates nothing. It is the thing being CHECKED, not the basis of a decision.

    HARD BOUNDARY, enforced in code and asserted by J2d-8: an expectation may never enter candidate
    ranking, family allocation, trial budget, or acceptance. It has exactly two outlets, the ledger
    and the next round's prompt.

    `extra="forbid"` is load-bearing, not tidiness. Pydantic's default silently DROPS an unknown
    field, so an agent that answered `{"dimension": "n_regs", "expect": "down", "expected_pct": 40}`
    would be accepted while the 40 vanished -- and the agent would have reasoned from a magnitude
    nobody ever checked, which is the precise thing this schema exists to prevent. Forbidding makes it
    a validation error, which the retry path returns to the agent with the reason.
    """

    model_config = ConfigDict(extra="forbid")

    # Must name a dimension in the shared vocabulary. Validated in the agent module's `check_output`
    # rather than here, so a wrong name is returned to the agent WITH the vocabulary listed instead
    # of raising a pydantic error whose text does not say what the legal values are.
    dimension: str
    expect: Literal["up", "down", "unchanged", "unknown"]
    why: str = ""


class RewriteCandidate(BaseModel):
    file: str
    # A rewrite may change backend, and until this field existed it could not SAY so: the
    # generator and novelty schemas both carried `backend`, the rewriter did not, so the one
    # module whose whole job is "restructure to unlock a blocked direction" had no way to
    # express the most structural change available. The field is a declaration only -- the
    # harness overrides it from the source via `_detect_backend`, because a label is not
    # evidence and `structural_signature` hashes the backend. Its real purpose is in the
    # PROMPT: a schema field is what makes switching backend a visible option rather than an
    # unmentioned one.
    backend: Literal["triton", "cuda"] = "triton"
    hypothesis_id: str = ""
    change_summary: str
    # S2d(a). Defaults to empty so an agent that declares nothing is not blocked -- but an empty
    # list is itself recorded and counted, because "declared nothing" and "declared and was wrong"
    # are different states and only one of them can be learned from.
    expectations: list[ResourceExpectation] = Field(default_factory=list)


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
