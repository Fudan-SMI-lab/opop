"""Immutable task boundary records shared with the manual evaluator."""

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, JsonValue

from kernel_optimizer.evaluation.task_eval import TaskEvaluation

type GoalId = Literal["ttft", "single", "multi"]
type Phase = Literal["prefill", "decode"]


class RunnerError(RuntimeError):
    def __init__(self, detail: str) -> None:
        self.detail = detail
        super().__init__(detail)


class FrozenRecord(BaseModel):
    model_config = ConfigDict(frozen=True, allow_inf_nan=False)


class GoalSpec(FrozenRecord):
    id: GoalId
    direction: Literal["minimize", "maximize"]
    unit: Literal["ms", "tokens/s"]
    input_tokens: int = Field(validation_alias=AliasChoices("input_tokens", "input_tokens_per_request"))
    output_tokens: int = Field(validation_alias=AliasChoices("output_tokens", "output_tokens_per_request"))
    requests: int = Field(validation_alias=AliasChoices("requests_per_trial_or_final_block", "independent_requests", "requests"))
    search_prompt_ids: tuple[str, ...]
    final_prompt_groups: tuple[tuple[str, ...], ...]


class Prompt(FrozenRecord):
    id: str
    split: Literal["calibration", "search", "heldout"]
    goal: Literal["calibration", "ttft", "single", "multi"]
    input_ids: tuple[int, ...]
    input_tokens: int
    rendered_text: str
    rendered_sha256: str
    token_ids_sha256: str
    padding_tokens: Literal[0]


class Assets(FrozenRecord):
    repo_id: str
    revision: str
    local_path_both_hosts: str
    corpus_file: str
    corpus_sha256: str


class QualityPolicy(FrozenRecord):
    """The numerical acceptance criteria, as the contract declares them.

    These were previously hardcoded in `runner_quality` while the contract declared them separately,
    so the enforced limit and the published limit could disagree without anything noticing. Reading
    them here makes the contract's declaration load-bearing: the bytes are hash-verified in `prepare`,
    so the gate enforces exactly what the approved artifact says.

    `logits_relative_l2_floor_multiple` and the floor evidence exist because an ABSOLUTE logits limit
    is not a property of a candidate alone -- it depends on how many modules the selected site group
    replaces. Measured on the Qwen3 TTFT task (B12), one equivalence-preserving Triton RMSNorm bound to
    a nested prefix of the site group produced:

        1 module 0.0   4 modules 0.0   16 modules 0.004489   36 modules 0.004489   73 modules 0.010349

    The kernel is identical at every point; only the count changes. And B13 measured that perturbing
    the ORIGINAL operator output by exactly the contract's own declared per-operator tolerance
    (rtol=0.02) yields 0.035628 at the logits -- 3.6x the 0.01 logits limit -- so the two declared
    criteria are mutually unsatisfiable on this site group, independently of any kernel.

    A floor therefore has to be MEASURED on the actual site group rather than assumed, in the same way
    `TargetSpec.reference_noise_us` measures timing noise from an A/A pair instead of assuming it. When
    no floor evidence is supplied the behaviour is exactly the declared absolute limit, so nothing
    changes for a task that never measures one.
    """

    local_operator_rtol: float = Field(gt=0)
    local_operator_atol: float = Field(ge=0)
    logits_relative_l2_max_per_prompt: float = Field(gt=0)
    logits_relative_l2_denominator_floor: float = Field(gt=0)
    paired_mean_nll_delta_max_nat_per_token: float = Field(gt=0)
    # How many times the MEASURED floor a candidate may be, when floor evidence exists. 1.0 means "no
    # worse than an equivalence-preserving implementation of this same operator on this same group".
    logits_relative_l2_floor_multiple: float = Field(default=1.0, ge=1.0)


class Contract(FrozenRecord):
    interface_version: Literal["c3-task-local-v1.0.0", "c3-fast-inputs-v1.0.0"]
    assets: Assets
    goals: tuple[GoalSpec, ...]
    runtime_versions: dict[str, str]
    # Optional so both the operator-path and fast-path contracts parse unchanged. When present, its
    # limits are the ones enforced; when absent, `runner_quality` falls back to its declared defaults.
    shared_policy: dict[str, JsonValue] | None = None

    def quality_policy(self) -> QualityPolicy | None:
        """The declared numerical criteria, or None when this contract does not carry them."""
        if self.shared_policy is None:
            return None
        quality = self.shared_policy.get("quality")
        return None if quality is None else QualityPolicy.model_validate(quality)


@dataclass(frozen=True, slots=True)
class AssetSpec:
    local_path: str
    revision: str
    contract_path: Path
    assets_manifest: Path
    file_sizes: Mapping[str, int]


@dataclass(frozen=True, slots=True)
class PreparedTask:
    contract: Contract
    asset_spec: AssetSpec
    prompts_by_id: Mapping[str, Prompt]
    contract_sha256: str
    corpus_sha256: str


@dataclass(frozen=True, slots=True)
class BindingReceipt:
    bundle_sha256: str
    source_hashes: Mapping[str, str]
    params_sha256: str
    site_ids: tuple[str, ...]
    baseline_restored: bool


@dataclass(frozen=True, slots=True)
class QualityReport:
    valid: bool
    metrics: Mapping[str, float]
    raw_path: Path
    detail: str | None = None


@dataclass(frozen=True, slots=True)
class MeasurementReport:
    evaluation: TaskEvaluation
    raw_path: Path


class WaveRecord(FrozenRecord):
    prompt_ids: tuple[str, ...]
    input_tokens: tuple[int, ...]
    tokens: tuple[tuple[int, ...], ...]
    accepted_s: float
    token_ready_s: tuple[float, ...]
    finished_s: float
    cache_lengths: tuple[int, ...]
    warmup: bool
    complete: bool
    detail: str | None = None
    forward_calls: int = 0


class OraclePrompt(FrozenRecord):
    prompt_id: str
    input_ids: tuple[int, ...]
    tokens: tuple[int, ...]
    logits_file: str
    logits_sha256: str
    vocab_size: int = Field(gt=0)
    nll: tuple[float, ...]


class OracleManifest(FrozenRecord):
    contract_sha256: str
    corpus_sha256: str
    baseline: Literal[True]
    prompts: tuple[OraclePrompt, ...]


class RawRecord(FrozenRecord):
    kind: str
    binding: dict[str, JsonValue]
    data: dict[str, JsonValue]
