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


class Contract(FrozenRecord):
    interface_version: Literal["c3-task-local-v1.0.0", "c3-fast-inputs-v1.0.0"]
    assets: Assets
    goals: tuple[GoalSpec, ...]
    runtime_versions: dict[str, str]


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
