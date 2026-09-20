"""Verify stored local and model evidence; source declarations alone never establish DEV_GO."""

import json
from pathlib import Path

from pydantic import Field, JsonValue

from kernel_optimizer.evaluation.task_eval import TaskEvaluation
from .device_records import StageIdentity
from .fast_evaluation import proven_local
from .fast_records import CandidateResult
from .runner_records import FrozenRecord, GoalSpec


class ModelReceiptData(FrozenRecord):
    stage: StageIdentity
    evaluation: TaskEvaluation
    forward_calls: int = Field(ge=0)


class ModelReceipt(FrozenRecord):
    kind: str
    binding: dict[str, JsonValue]
    data: ModelReceiptData


def prove_candidate(candidate: CandidateResult, stage: StageIdentity, goal: GoalSpec) -> bool:
    artifact, evaluation = candidate.artifact, candidate.model
    if artifact is None or artifact.author != "agent" or not artifact.session_id or evaluation is None:
        return False
    if (not evaluation.valid or evaluation.score is None or evaluation.score <= 0
            or evaluation.metrics.get("local_checks_passed") != 1
            or not 0 <= evaluation.metrics.get("logits_relative_l2_max", float("inf")) <= .01
            or evaluation.metrics.get("paired_mean_nll_delta_nat_per_token", float("inf")) > .02
            or evaluation.metrics.get("completed_requests") != goal.requests
            or evaluation.metrics.get("output_tokens") != goal.requests * goal.output_tokens):
        return False
    matching = [r for r in candidate.locals if proven_local(r) and r.bundle_sha256 == artifact.bundle_sha256
                and r.params_sha256 == artifact.params_sha256 and r.stage == stage
                and all(f.description.goal == goal.id for f in r.fixtures)]
    if not matching:
        return False
    try:
        receipt = ModelReceipt.model_validate_json(Path(json.loads(evaluation.detail or "{}")["receipt"]).read_bytes())
        return (receipt.kind == "fast-model" and receipt.data.stage == stage and receipt.data.evaluation.valid
            and receipt.data.forward_calls > 0 and receipt.data.evaluation.score == evaluation.score
            and receipt.binding.get("bundle_sha256") == artifact.bundle_sha256
            and receipt.binding.get("params_sha256") == artifact.params_sha256
            and receipt.binding.get("source_hashes") == artifact.source_hashes)
    except (OSError, ValueError, KeyError, TypeError):
        return False
