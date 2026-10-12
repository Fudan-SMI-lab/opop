"""Verify stored local and model evidence; source declarations alone never establish DEV_GO."""

import json
import math
from pathlib import Path

from pydantic import Field, JsonValue

from kernel_optimizer.evaluation.task_eval import TaskEvaluation
from .device_records import StageIdentity
from .fast_evaluation import proven_local
from .fast_records import CandidateResult
from .runner_quality import FLOOR_METRIC
from .runner_records import FrozenRecord, GoalSpec


class ModelReceiptData(FrozenRecord):
    stage: StageIdentity
    evaluation: TaskEvaluation
    forward_calls: int = Field(ge=0)
    # The quality receipt this model evaluation judged against. `evaluate_fast_model` has always
    # written it; it is declared here because `prove_candidate` now follows it to read the limit that
    # was ENFORCED rather than restating a literal. Required, not optional: a model receipt with no
    # quality receipt behind it cannot be proven, and defaulting it to None would turn that into a
    # silent pass at whatever limit happened to be hardcoded.
    quality_raw: str


class ModelReceipt(FrozenRecord):
    kind: str
    binding: dict[str, JsonValue]
    data: ModelReceiptData

    @property
    def quality_raw(self) -> str:
        return self.data.quality_raw


def logits_limit_enforced(quality_raw: Path) -> float:
    """The logits limit the quality gate ACTUALLY enforced, read from its own raw receipt.

    WHY THIS IS READ RATHER THAN HARDCODED. This check used to compare against a literal `.01`, the
    contract's declared limit. That was correct only while the limit was constant. It is not: a
    measured equivalence floor RAISES it (`runner_quality.criteria`), so a candidate the quality gate
    admitted at 0.011807 under an enforced limit of 0.016163 was then re-rejected here against 0.01 --
    two gates in one pipeline disagreeing about the same candidate. Measured in dev epoch
    F5-dev-0921-185739: `evaluate_quality` returned `valid: true, score 270.725` for exactly that
    candidate, and the epoch kept sampling because this function said no. The floor mechanism was
    inert at the only decision that consumes it.

    The limit is taken from the quality receipt's OWN `criteria` block, which `runner_quality` writes
    beside the metrics it judged, so it cannot drift from what was enforced. It cannot be relaxed by a
    candidate either: the receipt is written by the framework after the measurement, the candidate
    never supplies it, and a receipt whose criteria block is missing or malformed raises rather than
    defaulting to something permissive.

    This does NOT weaken the gate. With no floor evidence `criteria()` returns the declared limits
    unchanged, so a task that measures no floor is judged exactly as before.
    """
    document = json.loads(quality_raw.read_text(encoding="utf-8"))
    # Raw receipts are written by `ResidentRunner._raw` as {kind, binding, data}; the criteria block
    # lives under `data`. Accept a bare document too so this reads a receipt written either way.
    body = document.get("data") if isinstance(document.get("data"), dict) else document
    criteria = body.get("criteria")
    if not isinstance(criteria, list):
        raise ValueError("quality receipt has no criteria block")
    for row in criteria:
        if isinstance(row, dict) and row.get("metric") == FLOOR_METRIC:
            limit = row.get("limit")
            if isinstance(limit, bool) or not isinstance(limit, (int, float)):
                raise ValueError("quality receipt declares no numeric logits limit")
            if not math.isfinite(limit) or limit < 0:
                raise ValueError("quality receipt declares no finite non-negative logits limit")
            return float(limit)
    raise ValueError(f"quality receipt does not enforce {FLOOR_METRIC}")


def prove_candidate(candidate: CandidateResult, stage: StageIdentity, goal: GoalSpec) -> bool:
    artifact, evaluation = candidate.artifact, candidate.model
    if artifact is None or artifact.author != "agent" or not artifact.session_id or evaluation is None:
        return False
    # The model receipt is parsed FIRST, because it carries the pointer to the quality receipt whose
    # criteria decide the limit. `evaluate_fast_model` rebuilds the outer `detail` as
    # {receipt, detail} only, so `quality_raw` is reachable only from inside the receipt -- reading it
    # from the outer detail silently rejected every candidate.
    try:
        receipt = ModelReceipt.model_validate_json(
            Path(json.loads(evaluation.detail or "{}")["receipt"]).read_bytes())
        limit = logits_limit_enforced(Path(receipt.quality_raw))
    except (OSError, ValueError, KeyError, TypeError):
        return False
    if (not evaluation.valid or evaluation.score is None or evaluation.score <= 0
            or evaluation.metrics.get("local_checks_passed") != 1
            or not 0 <= evaluation.metrics.get(FLOOR_METRIC, float("inf")) <= limit
            or evaluation.metrics.get("paired_mean_nll_delta_nat_per_token", float("inf")) > .02
            or evaluation.metrics.get("completed_requests") != goal.requests
            or evaluation.metrics.get("output_tokens") != goal.requests * goal.output_tokens):
        return False
    matching = [r for r in candidate.locals if proven_local(r) and r.bundle_sha256 == artifact.bundle_sha256
                and r.params_sha256 == artifact.params_sha256 and r.stage == stage
                and all(f.description.goal == goal.id for f in r.fixtures)]
    if not matching:
        return False
    # Same receipt object parsed above; the identity chain is checked here, after the cheap metric
    # clauses, so the ordering of the original function is preserved.
    return (receipt.kind == "fast-model" and receipt.data.stage == stage and receipt.data.evaluation.valid
        and receipt.data.forward_calls > 0 and receipt.data.evaluation.score == evaluation.score
        and receipt.binding.get("bundle_sha256") == artifact.bundle_sha256
        and receipt.binding.get("params_sha256") == artifact.params_sha256
        and receipt.binding.get("source_hashes") == artifact.source_hashes)
