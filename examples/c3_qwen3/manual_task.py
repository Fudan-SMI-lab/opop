"""Provided-eval C3-B entry; model state is supplied by the caller, never loaded here."""

from collections.abc import Mapping
import json
from math import isfinite
from pathlib import Path
from typing import assert_never

from pydantic import JsonValue, ValidationError

from kernel_optimizer.evaluation.task_eval import TaskEvaluation, TaskEvaluationError

from .manual_data import admit_files, verify_frozen
from .manual_types import CallBudget, Context


def evaluate[T](candidate_path: Path, params: Mapping[str, JsonValue], context: Mapping[str, T]) -> TaskEvaluation:
    quality_metrics: dict[str, float] = {}
    details: dict[str, JsonValue] = {"mode": "manual-C3-B"}
    try:
        budget = context.get("call_budget")
        if not isinstance(budget, CallBudget) or not budget.admit():
            raise TaskEvaluationError("missing call budget or admission denied")
        task = Context.model_validate(context)
        admitted = admit_files(task, candidate_path, params)
        runner = task.resident_runner
        receipt = runner.bind(admitted.bundle)
        if (not receipt.baseline_restored or receipt.bundle_sha256 != admitted.bundle.bundle_sha256
                or receipt.params_sha256 != admitted.bundle.params_sha256
                or dict(receipt.source_hashes) != dict(admitted.source_hashes)
                or receipt.site_ids != admitted.site_ids):
            raise TaskEvaluationError("binding receipt identity/restoration mismatch")
        runner.reset(admitted.quality_requests)
        quality = runner.quality(admitted.quality_ids, task.oracle_refs)
        quality_metrics = dict(quality.metrics)
        details.update(quality_raw=str(quality.raw_path), quality_detail=quality.detail)
        if not quality.raw_path.is_absolute() or quality.raw_path.resolve() in admitted.hashes:
            raise TaskEvaluationError("quality evidence must be absolute and separate from frozen inputs")
        if (not quality.valid or not all(isfinite(v) for v in quality_metrics.values())
                or quality_metrics.get("local_checks_passed") != 1.0
                or not 0 <= quality_metrics.get("logits_relative_l2_max", float("inf")) <= .01
                or quality_metrics.get("paired_mean_nll_delta_nat_per_token", float("inf")) > .02):
            raise TaskEvaluationError("independent local/logits/NLL quality gate failed")
        verify_frozen(admitted.hashes)
        runner.reset(admitted.requests)
        measured = runner.measure(admitted.goal, task.prompt_ids)
        details.update(measurement_raw=str(measured.raw_path), measurement_detail=measured.evaluation.detail,
                       goal_id=task.goal_id, direction=admitted.goal.direction, unit=admitted.goal.unit,
                       bundle_sha256=receipt.bundle_sha256, params_sha256=receipt.params_sha256)
        result = measured.evaluation
        metrics = result.metrics
        if (not measured.raw_path.is_absolute() or measured.raw_path.resolve() in admitted.hashes
                or not result.valid or result.score is None
                or not isfinite(result.score) or result.score <= 0
                or metrics.get("completed_requests") != admitted.goal.requests
                or metrics.get("output_tokens") != admitted.goal.requests * admitted.goal.output_tokens
                or not all(isfinite(value) for value in metrics.values())
                or metrics.get("total_wall_ms", 0) <= 0):
            raise TaskEvaluationError("incomplete or invalid native measurement")
        match task.goal_id:
            case "single":
                if metrics.get("ttft_ms", 0) <= 0 or metrics.get("decode_wall_ms", 0) <= 0:
                    raise TaskEvaluationError("single decode requires TTFT and decode wall metrics")
            case "ttft" | "multi":
                pass
            case unreachable:
                assert_never(unreachable)
        verify_frozen(admitted.hashes)
        return TaskEvaluation(score=result.score, metrics={**metrics, **quality_metrics},
                              detail=json.dumps(details, sort_keys=True))
    except (TaskEvaluationError, ValidationError, OSError, KeyError, ValueError, TypeError, RuntimeError) as exc:
        details["failure"] = f"{type(exc).__name__}: {exc}"
        return TaskEvaluation(valid=False, metrics={key: value for key, value in quality_metrics.items()
                                                    if isfinite(value)}, detail=json.dumps(details, sort_keys=True))
