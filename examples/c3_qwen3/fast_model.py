import json
from pathlib import Path
from time import perf_counter

from pydantic import JsonValue

from kernel_optimizer.evaluation.task_eval import TaskEvaluation
from .device_records import LocalReport, LocalRuntime, StageIdentity
from .fast_prepare import FastAccess, prepare_fast
from .model_binding import load_bundle
from .model_runner import ResidentRunner
from .runner_records import FrozenRecord, GoalId, RunnerError


class FastModelRequest(FrozenRecord):
    stage: StageIdentity
    access: FastAccess
    bundle_path: Path
    params: dict[str, JsonValue]
    goal: GoalId
    prompt_ids: tuple[str, ...]
    oracle_refs: Path
    local_reports: tuple[LocalReport, ...] = ()


def evaluate_fast_model(request: FastModelRequest, runtime: LocalRuntime) -> TaskEvaluation:
    runner, admission = runtime.runner, runtime.admission
    started, before = perf_counter(), runner.backend.forward_calls
    detail: dict[str, JsonValue] = {}
    result = TaskEvaluation(valid=False, detail="not admitted")
    try:
        phase = {"setup": "development", "development": "development", "formal_search": "formal_search", "final": "final"}
        if phase[request.stage.stage] != request.access.phase:
            raise RunnerError("stage and prepared phase disagree")
        if admission.admit(request.stage, "model") is None:
            raise RunnerError("stage budget denied full-model admission")
        spec = runner.prepared.asset_spec
        prepared = prepare_fast(spec.contract_path, spec.assets_manifest, request.access).task
        if (prepared.contract_sha256, prepared.corpus_sha256) != (runner.prepared.contract_sha256, runner.prepared.corpus_sha256):
            raise RunnerError("resident model must be attached to the same explicit phase before model evaluation")
        goal = next(g for g in prepared.contract.goals if g.id == request.goal)
        groups = goal.final_prompt_groups if request.access.phase == "final" else (goal.search_prompt_ids,)
        if request.prompt_ids not in groups:
            raise RunnerError("prompt group is outside the current phase")
        bundle = load_bundle(request.bundle_path, request.params)
        covered = set()
        for local in request.local_reports:
            if (not local.valid or local.bundle_sha256 != bundle.bundle_sha256 or local.params_sha256 != bundle.params_sha256
                    or local.stage.framework_id != request.stage.framework_id):
                raise RunnerError("local prerequisite identity/quality differs from this model request")
            covered.update(f.description.site_id for f in local.fixtures if f.quality_passed)
        if not {s.site_id for s in bundle.document.sites} <= covered:
            raise RunnerError("every active site needs matching local proof before model quality")
        quality_ids = request.prompt_ids if request.access.phase == "final" else (
            "calibration-calibration-00", "calibration-calibration-01")
        with runtime.device.execution():
            runner.bind(bundle)
            quality = runner.quality(quality_ids, request.oracle_refs)
            detail["quality_raw"] = str(quality.raw_path)
            if not quality.valid:
                raise RunnerError(quality.detail or "model quality failed")
            measured = runner.measure(goal, request.prompt_ids)
        detail["measurement_raw"] = str(measured.raw_path)
        result = measured.evaluation.model_copy(update={"metrics": {**measured.evaluation.metrics, **quality.metrics}})
    except (OSError, ValueError, RuntimeError, KeyError, StopIteration) as exc:
        result = TaskEvaluation(valid=False, detail=f"{type(exc).__name__}: {exc}")
    finally:
        runner.binding.restore()
        runner.backend.reset()
    path = runner._raw("fast-model", {"stage": request.stage.model_dump(mode="json"),
        "forward_calls": runner.backend.forward_calls - before, "wall_ms": (perf_counter() - started) * 1000,
        "evaluation": result.model_dump(mode="json"), **detail})
    return result.model_copy(update={"detail": json.dumps({"receipt": str(path), "detail": result.detail})})
