"""Bounded local configuration screening using the existing Optuna tuner, never model J."""

from statistics import median

from kernel_optimizer.evaluation.task_eval import TaskEvaluation
from kernel_optimizer.models.core import DeviceLimits, ParamSet, TrialRecord
from kernel_optimizer.paramspace.guard import check_config
from kernel_optimizer.tuning.objective import Objective
from kernel_optimizer.tuning.tpe import OptunaTPETuner
from .device_records import LocalReport
from .fast_evaluation import FastEvaluator, proven_local
from .fast_records import Artifact
from .model_binding import load_bundle
from .search_bundle import parameter_space


def local_gain(report: LocalReport, noise_us: float) -> bool:
    return proven_local(report) and all(median(f.reference_us) - median(f.candidate_us) > noise_us for f in report.fixtures)


def tune_local(engine: FastEvaluator, artifact: Artifact) -> tuple[Artifact | None, tuple[LocalReport, ...]]:
    bundle = load_bundle(artifact.bundle, artifact.params.values)
    space = parameter_space(bundle)
    reports = [r for r in engine.reports if r.bundle_sha256 == bundle.bundle_sha256 and r.params_sha256 == bundle.params_sha256]
    default = reports[-1] if reports else engine.local(engine.request(artifact))
    results = [default]
    if not proven_local(default):
        return None, tuple(results)
    best, best_report = artifact, default
    points = {artifact.params.key()}
    alternative = artifact.alternative
    if alternative is not None and check_config(space, alternative, DeviceLimits()) is None:
        if engine.budget.remaining("local") > 2 and engine.budget.available():
            report = engine.local(engine.request(artifact, alternative))
            results.append(report)
            points.add(alternative.key())
            if proven_local(report) and report.latency_us < best_report.latency_us:
                best, best_report = artifact.model_copy(update={"params": alternative, "params_sha256": report.params_sha256}), report
    noise = engine.target.reference_noise_us
    if noise is None or not any(local_gain(r, noise) for r in results) or not space.domains:
        return best, tuple(results)
    tuner = OptunaTPETuner(space, lambda p: check_config(space, p, DeviceLimits()) is None,
        budget=4, anchors=tuple(ParamSet.model_validate({"values": r}) for r in (
            artifact.params.values, alternative.values if alternative is not None and alternative.key() in points else artifact.params.values)),
        n_startup_trials=2, objective=Objective(direction="minimize", unit="us", label="local fixture latency"))
    measured = {artifact.params.key(): (artifact.params, default)}
    if alternative is not None and len(results) > 1:
        measured[alternative.key()] = (alternative, results[1])
    while engine.budget.available() and engine.budget.remaining("local") > 2:
        asked = tuner.ask()
        if asked is None:
            break
        trial_id, params = asked
        if params.key() in measured:
            report = measured[params.key()][1]
        else:
            if len(points) >= 4:
                break
            report = engine.local(engine.request(artifact, params))
            results.append(report)
            points.add(params.key())
        evaluation = TaskEvaluation(
            score=report.latency_us if report.valid else None, valid=report.valid, detail=report.detail)
        tuner.tell(trial_id, TrialRecord(trial_id=trial_id, candidate_id=bundle.bundle_sha256, space_id=space.space_id,
            params=params, status="complete" if report.valid else "fail", task_evaluation=evaluation))
        if proven_local(report) and report.latency_us < best_report.latency_us:
            best, best_report = artifact.model_copy(update={"params": params, "params_sha256": report.params_sha256}), report
    return best, tuple(results)
