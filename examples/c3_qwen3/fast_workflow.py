"""Fixed per-epoch generation/local/model sequence; framework revision decisions stay external."""

from pathlib import Path
from time import time

from kernel_optimizer.config import AppConfig
from kernel_optimizer.tuning.objective import Objective
from .fast_artifacts import framework_id, write_artifact
from .fast_generation import DraftGenerator
from .fast_proof import prove_candidate
from .fast_records import CandidateResult, ProbeResult
from .fast_search import tune_local
from .fast_setup import measure_reference
from .fast_target import require_target
from .model_binding import load_bundle
from .runner_records import RunnerError
from .search_records import SearchClock


def scoped_config(cfg: AppConfig) -> AppConfig:
    scoped = cfg.model_copy(deep=True)
    scoped.opencode.request_timeout_s = 480
    scoped.agents.rewriter.max_retries = 0
    scoped.agents.rewriter.max_transport_retries = 0
    return scoped


def run_probe(generator: DraftGenerator, output: Path, *, epoch: int = 0,
              clock: SearchClock | None = None) -> ProbeResult:
    engine = generator.evaluator
    prepared, frame = generator.context.prepared, generator.context.framework
    require_target(engine.target, noise=False)
    if prepared.access.phase not in ("development", "formal_search"):
        raise RunnerError("candidate workflow cannot receive sealed final data")
    if engine.stage.framework_id != framework_id(frame):
        raise RunnerError("generation and admission framework identities differ")
    output.mkdir(parents=True, exist_ok=False)
    started = engine.budget.now()
    candidates: list[CandidateResult] = []
    baseline = None
    selected = None
    phase = prepared.access.phase
    status = "PARENT_TRIAGE" if phase == "development" else "baseline"
    goal = next(g for g in prepared.task.contract.goals if g.id == engine.target.goal)
    objective = Objective(direction=goal.direction, unit=goal.unit)
    try:
        if not engine.fixtures:
            engine.capture()
        engine.target = measure_reference(engine)
        require_target(engine.target)
        previous = None
        for version in (0, 1):
            if not engine.budget.available() or engine.runtime.runner.closed:
                status = "censored"
                break
            candidate = generator.invoke(version, previous)
            if candidate.status in ("transport_failed", "censored"):
                candidates.append(candidate)
                break
            if candidate.artifact is None:
                candidates.append(candidate)
                previous = candidate
                continue
            artifact, reports = tune_local(engine, candidate.artifact)
            if artifact is None:
                candidate = candidate.model_copy(update={"status": "local_failed", "locals": reports,
                    "error": reports[-1].detail if reports else "no valid local point"})
                candidates.append(candidate)
                previous = candidate
                continue
            if baseline is None:
                baseline = engine.model(None)
            if not baseline.valid:
                candidates.append(candidate.model_copy(update={"status": "model_failed", "artifact": artifact,
                    "locals": reports, "error": baseline.detail}))
                break
            measured = engine.model(artifact)
            candidate = candidate.model_copy(update={"artifact": artifact, "model": measured,
                "locals": tuple(engine.reports), "status": "valid" if measured.valid else "model_failed", "error": measured.detail})
            if not prove_candidate(candidate, engine.stage, goal):
                candidates.append(candidate)
                previous = candidate
                continue
            if phase == "development":
                status = "DEV_GO"
                candidates.append(candidate)
                break
            if baseline.score is not None and measured.score is not None and objective.gain(baseline.score, measured.score) > 0:
                confirmation = engine.model(artifact)
                confirmed = candidate.model_copy(update={"model": confirmation, "locals": tuple(engine.reports)})
                candidate = candidate.model_copy(update={"confirmation": confirmation, "locals": tuple(engine.reports)})
                if prove_candidate(confirmed, engine.stage, goal) and confirmation.score is not None and objective.gain(baseline.score, confirmation.score) > 0:
                    selected = write_artifact(artifact, output / "selected")
                    status = "accepted"
            candidates.append(candidate)
            break
    except (OSError, ValueError, RuntimeError) as exc:
        status = "framework_invalid"
        candidates.append(CandidateResult(version=len(candidates), status="framework_invalid", error=f"{type(exc).__name__}: {exc}"))
    if not engine.budget.available() and status not in ("DEV_GO", "accepted"):
        status = "censored"
    result = ProbeResult(phase=phase, epoch=epoch, framework=frame, target=engine.target,
        original_baseline_sha256=load_bundle(engine.files.baseline, {}).bundle_sha256,
        started_unix_s=started, finished_unix_s=engine.budget.now(), deadline_unix_s=engine.budget.spec.deadline_unix_s,
        drain_s=max(0, engine.budget.now() - engine.budget.spec.deadline_unix_s), status=status, candidates=tuple(candidates),
        baseline=baseline, selected=selected, budget_path=engine.budget.path, clock=clock)
    (output / "result.json").write_text(result.model_dump_json(indent=2), encoding="utf-8")
    return result
