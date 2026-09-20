"""Explicit STOP/review/resume boundaries and F-star admission; never patches candidates."""

from pathlib import Path
from time import time
from typing import assert_never

from .device_records import StageIdentity
from .fast_artifacts import framework_id
from .fast_budget import BudgetState, StageBudget
from .fast_generation import DraftGenerator
from .fast_prepare import FastPrepared
from .fast_proof import prove_candidate
from .fast_records import DevelopmentState, Framework, FrameworkReview, FrozenFramework, ProbeResult, TargetSpec
from .fast_target import require_target
from .manual_data import fingerprint
from .model_binding import load_bundle
from .runner_records import GoalId, RunnerError
from .search_records import SearchClock


def target_key(target: TargetSpec) -> str:
    return target.goal + ":" + target.brief.site_id


def develop(generator: DraftGenerator, state_path: Path, output: Path) -> ProbeResult:
    from .fast_workflow import run_probe
    engine = generator.evaluator
    budget = engine.budget
    if budget.spec.lane != "development":
        raise RunnerError("development needs its shared D budget")
    parent = load_bundle(engine.files.baseline, {})
    if state_path.exists():
        state = DevelopmentState.model_validate_json(state_path.read_bytes())
    else:
        state = DevelopmentState(D=budget.spec.started_unix_s, deadline_unix_s=budget.spec.deadline_unix_s,
            target_key=target_key(engine.target), original_baseline_sha256=parent.bundle_sha256,
            budget_path=budget.path, pending_framework=engine.stage.framework_id)
    if (state.status != "ready" or state.active_epoch is not None or len(state.results) >= 3
            or state.pending_framework != engine.stage.framework_id or state.target_key != target_key(engine.target)
            or state.original_baseline_sha256 != parent.bundle_sha256 or state.budget_path != budget.path
            or (state.D, state.deadline_unix_s) != (budget.spec.started_unix_s, budget.spec.deadline_unix_s)):
        raise RunnerError("development requires explicit review/resume with unchanged task, baseline and D")
    epoch = len(state.results)
    state = state.model_copy(update={"status": "running", "active_epoch": epoch})
    state_path.write_text(state.model_dump_json(indent=2), encoding="utf-8")
    result = run_probe(generator, output, epoch=epoch)
    status = "DEV_GO" if result.status == "DEV_GO" else "PARENT_TRIAGE"
    if result.status == "censored" or epoch == 2 and status != "DEV_GO":
        status = "STOP"
    state = state.model_copy(update={"results": (*state.results, output.resolve() / "result.json"),
        "status": status, "active_epoch": None, "pending_framework": None})
    state_path.write_text(state.model_dump_json(indent=2), encoding="utf-8")
    return result


def resume_development(state_path: Path, review: FrameworkReview, framework: Framework) -> DevelopmentState:
    state = DevelopmentState.model_validate_json(state_path.read_bytes())
    if state.status != "PARENT_TRIAGE" or not state.results or review.result.resolve() != state.results[-1]:
        raise RunnerError("review must close the latest stopped development epoch")
    if review.ended_unix_s < review.started_unix_s or review.ended_unix_s >= state.deadline_unix_s or time() >= state.deadline_unix_s:
        raise RunnerError("review cannot extend the original development clock")
    match review.decision:
        case "ordinary_failure":
            updated = state.model_copy(update={"status": "STOP"})
        case "framework_defect":
            if (review.regression_evidence is None or review.c2_compatibility_evidence is None
                    or not review.regression_evidence.is_file() or not review.c2_compatibility_evidence.is_file()):
                raise RunnerError("framework revision requires regression and C2 compatibility receipts")
            saved = BudgetState.model_validate_json(state.budget_path.read_bytes())
            ledger = StageBudget(state.budget_path, saved.spec)
            ledger.activate(saved.spec.stage.model_copy(update={"framework_id": framework_id(framework)}))
            updated = state.model_copy(update={"status": "ready", "pending_framework": framework_id(framework)})
        case "freeze":
            raise RunnerError("freeze requires DEV_GO, not a failed epoch")
        case unreachable:
            assert_never(unreachable)
    state_path.with_name(f"review-{len(state.results)-1}.json").write_text(review.model_dump_json(indent=2), encoding="utf-8")
    state_path.write_text(updated.model_dump_json(indent=2), encoding="utf-8")
    return updated


def freeze_framework(result_path: Path, targets: dict[GoalId, TargetSpec], compatibility: Path, *,
                     prepared: FastPrepared) -> FrozenFramework:
    result = ProbeResult.model_validate_json(result_path.read_bytes())
    if result.status != "DEV_GO" or not result.candidates or set(targets) != {"ttft", "single", "multi"}:
        raise RunnerError("formal admission requires actual DEV_GO and three predeclared targets")
    if not compatibility.is_file():
        raise RunnerError("freeze requires a C2 compatibility receipt")
    candidate = result.candidates[-1]
    goal = next(g for g in prepared.task.contract.goals if g.id == result.target.goal)
    stage = StageIdentity(profile="c3_fast_device", stage="development", framework_id=framework_id(result.framework))
    require_target(result.target)
    if not prove_candidate(candidate, stage, goal):
        raise RunnerError("declarations or manual controls cannot freeze F-star")
    for target in targets.values():
        require_target(target, noise=False)
    return FrozenFramework(framework=result.framework, development_result=result_path.resolve(),
        development_result_sha256=fingerprint(result_path), targets=targets,
        compatibility_evidence=compatibility.resolve(), compatibility_sha256=fingerprint(compatibility))


def formal_goal(generator: DraftGenerator, frozen: FrozenFramework, clock: SearchClock, output: Path) -> ProbeResult:
    from .fast_workflow import run_probe
    engine = generator.evaluator
    if (generator.context.framework != frozen.framework or engine.budget.spec.lane != "formal_search"
            or fingerprint(frozen.development_result) != frozen.development_result_sha256
            or fingerprint(frozen.compatibility_evidence) != frozen.compatibility_sha256):
        raise RunnerError("formal epoch framework or freeze evidence changed")
    original = ProbeResult.model_validate_json(frozen.development_result.read_bytes())
    goal = next(g for g in generator.context.prepared.task.contract.goals if g.id == original.target.goal)
    stage = StageIdentity(profile="c3_fast_device", stage="development", framework_id=framework_id(frozen.framework))
    if original.status != "DEV_GO" or not original.candidates or not prove_candidate(original.candidates[-1], stage, goal):
        raise RunnerError("DEV_GO requires actual matching local and full-model proof")
    target = frozen.targets[engine.target.goal]
    if target.model_dump(exclude={"reference_noise_us", "noise_evidence", "noise_evidence_sha256"}) != engine.target.model_dump(
            exclude={"reference_noise_us", "noise_evidence", "noise_evidence_sha256"}):
        raise RunnerError("formal target/profile changed after freeze")
    if engine.budget.spec.deadline_unix_s != min(engine.budget.spec.started_unix_s + 5400, clock.search_cutoff_unix_s):
        raise RunnerError("formal per-goal clock must be clipped by the common search cutoff")
    return run_probe(generator, output, clock=clock)
