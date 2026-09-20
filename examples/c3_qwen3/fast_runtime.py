"""Native composition around existing fast preparation, device and dedicated agent factories."""

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from time import time

from kernel_optimizer.config import load_config
from kernel_optimizer.store.run_store import RunStore
from kernel_optimizer.wiring import Runtime, build_task_rewriter
from .device_records import LocalRuntime, StageIdentity
from .fast_artifacts import framework_id, identify_framework
from .fast_budget import BudgetSpec, BudgetState, Lane, StageBudget
from .fast_evaluation import EvaluationFiles, FastEvaluator
from .fast_generation import DraftGenerator, GenerationContext
from .fast_prepare import FastAccess, AccessPhase, load_fast, prepare_fast
from .fast_records import Framework, TargetSpec
from .fast_workflow import scoped_config
from .local_device import TorchLocalDevice
from .model_binding import load_bundle
from .runner_records import FrozenRecord, RunnerError
from .search_bundle import baseline_bundle
from .search_records import SearchClock


class RunInputs(FrozenRecord):
    contract: Path
    assets_manifest: Path
    agent_config: Path
    framework: Path
    target: Path
    output: Path
    budget: Path
    goal_text: str
    oracle_refs: Path | None = None
    started_unix_s: float | None = None
    deadline_unix_s: float | None = None
    development_state: Path | None = None
    frozen_framework: Path | None = None
    clock: SearchClock | None = None
    device: str = "cuda:0"
    correct_control: Path | None = None
    wrong_control: Path | None = None


def create_oracle(runtime: LocalRuntime, baseline: Path, ids: tuple[str, ...]) -> Path:
    budget = runtime.admission
    if not isinstance(budget, StageBudget) or budget.admit(budget.spec.stage, "capture") is None:
        raise RunnerError("reference oracle preparation admission denied")
    runner = runtime.runner
    before, started = runner.backend.forward_calls, time()
    runner.binding.restore()
    runner.bind(load_bundle(baseline, {}))
    path = runner.create_oracles(ids)
    runner._raw("fast-oracle-preparation", {"stage": budget.spec.stage.model_dump(mode="json"),
        "prompt_ids": list(ids), "oracle": str(path), "forward_calls": runner.backend.forward_calls - before,
        "started_unix_s": started, "ended_unix_s": time()})
    return path


@contextmanager
def open_evaluator(inputs: RunInputs, phase: AccessPhase, lane: Lane) -> Iterator[FastEvaluator]:
    prepared = prepare_fast(inputs.contract, inputs.assets_manifest, FastAccess(profile="c3_fast_device", phase=phase))
    cfg = scoped_config(load_config(inputs.agent_config))
    frame = Framework.model_validate_json(inputs.framework.read_bytes())
    if identify_framework(frame.revision, cfg, prepared) != frame:
        raise RunnerError("tested framework/profile/schema/config changed before admission")
    target = TargetSpec.model_validate_json(inputs.target.read_bytes())
    stage_name = "setup" if lane == "setup" else phase
    stage = StageIdentity(profile="c3_fast_device", stage=stage_name, framework_id=framework_id(frame))
    if inputs.budget.exists():
        saved = BudgetState.model_validate_json(inputs.budget.read_bytes())
        spec = saved.spec
        if (spec.lane != lane or spec.stage != stage or spec.goal not in (None, target.goal)
                or inputs.started_unix_s is not None and inputs.started_unix_s != spec.started_unix_s
                or inputs.deadline_unix_s is not None and inputs.deadline_unix_s != spec.deadline_unix_s):
            raise RunnerError("existing stage clock/identity cannot be replaced")
    else:
        if inputs.started_unix_s is None or inputs.deadline_unix_s is None:
            raise RunnerError("new stage requires explicitly recorded start and deadline")
        spec = BudgetSpec(lane=lane, stage=stage, started_unix_s=inputs.started_unix_s,
                          deadline_unix_s=inputs.deadline_unix_s, goal=target.goal if lane != "setup" else None)
    budget = StageBudget(inputs.budget, spec)
    if not budget.available():
        raise RunnerError("stage expired before resident load; preserve ledger and stop")
    parent = inputs.output.with_name(inputs.output.name + "-baseline")
    baseline = baseline_bundle(parent)
    resident = load_fast(prepared, inputs.device)
    try:
        resident.output_dir = inputs.output.with_name(inputs.output.name + "-preparation")
        runtime = LocalRuntime(resident, TorchLocalDevice(inputs.device), budget)
        oracle = inputs.oracle_refs
        if oracle is None:
            if lane != "setup":
                raise RunnerError("reuse the new-contract setup calibration oracle; do not create free references")
            oracle = create_oracle(runtime, baseline, ("calibration-calibration-00", "calibration-calibration-01"))
        engine = FastEvaluator(runtime, target, EvaluationFiles(baseline, oracle, inputs.output.with_name(inputs.output.name + "-evaluation")))
        def check() -> None:
            if identify_framework(frame.revision, cfg, prepared) != frame:
                raise RunnerError("framework changed during a fixed epoch; stop rather than pool versions")
        engine.check_framework = check
        yield engine
    finally:
        resident.close()


@contextmanager
def open_generator(inputs: RunInputs, engine: FastEvaluator) -> Iterator[DraftGenerator]:
    phase = "development" if engine.stage.stage == "development" else "formal_search"
    prepared = prepare_fast(inputs.contract, inputs.assets_manifest, FastAccess(profile="c3_fast_device", phase=phase))
    frame = Framework.model_validate_json(inputs.framework.read_bytes())
    cfg = scoped_config(load_config(inputs.agent_config))
    store = RunStore.create(inputs.output.parent, inputs.output.name + "-agents", {
        "execution_profile": "model_operator", "task_kind": "model_project_operator",
        "framework": frame.model_dump(mode="json"), "deadline_unix_s": engine.budget.spec.deadline_unix_s})
    cfg.opencode.launch_cwd = store.run_dir
    with Runtime(cfg, store.run_dir) as runtime:
        agent = build_task_rewriter(cfg, store, runtime, execution_profile="model_operator")
        yield DraftGenerator(agent, engine, GenerationContext(prepared, frame, inputs.goal_text))
