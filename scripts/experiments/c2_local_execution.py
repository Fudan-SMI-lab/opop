"""One structural opportunity, one native B40 search, then measurement without reranking."""

from contextlib import chdir
from pathlib import Path
from time import monotonic
from typing import Literal, assert_never

from pydantic import Field

from kernel_optimizer.agents.runtime import AgentCallError
from kernel_optimizer.config import AppConfig
from kernel_optimizer.control.direct_task import TaskSearch
from kernel_optimizer.control.families import structural_signature
from kernel_optimizer.control.task_rewrite import probe_responses
from kernel_optimizer.evaluation.task_eval import TaskEvaluator
from kernel_optimizer.models.core import ParameterSpace, TrialRecord, sha256_text
from kernel_optimizer.paramspace.guard import check_config
from kernel_optimizer.paramspace.materializer import MaterializeError, materialize
from kernel_optimizer.tuning.objective import rank_record
from kernel_optimizer.tuning.stats import TuningStatsAnalyzer
from kernel_optimizer.tuning.tpe import OptunaTPETuner
from kernel_optimizer.store.run_store import RunStore
from scripts.experiments import c2_local_adapter
from scripts.experiments.c2_local_adapter import GpuAdapter
from scripts.experiments.c2_local_agents import Child, Services, direct_child, legacy_child
from scripts.experiments.c2_local_costs import Costs, costs
from scripts.experiments.c2_local_inputs import InputError, Responses, RunOptions, Shared, Strict, protocol, stage_inputs


class ArmResult(Strict):
    shared_id: str
    arm: Literal["A", "B"]
    path: Literal["direct", "legacy"]
    state: str
    selected: Literal["parent", "child"]
    selected_trial: TrialRecord
    child_best: TrialRecord | None = None
    trials: list[TrialRecord] = Field(default_factory=list)
    fresh_parent: list[TrialRecord] = Field(default_factory=list)
    fresh_child: list[TrialRecord] = Field(default_factory=list)
    error: str | None = None
    wall_s: float
    costs: Costs
    acquisition_costs: Costs | None = None
    treatment_status: str = "withheld"


def save_result(result: ArmResult, store: RunStore) -> ArmResult:
    (store.run_dir / "result.json").write_text(result.model_dump_json(indent=2), encoding="utf-8")
    store.append("RUN_FINISHED", {"selected": result.selected, "error": result.error, "wall_s": result.wall_s})
    return result


def tune_legacy(child: Child, adapter: GpuAdapter, services: Services) -> list[TrialRecord]:
    source = child.path.read_text(encoding="utf-8")
    space = ParameterSpace(space_id="child-space", candidate_id="child", source_sha=sha256_text(source),
                           domains=child.space.params, constraints=child.space.constraints)
    services.store.append("SPACE_PUBLISHED", {"space": space.model_dump(mode="json"), "budget": 40})
    tuner = OptunaTPETuner(space, lambda p: check_config(space, p, services.cfg.device) is None,
                          budget=40, seed=services.cfg.run.seed)
    records: list[TrialRecord] = []
    while (asked := tuner.ask()) is not None:
        trial_id, params = asked
        measured = adapter.measure(child.path, params)
        record = measured.model_copy(update={"trial_id": trial_id, "candidate_id": "child", "space_id": space.space_id})
        tuner.tell(trial_id, record)
        records.append(record)
        services.store.append("TRIAL_DONE", {"trial": record.model_dump(mode="json")})
    services.store.append("TUNING_DONE", {
        "stats": TuningStatsAnalyzer(services.cfg.device).analyze(space, records).model_dump(mode="json"),
        "draws": [{"number": t.number, "state": t.state.name, "params": t.params} for t in tuner.study.trials],
    })
    return records


def run_arm(shared: Shared, options: RunOptions, services: Services) -> ArmResult:
    started = monotonic()
    if shared.protocol_id != protocol(services.cfg):
        raise InputError("run model/retries/seed/source/evaluation differs from prepared protocol")
    if options.acquisition is not None:
        options.acquisition.validate_for(shared)
    directory = services.store.run_dir / "common"
    responses = options.acquisition.responses if options.arm == "B" and options.acquisition else []
    acquisition_costs = options.acquisition.costs if options.acquisition else None
    treatment_status = ("observed" if any(r.delta_j is not None for r in responses) else "no_valid_contrast") if options.arm == "B" else "withheld"
    inputs = stage_inputs(shared, directory, responses)
    adapter = GpuAdapter(shared, services.cfg, services.store)
    parent_path = directory / "parent.py"
    adapter.full, adapter.phase = True, "parent_smoke"
    smoke = adapter.measure(parent_path, shared.parent.params)
    if smoke.status != "complete":
        return save_result(ArmResult(
            shared_id=shared.identity(), arm=options.arm, path=options.path, state=shared.state,
            selected="parent", selected_trial=shared.parent, wall_s=monotonic() - started,
            error=f"parent_smoke_failed: {smoke.failure_detail}", costs=costs(services.store),
            acquisition_costs=acquisition_costs, treatment_status=treatment_status,
        ), services.store)
    records: list[TrialRecord] = []
    child: Child | None = None
    error = None
    search = None
    tuning_start = len(adapter.records)
    adapter.full, adapter.phase = False, "child_tuning"
    try:
        with chdir(directory):
            match options.path:
                case "direct":
                    child = direct_child(inputs, services)
                case "legacy":
                    child = legacy_child(shared, services, inputs)
                case unreachable:
                    assert_never(unreachable)
        if structural_signature(child.path.read_text(encoding="utf-8")) == structural_signature(shared.source):
            raise InputError("returned child has no structural change")
        adapter.backend = child.backend
        match options.path:
            case "direct":
                with TaskEvaluator(Path(c2_local_adapter.__file__), "evaluate") as evaluator:
                    budgets = services.cfg.budgets.model_copy(update={"trials_per_space": 40})
                    search = TaskSearch(evaluator, inputs.objective, budgets)
                    services.store.append("SPACE_PUBLISHED", {"space": child.space.model_dump(mode="json"), "budget": 40})
                    _ = search.evaluate_candidate(child.path, child.space, {"adapter": adapter}, seed=services.cfg.run.seed)
                    records = list(search.trials)
                    services.store.append("TUNING_DONE", {"stats": [s.model_dump(mode="json") for s in search.stats]})
            case "legacy":
                records = tune_legacy(child, adapter, services)
            case unreachable:
                assert_never(unreachable)
    except (AgentCallError, InputError, MaterializeError, OSError, ValueError, RuntimeError) as exc:
        error = f"{type(exc).__name__}: {exc}"
        services.store.append("LOCAL_CHILD_FAILED", {"error": error})
    if search is not None:
        records = list(search.trials)
    attempts = adapter.records[tuning_start:]
    if len(records) < len(attempts):
        records.extend(attempts[len(records):])
    reported = {e.payload["trial"]["trial_id"] for e in services.store.iter_events() if e.type == "TRIAL_DONE"}
    for record in records:
        if record.trial_id not in reported:
            services.store.append("TRIAL_DONE", {"trial": record.model_dump(mode="json")})
    objective = inputs.objective if options.path == "direct" else None
    valid = [t for t in records if t.status == "complete"]
    best = min(valid, key=lambda t: rank_record(t, objective)) if valid else None
    if best is None and error is None:
        error = "child_space_has_no_valid_trial"
    parent_score = shared.parent.latency_ms.robust_ms if shared.parent.latency_ms else float("inf")
    selected: Literal["parent", "child"] = "child" if best and rank_record(best, objective) < parent_score else "parent"
    selected_trial = best if selected == "child" and best else shared.parent
    services.store.append("SELECTION_FROZEN", {"selected": selected, "trial": selected_trial.model_dump(mode="json")})
    frozen_child = None
    if child is not None and best is not None:
        frozen_child = child.path.parent / "frozen-child.py"
        try:
            frozen_child.write_text(materialize(child.path.read_text(encoding="utf-8"), best.params), encoding="utf-8")
        except (MaterializeError, OSError) as exc:
            error = f"selected_child_materialization_failed: {exc}"
            frozen_child = None
    fresh_parent: list[TrialRecord] = []
    fresh_child: list[TrialRecord] = []
    adapter.full = True
    for block in range(options.final_blocks):
        order = ("parent", "child") if block % 2 == 0 else ("child", "parent")
        for which in order:
            if which == "parent":
                adapter.backend, adapter.phase = shared.backend, f"final_parent_{block}"
                fresh_parent.append(adapter.measure(parent_path, shared.parent.params))
            elif frozen_child is not None and best is not None and child is not None:
                adapter.backend, adapter.phase = child.backend, f"final_child_{block}"
                fresh_child.append(adapter.measure(frozen_child, best.params))
    result = ArmResult(shared_id=shared.identity(), arm=options.arm, path=options.path, state=shared.state,
                       selected=selected, selected_trial=selected_trial, child_best=best, trials=records,
                       fresh_parent=fresh_parent, fresh_child=fresh_child, error=error, wall_s=monotonic() - started,
                       costs=costs(services.store), acquisition_costs=acquisition_costs, treatment_status=treatment_status)
    return save_result(result, services.store)


def acquire(shared: Shared, cfg_store: tuple[AppConfig, RunStore], probe_budget: int) -> Responses:
    cfg, store = cfg_store
    directory = store.run_dir / "common"
    inputs = stage_inputs(shared, directory, [])
    adapter = GpuAdapter(shared, cfg, store)
    adapter.phase = "response_acquisition"
    with TaskEvaluator(Path(c2_local_adapter.__file__), "evaluate") as evaluator:
        search = TaskSearch(evaluator, inputs.objective, cfg.budgets)
        cid = shared.parent.candidate_id
        search.paths[cid], search.spaces[cid] = directory / "parent.py", shared.space
        search.resource_metrics, search.probe_budget = shared.resource_metrics, probe_budget
        responses = probe_responses(search, shared.parent, {"adapter": adapter})
    result = Responses(shared_id=shared.identity(), responses=responses, probe_calls=search.probe_calls, costs=costs(store))
    result.validate_for(shared)
    (store.run_dir / "responses.json").write_text(result.model_dump_json(indent=2), encoding="utf-8")
    store.append("RUN_FINISHED", {"probe_calls": result.probe_calls})
    return result
