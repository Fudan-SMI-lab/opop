"""Current shared routes only; explicit isolation belongs to T2 RED tests."""

import argparse
import json
import os
from contextlib import closing, nullcontext
from inspect import signature
from pathlib import Path
from typing import Final

import pytest
from optuna.samplers import TPESampler

from kernel_optimizer import task_cli, wiring
from kernel_optimizer.agents.modules import StructureRewriterAgent
from kernel_optimizer.agents.task_rewriter import TaskRewriterAgent
from kernel_optimizer.config import AppConfig
from kernel_optimizer.control.direct_task import TaskSearch
from kernel_optimizer.models.core import TaskSpec, sha256_text
from kernel_optimizer.store.run_store import RunStore
from kernel_optimizer.tuning import tpe
from scripts.experiments.c2_local_agents import Services, direct_child
from tests.c2_contract_capture import (
    REFERENCE, Captured, RecordingProvider, RequestCapture, cpu_stack, load_anchor, request_at, structured_files,
)
from tests.c2_opportunity_fakes import CampaignFixture, campaign, prepared

pytest_plugins = ["tests.c3_search_fakes"]
GOALS: Final = ("Minimize latency", "Maximize arbitrary J for KernelBench", "Optimize Qwen TTFT")


def save_evidence(name: str, request: RequestCapture) -> None:
    destination = os.environ.get("T1_EVIDENCE")
    if destination:
        path = Path(destination)
        path.mkdir(parents=True, exist_ok=True)
        (path / f"{name}.json").write_text(request.model_dump_json(indent=2) + "\n", encoding="utf-8")
        (path / f"{name}-normalization.json").write_text(
            json.dumps(request.normalization_map, indent=2) + "\n", encoding="utf-8")


def test_factory_classes_when_no_selector_is_supplied(tmp_path: Path) -> None:
    # Given
    inputs = request_at(tmp_path / "case")
    cfg = AppConfig()
    task = TaskSpec(level=3, problem_id=21, name="fixture", ref_path=inputs.project_root / "reference.py",
                    ref_src_sha=sha256_text(REFERENCE))
    with closing(RecordingProvider(tmp_path)) as provider:
        runtime = wiring.Runtime(cfg)
        runtime.client = provider
        store = RunStore(tmp_path / "run")
        # When
        legacy = wiring.build_orchestrator(cfg, store, task, runtime).deps.rewriter
        generic = wiring.build_task_rewriter(cfg, store, runtime)
        # Then
        assert type(legacy) is StructureRewriterAgent
        assert type(generic) is TaskRewriterAgent
        assert legacy.self_test_context is not None and generic.self_test_context is None
        assert generic.cfg == legacy.cfg == cfg.agents.module("rewriter")
        assert (generic.cfg.max_retries, generic.cfg.max_transport_retries, cfg.run.seed) == (2, 2, 0)
        assert cfg.budgets.trials_per_space == 40
        assert cfg.budgets.space_expansions_per_candidate == 0
        assert signature(TaskSearch.evaluate_candidate).parameters["startup_trials"].default == 10


def test_c2_budget_when_actual_controller_applies_experiment_overrides(campaign: CampaignFixture) -> None:
    # Given: the existing CPU campaign fixture replaces provider/worker boundaries, not the controller.
    # When
    result = campaign.run(1, "A1")
    # Then: B40 and one expansion are experiment policies, not the generic expansion default.
    assert result.information.asked == 80 and result.information.expanded_count == 1
    assert result.acquisition_calls == 12
    assert result.sampler_seed == 0
    assert campaign.cfg.agents.rewriter.max_retries == 0


@pytest.mark.parametrize("goal", GOALS)
def test_direct_c2_when_goal_wording_changes(tmp_path: Path, goal: str) -> None:
    # Given
    inputs = request_at(tmp_path / "case").model_copy(update={"goal": goal})
    with closing(RecordingProvider(inputs.project_root)) as provider:
        runtime = wiring.Runtime(AppConfig())
        runtime.client = provider
        services = Services(runtime.cfg, RunStore(tmp_path / "run"), runtime)
        # When: the actual C2 entry function reaches the provider, without generation.
        with pytest.raises(Captured):
            direct_child(inputs, services)
        # Then: current direct C2 is still the five-field shared schema.
        actual = provider.requests[0]
        assert actual.output_schema == load_anchor("8526a13-generic").output_schema
        payload = structured_files(actual)["analysis/task_response.json"]
        assert isinstance(payload, dict) and payload["goal"] == goal
        assert {"bundle_sources", "bundle_document"} <= payload.keys()
        save_evidence("current-direct", actual)


def test_generic_cli_when_provided_eval_and_rewrite_are_selected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Given: a real CPU callback, empty space and one rewrite; no EvalBuilder/provider service.
    inputs = request_at(tmp_path / "case")
    (inputs.project_root / "space.json").write_text('{"params":[]}', encoding="utf-8")
    (inputs.project_root / "eval.py").write_text("def evaluate(path, params, context): return 4.0\n", encoding="utf-8")
    sampler_calls: list[tuple[int, int]] = []

    def sampler(*, seed: int, multivariate: bool, group: bool, n_startup_trials: int,
                constant_liar: bool, categorical_distance_func: None) -> TPESampler:
        # Exact external sampler constructor boundary; preserve all supplied arguments.
        sampler_calls.append((seed, n_startup_trials))
        return TPESampler(seed=seed, multivariate=multivariate, group=group, n_startup_trials=n_startup_trials,
                          constant_liar=constant_liar, categorical_distance_func=categorical_distance_func)

    monkeypatch.setattr(tpe, "TPESampler", sampler)
    with closing(RecordingProvider(inputs.project_root)) as provider:
        runtime = wiring.Runtime(AppConfig())
        runtime.client = provider
        monkeypatch.setattr(wiring, "Runtime", lambda *args, **kwargs: nullcontext(runtime))
        parser = argparse.ArgumentParser()
        task_cli.add_task_parser(parser)
        args = parser.parse_args(["--project", str(inputs.project_root), "--candidate", "parent.py",
            "--space", "space.json", "--eval-file", "eval.py", "--direction", "maximize",
            "--goal", "Qwen arbitrary J", "--rewrite-rounds", "1", "--output", str(tmp_path / "output")])
        # When: actual CLI/search/run_rewrites/factory reach the recording boundary.
        with pytest.raises(Captured):
            task_cli.cmd_optimize_task(args)
        # Then
        actual = provider.requests[0]
        assert actual.output_schema == load_anchor("8526a13-generic").output_schema
        payload = structured_files(actual)["analysis/task_response.json"]
        assert isinstance(payload, dict) and payload["objective"] == {"direction": "maximize", "label": "J", "unit": None}
        assert args.trials == 40 and args.probe_budget == 8
        assert sampler_calls == [(0, 10)]
        save_evidence("current-generic-cli", actual)


def test_c3_cli_when_current_operator_entrypoint_is_selected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Given: existing numerical CPU fake supplies the resident model boundary only.
    from examples.c3_qwen3 import operator_cli
    from tests.c3_search_fakes import search_case

    session, unused_provider, unused_agent, inputs = search_case(tmp_path)
    unused_provider.close()
    input_path = tmp_path / "search-inputs.json"
    input_path.write_text(inputs.model_dump_json(), encoding="utf-8")
    cfg = AppConfig()
    built: list[TaskRewriterAgent] = []

    def factory(config: AppConfig, store: RunStore, runtime: wiring.Runtime) -> TaskRewriterAgent:
        agent = wiring.build_task_rewriter(config, store, runtime)
        built.append(agent)
        return agent

    with closing(RecordingProvider(tmp_path)) as provider:
        runtime = wiring.Runtime(cfg)
        runtime.client = provider
        observed_retries: list[tuple[int, int]] = []
        create_session = provider.create_session

        def record_session(directory: Path, title: str) -> str:
            observed_retries.append((built[0].cfg.max_retries, built[0].cfg.max_transport_retries))
            return create_session(directory, title)

        monkeypatch.setattr(provider, "create_session", record_session)
        monkeypatch.setattr(operator_cli, "load_config", lambda path: cfg)
        monkeypatch.setattr(operator_cli, "prepare", lambda *args: session.runner.prepared)
        monkeypatch.setattr(operator_cli, "load", lambda *args: session.runner)
        monkeypatch.setattr(operator_cli, "OperatorSession", lambda *args, **kwargs: session)
        monkeypatch.setattr(operator_cli, "Runtime", lambda *args: nullcontext(runtime))
        monkeypatch.setattr(operator_cli, "build_task_rewriter", factory)
        # When: actual CLI/factory/optimize_goal/helper/invoke stop at the provider.
        with pytest.raises(Captured):
            operator_cli.main(["optimize-goal", "--inputs", str(input_path), "--config", "fixture-unused.yaml"])
        # Then: T1 documents shared routing, not future model_operator isolation.
        actual = provider.requests[0]
        assert actual.output_schema == load_anchor("8526a13-bundle").output_schema
        payload = structured_files(actual)["analysis/task_response.json"]
        assert isinstance(payload, dict) and payload["bundle_document"] is not None
        assert "task/operator-helper.json" in actual.files
        assert actual.sequence == ("create_session", "prompt")
        assert len(built) == 1 and type(built[0]) is TaskRewriterAgent
        assert observed_retries == [(0, 0)]
        assert (built[0].cfg.max_retries, built[0].cfg.max_transport_retries) == (2, 2)
        save_evidence("current-c3-cli", actual)
