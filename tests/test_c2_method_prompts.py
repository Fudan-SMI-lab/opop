"""Prompt contracts test delivery and routing, never natural-language wording."""

from dataclasses import replace
from pathlib import Path
from typing import Literal

import pytest

from kernel_optimizer.agents.modules import (
    AnalystInputs,
    BottleneckAnalystAgent,
    ParameterizerAgent,
    ParameterizerInputs,
    RewriterInputs,
    StructureRewriterAgent,
)
from kernel_optimizer.agents.sandbox import Sandbox
from kernel_optimizer.models.core import DeviceLimits, TaskSpec
from kernel_optimizer.models.reports import BottleneckReport, TuningStats


@pytest.fixture
def task() -> TaskSpec:
    return TaskSpec(level=3, problem_id=43, name="fixture", ref_path=Path("ref.py"),
                    ref_src_sha="0" * 64)


@pytest.fixture
def rewrite(task: TaskSpec) -> RewriterInputs:
    return RewriterInputs(task, "PARAMS = {}\n", BottleneckReport(summary="fixture"),
                          [], DeviceLimits(), 1)


@pytest.fixture
def analyst(task: TaskSpec) -> AnalystInputs:
    return AnalystInputs(task, "PARAMS = {}\n", TuningStats(
        space_id="s", candidate_id="c", n_trials=0, n_complete=0, n_fail=0,
    ), "status,latency\n", DeviceLimits())


def test_baseline_wall_payload_when_supplied(rewrite: RewriterInputs, tmp_path: Path) -> None:
    # Given
    inputs = replace(rewrite, wall_text='{"param":"X","status":"compile_error"}')
    agent = StructureRewriterAgent.__new__(StructureRewriterAgent)
    sandbox = Sandbox(tmp_path)
    # When
    agent.seed_sandbox(inputs, sandbox)
    prompt = agent.render_prompt(inputs, sandbox)
    # Then
    assert sandbox.read_output("analysis/resource_walls.md") == inputs.wall_text
    assert sandbox.read_output("analysis/bottleneck.json") == inputs.report.model_dump_json(indent=2)
    assert "analysis/resource_walls.md" in prompt


def test_baseline_parameterizer_when_old_inputs(task: TaskSpec, tmp_path: Path) -> None:
    # Given
    inputs = ParameterizerInputs(task, "PARAMS = {}\n", DeviceLimits())
    agent = ParameterizerAgent.__new__(ParameterizerAgent)
    sandbox = Sandbox(tmp_path)
    # When
    agent.seed_sandbox(inputs, sandbox)
    prompt = agent.render_prompt(inputs, sandbox)
    # Then
    assert sandbox.read_output("candidate/source.py") == inputs.candidate_source
    assert "candidate/parameterized.py" in prompt
    legacy = agent.output_model.model_validate_json('{"file":"old.py","space":{"params":[]}}')
    assert legacy.file == "old.py" and legacy.space.params == []
    assert legacy.recommended_configs == []


@pytest.mark.parametrize("wall", [None, '{"status":"infeasible_shared_memory"}'])
@pytest.mark.parametrize("direction", ["minimize", "maximize"])
def test_response_delivery_when_wall_is_optional(
    rewrite: RewriterInputs, tmp_path: Path, wall: str | None, direction: str,
) -> None:
    # Given
    response = '{"direction":"' + direction + '","delta_j":-2,"missing":null}'
    inputs = replace(rewrite, wall_text=wall, conditional_response_text=response)
    agent = StructureRewriterAgent.__new__(StructureRewriterAgent)
    sandbox = Sandbox(tmp_path)
    # When
    agent.seed_sandbox(inputs, sandbox)
    prompt = agent.render_prompt(inputs, sandbox)
    # Then
    assert sandbox.read_output("analysis/conditional_responses.md") == response
    assert "analysis/conditional_responses.md" in prompt
    assert sandbox.exists("analysis/resource_walls.md") == bool(wall)
    if wall:
        assert sandbox.read_output("analysis/resource_walls.md") == wall
    assert sandbox.read_output("analysis/bottleneck.json") == inputs.report.model_dump_json(indent=2)


@pytest.mark.parametrize("intent", [None, "", '{"action":"fuse","region":null}'])
def test_intent_delivery_when_optional(
    task: TaskSpec, tmp_path: Path, intent: str | None,
) -> None:
    # Given
    inputs = ParameterizerInputs(task, "PARAMS = {}\n", DeviceLimits(), rewrite_intent=intent)
    agent = ParameterizerAgent.__new__(ParameterizerAgent)
    sandbox = Sandbox(tmp_path)
    # When
    agent.seed_sandbox(inputs, sandbox)
    prompt = agent.render_prompt(inputs, sandbox)
    # Then
    assert sandbox.exists("analysis/rewrite_intent.md") == bool(intent)
    assert ("analysis/rewrite_intent.md" in prompt) == bool(intent)
    if intent:
        assert sandbox.read_output("analysis/rewrite_intent.md") == intent
    legacy = agent.output_model.model_validate_json('{"file":"old.py","space":{"params":[]}}')
    assert legacy.file == "old.py" and legacy.space.params == []
    assert legacy.recommended_configs == []


@pytest.mark.parametrize("mode", ["whole_task", "legacy_local"])
def test_reasoning_route_when_mode_selected(
    analyst: AnalystInputs, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    mode: Literal["whole_task", "legacy_local"],
) -> None:
    # Given
    inputs = replace(analyst, reasoning_mode=mode, reference_source="REF_SENTINEL = 71\n",
                     conditional_response_text='{"axis":"X","delta_j":2}')
    from kernel_optimizer.agents import method_prompts

    monkeypatch.setattr(method_prompts, "WHOLE_TASK_GUIDANCE", "<route:whole_task>")
    monkeypatch.setattr(method_prompts, "LEGACY_LOCAL_GUIDANCE", "<route:legacy_local>")
    agent = BottleneckAnalystAgent.__new__(BottleneckAnalystAgent)
    sandbox = Sandbox(tmp_path)
    # When
    agent.seed_sandbox(inputs, sandbox)
    prompt = agent.render_prompt(inputs, sandbox)
    # Then
    assert f"<route:{mode}>" in prompt
    assert prompt.count("<route:") == 1
    assert sandbox.read_output("task/ref.py") == inputs.reference_source
    assert sandbox.read_output("analysis/conditional_responses.md") == inputs.conditional_response_text
    assert "task/ref.py" in prompt
    assert "analysis/conditional_responses.md" in prompt


@pytest.mark.parametrize("mode", ["whole_task", "legacy_local"])
def test_rewriter_route_when_mode_selected(
    rewrite: RewriterInputs, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    mode: Literal["whole_task", "legacy_local"],
) -> None:
    # Given
    inputs = replace(rewrite, reasoning_mode=mode, reference_source="REF_SENTINEL = 83\n")
    from kernel_optimizer.agents import method_prompts

    monkeypatch.setattr(method_prompts, "WHOLE_TASK_GUIDANCE", "<route:whole_task>")
    monkeypatch.setattr(method_prompts, "LEGACY_LOCAL_GUIDANCE", "<route:legacy_local>")
    agent = StructureRewriterAgent.__new__(StructureRewriterAgent)
    sandbox = Sandbox(tmp_path)
    # When
    agent.seed_sandbox(inputs, sandbox)
    prompt = agent.render_prompt(inputs, sandbox)
    # Then
    assert f"<route:{mode}>" in prompt
    assert prompt.count("<route:") == 1
    assert sandbox.read_output("task/ref.py") == inputs.reference_source
    assert "task/ref.py" in prompt


def test_defaults_when_existing_callers_omit_new_fields(
    analyst: AnalystInputs, rewrite: RewriterInputs, tmp_path: Path,
) -> None:
    # Given
    sandbox = Sandbox(tmp_path)
    agent = StructureRewriterAgent.__new__(StructureRewriterAgent)
    # When
    agent.seed_sandbox(rewrite, sandbox)
    prompt = agent.render_prompt(rewrite, sandbox)
    # Then
    assert analyst.reasoning_mode == rewrite.reasoning_mode == "whole_task"
    assert analyst.reference_source is rewrite.reference_source is None
    assert analyst.conditional_response_text is rewrite.conditional_response_text is None
    assert "task/ref.py" not in prompt
    assert "analysis/conditional_responses.md" not in prompt
    assert "analysis/resource_walls.md" not in prompt
    assert not sandbox.exists("task/ref.py")
    assert not sandbox.exists("analysis/conditional_responses.md")


def test_analyst_payload_when_new_fields_are_absent(
    analyst: AnalystInputs, tmp_path: Path,
) -> None:
    # Given
    agent = BottleneckAnalystAgent.__new__(BottleneckAnalystAgent)
    sandbox = Sandbox(tmp_path)
    # When
    agent.seed_sandbox(analyst, sandbox)
    prompt = agent.render_prompt(analyst, sandbox)
    # Then
    assert sandbox.read_output("tuning/stats.json") == analyst.stats.model_dump_json(indent=2)
    assert sandbox.read_output("tuning/trials.csv") == analyst.trials_csv
    assert "task/ref.py" not in prompt
    assert "analysis/conditional_responses.md" not in prompt


def test_intent_delivery_when_expanding(task: TaskSpec, tmp_path: Path) -> None:
    # Given
    inputs = ParameterizerInputs(task, "PARAMS = {'X': 2}\n", DeviceLimits(),
                                 expand_directive="X: 2 -> 4", rewrite_intent="intent-42")
    agent = ParameterizerAgent.__new__(ParameterizerAgent)
    sandbox = Sandbox(tmp_path)
    # When
    agent.seed_sandbox(inputs, sandbox)
    prompt = agent.render_prompt(inputs, sandbox)
    # Then
    assert sandbox.read_output("analysis/rewrite_intent.md") == inputs.rewrite_intent
    assert "analysis/rewrite_intent.md" in prompt
    assert inputs.expand_directive in prompt
    legacy = agent.output_model.model_validate_json('{"file":"old.py","space":{"params":[]}}')
    assert legacy.file == "old.py" and legacy.space.params == []
    assert legacy.recommended_configs == []
