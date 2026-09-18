"""Opportunity prompt contracts: input delivery and routing, not prose wording."""

import json
from dataclasses import replace
from pathlib import Path

import pytest

from kernel_optimizer.agents import method_prompts
from kernel_optimizer.agents.modules import (
    AnalystInputs,
    BottleneckAnalystAgent,
    ParameterizerAgent,
    ParameterizerInputs,
    RewriterInputs,
    StructureRewriterAgent,
)
from kernel_optimizer.agents.sandbox import Sandbox
from kernel_optimizer.models.core import DeviceLimits, ParamSet, TaskSpec
from kernel_optimizer.models.reports import BottleneckReport, TuningStats
from kernel_optimizer.paramspace.materializer import extract_defaults, materialize


@pytest.fixture
def analyst() -> AnalystInputs:
    return AnalystInputs(
        task=TaskSpec(level=3, problem_id=43, name="opportunity", ref_path=Path("ref.py"),
                      ref_src_sha="0" * 64),
        candidate_source="PARAMS = {'TILE': 16, 'STAGES': 1}\n",
        stats=TuningStats(space_id="s", candidate_id="c", n_trials=1, n_complete=1, n_fail=0),
        trials_csv="TILE,STAGES,latency\n64,3,2.5\n", device=DeviceLimits(),
    )


@pytest.fixture
def rewrite(analyst: AnalystInputs) -> RewriterInputs:
    return RewriterInputs(analyst.task, analyst.candidate_source,
                          BottleneckReport(summary="fixture"), [], analyst.device, 1)


def test_baseline_analyst_when_old_call(analyst: AnalystInputs, tmp_path: Path) -> None:
    # Given
    agent = BottleneckAnalystAgent.__new__(BottleneckAnalystAgent)
    sandbox = Sandbox(tmp_path)
    # When
    agent.seed_sandbox(analyst, sandbox)
    prompt = agent.render_prompt(analyst, sandbox)
    # Then
    assert sandbox.read_output("candidate/source.py") == analyst.candidate_source
    assert sandbox.read_output("tuning/trials.csv") == analyst.trials_csv
    assert "tuning/selected_params.json" not in prompt
    assert not sandbox.exists("tuning/selected_params.json")


@pytest.mark.parametrize("response", [None, "", '{"delta_j":-0.5,"fixed":{"STAGES":3}}'])
def test_analyst_c2_route_when_responses_supplied(
    analyst: AnalystInputs, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    response: str | None,
) -> None:
    # Given
    inputs = replace(analyst, conditional_response_text=response)
    monkeypatch.setattr(method_prompts, "C2_OPPORTUNITY_GUIDANCE", "<c2-route>", raising=False)
    monkeypatch.setattr(method_prompts, "WHOLE_TASK_GUIDANCE", "<general-route>")
    agent = BottleneckAnalystAgent.__new__(BottleneckAnalystAgent)
    sandbox = Sandbox(tmp_path)
    # When
    agent.seed_sandbox(inputs, sandbox)
    prompt = agent.render_prompt(inputs, sandbox)
    # Then
    assert "<general-route>" in prompt
    assert ("<c2-route>" in prompt) == bool(response)
    assert sandbox.exists("analysis/conditional_responses.md") == bool(response)
    assert sandbox.read_output("tuning/trials.csv") == analyst.trials_csv
    if response:
        assert sandbox.read_output("analysis/conditional_responses.md") == response


@pytest.mark.parametrize("response", [None, "", '{"delta_j":0.5,"missing":null}'])
def test_rewriter_c2_route_when_responses_supplied(
    rewrite: RewriterInputs, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    response: str | None,
) -> None:
    # Given
    inputs = replace(rewrite, conditional_response_text=response, wall_text="compiler-refusal-7")
    monkeypatch.setattr(method_prompts, "C2_OPPORTUNITY_GUIDANCE", "<c2-route>", raising=False)
    monkeypatch.setattr(method_prompts, "WHOLE_TASK_GUIDANCE", "<general-route>")
    agent = StructureRewriterAgent.__new__(StructureRewriterAgent)
    sandbox = Sandbox(tmp_path)
    # When
    agent.seed_sandbox(inputs, sandbox)
    prompt = agent.render_prompt(inputs, sandbox)
    # Then
    assert "<general-route>" in prompt
    assert ("<c2-route>" in prompt) == bool(response)
    assert sandbox.read_output("analysis/resource_walls.md") == inputs.wall_text
    assert sandbox.exists("analysis/conditional_responses.md") == bool(response)
    if response:
        assert sandbox.read_output("analysis/conditional_responses.md") == response


def test_analyst_selected_values_when_source_defaults_differ(
    analyst: AnalystInputs, tmp_path: Path,
) -> None:
    # Given
    selected = ParamSet(values={"TILE": 64, "STAGES": 3})
    response = '{"fixed":{"STAGES":3},"endpoints":[64,128],"axis":"TILE"}'
    inputs = replace(analyst, selected_params=selected, conditional_response_text=response)
    agent = BottleneckAnalystAgent.__new__(BottleneckAnalystAgent)
    sandbox = Sandbox(tmp_path)
    # When
    agent.seed_sandbox(inputs, sandbox)
    prompt = agent.render_prompt(inputs, sandbox)
    # Then
    assert json.loads(sandbox.read_output("tuning/selected_params.json")) == selected.model_dump()
    assert extract_defaults(sandbox.read_output("candidate/source.py")) == {"TILE": 16, "STAGES": 1}
    assert sandbox.read_output("analysis/conditional_responses.md") == response
    assert "tuning/selected_params.json" in prompt


@pytest.mark.parametrize("materialized", [False, True])
def test_rewriter_selected_context_when_materialization_declared(
    rewrite: RewriterInputs, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    materialized: bool,
) -> None:
    # Given
    selected = ParamSet(values={"TILE": 64, "STAGES": 3})
    source = materialize(rewrite.best_source, selected) if materialized else rewrite.best_source
    response = '{"fixed":{"STAGES":3},"endpoints":[64,128],"axis":"TILE"}'
    inputs = replace(rewrite, selected_params=selected, source_materialized=materialized,
                     best_source=source, conditional_response_text=response)
    monkeypatch.setattr(method_prompts, "RAW_SOURCE_GUIDANCE", "<raw-route>", raising=False)
    monkeypatch.setattr(method_prompts, "MATERIALIZED_SOURCE_GUIDANCE", "<materialized-route>",
                        raising=False)
    agent = StructureRewriterAgent.__new__(StructureRewriterAgent)
    sandbox = Sandbox(tmp_path)
    # When
    agent.seed_sandbox(inputs, sandbox)
    prompt = agent.render_prompt(inputs, sandbox)
    # Then
    assert sandbox.read_output("candidate/best.py") == source
    assert json.loads(sandbox.read_output("tuning/selected_params.json")) == selected.model_dump()
    assert sandbox.read_output("analysis/conditional_responses.md") == response
    assert ("<materialized-route>" in prompt) == materialized
    assert ("<raw-route>" in prompt) == (not materialized)
    assert "tuning/selected_params.json" in prompt
    assert extract_defaults(source) == (selected.values if materialized else {"TILE": 16, "STAGES": 1})


def test_selected_defaults_when_old_call(analyst: AnalystInputs, rewrite: RewriterInputs) -> None:
    # Given / When: existing constructor calls omit all new fields.
    # Then
    assert analyst.selected_params is rewrite.selected_params is None
    assert rewrite.source_materialized is False


@pytest.mark.parametrize("expansion", ["", "TILE: 64 -> 128"])
def test_joint_region_intent_when_parameterizing(
    analyst: AnalystInputs, tmp_path: Path, expansion: str,
) -> None:
    # Given
    intent = '{"region":{"TILE":[64,128]},"partners":{"STAGES":[3,4]}}'
    inputs = ParameterizerInputs(analyst.task, analyst.candidate_source, analyst.device,
                                 rewrite_intent=intent, expand_directive=expansion)
    agent = ParameterizerAgent.__new__(ParameterizerAgent)
    sandbox = Sandbox(tmp_path)
    # When
    agent.seed_sandbox(inputs, sandbox)
    prompt = agent.render_prompt(inputs, sandbox)
    # Then
    assert sandbox.read_output("analysis/rewrite_intent.md") == intent
    assert "analysis/rewrite_intent.md" in prompt
    assert set(agent.output_model.model_fields) == {"file", "space"}


def test_selected_payload_when_empty_paramset(analyst: AnalystInputs, tmp_path: Path) -> None:
    # Given
    inputs = replace(analyst, selected_params=ParamSet(values={}))
    agent = BottleneckAnalystAgent.__new__(BottleneckAnalystAgent)
    sandbox = Sandbox(tmp_path)
    # When
    agent.seed_sandbox(inputs, sandbox)
    prompt = agent.render_prompt(inputs, sandbox)
    # Then
    assert json.loads(sandbox.read_output("tuning/selected_params.json")) == {"values": {}}
    assert "tuning/selected_params.json" in prompt


def test_baseline_rewriter_when_old_call(rewrite: RewriterInputs, tmp_path: Path) -> None:
    # Given
    agent = StructureRewriterAgent.__new__(StructureRewriterAgent)
    sandbox = Sandbox(tmp_path)
    # When
    agent.seed_sandbox(rewrite, sandbox)
    prompt = agent.render_prompt(rewrite, sandbox)
    # Then
    assert sandbox.read_output("candidate/best.py") == rewrite.best_source
    assert "tuning/selected_params.json" not in prompt
    assert not sandbox.exists("tuning/selected_params.json")
