"""Recommended-config schema, sandbox delivery and advisory acceptance contracts."""

import json
from dataclasses import replace
from pathlib import Path
from typing import Final

import pytest
from pydantic import ValidationError

from kernel_optimizer.agents import modules
from kernel_optimizer.agents.modules import (
    ParameterizerAgent, ParameterizerInputs, RewriterInputs, StructureRewriterAgent,
)
from kernel_optimizer.agents.sandbox import Sandbox
from kernel_optimizer.models.core import DeviceLimits, ParamSet, TaskSpec
from kernel_optimizer.models.reports import (
    BottleneckReport, ParameterizationResult, RewriteCandidate, RewriteResult,
)


OUTPUTS: Final = (
    (RewriteCandidate, '{"file":"candidate.py","change_summary":"fixture"}'),
    (ParameterizationResult, '{"file":"candidate.py","space":{"params":['
     '{"name":"X","kind":"int","choices":[1,2]},'
     '{"name":"Y","kind":"int","choices":[1,2]}],'
     '"constraints":[{"expr":"X <= Y","rationale":"fixture"}]}}'),
)
SOURCE: Final = """import triton
import triton.language as tl
PARAMS = {'X': 1, 'Y': 2}
@triton.jit
def kernel(ptr):
    tl.store(ptr, 1)
"""


@pytest.fixture
def inputs() -> ParameterizerInputs:
    task = TaskSpec(level=1, problem_id=1, name="fixture", ref_path=Path("ref.py"),
                    ref_src_sha="0" * 64)
    return ParameterizerInputs(task, SOURCE, DeviceLimits())


@pytest.mark.parametrize("model,payload", OUTPUTS)
def test_baseline_output_when_old_json(
    model: type[RewriteCandidate] | type[ParameterizationResult], payload: str,
) -> None:
    # Given / When
    output = model.model_validate_json(payload)
    # Then
    assert output.file == "candidate.py"
    assert output.recommended_configs == []


def test_baseline_seed_when_old_inputs(inputs: ParameterizerInputs, tmp_path: Path) -> None:
    # Given
    agent = ParameterizerAgent.__new__(ParameterizerAgent)
    sandbox = Sandbox(tmp_path)
    # When
    agent.seed_sandbox(inputs, sandbox)
    prompt = agent.render_prompt(inputs, sandbox)
    # Then
    assert sandbox.read_output("candidate/source.py") == SOURCE
    assert not sandbox.exists("analysis/recommended_configs.json")
    assert "analysis/recommended_configs.json" not in prompt
    assert inputs.recommended_configs == ()


def test_baseline_acceptance_when_valid_candidate(tmp_path: Path) -> None:
    # Given
    sandbox = Sandbox(tmp_path)
    sandbox.write_input("candidate.py", SOURCE)
    output = ParameterizationResult.model_validate_json(OUTPUTS[1][1])
    agent = ParameterizerAgent.__new__(ParameterizerAgent)
    # When
    problem = agent.check_output(output, sandbox)
    # Then
    assert problem is None


@pytest.mark.parametrize("model,payload", OUTPUTS)
@pytest.mark.parametrize("count", [0, 1, 2])
def test_configs_roundtrip_when_within_limit(
    model: type[RewriteCandidate] | type[ParameterizationResult], payload: str, count: int,
) -> None:
    # Given
    configs = [{"values": {"X": 2, "Y": 2, "MODE": "nominal", "SCALE": 0.5}}] * count
    # When
    output = model.model_validate({**json.loads(payload), "recommended_configs": configs})
    # Then
    assert [config.model_dump() for config in output.recommended_configs] == configs
    assert json.loads(output.model_dump_json())["recommended_configs"] == configs


@pytest.mark.parametrize("model,payload", OUTPUTS)
def test_schema_limit_when_three_configs(
    model: type[RewriteCandidate] | type[ParameterizationResult], payload: str,
) -> None:
    # Given
    configs = [{"values": {"X": 1, "Y": 2}}] * 3
    # When / Then
    with pytest.raises(ValidationError) as rejected:
        model.model_validate({**json.loads(payload), "recommended_configs": configs})
    assert rejected.value.errors()[0]["type"] == "too_long"
    assert rejected.value.errors()[0]["loc"] == ("recommended_configs",)


@pytest.mark.parametrize("model,payload", OUTPUTS)
@pytest.mark.parametrize("raw", ['[{"values":{"X":null}}]', '[{"values":{"X":[]}}]',
                                 '[{"values":{"X":{}}}]', '[{}]', 'null'])
def test_invalid_structure_when_config_is_malformed(
    model: type[RewriteCandidate] | type[ParameterizationResult], payload: str, raw: str,
) -> None:
    # Given
    data = {**json.loads(payload), "recommended_configs": json.loads(raw)}
    # When / Then
    with pytest.raises(ValidationError) as rejected:
        model.model_validate(data)
    assert all(error["loc"][0] == "recommended_configs" for error in rejected.value.errors())


@pytest.mark.parametrize("model,payload", OUTPUTS)
def test_json_schema_when_optional_field_added(
    model: type[RewriteCandidate] | type[ParameterizationResult], payload: str,
) -> None:
    # Given / When
    schema = model.model_json_schema()
    # Then
    assert schema["properties"]["recommended_configs"]["maxItems"] == 2
    assert "recommended_configs" not in schema["required"]
    assert "values" in schema["$defs"]["ParamSet"]["properties"]
    assert set(schema["$defs"]["ParamSet"]["properties"]) == {"values"}
    assert model.model_validate_json(payload).recommended_configs == []


@pytest.mark.parametrize("count", [0, 1, 2])
@pytest.mark.parametrize("expansion", ["", "X: 2 -> 3"])
def test_request_delivery_when_parameterizing(
    inputs: ParameterizerInputs, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    count: int, expansion: str,
) -> None:
    # Given
    configs = (ParamSet(values={"X": 2, "Y": 2}), ParamSet(values={"X": 1}))[:count]
    request = replace(inputs, recommended_configs=configs, expand_directive=expansion)
    monkeypatch.setattr(modules, "PARAMETERIZER_RECOMMENDATIONS_GUIDANCE", "<resolve-configs>",
                        raising=False)
    agent = ParameterizerAgent.__new__(ParameterizerAgent)
    sandbox = Sandbox(tmp_path)
    # When
    agent.seed_sandbox(request, sandbox)
    prompt = agent.render_prompt(request, sandbox)
    # Then
    assert "<resolve-configs>" in prompt
    assert sandbox.exists("analysis/recommended_configs.json") == bool(configs)
    assert ("analysis/recommended_configs.json" in prompt) == bool(configs)
    if configs:
        assert json.loads(sandbox.read_output("analysis/recommended_configs.json")) == [
            config.model_dump() for config in configs
        ]


@pytest.mark.parametrize("raw", ['{"WRONG":1}', '{"X":1}', '{"X":999,"Y":2}', '{"X":2,"Y":1}'])
def test_parameterizer_accepts_candidate_when_request_is_semantically_invalid(
    tmp_path: Path, raw: str,
) -> None:
    # Given
    sandbox = Sandbox(tmp_path)
    sandbox.write_input("candidate.py", SOURCE)
    data = {**json.loads(OUTPUTS[1][1]), "recommended_configs": [{"values": json.loads(raw)}]}
    output = ParameterizationResult.model_validate(data)
    agent = ParameterizerAgent.__new__(ParameterizerAgent)
    # When
    problem = agent.check_output(output, sandbox)
    # Then
    assert problem is None
    assert output.recommended_configs[0].values == json.loads(raw)


@pytest.mark.parametrize("raw", ['{"WRONG":1}', '{"X":1}', '{"X":999,"Y":2}', '{"X":2,"Y":1}'])
def test_rewriter_accepts_candidate_when_request_is_semantically_invalid(tmp_path: Path, raw: str) -> None:
    # Given
    sandbox = Sandbox(tmp_path)
    sandbox.write_input("candidate.py", SOURCE)
    candidate = RewriteCandidate.model_validate({**json.loads(OUTPUTS[0][1]),
        "recommended_configs": [{"values": json.loads(raw)}]})
    output = RewriteResult(candidates=[candidate])
    agent = StructureRewriterAgent.__new__(StructureRewriterAgent)
    # When
    problem = agent.check_output(output, sandbox)
    # Then
    assert problem is None
    assert candidate.recommended_configs[0].values == json.loads(raw)


def test_rewriter_routes_optional_request_guidance(
    inputs: ParameterizerInputs, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given
    request = RewriterInputs(inputs.task, SOURCE, BottleneckReport(summary="fixture"),
                             [], inputs.device, 1)
    monkeypatch.setattr(modules, "REWRITER_RECOMMENDATIONS_GUIDANCE", "<recommend-configs>",
                        raising=False)
    agent = StructureRewriterAgent.__new__(StructureRewriterAgent)
    # When
    prompt = agent.render_prompt(request, Sandbox(tmp_path))
    # Then
    assert "<recommend-configs>" in prompt
