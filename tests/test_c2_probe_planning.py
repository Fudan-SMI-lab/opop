"""Probe planning schemas and route selection without natural-language assertions."""

import json
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from kernel_optimizer.agents import method_prompts, modules
from kernel_optimizer.agents.modules import (
    AnalystInputs, BottleneckAnalystAgent, RewriterInputs, StructureRewriterAgent,
)
from kernel_optimizer.agents.runtime import OpencodeClient, PromptResult
from kernel_optimizer.agents.sandbox import Sandbox, SandboxFactory
from kernel_optimizer.config import AgentModuleConfig
from kernel_optimizer.models import reports
from kernel_optimizer.models.core import DeviceLimits, ParamSet, TaskSpec
from kernel_optimizer.models.reports import BottleneckReport, TuningStats
from kernel_optimizer.store.run_store import RunStore


@pytest.fixture
def analyst() -> AnalystInputs:
    task = TaskSpec(level=1, problem_id=1, name="fixture", ref_path=Path("ref.py"), ref_src_sha="0" * 64)
    return AnalystInputs(task, "PARAMS = {'X': 1, 'Y': 2}\n", TuningStats(
        candidate_id="c", space_id="s", n_complete=1, n_fail=0,
    ), "X,Y,latency\n3,4,1.5\n", DeviceLimits(), selected_params=ParamSet(values={"X": 3, "Y": 4}))


def test_legacy_report_when_requests_absent() -> None:
    # Given / When
    report = BottleneckReport.model_validate_json('{"summary":"fixture","hypotheses":[]}')
    # Then
    assert report.summary == "fixture"
    assert report.hypotheses == []
    assert report.probe_requests == []


def test_legacy_analyst_when_planning_omitted(analyst: AnalystInputs, tmp_path: Path) -> None:
    # Given
    agent = BottleneckAnalystAgent.__new__(BottleneckAnalystAgent)
    sandbox = Sandbox(tmp_path)
    # When
    agent.seed_sandbox(analyst, sandbox)
    # Then
    assert sandbox.read_output("candidate/source.py") == analyst.candidate_source
    assert sandbox.read_output("tuning/trials.csv") == analyst.trials_csv
    assert not sandbox.exists("analysis/conditional_responses.md")
    assert analyst.plan_probes is False and analyst.probe_budget == 12


@pytest.mark.parametrize("enabled", [False, True])
def test_single_analyst_call_when_planning_enabled(
    analyst: AnalystInputs, tmp_path: Path, enabled: bool,
) -> None:
    # Given
    request = replace(analyst, plan_probes=enabled)
    sandbox = Sandbox(tmp_path / "sandbox")
    factory = SandboxFactory(tmp_path)
    client = OpencodeClient.__new__(OpencodeClient)
    result = PromptResult(text="", session_id="offline", structured={"summary": "fixture"})
    agent = BottleneckAnalystAgent(client, factory, RunStore(tmp_path), AgentModuleConfig())
    with patch.object(factory, "create", return_value=sandbox), \
         patch.object(client, "create_session", return_value="offline") as session, \
         patch.object(client, "prompt", return_value=result) as prompt:
        # When
        outcome = agent.invoke(request)
    # Then
    assert session.call_count == prompt.call_count == outcome.attempts == 1
    assert outcome.output.probe_requests == []


@pytest.mark.parametrize("count", [0, 1, 6])
def test_probe_requests_roundtrip_when_bounded(count: int) -> None:
    # Given
    requests = [{"axis": "MODE", "a_value": "a", "b_value": "b", "partners": {"Y": 4}}] * count
    # When
    report = BottleneckReport.model_validate({"summary": "fixture", "probe_requests": requests})
    # Then
    assert [request.model_dump() for request in report.probe_requests] == requests


def test_probe_request_limit_when_seven() -> None:
    # Given
    requests = [{"axis": "X", "a_value": 1, "b_value": 3}] * 7
    # When / Then
    with pytest.raises(ValidationError) as rejected:
        BottleneckReport.model_validate({"summary": "fixture", "probe_requests": requests})
    assert rejected.value.errors()[0]["type"] == "too_long"


def test_partners_default_when_omitted() -> None:
    # Given / When
    request = reports.ProbeRequest.model_validate({"axis": "X", "a_value": 1, "b_value": 3})
    # Then
    assert request.partners == {}
    assert set(request.model_dump()) == {"axis", "a_value", "b_value", "partners"}


def test_axis_cannot_override_itself_in_partners() -> None:
    # Given
    payload = {"axis": "X", "a_value": 1, "b_value": 3, "partners": {"X": 2}}
    # When / Then
    with pytest.raises(ValidationError) as rejected:
        reports.ProbeRequest.model_validate(payload)
    assert rejected.value.errors()[0]["type"] == "probe_axis_in_partners"


@pytest.mark.parametrize("raw", ['{"axis":"X","a_value":null,"b_value":3}',
                                 '{"axis":"X","a_value":1,"b_value":[]}',
                                 '{"axis":"X","a_value":1,"b_value":3,"partners":{"Y":{}}}'])
def test_malformed_probe_values_rejected(raw: str) -> None:
    # Given / When / Then
    with pytest.raises(ValidationError):
        reports.ProbeRequest.model_validate_json(raw)


@pytest.mark.parametrize("budget", [0, 1, 5, 12, 20])
@pytest.mark.parametrize("enabled", [False, True])
def test_planning_route_when_explicitly_requested(
    analyst: AnalystInputs, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    budget: int, enabled: bool,
) -> None:
    # Given
    request = replace(analyst, plan_probes=enabled, probe_budget=budget)
    monkeypatch.setattr(modules, "PROBE_PLANNING_GUIDANCE", "<planning:{endpoint_budget}:{request_limit}>",
                        raising=False)
    agent = BottleneckAnalystAgent.__new__(BottleneckAnalystAgent)
    sandbox = Sandbox(tmp_path)
    # When
    agent.seed_sandbox(request, sandbox)
    prompt = agent.render_prompt(request, sandbox)
    # Then
    assert (f"<planning:{budget}:{min(6, budget // 2)}>" in prompt) == enabled
    assert json.loads(sandbox.read_output("tuning/selected_params.json")) == {"values": {"X": 3, "Y": 4}}
    assert sandbox.read_output("tuning/trials.csv") == analyst.trials_csv
    assert not sandbox.exists("analysis/conditional_responses.md")


@pytest.mark.parametrize("preliminary", [False, True])
@pytest.mark.parametrize("response", [None, '{"axis":"X","delta_j":-0.5}'])
def test_preliminary_report_route_when_declared(
    analyst: AnalystInputs, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    preliminary: bool, response: str | None,
) -> None:
    # Given
    report = BottleneckReport(summary="pre-probe-fixture")
    request = RewriterInputs(analyst.task, analyst.candidate_source, report, [], analyst.device, 1,
        report_precedes_responses=preliminary, conditional_response_text=response, wall_text="wall-42")
    monkeypatch.setattr(modules, "PRELIMINARY_REPORT_GUIDANCE", "<preliminary-report>", raising=False)
    agent = StructureRewriterAgent.__new__(StructureRewriterAgent)
    sandbox = Sandbox(tmp_path)
    # When
    agent.seed_sandbox(request, sandbox)
    prompt = agent.render_prompt(request, sandbox)
    # Then
    assert ("<preliminary-report>" in prompt) == preliminary
    assert json.loads(sandbox.read_output("analysis/bottleneck.json")) == report.model_dump(mode="json")
    assert sandbox.read_output("analysis/resource_walls.md") == "wall-42"
    assert sandbox.exists("analysis/conditional_responses.md") == bool(response)
    if response:
        assert sandbox.read_output("analysis/conditional_responses.md") == response


def test_planning_excludes_response_first_context(
    analyst: AnalystInputs, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given
    request = replace(analyst, plan_probes=True, conditional_response_text="later-response")
    monkeypatch.setattr(method_prompts, "C2_OPPORTUNITY_GUIDANCE", "<response-first>")
    agent = BottleneckAnalystAgent.__new__(BottleneckAnalystAgent)
    sandbox = Sandbox(tmp_path)
    # When
    agent.seed_sandbox(request, sandbox)
    prompt = agent.render_prompt(request, sandbox)
    # Then
    assert "<response-first>" not in prompt
    assert not sandbox.exists("analysis/conditional_responses.md")
