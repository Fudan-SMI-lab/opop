"""Expansion context delivery uses real models and sandbox files, not prose pins."""

import json
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import pytest

from kernel_optimizer.agents import modules
from kernel_optimizer.agents.modules import ParameterizerAgent, ParameterizerInputs
from kernel_optimizer.agents.runtime import OpencodeClient, PromptResult
from kernel_optimizer.agents.sandbox import Sandbox, SandboxFactory
from kernel_optimizer.config import AgentModuleConfig
from kernel_optimizer.models.core import DeviceLimits, LatencyStats, ParamSet, ProfileRecord, TaskSpec, TrialRecord
from kernel_optimizer.models.reports import FailureCluster, ParamStat, TuningStats
from kernel_optimizer.store.run_store import RunStore


@pytest.fixture
def inputs() -> ParameterizerInputs:
    task = TaskSpec(level=1, problem_id=1, name="fixture", ref_path=Path("missing-ref.py"),
                    ref_src_sha="0" * 64)
    return ParameterizerInputs(task, "PARAMS = {'X': 1, 'Y': 2}\n", DeviceLimits(),
                               expand_directive="X: 2 -> 3", prior_constraints=(("X <= Y", "fixture"),))


def test_legacy_expansion_when_evidence_absent(inputs: ParameterizerInputs, tmp_path: Path) -> None:
    # Given
    agent = ParameterizerAgent.__new__(ParameterizerAgent)
    sandbox = Sandbox(tmp_path)
    # When
    agent.seed_sandbox(inputs, sandbox)
    prompt = agent.render_prompt(inputs, sandbox)
    # Then
    assert sandbox.read_output("candidate/source.py") == inputs.candidate_source
    for path in ("tuning/selected_trial.json", "candidate/selected.py", "tuning/stats.json",
                 "tuning/trials.json", "task/ref.py"):
        assert not sandbox.exists(path)
        assert path not in prompt
    assert inputs.expand_directive in prompt
    assert inputs.selected_trial is inputs.selected_source is inputs.expansion_stats is None
    assert inputs.expansion_trials == () and inputs.reference_source is None


@pytest.fixture
def selected() -> TrialRecord:
    return TrialRecord(trial_id="old-winner", candidate_id="c", space_id="initial-space",
        params=ParamSet(values={"X": 3, "Y": 4}), status="complete",
        latency_ms=LatencyStats(mean=1.5, median=1.4, std=0.1, min=1.3, max=1.7, n_samples=20),
        profile=ProfileRecord(n_regs=93, n_spills=2, shared_bytes=8192), fp64_rescued_trials=1)


@pytest.mark.parametrize("expansion", ["", "X: 3 -> 5"])
def test_actual_evidence_delivery_when_supplied(
    inputs: ParameterizerInputs, selected: TrialRecord, tmp_path: Path, expansion: str,
) -> None:
    # Given
    failed = TrialRecord(trial_id="new-failure", candidate_id="c", space_id="expanded-space",
        params=ParamSet(values={"X": 5, "Y": 4}), status="fail", failure_kind="compile_error",
        failure_detail="compiler-refusal-42", job_wall_s=2.25)
    stats = TuningStats(candidate_id="c", space_id="expanded-space", n_complete=1, n_fail=1,
        best=selected, param_stats=[ParamStat(name="X", best_value=3, at_boundary=False,
            latency_by_value={"1": 2.0, "3": 1.4}, failure_rate_by_value={"5": 1.0})],
        failure_clusters=[FailureCluster(param="X", value="5", failure_rate=1.0,
                                        dominant_kind="compile_error")])
    request = replace(inputs, expand_directive=expansion, selected_trial=selected,
        selected_source="PARAMS = {'X': 3, 'Y': 4}\n", expansion_stats=stats,
        expansion_trials=(selected, failed), reference_source="SHAPE = (17, 31)\n")
    agent = ParameterizerAgent.__new__(ParameterizerAgent)
    sandbox = Sandbox(tmp_path)
    # When
    agent.seed_sandbox(request, sandbox)
    prompt = agent.render_prompt(request, sandbox)
    # Then
    assert json.loads(sandbox.read_output("tuning/selected_trial.json")) == selected.model_dump(mode="json")
    assert json.loads(sandbox.read_output("tuning/stats.json")) == stats.model_dump(mode="json")
    assert json.loads(sandbox.read_output("tuning/trials.json")) == [
        selected.model_dump(mode="json"), failed.model_dump(mode="json")]
    assert sandbox.read_output("candidate/source.py") == inputs.candidate_source
    assert sandbox.read_output("candidate/selected.py") == request.selected_source
    assert sandbox.read_output("task/ref.py") == request.reference_source
    for path in ("tuning/selected_trial.json", "candidate/selected.py", "tuning/stats.json",
                 "tuning/trials.json", "task/ref.py"):
        assert path in prompt


def test_reference_only_when_other_evidence_unknown(inputs: ParameterizerInputs, tmp_path: Path) -> None:
    # Given
    request = replace(inputs, reference_source="REFERENCE = 73\n")
    agent = ParameterizerAgent.__new__(ParameterizerAgent)
    sandbox = Sandbox(tmp_path)
    # When
    agent.seed_sandbox(request, sandbox)
    prompt = agent.render_prompt(request, sandbox)
    # Then
    assert sandbox.read_output("task/ref.py") == request.reference_source
    for path in ("tuning/selected_trial.json", "candidate/selected.py", "tuning/stats.json", "tuning/trials.json"):
        assert not sandbox.exists(path)
        assert path not in prompt


def test_expansion_evidence_guidance_route(
    inputs: ParameterizerInputs, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given
    monkeypatch.setattr(modules, "EXPANSION_EVIDENCE_GUIDANCE", "<expansion-evidence>", raising=False)
    agent = ParameterizerAgent.__new__(ParameterizerAgent)
    # When
    prompt = agent.render_prompt(inputs, Sandbox(tmp_path))
    # Then
    assert "<expansion-evidence>" in prompt


@pytest.mark.parametrize("evidence", [False, True])
def test_single_parameterizer_call_when_evidence_optional(
    inputs: ParameterizerInputs, selected: TrialRecord, tmp_path: Path, evidence: bool,
) -> None:
    # Given
    request = replace(inputs, selected_trial=selected if evidence else None)
    sandbox = Sandbox(tmp_path / "sandbox")
    sandbox.write_input("candidate/parameterized.py", "import triton\nimport triton.language as tl\n"
        "PARAMS = {'X': 1, 'Y': 2}\n@triton.jit\ndef kernel(ptr):\n    tl.store(ptr, 1)\n")
    result = PromptResult(text="", session_id="offline", structured={
        "file": "candidate/parameterized.py", "space": {"params": [
            {"name": "X", "kind": "int", "choices": [1, 2]},
            {"name": "Y", "kind": "int", "choices": [2, 4]}]}})
    factory = SandboxFactory(tmp_path)
    client = OpencodeClient.__new__(OpencodeClient)
    agent = ParameterizerAgent(client, factory, RunStore(tmp_path), AgentModuleConfig())
    with patch.object(factory, "create", return_value=sandbox), \
         patch.object(client, "create_session", return_value="offline") as session, \
         patch.object(client, "prompt", return_value=result) as prompt:
        # When
        outcome = agent.invoke(request)
    # Then
    assert session.call_count == prompt.call_count == outcome.attempts == 1
    assert outcome.output.file == "candidate/parameterized.py"
