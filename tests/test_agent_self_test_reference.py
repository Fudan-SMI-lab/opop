"""Optional reference resolution through real agent seeding, without GPU or provider calls."""

import hashlib
import json
from pathlib import Path

import pytest

from kernel_optimizer.agents.modules import (
    AnalystInputs, BottleneckAnalystAgent, RewriterInputs, StructureRewriterAgent,
)
from kernel_optimizer.agents.runtime import OpencodeClient, PromptResult
from kernel_optimizer.agents.sandbox import Sandbox, SandboxFactory
from kernel_optimizer.agents.self_test_context import SelfTestContext, self_test_context_factory
from kernel_optimizer.config import AgentModuleConfig, EvalConfig, GpuConcurrencyConfig, WslConfig
from kernel_optimizer.evaluation.correctness import CorrectnessEvaluator
from kernel_optimizer.gpu.worker_client import WslGpuWorker
from kernel_optimizer.models.core import DeviceLimits, TaskSpec
from kernel_optimizer.models.reports import BottleneckReport, TuningStats
from kernel_optimizer.store.run_store import RunStore


@pytest.fixture
def context_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[CorrectnessEvaluator, Sandbox]:
    worker = WslGpuWorker(WslConfig(venv="/configured/worker"), GpuConcurrencyConfig(), tmp_path / "jobs")
    evaluator = CorrectnessEvaluator(worker, EvalConfig(fp64_relative_gate=True), worker.conc, seed=0)
    sb = Sandbox(tmp_path / "sandbox")
    sb.write_input("rewrites/child.py", "import triton\nPARAMS = {'TILE': 8}\n@triton.jit\ndef child(): return 3\n")
    monkeypatch.setattr("kernel_optimizer.agents.self_test_context.to_wsl_path", str)
    monkeypatch.setattr(SandboxFactory, "create", lambda self, call_id: sb)
    monkeypatch.setattr(OpencodeClient, "create_session", lambda *args, **kwargs: "fake-session")
    monkeypatch.setattr(OpencodeClient, "prompt", lambda *args, **kwargs: PromptResult(
        session_id="fake-session", text="", structured={"summary": "fixture", "candidates": [
            {"file": "rewrites/child.py", "hypothesis_id": "H1", "change_summary": "fixture"}]}))
    return evaluator, sb


@pytest.mark.parametrize("rewrite", [False, True])
@pytest.mark.parametrize("decoy", [False, True])
@pytest.mark.parametrize("placeholder_hash", [False, True])
def test_payload_reference_when_configured_relative_path_is_unrelated(
    context_env: tuple[CorrectnessEvaluator, Sandbox], monkeypatch: pytest.MonkeyPatch,
    rewrite: bool, decoy: bool, placeholder_hash: bool,
) -> None:
    # Given typed inputs carrying the reference, and a wrong cwd with an optional decoy.
    evaluator, sb = context_env
    cwd = sb.root.parent / "other-opportunity"
    cwd.mkdir()
    if decoy:
        (cwd / "reference.py").write_text("SHAPE = (999,)\n")
    monkeypatch.chdir(cwd)
    payload = "from helper import N\nSHAPE = (N,)\n"
    sb.write_input("task/helper.py", "N = 13\n")
    task = TaskSpec(level=3, problem_id=21, name="typed", ref_path=Path("reference.py"),
                    ref_src_sha="old" if placeholder_hash else hashlib.sha256(payload.encode()).hexdigest())
    client = OpencodeClient.__new__(OpencodeClient)
    common = (client, SandboxFactory(sb.root.parent), RunStore(sb.root.parent), AgentModuleConfig())
    hook = self_test_context_factory(task, evaluator)
    if rewrite:
        agent = StructureRewriterAgent(*common, self_test_context=hook)
        inputs = RewriterInputs(task, "PARAMS={'TILE': 8}\n", BottleneckReport(summary="fixture"),
                                [], DeviceLimits(), 1, reference_source=payload)
        # When the real module seeds its explicit payload before the optional hook.
        agent.invoke(inputs)
    else:
        analyst = BottleneckAnalystAgent(*common, self_test_context=hook)
        inputs = AnalystInputs(task, "PARAMS={'TILE': 8}\n", TuningStats(
            candidate_id="candidate", space_id="space", n_complete=0, n_fail=0),
            "", DeviceLimits(), reference_source=payload)
        analyst.invoke(inputs)
    # Then the executable context identifies the actual payload and preserves its dependency.
    context = SelfTestContext.model_validate_json(sb.read_output("task/self_test.json"))
    assert context.task.ref_path.read_text() == payload
    assert context.task.ref_src_sha == hashlib.sha256(context.task.ref_path.read_bytes()).hexdigest()
    assert (context.reference_dependencies / "helper.py").read_text() == "N = 13\n"
    assert context.seed == 0 and context.configured_worker_python == "/configured/worker/bin/python"
    assert context.evaluation.fp64_relative_gate


def test_agent_returns_output_when_both_references_are_missing(
    context_env: tuple[CorrectnessEvaluator, Sandbox], monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given a legacy caller without an on-disk or input-payload reference.
    evaluator, sb = context_env
    monkeypatch.chdir(sb.root.parent)
    task = TaskSpec(level=3, problem_id=21, name="legacy", ref_path=Path("absent.py"), ref_src_sha="old")
    agent = BottleneckAnalystAgent(OpencodeClient.__new__(OpencodeClient), SandboxFactory(sb.root.parent),
        RunStore(sb.root.parent), AgentModuleConfig(), self_test_context=self_test_context_factory(task, evaluator))
    inputs = AnalystInputs(task, "PARAMS={'TILE': 8}\n", TuningStats(
        candidate_id="candidate", space_id="space", n_complete=0, n_fail=0), "", DeviceLimits())
    # When the actual agent invokes its optional hook, then output is still accepted.
    result = agent.invoke(inputs)
    assert result.output.summary == "fixture"
    assert json.loads(sb.read_output("task/self_test.json")) == {
        "label": "agent_self_test", "available": False, "reason": "reference_unavailable"}
    assert not sb.exists("task/self_test_usage.md")
    assert not sb.exists("task/self_test_reference.py")


def test_configured_reference_when_cwd_changes_after_factory(
    context_env: tuple[CorrectnessEvaluator, Sandbox], monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given a real configured relative reference, bind its origin before cwd changes.
    evaluator, sb = context_env
    monkeypatch.chdir(sb.root.parent)
    reference = sb.root.parent / "reference.py"
    reference.write_bytes(b"SHAPE = (13,)\n")
    task = TaskSpec(level=3, problem_id=21, name="bound", ref_path=Path("reference.py"),
                    ref_src_sha=hashlib.sha256(reference.read_bytes()).hexdigest())
    hook = self_test_context_factory(task, evaluator)
    other = sb.root.parent / "other"
    other.mkdir()
    (other / "reference.py").write_bytes(b"SHAPE = (999,)\n")
    monkeypatch.chdir(other)
    # When seeded, then the captured configured location wins, not the current directory.
    hook(sb)
    context = SelfTestContext.model_validate_json(sb.read_output("task/self_test.json"))
    assert context.task.ref_path.read_bytes() == reference.read_bytes()


def test_unavailable_when_authoritative_hash_matches_neither_reference(
    context_env: tuple[CorrectnessEvaluator, Sandbox],
) -> None:
    # Given two real files but neither matches the task's authoritative content hash.
    evaluator, sb = context_env
    reference = sb.root.parent / "wrong.py"
    reference.write_bytes(b"SHAPE = (999,)\n")
    sb.write_input("task/ref.py", "SHAPE = (888,)\n")
    task = TaskSpec(level=3, problem_id=21, name="mismatch", ref_path=reference,
                    ref_src_sha=hashlib.sha256(b"SHAPE = (13,)\n").hexdigest())
    # When selecting context, then neither wrong reference is advertised as usable.
    self_test_context_factory(task, evaluator)(sb)
    data = json.loads(sb.read_output("task/self_test.json"))
    assert data["available"] is False and data["reason"] == "reference_hash_mismatch"
    assert not sb.exists("task/self_test_usage.md")


def test_cli_rejects_changed_reference_after_context_was_seeded(
    context_env: tuple[CorrectnessEvaluator, Sandbox], capsys: pytest.CaptureFixture[str],
) -> None:
    from kernel_optimizer.agents.self_test import main
    # Given a usable snapshot that changes before an explicit formal invocation.
    evaluator, sb = context_env
    reference = sb.root.parent / "reference.py"
    reference.write_bytes(b"SHAPE = (13,)\n")
    task = TaskSpec(level=3, problem_id=21, name="changed", ref_path=reference, ref_src_sha="old")
    self_test_context_factory(task, evaluator)(sb)
    sb.write_input("task/self_test_reference.py", "SHAPE = (999,)\n")
    candidate = sb.write_input("candidate.py", "PARAMS = {'TILE': 8}\n")
    # When invoked, then it fails before any GPU dispatch with no fabricated score.
    code = main(["--context", str(sb.root / "task/self_test.json"), "--candidate", str(candidate),
                 "--mode", "full", "--output", str(sb.root / "result")])
    assert code == 1
    result = json.loads(capsys.readouterr().out)
    assert result["result_valid"] is False and result["score_ms"] is None
    assert result["error"].startswith("REFERENCE_CHANGED:")


def test_reseed_removes_executable_usage_when_reference_becomes_unavailable(
    context_env: tuple[CorrectnessEvaluator, Sandbox],
) -> None:
    # Given an earlier usable helper in the same sandbox, followed by removal of its source.
    evaluator, sb = context_env
    reference = sb.root.parent / "reference.py"
    reference.write_bytes(b"SHAPE = (13,)\n")
    task = TaskSpec(level=3, problem_id=21, name="reseed", ref_path=reference, ref_src_sha="old")
    hook = self_test_context_factory(task, evaluator)
    hook(sb)
    reference.unlink()
    # When the optional context is reseeded, then stale executable instructions are withdrawn.
    hook(sb)
    assert json.loads(sb.read_output("task/self_test.json"))["available"] is False
    assert not sb.exists("task/self_test_usage.md")
