"""CPU-only formal self-test contract; no GPU, model, Git, or network calls."""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from kernel_optimizer.agents.sandbox import Sandbox
from kernel_optimizer.config import EvalConfig, GpuConcurrencyConfig, WslConfig
from kernel_optimizer.evaluation.correctness import CorrectnessEvaluator
from kernel_optimizer.gpu.worker_client import WslGpuWorker
from kernel_optimizer.models.core import TaskSpec


@pytest.fixture
def seeded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    # Given a live evaluator whose correctness seed differs from the tuning seed.
    from kernel_optimizer.agents import self_test_context as context_module

    monkeypatch.setattr(context_module, "to_wsl_path", lambda p: str(Path(p).resolve()))
    ref = tmp_path / "original.py"
    ref.write_bytes(b"SHAPE = (7, 13)\n")
    task = TaskSpec(level=3, problem_id=21, name="fixture", ref_path=ref,
                    ref_src_sha="old-hash-is-not-authoritative")
    worker = WslGpuWorker(WslConfig(venv="/configured/worker", kernelbench_src="/kb/src"),
                          GpuConcurrencyConfig(timing_cooldown_s=0), tmp_path / "jobs")
    evaluator = CorrectnessEvaluator(worker, EvalConfig(), worker.conc, seed=2)
    hook = context_module.self_test_context_factory(task, evaluator)
    evaluator.seed = 0
    evaluator.cfg = EvalConfig(correctness_mode="dual_witness_relaxed", fp64_relative_gate=True)
    sb = Sandbox(tmp_path / "sandbox")
    sb.write_input("task/ref.py", "KEEP = True\n")
    sb.write_input("task/eval_semantics.md", "training=False")
    # When the optional hook is called after module seeding.
    hook(sb)
    # Then the original module reference is not replaced.
    assert sb.read_output("task/ref.py") == "KEEP = True\n"
    return sb.root / "task/self_test.json"


def test_context_when_live_evaluator_changes(seeded: Path) -> None:
    # Given a freshly seeded context; when parsed; then only evaluation data is exposed.
    data = json.loads(seeded.read_text(encoding="utf-8"))
    assert data["seed"] == 0
    assert data["evaluation"]["fp64_relative_gate"] is True
    assert data["configured_worker_python"] == "/configured/worker/bin/python"
    assert data["wsl"]["kernelbench_src"] == "/kb/src"
    assert data["task"]["ref_src_sha"] == hashlib.sha256(b"SHAPE = (7, 13)\n").hexdigest()
    assert data["eval_semantics"] == "training=False"
    assert set(data) == {"label", "task", "evaluation", "wsl", "concurrency", "seed",
                         "jobs_dir", "worker_main_path", "configured_worker_python",
                         "source_path", "input_mode", "eval_semantics", "reference_dependencies"}


FAKE_WORKER = '''
import argparse, json, runpy, sys
from pathlib import Path
p = argparse.ArgumentParser()
p.add_argument('--job'); p.add_argument('--out')
a = p.parse_args()
j = json.loads(Path(a.job).read_text())
if j['job_type'] == 'static_check':
    r = {'ok': True}
else:
    assert j['job_type'] == 'eval_correctness_relaxed'
    assert j['fp64_relative_gate'] is True and j['seed'] == 0
    source = Path(j['kernel_src_path'])
    sys.path.insert(0, str(source.parent))
    values = runpy.run_path(str(source))['PARAMS']
    correct = values['TILE'] != 99
    latency = {98: float('nan'), 97: 0., 96: -2.}.get(values['TILE'], 2.)
    r = {'ok': correct, 'compiled': True, 'correct': correct,
         'latency_ms': {'mean': latency, 'std': 0., 'min': latency, 'max': latency, 'n': j['num_perf_trials']},
         'fp64_gate_enabled': True, 'fp64_rescued_trials': 1,
         'observed_job': j, 'observed_params': values}
Path(a.out).write_text(json.dumps(r))
'''


@pytest.mark.parametrize(("mode", "perf", "correct", "tile", "exit_code"), [
    ("quick", 20, 3, 8, 0), ("full", 100, 5, 16, 0), ("full", 100, 5, 99, 1),
    ("quick", 20, 3, 98, 1), ("quick", 20, 3, 97, 1), ("quick", 20, 3, 96, 1),
])
def test_cli_when_formal_worker_returns(seeded: Path, mode: str, perf: int,
                                       correct: int, tile: int, exit_code: int) -> None:
    # Given local sibling dependencies and an unrelated file that must not be staged.
    root = seeded.parent.parent
    candidate = root / "candidate/kernel.py"
    candidate.parent.mkdir()
    candidate.write_text("from helper import VALUE\nPARAMS = {'TILE': 8, 'dtype': 'fp32'}\n",
                         encoding="utf-8")
    (candidate.parent / "helper.py").write_text("VALUE = 1\n", encoding="utf-8")
    (candidate.parent / "unrelated.py").write_text("raise RuntimeError('not a dependency')\n")
    params = root / "params.json"
    params.write_text(json.dumps({"values": {"TILE": tile, "dtype": "bf16"}}))
    fake = root / "fake_worker.py"
    fake.write_text(FAKE_WORKER, encoding="utf-8")
    output = root / "result"
    bootstrap = '''
import runpy, sys
from kernel_optimizer.gpu import worker_client
worker_client._wsl_hop_needed = lambda: False
original = worker_client.WslGpuWorker._build_command
fake_worker = sys.argv[1]
def command(self, job, out):
    argv, env = original(self, job, out)
    assert argv[0] == '/configured/worker/bin/python'
    return [sys.executable, fake_worker, '--job', str(job), '--out', str(out)], env
worker_client.WslGpuWorker._build_command = command
sys.argv = ['self_test', *sys.argv[2:]]
runpy.run_module('kernel_optimizer.agents.self_test', run_name='__main__', alter_sys=True)
'''
    env = {**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src")}
    # When the real CLI executes with only worker process launch replaced.
    proc = subprocess.run([sys.executable, "-c", bootstrap, str(fake), "--context", str(seeded),
                           "--candidate", str(candidate), "--params", str(params),
                           "--mode", mode, "--output", str(output)],
                          env=env, capture_output=True, text=True, timeout=30)
    # Then the formal job contract and validity, not an echo, determine success.
    assert proc.returncode == exit_code, proc.stderr
    result = json.loads((output / "result.json").read_text())
    raw = json.loads((output / "raw_worker.json").read_text())
    assert raw["observed_job"]["num_perf_trials"] == perf
    assert raw["observed_job"]["num_correct_trials"] == correct
    assert raw["observed_params"] == {"TILE": tile, "dtype": "bf16"}
    assert result["result_valid"] is (exit_code == 0)
    assert result["score_ms"] == (2.0 if exit_code == 0 else None)
    assert result["fp64_rescued_trials"] == 1
    assert result["label"] == "agent_self_test"
    commands = [json.loads(line) for line in (output / "commands.jsonl").read_text().splitlines()]
    assert len(commands) == 2
    assert Path(commands[0]["job"]).parent == Path(json.loads(seeded.read_text())["jobs_dir"])
    assert set(commands[0]["environment"]) == {"PYTHONPATH", "TRITON_CACHE_DIR"}
    assert not list((output / "source").rglob("unrelated.py"))
    staged = Path(raw["observed_job"]["kernel_src_path"])
    assert staged != candidate
    assert hashlib.sha256(staged.read_bytes()).hexdigest() == result["materialized_source_sha256"]


def test_defaults_when_params_omitted(seeded: Path) -> None:
    from kernel_optimizer.agents.self_test_artifacts import stage_candidate
    # Given native string-valued defaults, when staged, then retain their exact keys and values.
    candidate = seeded.parent.parent / "native.py"
    candidate.write_text("PARAMS = {'BLOCK_M': 32, 'dtype': 'tf32'}\n")
    path, params = stage_candidate(candidate, seeded.parent.parent / "staged", None)
    assert params.values == {"BLOCK_M": 32, "dtype": "tf32"}
    assert path.read_bytes() == candidate.read_bytes()


def test_mapping_when_foreign_keys_supplied(seeded: Path) -> None:
    from kernel_optimizer.agents.self_test_artifacts import stage_candidate
    from kernel_optimizer.models.core import ParamSet
    from kernel_optimizer.paramspace.materializer import MaterializeError
    # Given a foreign key, when materialized, then reject rather than guess a mapping.
    candidate = seeded.parent.parent / "native.py"
    candidate.write_text("PARAMS = {'BLOCK_M': 32}\n")
    with pytest.raises(MaterializeError, match="KEY_MISMATCH"):
        stage_candidate(candidate, seeded.parent.parent / "staged", ParamSet(values={"TILE": 32}))


@pytest.mark.parametrize("configured", [False, True])
def test_agent_when_optional_hook_is_configured(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                               configured: bool) -> None:
    from kernel_optimizer.agents.base import AgentModule
    from kernel_optimizer.agents.runtime import OpencodeClient, PromptResult
    from kernel_optimizer.agents.sandbox import SandboxFactory
    from kernel_optimizer.config import AgentModuleConfig
    from kernel_optimizer.models.core import ParamSet
    from kernel_optimizer.store.run_store import RunStore

    # Given an existing constructor and a sandbox factory with Git creation bypassed.
    class Module(AgentModule[str, ParamSet]):
        output_model = ParamSet

        def seed_sandbox(self, inputs: str, sb: Sandbox) -> None:
            sb.write_input("seeded", inputs)

        def render_prompt(self, inputs: str, sb: Sandbox) -> str:
            assert sb.exists("context") is configured
            return inputs

    sb = Sandbox(tmp_path / "sandbox")
    monkeypatch.setattr(SandboxFactory, "create", lambda self, call_id: sb)
    client = OpencodeClient.__new__(OpencodeClient)
    monkeypatch.setattr(client, "create_session", lambda *a, **kw: "fake-session")
    monkeypatch.setattr(client, "prompt", lambda *a, **kw: PromptResult(
        session_id="fake-session", text="", structured={"values": {"X": 1}}, tokens={}, cost=0))
    module = Module(client, SandboxFactory(tmp_path), RunStore(tmp_path), AgentModuleConfig())

    def hook(sandbox: Sandbox) -> str:
        assert sandbox.exists("seeded")
        sandbox.write_input("context", "{}")
        return ""

    if configured:
        module.self_test_context = hook
    # When invoked, then the hook runs after seed and before render, or remains absent.
    outcome = module.invoke("fixture")
    assert outcome.output.values == {"X": 1}
    assert sb.exists("context") is configured


def test_cli_when_parameter_mapping_fails(seeded: Path, capsys: pytest.CaptureFixture[str]) -> None:
    from kernel_optimizer.agents.self_test import main
    # Given a foreign parameter key, when CLI parsing succeeds, then preserve a failed result artifact.
    candidate = seeded.parent.parent / "candidate.py"
    candidate.write_text("PARAMS = {'BLOCK': 8}\n")
    params = candidate.with_suffix(".json")
    params.write_text('{"values": {"TILE": 8}}')
    output = seeded.parent.parent / "failure"
    code = main(["--context", str(seeded), "--candidate", str(candidate), "--params", str(params),
                 "--mode", "full", "--output", str(output)])
    assert code == 1
    assert json.loads(capsys.readouterr().out)["score_ms"] is None
    assert json.loads((output / "result.json").read_text())["result_valid"] is False


def test_reference_when_importing_sibling_package(tmp_path: Path) -> None:
    from kernel_optimizer.agents.self_test_artifacts import copy_dependencies
    # Given nested imports and an unrelated probe, when copied, then preserve only reachable layout.
    source = tmp_path / "inputs"
    (source / "helpers").mkdir(parents=True)
    ref = source / "reference.py"
    ref.write_text("from helpers.shape import SHAPE\n")
    (source / "helpers/__init__.py").write_text("")
    (source / "helpers/shape.py").write_text("from .sizes import N\nSHAPE = (N,)\n")
    (source / "helpers/sizes.py").write_text("N = 13\n")
    (source / "probe.py").write_text("raise RuntimeError('unrelated')\n")
    copied = copy_dependencies(ref, tmp_path / "snapshot")
    assert copied.read_bytes() == ref.read_bytes()
    assert {p.relative_to(copied.parent).as_posix() for p in copied.parent.rglob("*.py")} == {
        "reference.py", "helpers/__init__.py", "helpers/shape.py", "helpers/sizes.py"}


def test_wiring_when_parameterizer_has_no_reference(tmp_path: Path) -> None:
    from kernel_optimizer.agents.runtime import OpencodeClient
    from kernel_optimizer.config import AppConfig
    from kernel_optimizer.store.run_store import RunStore
    from kernel_optimizer.wiring import Runtime, build_orchestrator
    # Given a provider-bearing configuration, when seeded by the wired hook, then expose only evaluation.
    cfg = AppConfig()
    cfg.opencode.sandbox_extra_config = {"provider": {"secret": {"apiKey": "do-not-copy"}}}
    ref = tmp_path / "ref.py"
    ref.write_text("SHAPE = (13,)\n")
    task = TaskSpec(level=3, problem_id=21, name="fixture", ref_path=ref, ref_src_sha="old")
    runtime = Runtime.__new__(Runtime)
    runtime.client = OpencodeClient.__new__(OpencodeClient)
    orchestrator = build_orchestrator(cfg, RunStore(tmp_path), task, runtime)
    deps = orchestrator.deps
    assert all(agent.self_test_context is not None for agent in (
        deps.generator, deps.parameterizer, deps.rewriter, deps.repair, deps.novelty, deps.analyst))
    deps.evaluator.seed = 17
    sb = Sandbox(tmp_path / "parameterizer")
    assert deps.parameterizer.self_test_context is not None
    deps.parameterizer.self_test_context(sb)
    data = json.loads(sb.read_output("task/self_test.json"))
    assert data["seed"] == 17
    assert sb.exists("task/self_test_reference.py") and not sb.exists("task/ref.py")
    assert "do-not-copy" not in sb.read_output("task/self_test.json")
