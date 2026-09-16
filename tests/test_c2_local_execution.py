"""Only provider/worker are faked: these tests are NOT GPU performance evidence."""

from pathlib import Path
import json
import os

import pytest

from kernel_optimizer.agents.runtime import AgentCallError, OpencodeClient, OpencodeServer, PromptResult
from kernel_optimizer.config import AppConfig
from kernel_optimizer.gpu.worker_client import WslGpuWorker
from kernel_optimizer.paramspace.materializer import extract_defaults
from kernel_optimizer.store.run_store import RunStore
from kernel_optimizer.wiring import Runtime
from test_c2_local_inputs import runner_module


@pytest.fixture
def shared():
    inputs = runner_module("inputs")
    return inputs.Shared.model_validate({
        "task": "level3:43", "state": "off", "source": "PARAMS={'x': 77, 'fixed': 3}\ndef work(): return 1\n",
        "reference_source": "def reference(): return 1\n", "parent": {
            "trial_id": "p", "candidate_id": "parent", "space_id": "s",
            "params": {"values": {"x": 77, "fixed": 3}}, "status": "complete",
            "latency_ms": {"mean": 5, "std": 0, "min": 5, "max": 5, "n_samples": 20}},
        "space": {"params": [{"name": "x", "kind": "int", "choices": list(range(200))},
                              {"name": "fixed", "kind": "int", "choices": [3]}]},
        "trials": [], "semantics": {"training": True}, "device": {},
    })


@pytest.mark.parametrize("path", ["direct", "legacy"])
@pytest.mark.parametrize("failure", ["none", "generation", "invalid", "smoke"])
@pytest.mark.parametrize("arm", ["A", "B"])
def test_one_child_b40_and_frozen_selection_when_final_reverses_ranking(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, shared, path: str, failure: str, arm: str,
):
    # Given: real agents/tuners/evaluators with only provider transport and GPU worker replaced.
    execution = runner_module("execution")
    inputs = runner_module("inputs")
    cfg = AppConfig()
    cfg.agents.rewriter.max_retries = 0
    cfg.agents.rewriter.max_transport_retries = 0
    cfg.agents.parameterizer.max_retries = 0
    cfg.agents.analyst.max_retries = 0
    shared = shared.model_copy(update={"protocol_id": inputs.protocol(cfg)})
    calls = []
    jobs = []

    def session(self, directory, title):
        return title

    def prompt(self, session_id, text, **kwargs):
        directory = kwargs["directory"]
        title = kwargs["schema"]["title"]
        calls.append(title)
        if title == "TaskRewriteResult":
            seeded = json.loads((directory / "analysis" / "task_response.json").read_text())
            assert seeded["project_root"] == "." and seeded["candidate_path"] == "parent.py"
            assert bool(seeded["responses"]) == (arm == "B")
            assert (directory / "project" / "ordinary.json").is_file()
        if failure == "generation":
            raise AgentCallError("CPU fixture provider failed")
        source = "import triton\nPARAMS={'new_key': 1}\n@triton.jit\ndef work(): return 2\n"
        space = {"params": [{"name": "new_key", "kind": "int", "choices": list(range(200))}]}
        (directory / "child.py").write_text(source)
        answers = {
            "TaskRewriteResult": {"candidate_file": "child.py", "space": space},
            "BottleneckReport": {"summary": "CPU fixture", "hypotheses": []},
            "RewriteResult": {"candidates": [{"file": "child.py", "backend": "triton",
                                               "hypothesis_id": "H1", "change_summary": "fixture"}]},
            "ParameterizationResult": {"file": "child.py", "space": space},
        }
        return PromptResult(text="", structured=answers[title], cost=0.1, tokens={"input": 10}, session_id=session_id)

    def worker(self, job, timeout_s, tag, **kwargs):
        jobs.append(job)
        if job["job_type"] == "static_check":
            return {"ok": True}
        candidate = Path(job["kernel_src_path"])
        params = extract_defaults(candidate.read_text())
        is_child = "new_key" in params
        full = job.get("num_perf_trials") == 100
        value = (50.0 if full else 1.0) if is_child else 5.0
        if (is_child and failure == "invalid") or failure == "smoke":
            return {"ok": False, "failure_kind": "correctness_mismatch", "log_tail": "fixture"}
        return {"ok": True, "latency_ms": {"mean": value, "median": value, "min": value,
                "max": value, "std": 0, "n": job.get("num_perf_trials", 20),
                "samples": [value] * job.get("num_perf_trials", 20)}}

    monkeypatch.setattr(OpencodeClient, "create_session", session)
    monkeypatch.setattr(OpencodeClient, "prompt", prompt)
    monkeypatch.setattr(WslGpuWorker, "run_job", worker)
    runtime = Runtime(cfg)
    runtime.client = OpencodeClient("http://unused.invalid")
    store = RunStore.create(tmp_path, "arm", {})
    acquisition = None
    if arm == "B":
        acquisition = execution.acquire(shared, (cfg, RunStore.create(tmp_path, "acquisition", {})), 2)
        assert acquisition.probe_calls == 2
        assert acquisition.responses[0].a_params.values["fixed"] == 3
        assert acquisition.responses[0].b_params.values["fixed"] == 3
    # When: exactly one structural opportunity runs, then frozen artifacts are remeasured.
    try:
        result = execution.run_arm(shared, inputs.RunOptions(arm=arm, path=path, acquisition=acquisition),
                                   execution.Services(cfg, store, runtime))
    finally:
        runtime.client.close()
    # Then: failures remain outcomes, and the deliberately worse final child cannot rerank.
    if failure in {"generation", "smoke"}:
        assert result.selected == "parent" and result.error
        assert result.trials == []
    else:
        assert len(result.trials) == 40, result.error
        assert all(set(t.params.values) == {"new_key"} for t in result.trials)
        assert result.selected == ("parent" if failure == "invalid" else "child")
    assert len(result.fresh_parent) == (0 if failure == "smoke" else 3)
    assert len(result.fresh_child) == (3 if failure == "none" else 0)
    if result.child_best:
        assert all(t.params == result.child_best.params for t in result.fresh_child)
    assert calls.count("TaskRewriteResult") + calls.count("RewriteResult") <= 1
    assert all(j["num_perf_trials"] == 100 for j in jobs if j.get("measure_launch_overhead"))
    events = store.iter_events()
    assert len([e for e in events if e.type == "TRIAL_DONE"]) == len(result.trials)
    assert not any(e.type.startswith("SCAN_") for e in events)
    assert result.costs.worker_attempts == len([e for e in events if e.type == "LOCAL_EVAL_STARTED"])
    if failure not in {"generation", "smoke"}:
        assert result.costs.agent_cost == pytest.approx(0.1 * len(calls))


def test_cli_help_is_cpu_only_when_invoked_as_module():
    # Given / When / Then: argparse construction requires no provider or GPU.
    cli = runner_module("runner")
    with pytest.raises(SystemExit) as exc:
        cli.main(["--help"])
    assert exc.value.code == 0


def test_runtime_state_is_separate_when_config_is_shared(tmp_path: Path):
    # Given: one unchanged operator config.
    cli = runner_module("runner")
    cfg = AppConfig()
    original = cfg.model_dump()
    # When: runtime state is derived independently for A and B.
    a = cli.isolated_config(cfg, tmp_path / "A")
    b = cli.isolated_config(cfg, tmp_path / "B")
    # Then: mutable state is distinct, while original config and agent policy remain equal.
    assert a.wsl.triton_cache_dir != b.wsl.triton_cache_dir
    assert a.opencode.server_env["XDG_DATA_HOME"] != b.opencode.server_env["XDG_DATA_HOME"]
    assert a.agents == b.agents
    assert cfg.model_dump() == original


def test_worker_cache_environment_is_restored_when_local_scope_ends(tmp_path: Path):
    # Given: a process environment which must not become shared mutable arm state.
    cli = runner_module("runner")
    cfg = cli.isolated_config(AppConfig(), tmp_path)
    original = os.environ.get("TORCH_EXTENSIONS_DIR")
    # When: worker subprocesses inherit the per-run cache environment.
    with cli.worker_environment(cfg):
        assert os.environ["TORCH_EXTENSIONS_DIR"] == str(tmp_path / "cache" / "torch")
    # Then: the orchestrator's previous environment is restored.
    assert os.environ.get("TORCH_EXTENSIONS_DIR") == original


def test_cli_preserves_outcome_when_provider_startup_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, shared):
    # Given: valid prepared inputs, but the external provider server cannot start.
    cli = runner_module("runner")
    inputs = runner_module("inputs")
    cfg = AppConfig()
    config_path = tmp_path / "config.yaml"
    config_path.write_text(cfg.model_dump_json())
    shared_path = tmp_path / "shared.json"
    prepared = shared.model_copy(update={"evaluation": cfg.evaluation.model_dump(mode="json"),
                                          "protocol_id": inputs.protocol(cfg)})
    shared_path.write_text(prepared.model_dump_json())

    def unavailable(self):
        raise AgentCallError("CPU fixture provider startup failure")

    monkeypatch.setattr(OpencodeServer, "start", unavailable)
    # When: the real CLI enters its provider boundary, before any GPU work.
    code = cli.main(["--config", str(config_path), "run-arm", "--shared", str(shared_path),
                     "--run-dir", str(tmp_path / "run"), "--arm", "A", "--path", "direct"])
    # Then: it exits unsuccessfully and retains a terminal arm outcome.
    assert code == 1
    result = json.loads((tmp_path / "run" / "result.json").read_text())
    assert result["error"] and result["costs"]["worker_attempts"] == 0
