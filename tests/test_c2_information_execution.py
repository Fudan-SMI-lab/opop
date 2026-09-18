"""Real legacy agents/native retune; only provider transport and GPU worker are faked."""

import json
import sys
from pathlib import Path

import pytest

from kernel_optimizer.agents.runtime import AgentCallError, OpencodeClient, OpencodeServer, PromptResult
from kernel_optimizer.control.orchestrator import Orchestrator
from kernel_optimizer.gpu.worker_client import WslGpuWorker
from kernel_optimizer.paramspace.materializer import extract_defaults
from scripts.experiments.c2_retune import retune
from tests.test_c2_information_inputs import information_module, prepared as existing_prepared

prepared = existing_prepared


@pytest.mark.parametrize(("group", "outcome"), [
    ("G0", "good"), ("G1", "good"), ("G2", "good"),
    ("G0", "witness"), ("G1", "witness"), ("G2", "witness"),
    ("G1", "generation"), ("G1", "unsupported"), ("G1", "slow"),
    ("G1", "final_fail"), ("G0", "parent_fail"), ("G1", "no_best"),
])
def test_single_opportunity_uses_native_retune_and_tuning_only_selection(tmp_path, monkeypatch, prepared, group, outcome):
    # Given: actual common/acquisition schemas and identical bounded agent settings.
    module = information_module()
    shared, acquisition, cfg = prepared
    for name in ("analyst", "rewriter", "parameterizer"):
        getattr(cfg.agents, name).max_retries = 0
        getattr(cfg.agents, name).max_transport_retries = 0
    runtime_active = []
    roles = []
    native_calls = []

    def start(self):
        runtime_active.append(True)
        return "http://unused.invalid"

    def stop(self):
        runtime_active.pop()

    def session(self, directory, title):
        return title

    def prompt(self, session_id, text, **kwargs):
        assert runtime_active
        title = kwargs["schema"]["title"]
        roles.append(title)
        if outcome == "generation":
            raise AgentCallError("CPU fixture generation failure")
        directory = kwargs["directory"]
        if title == "RewriteResult":
            brief = directory / "analysis/conditional_responses.md"
            assert not (directory / "analysis/resource_walls.md").exists()
            if group == "G0":
                assert not brief.exists()
            else:
                responses = [json.loads(line) for line in brief.read_text().splitlines()]
                if group == "G1":
                    assert responses[0]["a"]["metrics"] == {} and responses[0]["a"]["detail"] is None
                else:
                    assert responses == [r.model_dump(mode="json") for r in acquisition.responses]
        source = "import triton\nPARAMS={'z': 7}\n@triton.jit\ndef work(): return 2\n"
        if outcome == "unsupported" and title == "ParameterizationResult":
            source = "import triton\n@triton.jit\ndef work(): return 2\n"
        (directory / "child.py").write_text(source)
        answers = {
            "BottleneckReport": {"summary": "CPU fixture", "hypotheses": []},
            "RewriteResult": {"candidates": [{"file": "child.py", "backend": "triton",
                                               "hypothesis_id": "H1", "change_summary": "fixture"}]},
            "ParameterizationResult": {"file": "child.py", "space": {
                "params": [{"name": "z", "kind": "int", "choices": list(range(100))}], "constraints": []}},
        }
        return PromptResult(text="", structured=answers[title], session_id=session_id, cost=0.1)

    def worker(self, job, timeout_s, tag, **kwargs):
        assert bool(runtime_active) == ("retune" in self.jobs_dir.parts)
        if job["job_type"] == "static_check":
            return {"ok": True}
        if job["job_type"] == "compile_probe":
            paths = [job["kernel_src_path"], *job.get("extra_kernel_src_paths", [])]
            return {"ok": True, "results": {p: {"ok": True, "max_shared": 0} for p in paths}}
        assert job["seed"] == 0
        params = extract_defaults(Path(job["kernel_src_path"]).read_text())
        child = "z" in params
        full = job.get("num_perf_trials") == 100
        if child and outcome == "no_best":
            if "-wit-" in tag:
                return {"ok": True}
            return {"ok": False, "failure_kind": "correctness_mismatch", "log_tail": "CPU fixture no valid trial"}
        if (child and outcome == "witness") or (child and full and outcome == "final_fail") or outcome == "parent_fail":
            return {"ok": False, "failure_kind": "correctness_mismatch", "log_tail": "CPU fixture invalid output"}
        value = 2.0
        if child:
            value = (6.0 if outcome == "slow" else 4.0) if params["z"] == 7 else 10.0
            if full:
                value = 1.0 if outcome == "slow" else 99.0
        return {"ok": True, "latency_ms": {"mean": value, "median": value, "min": value, "max": value,
                "std": 0.0, "n": job["num_perf_trials"], "samples": [value] * job["num_perf_trials"]}}

    def profile(frame, event, arg):
        if event == "call" and frame.f_code is retune.__code__:
            assert runtime_active and frame.f_locals["runtime"].client is not None
            spec = frame.f_locals["inputs"]
            assert spec.sampler_seed == spec.evaluation_seed == 0
            assert extract_defaults(spec.source.read_text()) == {"z": 7}
            assert json.loads(spec.space.read_text())["domains"][0]["choices"] == list(range(100))
            native_calls.append("retune")
        if event == "call" and frame.f_code is Orchestrator._tune.__code__:
            native_calls.append("tune")

    monkeypatch.setattr(OpencodeServer, "start", start)
    monkeypatch.setattr(OpencodeServer, "stop", stop)
    monkeypatch.setattr(OpencodeClient, "create_session", session)
    monkeypatch.setattr(OpencodeClient, "prompt", prompt)
    monkeypatch.setattr(WslGpuWorker, "run_job", worker)
    previous_profile = sys.getprofile()
    # When: one group runs the existing generation and native retune paths.
    try:
        sys.setprofile(profile)
        result = module.run_information(shared, acquisition, module.InformationRun(cfg, group, tmp_path / "run"))
    finally:
        sys.setprofile(previous_profile)
    # Then: failures are outcomes; final measurements cannot change tuning-vs-parent selection.
    assert not runtime_active and len(result.parent_finals) == 3
    assert result.parent_baseline_ms == 5.0
    assert result.selected == ("child" if outcome in {"good", "final_fail"} else "parent")
    assert result.acquisition_costs == acquisition.costs
    assert result.parent_costs.worker_attempts == 3
    if outcome in {"generation", "unsupported", "parent_fail"}:
        assert native_calls == [] and result.error and result.child is None
    elif outcome == "witness":
        assert native_calls == ["retune"] and result.child.status == "rejected" and result.error
    elif outcome == "no_best":
        assert native_calls == ["retune", "tune"] and result.child.status == "no_best" and result.error
        assert len(result.child.trials) == 40 and result.child.finals == []
    else:
        assert native_calls == ["retune", "tune"] and len(result.child.trials) == 40
        assert len(result.child.finals) == 3
    if outcome not in {"generation", "parent_fail"}:
        assert roles == ["BottleneckReport", "RewriteResult", "ParameterizationResult"]
        assert result.generation_costs.agent_calls == 3
    assert (tmp_path / "run/result.json").is_file()


def test_cli_help_is_available_without_runtime_calls():
    module = information_module()
    with pytest.raises(SystemExit) as exc:
        module.main(["--help"])
    assert exc.value.code == 0
