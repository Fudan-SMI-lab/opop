"""Native validation/_tune tests; only the GPU worker is a CPU fake."""

import importlib
import os
import sys
from pathlib import Path
from types import ModuleType

import pytest

from kernel_optimizer.agents.base import AgentModule
from kernel_optimizer.agents.runtime import OpencodeServer
from kernel_optimizer.config import AppConfig
from kernel_optimizer.control.orchestrator import Orchestrator
from kernel_optimizer.gpu.worker_client import WslGpuWorker
from kernel_optimizer.models.core import ParamDomain, ParameterSpace
from kernel_optimizer.paramspace.materializer import extract_defaults
from kernel_optimizer.paramspace.validation import SpaceValidator
from kernel_optimizer.store.run_store import RunStore


def retune_module():
    assert (Path(__file__).parents[1] / "scripts/experiments/c2_retune.py").is_file(), "retune entry is missing"
    return importlib.import_module("scripts.experiments.c2_retune")


@pytest.mark.parametrize(("sampler_seed", "outcome"), [(0, "complete"), (1, "complete"),
                                                     (2, "complete"), (0, "rejected"), (0, "no_best")])
@pytest.mark.parametrize("final_blocks", [0, 3])
def test_native_retune_preserves_witnesses_seeds_and_final_selection(tmp_path, monkeypatch, sampler_seed, outcome, final_blocks):
    # Given: original defaults distinct from the minimal witness, and a full space.
    module = retune_module()
    source = tmp_path / "parameterized.py"
    source.write_text("PARAMS = {'x': 7, 'y': 2}\ndef run(): return PARAMS['x'] + PARAMS['y']\n")
    reference = tmp_path / "reference.py"
    reference.write_text("def get_inputs(): return []\n")
    published = ParameterSpace(space_id="original", candidate_id="original", source_sha="original",
                               domains=[ParamDomain(name="x", kind="int", choices=list(range(100))),
                                        ParamDomain(name="y", kind="int", choices=[1, 2])])
    space_file = tmp_path / "space.json"
    space_file.write_text(published.model_dump_json())
    cfg = AppConfig()
    cfg.run.seed = 19
    cfg.gpu.compile_screen_enabled = True
    cfg.gpu.concurrency.enabled = True
    cfg.v3.ordered_categoricals.enabled = True
    cfg.v3.search.deweight_unconditional_failures = True
    cfg.v4.conditional_scan.mode = "active"
    cfg.budgets.space_expansions_per_candidate = 2
    original_config = cfg.model_dump()
    spec = module.RetuneInputs(task="level3:21", source=source, space=space_file, reference=reference,
                               sampler_seed=sampler_seed, evaluation_seed=73, output=tmp_path / "run", final_blocks=final_blocks)
    calls = []
    native_calls = []
    anchors = []
    cached = []

    def forbidden(*args, **kwargs):
        pytest.fail("retuning entered an LLM/server path")

    def worker(self, job, timeout_s, tag, **kwargs):
        calls.append((tag, job))
        assert self.worker_main_path.name == "c2_retune_worker.py"
        assert os.environ["C2_RETUNE_EVALUATION_SEED"] == "73"
        if job["job_type"] == "static_check":
            return {"ok": True}
        if job["job_type"] == "compile_probe":
            paths = [job["kernel_src_path"], *job.get("extra_kernel_src_paths", [])]
            return {"ok": True, "results": {
                p: {"ok": True, "max_shared": 0 if int(extract_defaults(Path(p).read_text())["x"]) < 50 else 999999}
                for p in paths}}
        assert job["seed"] == 73
        if outcome == "rejected" or (outcome == "no_best" and "-wit-" not in tag):
            return {"ok": False, "failure_kind": "correctness_mismatch", "log_tail": "CPU fixture mismatch"}
        if outcome == "no_best":
            return {"ok": True}
        params = extract_defaults(Path(job["kernel_src_path"]).read_text())
        value = 1.0 if params == {"x": 7, "y": 2} else 2.0 + int(params["x"])
        if job["num_perf_trials"] == 100:
            value = 999.0
        return {"ok": True, "latency_ms": {"mean": value, "std": 0.0, "min": value, "max": value,
                "median": value, "n": job["num_perf_trials"], "samples": [value] * job["num_perf_trials"]}}

    def profile(frame, event, arg):
        if event != "call":
            return
        if frame.f_code is SpaceValidator.validate_and_publish.__code__:
            native_calls.append("validate")
            assert frame.f_locals["self"].seed == 73
        if frame.f_code is Orchestrator._continue_accepted_candidate.__code__:
            native_calls.append("continue")
            assert frame.f_locals["run_analysis"] is False
            assert frame.f_locals["self"].cfg.budgets.space_expansions_per_candidate == 0
        if frame.f_code is Orchestrator._tune.__code__:
            native_calls.append("tune")
            orch = frame.f_locals["self"]
            assert orch.cfg.run.seed == sampler_seed
            assert orch.deps.evaluator.seed == 73
            assert orch.cfg.gpu.concurrency.enabled
            assert orch.cfg.v3.ordered_categoricals.enabled
            assert orch.deweight_ledger is not None
            anchors.extend(frame.f_locals["anchors"])
            cached.extend(frame.f_locals["measured_cache"].values())
        if frame.f_code is Orchestrator._prescreen_space.__code__:
            native_calls.append("prescreen")

    monkeypatch.setattr(AgentModule, "invoke", forbidden)
    monkeypatch.setattr(OpencodeServer, "start", forbidden)
    monkeypatch.setattr(WslGpuWorker, "run_job", worker)
    previous_profile = sys.getprofile()
    # When: the real validator and native tuning loop run without monkeypatched optimization logic.
    try:
        sys.setprofile(profile)
        result = module.retune(spec, cfg)
    finally:
        sys.setprofile(previous_profile)
    # Then: gate failures stop, accepted witnesses are the only anchors, and finals cannot rerank.
    assert result.status == outcome
    assert result.final_blocks_requested == final_blocks
    assert cfg.model_dump() == original_config
    events = RunStore.open(spec.output).iter_events()
    assert not any(e.type.startswith("AGENT_") or e.type.startswith("SCAN_") for e in events)
    if outcome == "rejected":
        assert native_calls == ["validate"]
        assert result.rejection.reason == "witness_default_failed"
        assert result.trials == [] and result.finals == []
        return
    assert native_calls == ["validate", "continue", "tune", "prescreen"]
    assert sum(e.type == "SPACE_PUBLISHED" for e in events) == 1
    assert sum(e.type == "STATS_DONE" for e in events) == 1
    assert [p.values for p in anchors] == [{"x": 7, "y": 2}, {"x": 0, "y": 1}]
    assert len(result.trials) == 40
    tuning = next(e.payload for e in events if e.type == "TUNING_DONE")
    assert tuning["snapshot"]["asked"] == tuning["snapshot"]["budget"] == 40
    assert tuning["conditional_scan"] is None and tuning["slope_guide"] is None
    assert any(e.type == "SPACE_PRESCREENED" and e.payload["answered"] > 0 for e in events)
    if outcome == "no_best":
        assert cached == [] and result.selected is None and result.finals == []
        return
    assert [t.params for t in cached] == anchors
    assert [t.params for t in result.trials[:2]] == anchors
    reused = [e for e in events if e.type == "TRIAL_DONE" and e.payload.get("reused_measurement")]
    assert len(reused) == 2
    quick_calls = [j for _, j in calls if j.get("num_perf_trials") == cfg.evaluation.quick_perf_trials]
    expected_fresh = sum(t.failure_kind != "infeasible_shared_memory" for t in result.trials[2:])
    assert len(quick_calls) == 2 + expected_fresh
    assert result.selected.params.values == {"x": 7, "y": 2}
    assert result.selected.latency_ms == 1.0
    assert len(result.finals) == final_blocks
    assert sum(j.get("num_perf_trials") == 100 for _, j in calls) == final_blocks
    assert all(t.params == result.selected.params and t.latency_ms.robust_ms == 999.0 for t in result.finals)
    assert extract_defaults((spec.output / "report" / "selected.py").read_text()) == {"x": 7, "y": 2}


def test_retune_cli_help_never_starts_runtime():
    module = retune_module()
    with pytest.raises(SystemExit) as exc:
        module.main(["--help"])
    assert exc.value.code == 0


def test_worker_entry_seeds_before_delegating_without_gpu(monkeypatch):
    # Given: the external worker and KernelBench seed boundary, replaced without importing torch.
    retune_module()
    wrapper = importlib.import_module("scripts.experiments.c2_retune_worker")
    from kernel_optimizer.gpu import worker_main

    observed = []
    fake_eval = ModuleType("kernelbench.eval")
    monkeypatch.setattr(fake_eval, "set_seed", lambda seed: observed.append(seed), raising=False)
    monkeypatch.setitem(sys.modules, "kernelbench", ModuleType("kernelbench"))
    monkeypatch.setitem(sys.modules, "kernelbench.eval", fake_eval)
    monkeypatch.setenv("C2_RETUNE_EVALUATION_SEED", "73")
    monkeypatch.setattr(worker_main, "_ensure_optional_deps", lambda: observed.append("dependencies"))

    def delegated():
        observed.append("worker")
        return 7

    monkeypatch.setattr(worker_main, "main", delegated)
    # When: the retune-only entry delegates to the native worker.
    result = wrapper.main()
    # Then: fixed seeding precedes all native work, and the native exit result is preserved.
    assert observed == ["dependencies", 73, "worker"]
    assert result == 7
