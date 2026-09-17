"""Adapter contracts with real agents/continuation and CPU-only external boundaries."""

import json
import sys
from contextlib import closing
from pathlib import Path

import pytest

from kernel_optimizer.agents.modules import BottleneckAnalystAgent, StructureRewriterAgent
from kernel_optimizer.agents.runtime import OpencodeClient, PromptResult
from kernel_optimizer.config import AppConfig
from kernel_optimizer.control.orchestrator import Orchestrator
from kernel_optimizer.gpu.worker_client import WslGpuWorker
from kernel_optimizer.models.core import ParamDomain, ParameterSpace, sha256_text
from kernel_optimizer.paramspace.materializer import extract_defaults
from kernel_optimizer.store.run_store import RunStore
from kernel_optimizer.wiring import Runtime
from scripts.experiments import c2_local_agents as agents
from scripts.experiments.c2_local_inputs import Shared, stage_inputs
from scripts.experiments.c2_closed_loop import retained_feedback
from scripts.experiments.c2_information_inputs import InformationResult
from scripts.experiments.c2_retune import RetuneInputs, retune
from tests.test_c2_information_inputs import prepared as existing_prepared

prepared = existing_prepared


@pytest.mark.parametrize("mode", ["whole_task", "legacy_local"])
@pytest.mark.parametrize("explicit_empty", [False, True])
def test_shared_proposal_gets_two_parameterizations_and_scoped_feedback(tmp_path, monkeypatch, prepared, mode, explicit_empty):
    # Given: one actual attempted action, plus an untried analyst suggestion.
    assert hasattr(agents, "generate_legacy_proposal"), "split generation API is missing"
    shared, acquisition, cfg = prepared
    history = [{"id": "previous", "change": "previous attempted action", "parent_candidate_id": "parent"}]
    shared = shared.model_copy(update={"failed_hypotheses": history})
    inputs = stage_inputs(shared, tmp_path / "common", acquisition.responses)
    calls, parameter_sources, intents, delivered_history, captured = [], [], [], [], []
    source = "import triton\nPARAMS={'x': 7}\n@triton.jit\ndef work(): return 2\n"

    def prompt(self, session_id, text, **kwargs):
        root, title = kwargs["directory"], kwargs["schema"]["title"]
        calls.append(title)
        if title in {"BottleneckReport", "RewriteResult"}:
            assert (root / "task/ref.py").read_text() == shared.reference_source
            assert (root / "analysis/conditional_responses.md").read_text() == "\n".join(r.model_dump_json() for r in inputs.responses)
            assert not (root / "analysis/resource_walls.md").exists()
        if title == "RewriteResult":
            delivered_history.append(json.loads((root / "history/failed_hypotheses.json").read_text()))
        if title == "ParameterizationResult":
            parameter_sources.append((root / "candidate/source.py").read_text())
            intent = root / "analysis/rewrite_intent.md"
            intents.append(intent.read_text() if intent.exists() else None)
        (root / "child.py").write_text(source)
        answers = {
            "BottleneckReport": {"summary": "fixture", "hypotheses": [
                {"id": "attempted", "change": "chosen action", "expected_effect": "fixture"},
                {"id": "untried", "change": "untried action", "expected_effect": "fixture"}]},
            "RewriteResult": {"candidates": [{"file": "child.py", "backend": "triton",
                "hypothesis_id": "attempted", "change_summary": "chosen action"}]},
            "ParameterizationResult": {"file": "child.py", "space": {"params": [
                {"name": "x", "kind": "int", "choices": [0, 7]}]}},
        }
        return PromptResult(text="", structured=answers[title], session_id=session_id)

    def profile(frame, event, arg):
        if event == "call" and frame.f_code in {BottleneckAnalystAgent.seed_sandbox.__code__, StructureRewriterAgent.seed_sandbox.__code__}:
            captured.append(frame.f_locals["inputs"])

    monkeypatch.setattr(OpencodeClient, "create_session", lambda self, directory, title: title)
    monkeypatch.setattr(OpencodeClient, "prompt", prompt)
    previous = sys.getprofile()
    with closing(OpencodeClient("http://unused.invalid")) as client:
        runtime = Runtime(cfg)
        runtime.client = client
        service = agents.Services(cfg, RunStore.create(tmp_path, "generation", {}), runtime)
        # When: P/H share exactly one generated proposal and independently parameterize it.
        try:
            sys.setprofile(profile)
            proposal = agents.generate_legacy_proposal(shared, service, inputs, reasoning_mode=mode,
                                                       failed_hypotheses=[] if explicit_empty else None)
            p = agents.parameterize_legacy_proposal(shared, service, proposal, pass_intent=False)
            h = agents.parameterize_legacy_proposal(shared, service, proposal, pass_intent=True)
        finally:
            sys.setprofile(previous)
    # Then: no proposal redraw, no invented failures, identical source and selective intent delivery.
    assert calls == ["BottleneckReport", "RewriteResult", "ParameterizationResult", "ParameterizationResult"]
    assert parameter_sources == [proposal.source, proposal.source] == [source, source]
    assert intents == [None, proposal.change_summary]
    assert delivered_history == [[] if explicit_empty else history]
    assert all(item.reasoning_mode == mode and item.reference_source == shared.reference_source for item in captured)
    assert all(item.conditional_response_text for item in captured)
    assert captured[-1].wall_text is None
    assert proposal.hypothesis_id == "attempted" and p.path != h.path


def test_empty_feedback_preserves_old_shared_identity(prepared):
    # Given / When / Then: older prepared bundles have no feedback field.
    shared, _, _ = prepared
    old = shared.model_dump_json(exclude={"failed_hypotheses"})
    restored = Shared.model_validate_json(old)
    assert restored.failed_hypotheses == []
    assert restored.identity() == sha256_text(old)


def test_feedback_marks_only_the_executed_proposal_when_parent_is_retained(tmp_path, prepared):
    # Given: a report with two suggestions, but only one actual generated proposal.
    shared, _, _ = prepared
    store = RunStore.create(tmp_path, "generation", {})
    store.append("BOTTLENECK_REPORTED", {"hypotheses": [{"id": "used"}, {"id": "untried"}]})
    store.append("LEGACY_PROPOSAL_GENERATED", {"parent_candidate_id": "parent",
                 "hypothesis_id": "used", "change_summary": "executed action"})
    outcome = InformationResult.model_validate({
        "group": "G0", "state": "off", "selected": "parent", "selected_artifact": "parent.py",
        "parent_baseline_ms": 5.0, "parent_params": shared.parent.params, "parent_finals": [],
        "child": {"status": "complete", "selected": {"candidate_id": "child", "params": shared.parent.params,
                                                       "latency_ms": 6.0}}, "error": None,
        "generation_costs": {}, "acquisition_costs": {}, "parent_costs": {}, "wall_s": 1.0,
    })
    # When: the completed slower attempt is carried forward on this same parent.
    updated = retained_feedback(shared, outcome, tmp_path)
    # Then: the unexecuted suggestion is not marked failed and the incumbent is unchanged.
    assert updated.parent == shared.parent and updated.source == shared.source
    assert [item["id"] for item in updated.failed_hypotheses] == ["used"]
    assert updated.failed_hypotheses[0]["outcome"] == "valid_but_not_faster"
    assert shared.failed_hypotheses == []


@pytest.mark.parametrize(("shift", "missing"), [(100, False), (-10, False), (100, True)])
def test_selected_artifact_tracks_actual_measurement_across_source_changing_expansion(tmp_path, monkeypatch, shift, missing):
    # Given: a native eligible boundary optimum and one changed-body expansion.
    source = tmp_path / "source.py"
    source.write_text("import triton\nSHIFT = 0\nPARAMS={'x': 79}\n@triton.jit\ndef work(): return SHIFT\n")
    reference = tmp_path / "ref.py"
    reference.write_text("def get_inputs(): return []\n")
    space = ParameterSpace(space_id="s", candidate_id="c", source_sha="s",
                           domains=[ParamDomain(name="x", kind="int", choices=list(range(80)))])
    path = tmp_path / "space.json"
    path.write_text(space.model_dump_json())
    cfg = AppConfig()
    cfg.gpu.compile_screen_enabled = False
    spec = RetuneInputs(task="level3:21", source=source, space=path, reference=reference,
                        sampler_seed=0, evaluation_seed=0, output=tmp_path / "retune",
                        space_expansions_per_candidate=1, final_blocks=3)
    models, finals = [], []

    def prompt(self, session_id, text, **kwargs):
        models.append(kwargs["schema"]["title"])
        expanded = source.read_text().replace("SHIFT = 0", f"SHIFT = {shift}").replace("'x': 79", "'x': 89")
        (kwargs["directory"] / "expanded.py").write_text(expanded)
        return PromptResult(text="", session_id=session_id, structured={"file": "expanded.py", "space": {
            "params": [{"name": "x", "kind": "int", "choices": list(range(90))}]}})

    def worker(self, job, timeout_s, tag, **kwargs):
        if job["job_type"] == "static_check":
            return {"ok": True}
        text = Path(job["kernel_src_path"]).read_text()
        value = 100.0 - int(extract_defaults(text)["x"]) + int(text.split("SHIFT = ")[1].splitlines()[0])
        if job["num_perf_trials"] == 100:
            finals.append(text)
        return {"ok": True, "latency_ms": {"mean": value, "median": value, "min": value,
                "max": value, "std": 0.0, "n": job["num_perf_trials"]},
                "triton": {"kernels": [{"name": "work", "n_regs": 8, "shared": 0, "n_spills": 0}]}}

    def profile(frame, event, arg):
        if missing and event == "return" and frame.f_code is Orchestrator._continue_accepted_candidate.__code__:
            crun = frame.f_locals["crun"]
            best = min((t for t in crun.trials if t.latency_ms), key=lambda t: t.latency_ms.robust_ms)
            (frame.f_locals["self"].store.candidate_dir(best.candidate_id) / "trials" / f"{best.trial_id}.py").unlink()

    monkeypatch.setattr(OpencodeClient, "create_session", lambda self, directory, title: title)
    monkeypatch.setattr(OpencodeClient, "prompt", prompt)
    monkeypatch.setattr(WslGpuWorker, "run_job", worker)
    previous = sys.getprofile()
    with closing(OpencodeClient("http://unused.invalid")) as client:
        runtime = Runtime(cfg)
        runtime.client = client
        # When: the unchanged native expansion may retain an earlier source's best point.
        try:
            sys.setprofile(profile)
            result = retune(spec, cfg, runtime=runtime)
        finally:
            sys.setprofile(previous)
    # Then: finalization uses the measured artifact, never new-source/old-score fabrication.
    assert models == ["ParameterizationResult"] and len(result.trials) == 80
    if missing:
        assert result.status == "artifact_error" and result.error and finals == []
        assert not (spec.output / "report/selected.py").exists()
    else:
        trial = result.selected_trial
        assert trial is not None and result.selected is not None
        measured = spec.output / "candidates" / trial.candidate_id / "trials" / f"{trial.trial_id}.py"
        assert (spec.output / "report/selected.py").read_bytes() == measured.read_bytes()
        assert finals == [measured.read_text()] * 3
        assert result.selected.latency_ms == (21.0 if shift > 0 else 1.0)
        assert result.selected.params.values == {"x": 79 if shift > 0 else 89}
