"""Actual information/native expansion flow; only external agent/worker calls are faked."""

import json
import sys
from pathlib import Path

import pytest

from kernel_optimizer.agents.runtime import AgentCallError, OpencodeClient, OpencodeServer, PromptResult
from kernel_optimizer.control.orchestrator import Orchestrator
from kernel_optimizer.gpu.worker_client import WslGpuWorker
from kernel_optimizer.paramspace.materializer import extract_defaults, materialize
from kernel_optimizer.store.run_store import RunStore
from scripts.experiments.c2_information import run_information
from scripts.experiments.c2_information_inputs import InformationRun
from scripts.experiments.c2_local_inputs import InputError
from scripts.experiments.c2_retune import retune
from tests.test_c2_information_inputs import prepared


@pytest.mark.parametrize("mode", ["win", "same", "changed", "noop", "reject", "flat", "agent_error", "partial", "legacy"])
def test_live_consumer_reaches_native_expansion_and_preserves_all_spaces(tmp_path, monkeypatch, prepared, mode):
    # Given: an initial child whose best quick score is worse than the original parent.
    shared, acquisition, cfg = prepared
    cfg.gpu.compile_screen_enabled = False
    original_cap = 1 if mode == "legacy" else 0
    cfg.budgets.space_expansions_per_candidate = original_cap
    for name in ("analyst", "rewriter", "parameterizer"):
        getattr(cfg.agents, name).max_retries = 0
        getattr(cfg.agents, name).max_transport_retries = 0
    active, clients, roles, intents, jobs = [], [], [], [], []
    native_caps = []
    default = 2 if mode == "partial" else 7
    initial = f"import triton\nPARAMS={{'x': {default}}}\n@triton.jit\ndef work(): return PARAMS['x']\n"
    intent = "fixture-region-and-partners"
    run = InformationRun(cfg, "H", tmp_path / "run", sampler_seed=2, require_b40=True,
                         space_expansions_per_candidate=1)
    if mode == "legacy":
        run = InformationRun(cfg, "H", tmp_path / "run", sampler_seed=2, require_b40=True)
    cap = run.space_expansions_per_candidate
    assert cap == (0 if mode == "legacy" else 1)

    def start(self):
        active.append(True)
        return "http://unused.invalid"

    def session(self, directory, title):
        assert active and not self._http.is_closed
        if self not in clients:
            clients.append(self)
        return title

    def prompt(self, session_id, text, **kwargs):
        assert active and not self._http.is_closed
        root, title = kwargs["directory"], kwargs["schema"]["title"]
        expansion = "retune" in root.parts
        roles.append((title, expansion))
        source = initial
        choices = list(range(4 if mode == "partial" and not expansion else 60))
        if expansion:
            if mode == "agent_error":
                raise AgentCallError("fixture expansion transport failure")
            source = initial.replace(f"'x': {default}", "'x': 59")
            if mode == "changed":
                source = source.replace("return PARAMS['x']", "return PARAMS['x'] + 100")
            if mode != "noop":
                choices.insert(0, -1)
        if title == "ParameterizationResult":
            intents.append((root / "analysis/rewrite_intent.md").read_text(encoding="utf-8"))
        (root / "child.py").write_text(source, encoding="utf-8")
        responses = {
            "BottleneckReport": {"summary": "fixture", "hypotheses": []},
            "RewriteResult": {"candidates": [{"file": "child.py", "backend": "triton",
                                               "hypothesis_id": "H1", "change_summary": intent}]},
            "ParameterizationResult": {"file": "child.py", "space": {"params": [{
                "name": "wrong_key" if expansion and mode == "reject" else "x", "kind": "int", "choices": choices}]}},
        }
        return PromptResult(text="", structured=responses[title], session_id=session_id)

    def worker(self, job, timeout_s, tag, **kwargs):
        native = "retune" in self.jobs_dir.parts
        assert bool(active) == native
        if native:
            assert self.worker_main_path.name == "c2_retune_worker.py"
            assert self.jobs_dir == tmp_path / "run/retune/jobs"
        if job["job_type"] == "static_check":
            return {"ok": True}
        assert job["seed"] == 0
        source = Path(job["kernel_src_path"]).read_text(encoding="utf-8")
        x = int(extract_defaults(source)["x"])
        value = 10.0 + x if native else 2.0
        if mode == "flat" and native:
            value = 10.0
        if mode == "win" and x == -1:
            value = 1.0
        if "+ 100" in source:
            value += 100
        n = job["num_perf_trials"]
        jobs.append((native, n, source))
        if n == 100:
            assert job["num_correct_trials"] == 5
            value = 99.0 if native else 2.0
        return {"ok": True, "latency_ms": {"mean": value, "median": value, "min": value, "max": value,
                                            "std": 0.0, "n": n, "samples": [value] * n}}

    def profile(frame, event, arg):
        if event != "call":
            return
        if frame.f_code is retune.__code__:
            runtime = frame.f_locals["runtime"]
            assert active and runtime.client is clients[0] and not runtime.client._http.is_closed
            spec = frame.f_locals["inputs"]
            assert spec.space_expansions_per_candidate == frame.f_locals["cfg"].budgets.space_expansions_per_candidate == cap
            assert spec.rewrite_intent == intent and spec.final_blocks == 3 and spec.sampler_seed == 2
        if frame.f_code is Orchestrator._continue_accepted_candidate.__code__:
            orch = frame.f_locals["self"]
            native_caps.append(orch.cfg.budgets.space_expansions_per_candidate)
            assert frame.f_locals["run_analysis"] is False and orch.cfg.run.seed == 2
            assert orch.deps.parameterizer.client is clients[0]

    monkeypatch.setattr(OpencodeServer, "start", start)
    monkeypatch.setattr(OpencodeServer, "stop", lambda self: active.pop())
    monkeypatch.setattr(OpencodeClient, "create_session", session)
    monkeypatch.setattr(OpencodeClient, "prompt", prompt)
    monkeypatch.setattr(WslGpuWorker, "run_job", worker)
    previous_profile = sys.getprofile()
    original_cwd = Path.cwd()
    # When: the actual consumer performs generation, initial B40 and native eligibility/expansion.
    try:
        sys.setprofile(profile)
        result = run_information(shared, acquisition, run)
    finally:
        sys.setprofile(previous_profile)
    # Then: expansion consumes only its admitted study, not a second candidate pipeline or more finals.
    assert not active and len(clients) == 1 and clients[0]._http.is_closed
    assert Path.cwd() == original_cwd and native_caps == [cap]
    assert cfg.budgets.space_expansions_per_candidate == original_cap
    expanded = mode in {"win", "same", "changed", "partial"}
    assert result.expanded_count == result.child.expanded_count == int(expanded)
    counts = [4, 40] if mode == "partial" else [40, 40] if expanded else [40]
    assert result.asked == result.child.asked == sum(counts)
    assert [s.asked for s in result.studies] == counts
    assert result.studies == result.child.studies
    assert all(s.n_complete + s.n_fail == s.asked for s in result.studies)
    assert result.status == ("censored" if mode == "partial" else "valid") and result.child.status == "complete"
    assert result.selected == ("child" if mode == "win" else "parent")
    assert len(result.parent_finals) == len(result.child.finals) == 3
    assert sum(n == 100 for _, n, _ in jobs) == 6
    assert roles[:3] == [("BottleneckReport", False), ("RewriteResult", False), ("ParameterizationResult", False)]
    assert roles[3:] == [("ParameterizationResult", True)] * (0 if mode in {"flat", "legacy"} else 2 if mode == "reject" else 1)
    store = RunStore.open(tmp_path / "run/retune")
    events = store.iter_events()
    registered = next(e.payload["candidate"] for e in events if e.type == "CANDIDATE_REGISTERED")
    assert registered["origin"] == "rewrite" and registered["approach_summary"] == intent
    assert intents == [intent] * (len(roles) - 2 - int(mode == "agent_error"))
    eligibility = next((e for e in events if e.type == "SPACE_EXPANSION_ELIGIBILITY"), None)
    if mode == "legacy":
        assert eligibility is None
    else:
        assert eligibility.payload["eligible"] == (mode != "flat")
        assert eligibility.payload["base_best"]["latency_ms"]["median"] > result.parent_baseline_ms
    assert all(e.payload["slope_guide"] is None and e.payload["conditional_scan"] is None
               for e in events if e.type == "TUNING_DONE")
    selected = result.child.selected_trial
    assert selected.space_id == result.child.selected_space.space_id
    assert len(result.child.spaces) == (2 if expanded else 1)
    assert selected.space_id == result.child.spaces[1 if mode in {"win", "same", "partial"} else 0].space_id
    measured = store.candidate_dir(selected.candidate_id) / "trials" / f"{selected.trial_id}.py"
    assert (store.run_dir / "report/selected.py").read_bytes() == measured.read_bytes()
    if mode == "changed":
        assert measured.read_text(encoding="utf-8") == materialize(initial, selected.params)
        assert store.get_artifact(eligibility.payload["base_best_source_ref"]) == measured.read_bytes()
    manifest = json.loads((store.run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["inputs"]["space_expansions_per_candidate"] == cap


@pytest.mark.parametrize("cap", [-1, 2])
def test_information_rejects_out_of_program_expansion_cap(prepared, tmp_path, cap):
    # Given / When: an unsupported per-opportunity expansion allowance is requested.
    with pytest.raises(InputError):
        InformationRun(prepared[2], "G0", tmp_path / "run", space_expansions_per_candidate=cap)


@pytest.mark.parametrize(("counts", "total", "complete"), [([40, 40], 80, True), ([39, 40], 79, False),
                                                           ([40, None], None, False)])
def test_study_accounting_keeps_every_publication_when_completion_is_missing(tmp_path, counts, total, complete):
    # Given: native publications with complete, short, or absent completion records.
    from kernel_optimizer.models.core import ParamDomain, ParameterSpace
    from scripts.experiments.c2_retune_studies import read_studies
    store = RunStore.create(tmp_path, "native", {})
    for index, asked in enumerate(counts):
        space = ParameterSpace(candidate_id="child", space_id=str(index), source_sha="fixture",
                               domains=[ParamDomain(name="x", kind="int", choices=[0, 1])])
        store.append("SPACE_PUBLISHED", {"space": space.model_dump()})
        if asked is not None:
            store.append("TUNING_DONE", {"candidate_id": "child", "space_id": str(index), "snapshot": {"asked": asked}})
    # When: read the existing event stream without manufacturing a completion.
    accounting = read_studies(store)
    # Then: the final study cannot hide a shorter or unfinished earlier/admitted study.
    assert accounting.asked_total == total and accounting.b40_complete == complete
    assert [s.asked for s in accounting.studies] == counts
