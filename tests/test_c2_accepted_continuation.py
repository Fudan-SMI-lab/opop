"""CPU contracts for accepted-candidate continuation; no GPU or model calls."""

from collections.abc import Iterator
from contextlib import closing
from pathlib import Path
from types import SimpleNamespace

import pytest

from kernel_optimizer.agents.base import AgentModule
from kernel_optimizer.agents.runtime import OpencodeClient
from kernel_optimizer.config import AppConfig
from kernel_optimizer.control.orchestrator import CandidateRun, Orchestrator
from kernel_optimizer.gpu.worker_client import WslGpuWorker
from kernel_optimizer.models.core import ParamDomain, ParameterSpace, ParamSet, TaskSpec
from kernel_optimizer.models.reports import BottleneckReport
from kernel_optimizer.paramspace.materializer import extract_defaults
from kernel_optimizer.paramspace.validation import SpaceAccepted, WitnessResult
from kernel_optimizer.store.run_store import RunStore
from kernel_optimizer.wiring import Runtime, build_orchestrator


@pytest.fixture
def orch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Orchestrator]:
    cfg = AppConfig()
    cfg.budgets.trials_per_space = 40
    cfg.budgets.space_expansions_per_candidate = 0
    cfg.gpu.compile_screen_enabled = False
    cfg.v4.conditional_scan.mode = "off"
    cfg.v3.slope_guide.enabled = False
    ref = tmp_path / "ref.py"
    ref.write_text("def get_inputs(): return []\n", encoding="utf-8")
    task = TaskSpec(level=3, problem_id=21, name="fixture", ref_path=ref, ref_src_sha="ref")
    store = RunStore.create(tmp_path, "run", {})

    def forbidden(*args, **kwargs):
        pytest.fail("unexpected agent call")

    def worker(self, job, timeout_s, tag, **kwargs):
        source = Path(job["kernel_src_path"]).read_text()
        params = extract_defaults(source)
        value = 10.0 + int(params["x"]) + (100.0 if "+ 1" in source else 0.0)
        return {"ok": True, "latency_ms": {"mean": value, "median": value,
                "min": value, "max": value, "std": 0.0, "n": 20}}

    monkeypatch.setattr(AgentModule, "invoke", forbidden)
    monkeypatch.setattr(WslGpuWorker, "run_job", worker)
    with closing(OpencodeClient("http://127.0.0.1:1")) as client:
        runtime = Runtime(cfg)
        runtime.client = client
        yield build_orchestrator(cfg, store, task, runtime)


def candidate(orch: Orchestrator) -> tuple[CandidateRun, SpaceAccepted]:
    source = "PARAMS = {'x': 7}\ndef run(): return PARAMS['x']\n"
    cand = orch._register(source, "rewrite", [], "triton", "child dataflow intent")
    assert cand is not None
    space = ParameterSpace(space_id="space", candidate_id=cand.candidate_id, source_sha="s",
                           domains=[ParamDomain(name="x", kind="int", choices=list(range(60)))])
    witnesses = [WitnessResult(params=ParamSet(values={"x": x}), latency_mean_ms=ms,
                   worker_result={"ok": True, "latency_ms": {"mean": ms, "median": ms,
                                  "min": ms, "max": ms, "std": 0.0, "n": 20}})
                 for x, ms in [(7, 1.0), (0, 2.0)]]
    return orch.runs[cand.candidate_id], SpaceAccepted(space=space, witnesses=witnesses)


def test_native_tune_baseline_preserves_b40_and_witnesses(orch: Orchestrator) -> None:
    # Given: accepted default and second witness, distinct from native measurements.
    crun, accepted = candidate(orch)
    crun.space = accepted.space
    from kernel_optimizer.models.core import TrialRecord
    from kernel_optimizer.evaluation.correctness import latency_from_result
    cache = {w.params.key(): TrialRecord(trial_id=f"wit-{w.params.key()}",
             candidate_id=crun.candidate.candidate_id, space_id=accepted.space.space_id,
             params=w.params, status="complete", latency_ms=latency_from_result(w.worker_result))
             for w in accepted.witnesses}
    # When: the pre-existing native tuning seam consumes its witnesses.
    orch._tune(crun, tuple(w.params for w in accepted.witnesses), cache)
    # Then: witnesses consume budget, and candidate/family selection agrees.
    assert len(crun.trials) == 40
    assert [t.params for t in crun.trials[:2]] == [w.params for w in accepted.witnesses]
    assert crun.best_ms == 1.0
    assert orch.deps.families.families[crun.candidate.family_id].best.latency_ms == 1.0


def test_continuation_when_analysis_disabled_keeps_numeric_stats(orch: Orchestrator) -> None:
    # Given: an accepted result and a dummy agent client that must never be used.
    crun, accepted = candidate(orch)
    # When: the adapter continuation runs without post-tune analysis.
    orch._continue_accepted_candidate(crun, accepted, run_analysis=False)
    # Then: native tuning, stats and selection survive without additional agents.
    assert len(crun.trials) == 40
    assert [t.params for t in crun.trials[:2]] == [w.params for w in accepted.witnesses]
    assert crun.stats is not None and crun.best_ms == 1.0
    assert crun.report is None and crun.candidate.status == "tuned"


def test_pipeline_delegates_once_after_acceptance(orch: Orchestrator, monkeypatch: pytest.MonkeyPatch) -> None:
    # Given: parameterization has already validated exactly one accepted proposal.
    crun, accepted = candidate(orch)
    calls = []
    monkeypatch.setattr(orch, "_parameterize_with_repair", lambda run: calls.append("accept") or accepted)
    monkeypatch.setattr(orch, "_continue_accepted_candidate",
                        lambda run, result: calls.append((run, result)))
    # When: the production entrypoint handles this candidate.
    orch._candidate_pipeline(crun.candidate.candidate_id)
    # Then: it shares the continuation without another parameterization/validation.
    assert calls == ["accept", (crun, accepted)]


@pytest.mark.parametrize(("changed", "new_default"), [(False, False), (False, True), (True, False)])
def test_expansion_retests_prior_best_when_source_changes(
    orch: Orchestrator, monkeypatch: pytest.MonkeyPatch, changed: bool, new_default: bool,
) -> None:
    # Given: a prior optimum absent from the expanded witnesses.
    crun, accepted = candidate(orch)
    crun.space = accepted.space
    orch._tune(crun, (), {})
    prior = min(crun.trials, key=lambda t: t.latency_ms.robust_ms)
    old_ms = prior.latency_ms.robust_ms
    expanded_source = crun.source.replace("return PARAMS['x']", "return PARAMS['x'] + 1") if changed else crun.source
    if new_default:
        expanded_source = expanded_source.replace("'x': 7", "'x': 59")
    expanded = accepted.model_copy(deep=True)
    expanded.space.space_id = "expanded"
    expanded.space.domains[0].choices.append(60)
    expanded.witnesses = [WitnessResult(params=ParamSet(values={"x": x}), latency_mean_ms=500.0,
                          worker_result={"ok": True, "latency_ms": {"mean": 500.0, "median": 500.0,
                                         "min": 500.0, "max": 500.0, "std": 0.0, "n": 20}})
                          for x in [58, 59]]
    crun.stats = orch.deps.stats_analyzer.analyze(crun.space, crun.trials)
    orch.cfg.budgets.space_expansions_per_candidate = 1
    monkeypatch.setattr("kernel_optimizer.control.orchestrator.boundary_knobs_to_expand",
                        lambda *a, **k: [{"name": "x", "direction": "max"}])
    monkeypatch.setattr(orch.deps.parameterizer, "invoke", lambda inputs: SimpleNamespace(
        sandbox=SimpleNamespace(read_output=lambda file: expanded_source),
        output=SimpleNamespace(file="source.py")))
    monkeypatch.setattr(orch.deps.validator, "validate_and_publish", lambda *a, **k: expanded)
    # When: native expansion anchors the old optimum in the new study.
    orch._maybe_expand_space(crun, run_analysis=False)
    # Then: only equivalent materializations reuse the prior score; all witnesses remain.
    events = [e for e in orch.store.iter_events() if e.type == "TRIAL_DONE"
              and e.payload["trial"]["space_id"] == "expanded"]
    anchor = next(e for e in events if e.payload["trial"]["params"] == prior.params.model_dump())
    assert bool(anchor.payload.get("reused_measurement")) is not changed
    assert anchor.payload["trial"]["latency_ms"]["median"] == old_ms + (100.0 if changed else 0.0)
    assert len(events) == 40 and crun.best_ms == old_ms
    assert orch.deps.families.families[crun.candidate.family_id].best.latency_ms == old_ms
    assert all(any(t.params == w.params for t in crun.trials) for w in expanded.witnesses)


def test_parameterization_receives_registered_child_intent(orch: Orchestrator, monkeypatch: pytest.MonkeyPatch) -> None:
    # Given: the existing metadata holds the rewrite summary.
    crun, _ = candidate(orch)
    captured = []
    monkeypatch.setattr(orch.deps.parameterizer, "invoke", lambda inputs: captured.append(inputs))
    # When: the production parameterizer caller builds its input.
    orch._parameterize_agent_call(crun.source, "", crun.candidate.candidate_id)
    # Then: intent is taken from this child, not its parent or another candidate.
    assert captured[0].rewrite_intent == crun.candidate.approach_summary


def test_production_analysis_and_rewrite_keep_response_separate(
    orch: Orchestrator, monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given: independent compiler-wall and conditional-response inputs.
    crun, accepted = candidate(orch)
    orch._continue_accepted_candidate(crun, accepted, run_analysis=False)
    wall, response = "compiler-input", "response-input"
    orch.cfg.v3.wall_attribution.in_prompt = True
    monkeypatch.setattr(orch, "_measure_best_overhead", lambda run: None)
    monkeypatch.setattr(orch, "_classify_bottleneck", lambda run: None)
    monkeypatch.setattr(orch, "_attribute_resource_walls", lambda run: [wall])
    monkeypatch.setattr("kernel_optimizer.control.orchestrator.wall_attribution.for_prompt", lambda walls: walls[0])
    monkeypatch.setattr(orch, "_attribute_soft_walls", lambda run: None)
    monkeypatch.setattr(orch, "_conditional_brief", lambda run: response)
    monkeypatch.setattr(orch, "_dimension_digest", lambda *args: None)
    analyst_inputs, rewrite_inputs = [], []
    monkeypatch.setattr(orch.deps.analyst, "invoke", lambda inputs: analyst_inputs.append(inputs)
                        or SimpleNamespace(output=BottleneckReport(summary="fixture")))
    monkeypatch.setattr(orch.deps.rewriter, "invoke", lambda inputs: rewrite_inputs.append(inputs)
                        or SimpleNamespace(output=SimpleNamespace(candidates=[])))
    # When: production analysis feeds the subsequent production rewrite.
    orch._stats_and_analysis(crun)
    orch._do_rewrite(crun.candidate.family_id, crun)
    # Then: source fields, not rendered prompt prose, preserve their provenance.
    assert crun.wall_text == rewrite_inputs[0].wall_text == wall
    assert crun.conditional_response_text == rewrite_inputs[0].conditional_response_text == response
    assert analyst_inputs[0].conditional_response_text == response
    assert analyst_inputs[0].digest_text == wall
    expected_reference = orch.task.ref_path.read_text(encoding="utf-8")
    assert analyst_inputs[0].reference_source == rewrite_inputs[0].reference_source == expected_reference


def test_continuation_retains_unmeasured_witness_anchors(orch: Orchestrator) -> None:
    # Given: valid witnesses without cached timing measurements.
    crun, accepted = candidate(orch)
    accepted.witnesses = [w.model_copy(update={"latency_mean_ms": None}) for w in accepted.witnesses]
    # When: native continuation must measure both anchors inside its B40 budget.
    orch._continue_accepted_candidate(crun, accepted, run_analysis=False)
    # Then: defaults remain first and native evaluation supplies their actual score.
    assert len(crun.trials) == 40
    assert [t.params for t in crun.trials[:2]] == [w.params for w in accepted.witnesses]
    assert crun.trials[0].latency_ms.robust_ms == 17.0
    assert not any(e.payload.get("reused_measurement") for e in orch.store.iter_events()
                   if e.type == "TRIAL_DONE")
