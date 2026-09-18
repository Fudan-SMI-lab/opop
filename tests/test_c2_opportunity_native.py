"""Native opportunity flow with real validation, eligibility, TPE and CPU worker fakes."""

from types import SimpleNamespace
from typing import Literal

import pytest

from kernel_optimizer.control.orchestrator import Orchestrator
from kernel_optimizer.models.core import ParamSet
from kernel_optimizer.models.reports import BottleneckReport, ParameterizationResult
from kernel_optimizer.paramspace.materializer import extract_defaults, materialize
from kernel_optimizer.paramspace.validation import SpaceAccepted
from tests.test_c2_accepted_continuation import orch as orch


def prepare(orch: Orchestrator, monkeypatch: pytest.MonkeyPatch,
            mode: Literal["same", "changed", "noop", "reject"]):
    source = "PARAMS = {'x': 7}\ndef run(): return PARAMS['x']\n"
    cand = orch._register(source, "rewrite", [], "triton", "extend x with companion conditions")
    assert cand is not None
    crun = orch.runs[cand.candidate_id]
    proposal = ParameterizationResult.model_validate({"file": "source.py", "space": {
        "params": [{"name": "x", "kind": "int", "choices": list(range(60))}], "constraints": []}})
    accepted = orch.deps.validator.validate_and_publish(
        cand, source, proposal, orch.task, orch.store.candidate_dir(cand.candidate_id))
    assert isinstance(accepted, SpaceAccepted)
    orch.cfg.budgets.space_expansions_per_candidate = 1
    expanded_source = source.replace("'x': 7", "'x': 59")
    if mode == "changed":
        expanded_source = expanded_source.replace("return PARAMS['x']", "return PARAMS['x'] + 1")
    expanded = proposal.model_copy(deep=True)
    if mode != "noop":
        expanded.space.params[0].choices.insert(0, -1)
    if mode == "reject":
        expanded.space.params[0].name = "wrong_key"
    captured = []

    def invoke(inputs):
        captured.append(inputs)
        return SimpleNamespace(output=expanded,
                               sandbox=SimpleNamespace(read_output=lambda file: expanded_source))

    monkeypatch.setattr(orch.deps.parameterizer, "invoke", invoke)
    return crun, accepted, captured


@pytest.mark.parametrize("mode", ["same", "changed"])
def test_existing_native_flow_runs_two_b40_without_parent_rejection(orch, monkeypatch, mode):
    # Given: a better parent score in the family, and an actually eligible child.
    crun, accepted, captured = prepare(orch, monkeypatch, mode)
    family = orch.deps.families.families[crun.candidate.family_id]
    orch.deps.families.update_best(family.family_id, "parent", ParamSet(values={"x": 0}), 0.5)
    # When: numeric-only continuation completes all eligible native studies.
    orch._continue_accepted_candidate(crun, accepted, run_analysis=False)
    # Then: parent ranking never suppresses expansion and each accepted space gets B40.
    events = orch.store.iter_events()
    tuning = [e for e in events if e.type == "TUNING_DONE"]
    assert [e.payload["snapshot"]["asked"] for e in tuning] == [40, 40]
    assert len(crun.trials) == 80 and len(captured) == 1
    assert captured[0].rewrite_intent == crun.candidate.approach_summary
    assert all(e.payload["conditional_scan"] is None and e.payload["slope_guide"] is None for e in tuning)
    assert family.best.candidate_id == "parent" and crun.report is None
    assert crun.best_ms == (10.0 if mode == "changed" else 9.0)
    old_point = next(e for e in events if e.type == "TRIAL_DONE"
                     and e.payload["trial"]["space_id"] == crun.space.space_id
                     and e.payload["trial"]["params"]["values"] == {"x": 0})
    assert bool(old_point.payload.get("reused_measurement")) == (mode == "same")
    assert old_point.payload["trial"]["latency_ms"]["median"] == (110.0 if mode == "changed" else 10.0)


@pytest.mark.parametrize("mode", ["noop", "reject"])
def test_existing_noop_or_rejection_preserves_initial_study(orch, monkeypatch, mode):
    # Given: real eligibility, but an unchanged domain or invalid expansion proposal.
    crun, accepted, captured = prepare(orch, monkeypatch, mode)
    # When: the native bounded expansion path considers that proposal.
    orch._continue_accepted_candidate(crun, accepted, run_analysis=False)
    # Then: no new budget or published space appears, and original measured source survives.
    events = orch.store.iter_events()
    assert len(crun.trials) == 40 and crun.space.space_id == accepted.space.space_id
    assert len(captured) == (1 if mode == "noop" else 2)
    assert sum(e.type == "TUNING_DONE" for e in events) == 1
    assert sum(e.type == "SPACE_PUBLISHED" for e in events) == 1
    assert not any(e.type == "SPACE_EXPANDED" for e in events)
    reasons = [e.payload["reason"] for e in events if e.type == "SPACE_EXPANSION_REJECTED"]
    assert reasons == (["no_new_choices"] if mode == "noop" else ["key_mismatch", "key_mismatch"])
    selected = crun.stats.best
    measured = orch.store.candidate_dir(crun.candidate.candidate_id) / "trials" / f"{selected.trial_id}.py"
    assert measured.read_text() == materialize(crun.source, selected.params)


def test_expansion_archives_base_and_changed_source_before_new_trials(orch, monkeypatch):
    # Given: an eligible expansion that changes the body and is worse everywhere.
    crun, accepted, _ = prepare(orch, monkeypatch, "changed")
    initial_source = crun.source
    # When: initial and expanded B40 execute through the shared continuation.
    orch._continue_accepted_candidate(crun, accepted, run_analysis=False)
    # Then: the pre-expansion snapshot retains its own score, space and exact measured bytes.
    events = orch.store.iter_events()
    eligibility = next(e for e in events if e.type == "SPACE_EXPANSION_ELIGIBILITY")
    base = eligibility.payload["base_best"]
    assert eligibility.payload["eligible"] is True
    assert base["space_id"] == eligibility.payload["space"]["space_id"] == accepted.space.space_id
    assert base["latency_ms"]["median"] == 10.0 and base["params"]["values"] == {"x": 0}
    assert orch.store.get_artifact(eligibility.payload["base_source_ref"]).decode() == initial_source
    measured = orch.store.candidate_dir(crun.candidate.candidate_id) / "trials" / f"{base['trial_id']}.py"
    assert orch.store.get_artifact(eligibility.payload["base_best_source_ref"]) == measured.read_bytes()
    assert measured.read_text() == materialize(initial_source, ParamSet(values={"x": 0}))
    expanded = next(e for e in events if e.type == "SPACE_EXPANDED")
    assert eligibility.seq < expanded.seq < next(e.seq for e in events if e.type == "TRIAL_DONE"
                                                and e.payload["trial"]["space_id"] == crun.space.space_id)
    assert expanded.payload["previous_space_id"] == accepted.space.space_id
    assert expanded.payload["space_id"] == crun.space.space_id != accepted.space.space_id
    assert expanded.payload["source_changed"] and not expanded.payload["prior_best_source_matches"]
    assert orch.store.get_artifact(expanded.payload["source_after_ref"]).decode() == crun.source


def test_ineligible_space_is_recorded_without_extra_budget(orch, monkeypatch):
    # Given: actual flat timing evidence, hence no meaningful boundary effect.
    crun, accepted, captured = prepare(orch, monkeypatch, "same")
    from kernel_optimizer.gpu.worker_client import WslGpuWorker
    def worker(self, job, timeout_s, tag, **kwargs):
        return {"ok": True, "latency_ms": {"mean": 10.0, "median": 10.0,
                "min": 10.0, "max": 10.0, "std": 0.0, "n": 20}}
    monkeypatch.setattr(WslGpuWorker, "run_job", worker)
    accepted.witnesses = [w.model_copy(update={"latency_mean_ms": None}) for w in accepted.witnesses]
    # When: numeric eligibility evaluates the completed initial B40.
    orch._continue_accepted_candidate(crun, accepted, run_analysis=False)
    # Then: an explicit negative eligibility result explains the absence of expansion.
    eligibility = next(e for e in orch.store.iter_events() if e.type == "SPACE_EXPANSION_ELIGIBILITY")
    assert eligibility.payload["eligible"] is False and eligibility.payload["knobs"] == []
    assert len(crun.trials) == 40 and captured == []


def test_finalize_uses_old_measured_winner_not_latest_source(orch, monkeypatch):
    # Given: a retained old-space optimum after a worse body-changing expansion.
    crun, accepted, _ = prepare(orch, monkeypatch, "changed")
    initial_source = crun.source
    orch._continue_accepted_candidate(crun, accepted, run_analysis=False)
    seen = []
    def final(task, path, backend):
        seen.append(path.read_text())
        return {"ok": True}
    monkeypatch.setattr(orch.deps.benchmarker, "final_reeval", final)
    # When: production finalization exports its already selected candidate.
    result = orch._finalize()
    # Then: it evaluates the measured artifact and reports the selected old space.
    assert seen == [materialize(initial_source, ParamSet(values={"x": 0}))]
    assert result["best"]["space_id"] == accepted.space.space_id
    assert crun.space.space_id != accepted.space.space_id


def test_production_context_reports_actual_selected_point_and_source(orch, monkeypatch):
    # Given: the latest tunable defaults/body differ from the retained measured winner.
    crun, accepted, _ = prepare(orch, monkeypatch, "changed")
    initial_source = crun.source
    orch._continue_accepted_candidate(crun, accepted, run_analysis=False)
    for name in ("_measure_best_overhead", "_classify_bottleneck", "_attribute_resource_walls",
                 "_attribute_soft_walls", "_conditional_brief"):
        monkeypatch.setattr(orch, name, lambda run: None)
    monkeypatch.setattr(orch, "_dimension_digest", lambda *args: None)
    analyst, rewrite = [], []
    monkeypatch.setattr(orch.deps.analyst, "invoke", lambda inputs: analyst.append(inputs)
                        or SimpleNamespace(output=BottleneckReport(summary="fixture")))
    monkeypatch.setattr(orch.deps.rewriter, "invoke", lambda inputs: rewrite.append(inputs)
                        or SimpleNamespace(output=SimpleNamespace(candidates=[])))
    # When: production analysis feeds a subsequent rewrite.
    orch._stats_and_analysis(crun)
    orch._do_rewrite(crun.candidate.family_id, crun)
    # Then: the analyst keeps tunable source, while rewrite receives the measured old winner.
    assert analyst[0].candidate_source == crun.source
    assert analyst[0].selected_params == rewrite[0].selected_params == ParamSet(values={"x": 0})
    assert extract_defaults(analyst[0].candidate_source) == {"x": 59}
    assert rewrite[0].best_source == materialize(initial_source, rewrite[0].selected_params)
    assert rewrite[0].source_materialized is True


def test_rewrite_marks_raw_fallback_when_measured_artifact_is_missing(orch, monkeypatch):
    # Given: selected params are known but their measured artifact is unavailable.
    crun, accepted, _ = prepare(orch, monkeypatch, "changed")
    orch._continue_accepted_candidate(crun, accepted, run_analysis=False)
    selected = orch._selected_trial(crun)
    path = orch.store.candidate_dir(crun.candidate.candidate_id) / "trials" / f"{selected.trial_id}.py"
    path.unlink()
    captured = []
    monkeypatch.setattr(orch.deps.rewriter, "invoke", lambda inputs: captured.append(inputs)
                        or SimpleNamespace(output=SimpleNamespace(candidates=[])))
    # When: the existing raw-source fallback feeds rewrite.
    orch._do_rewrite(crun.candidate.family_id, crun)
    # Then: no claim says the latest tunable source is the measured old best.
    assert captured[0].best_source == crun.source
    assert captured[0].selected_params == selected.params
    assert captured[0].source_materialized is False
