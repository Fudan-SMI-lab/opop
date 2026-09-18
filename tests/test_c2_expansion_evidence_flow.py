"""Expansion evidence is candidate-local, measured, marginal and policy-neutral."""

import json

import pytest

from kernel_optimizer.agents.runtime import AgentCallError
from kernel_optimizer.control.orchestrator import boundary_knobs_to_expand
from kernel_optimizer.models.core import (
    Constraint, LatencyStats, ParamDomain, ParameterSpace, ParamSet, ProfileRecord, TrialRecord,
)
from kernel_optimizer.models.reports import ResourceSnapshot
from kernel_optimizer.paramspace.materializer import materialize
from tests.test_c2_accepted_continuation import orch as orch


@pytest.fixture
def evidence(orch):
    source = ("PARAMS = {'SK1': 3, 'partner': 'A', 'gate': 1}\n"
              "def run():\n    if PARAMS['gate'] == 1: return 1\n    return PARAMS['SK1']\n")
    cand = orch._register(source, "rewrite", [], "triton", "candidate intent")
    crun = orch.runs[cand.candidate_id]
    crun.space = ParameterSpace(space_id="current", candidate_id=cand.candidate_id, source_sha="current",
        domains=[ParamDomain(name="SK1", kind="int", choices=[1, 2, 3], description="used when gate=2"),
                 ParamDomain(name="partner", kind="str", choices=["A", "B", "C"]),
                 ParamDomain(name="gate", kind="int", choices=[1, 2])],
        constraints=[Constraint(expr="SK1 >= 1", rationale="fixture")])
    for value, samples in [(1, [4.0, 4.026, 4.1]), (2, [3.5, 4.129, 4.6]), (3, [4.02, 4.048, 4.2])]:
        for partner, ms in zip(["A", "B", "C"], samples):
            crun.trials.append(TrialRecord(trial_id=f"t-{value}-{partner}", candidate_id=cand.candidate_id,
                space_id="old", params=ParamSet(values={"SK1": value, "partner": partner, "gate": 1}),
                status="complete", latency_ms=LatencyStats(mean=ms, median=ms, min=ms, max=ms, std=0, n_samples=20),
                profile=ProfileRecord(n_regs=32, shared_bytes=1024, n_spills=0)))
    crun.trials.append(TrialRecord(trial_id="failed", candidate_id=cand.candidate_id, space_id="current",
        params=ParamSet(values={"SK1": 2, "partner": "B", "gate": 1}), status="fail",
        failure_kind="compile_error", failure_detail="fixture compile failure"))
    crun.best_ms = 3.5
    crun.stats = orch.deps.stats_analyzer.analyze(crun.space, crun.trials)
    selected = orch._selected_trial(crun)
    directory = orch.store.candidate_dir(cand.candidate_id) / "trials"
    directory.mkdir()
    measured = directory / f"{selected.trial_id}.py"
    measured.write_text(materialize(source.replace("return 1", "return 0"), selected.params), encoding="utf-8")
    orch.task.ref_path.write_text("SHAPE = (2, 19)\ndef get_inputs(): return []\n", encoding="utf-8")
    orch.cfg.budgets.space_expansions_per_candidate = 1
    crun.recommended_configs = (ParamSet(values={"SK1": 3, "partner": "B", "gate": 1}),)
    return crun, measured


@pytest.mark.parametrize(("effect", "saturated", "expected"), [
    (2.0, False, [{"name": "SK1", "direction": "min"}]), (3.0, False, []), (2.0, True, []),
])
def test_existing_fallback_and_resource_policy_baseline(evidence, effect, saturated, expected):
    # Given: the non-monotone 4.026 / 4.129 / 4.048 marginal table with an interior winner.
    crun, _ = evidence
    if saturated:
        crun.stats.resource_at_best = ResourceSnapshot(regs_frac_of_limit=1.0, shared_frac_of_limit=1.0)
    # When: the existing eligibility function applies its unchanged thresholds.
    requests = boundary_knobs_to_expand(crun.stats, 0.8, crun.space, min_effect_pct=effect)
    # Then: fallback survives at 2%, but not below effect qualification or without headroom.
    assert requests == expected


def test_existing_failure_preference_does_not_cancel_only_request(evidence):
    # Given: the sole otherwise eligible fallback has a failing edge.
    crun, _ = evidence
    crun.stats.param_stats[0].failure_rate_by_value[repr(1)] = 0.9
    # When: native failure preference has no healthier alternative.
    requests = boundary_knobs_to_expand(crun.stats, 0.8, crun.space, max_edge_failure_frac=0.2)
    # Then: it still delivers the existing request, rather than becoming a hard veto.
    assert requests == [{"name": "SK1", "direction": "min"}]


@pytest.mark.parametrize(("missing_source", "missing_reference"), [(False, False), (True, False), (False, True)])
def test_native_input_contains_real_selected_artifact_and_local_history(
    orch, evidence, monkeypatch, missing_source, missing_reference,
):
    # Given: a selected old-space artifact distinct from latest defaults/body, plus an unrelated record.
    crun, measured = evidence
    selected = orch._selected_trial(crun)
    source = measured.read_bytes().decode("utf-8")
    reference = orch.task.ref_path.read_text(encoding="utf-8")
    local_trials = tuple(crun.trials)
    crun.trials.append(selected.model_copy(update={"candidate_id": "foreign", "trial_id": "foreign"}))
    if missing_source:
        measured.unlink()
    if missing_reference:
        orch.task.ref_path.unlink()
    captured = []
    def capture(inputs):
        captured.append(inputs)
        raise AgentCallError("stop after capturing the real expansion input")
    monkeypatch.setattr(orch.deps.parameterizer, "invoke", capture)
    # When: real eligibility enters the single existing expansion agent seam.
    orch._maybe_expand_space(crun, run_analysis=False)
    # Then: available evidence is exact, missing evidence stays absent, and no extra measurements occur.
    assert len(captured) == 1
    inputs = captured[0]
    assert inputs.selected_trial is selected and inputs.selected_trial.space_id == "old"
    assert inputs.selected_source == (None if missing_source else source)
    assert inputs.selected_source != crun.source and inputs.candidate_source == crun.source
    assert inputs.reference_source == (None if missing_reference else reference)
    assert inputs.expansion_stats is crun.stats and inputs.expansion_stats.space_id == "current"
    assert inputs.expansion_trials == local_trials
    assert inputs.expansion_trials[-1].failure_kind == "compile_error"
    assert inputs.selected_trial.profile.n_regs == 32
    assert inputs.recommended_configs == crun.recommended_configs
    assert inputs.prior_constraints == tuple((c.expr, c.rationale) for c in crun.space.constraints)
    assert inputs.rewrite_intent == crun.candidate.approach_summary
    assert not any(e.type == "TRIAL_DONE" for e in orch.store.iter_events())


def test_fallback_directive_serializes_marginal_counts_without_conditional_claim(orch, evidence):
    # Given: real non-monotone marginal evidence; SK1 is inactive in the observed gate=1 branch.
    crun, _ = evidence
    requests = boundary_knobs_to_expand(crun.stats, 0.8, crun.space)
    # When: the directive renders evidence for the existing native request.
    text = orch._expand_directive_text(crun, requests)
    # Then: structured data identifies fallback, mixed spaces, varying partners and actual support.
    payload = json.loads(text.split("```json\n", 1)[1].split("\n```", 1)[0])
    assert payload["statistics_scope"] == "marginal_nonconditional"
    assert payload["current_space_id"] == payload["stats_space_id"] == "current"
    assert payload["selected_space_id"] == "old" and payload["trial_space_ids"] == ["current", "old"]
    knob = payload["requests"][0]
    assert knob["name"] == "SK1" and knob["requested_direction"] == "min"
    assert knob["indication"] == "median_edge_fallback" and knob["actual_best_trial_value"] == 2
    assert knob["description"] == crun.space.domains[0].description
    assert knob["varying_partners"] is True and knob["conditional_trend"] is None
    assert knob["active_branch_effect"] is None
    assert [v["median_ms"] for v in knob["per_value"]] == [4.026, 4.129, 4.048]
    assert [v["n_complete"] for v in knob["per_value"]] == [3, 3, 3]
    assert [v["n_fail"] for v in knob["per_value"]] == [0, 1, 0]
    assert requests == boundary_knobs_to_expand(crun.stats, 0.8, crun.space)


@pytest.mark.parametrize("has_stats", [True, False])
def test_directive_identifies_winner_anchor_or_uncertainty(orch, evidence, has_stats):
    # Given: an edge-winning monotone marginal table, or missing statistics for that request.
    crun, _ = evidence
    for trial in crun.trials:
        if trial.latency_ms is not None:
            ms = float(trial.params.values["SK1"]) + 3.0
            trial.latency_ms = trial.latency_ms.model_copy(update={"mean": ms, "median": ms})
    crun.best_ms = 4.0
    crun.stats = orch.deps.stats_analyzer.analyze(crun.space, crun.trials)
    requests = boundary_knobs_to_expand(crun.stats, 0.8, crun.space)
    assert requests == [{"name": "SK1", "direction": "min"}]
    if not has_stats:
        crun.stats = None
    # When: only the directive is rendered; no eligibility or source is changed.
    text = orch._expand_directive_text(crun, requests)
    # Then: labels follow available stats, not a universal monotonic-improvement assertion.
    payload = json.loads(text.split("```json\n", 1)[1].split("\n```", 1)[0])
    assert payload["requests"][0]["indication"] == ("winner_anchored_marginal" if has_stats else "marginal_uncertain")
    assert payload["requests"][0]["actual_best_trial_value"] == 1


def test_directive_preserves_unknown_choices_and_excludes_foreign_support(orch, evidence):
    # Given: an unobserved offered choice and a foreign record that must not fill its support.
    crun, _ = evidence
    crun.space.domains[0].choices.append(4)
    crun.trials.append(crun.trials[0].model_copy(update={"candidate_id": "foreign", "space_id": "foreign-space",
        "params": ParamSet(values={"SK1": 4, "partner": "A", "gate": 1})}))
    requests = boundary_knobs_to_expand(crun.stats, 0.8, crun.space)
    # When: directive evidence is rendered without running or changing the offered domain.
    text = orch._expand_directive_text(crun, requests)
    # Then: missing observations stay unknown rather than importing another candidate's measurement.
    payload = json.loads(text.split("```json\n", 1)[1].split("\n```", 1)[0])
    assert payload["trial_space_ids"] == ["current", "old"]
    unseen = payload["requests"][0]["per_value"][-1]
    assert unseen["value"] == 4 and unseen["median_ms"] is None and unseen["failure_rate"] is None
    assert unseen["n_complete"] == unseen["n_fail"] == unseen["n_missing_latency"] == 0
