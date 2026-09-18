"""Recommended points are requests for native measurements, never private scores."""

from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from kernel_optimizer.models.core import Constraint, ParamSet
from kernel_optimizer.models.reports import BottleneckReport, ParameterizationResult
from kernel_optimizer.paramspace.validation import SpaceAccepted
from scripts.experiments import c2_local_agents as bridge
from scripts.experiments.c2_retune import RetuneInputs, tune_existing
from tests.test_c2_accepted_continuation import orch as orch
from tests.test_c2_information_inputs import prepared as prepared


POINTS = (ParamSet.model_validate({"values": {"x": 97, "y": 83}, "private_latency_ms": 0.001}),
          ParamSet(values={"x": 91, "y": 87}))


def proposal(source: str, points=()):
    return ParameterizationResult.model_validate({"file": "child.py", "space": {"params": [
        {"name": "x", "kind": "int", "choices": list(range(100))},
        {"name": "z" if "'z'" in source else "y", "kind": "int", "choices": list(range(100))}]},
        "recommended_configs": [p.model_dump() for p in points]})


def joint(orch, points=()):
    source = "PARAMS = {'x': 7, 'y': 2}\ndef run(): return PARAMS['x']\n"
    cand = orch._register(source, "rewrite", [], "triton", "joint-region fixture",
                          recommended_configs=points)
    crun = orch.runs[cand.candidate_id]
    accepted = orch.deps.validator.validate_and_publish(
        cand, source, proposal(source), orch.task, orch.store.candidate_dir(cand.candidate_id))
    assert isinstance(accepted, SpaceAccepted)
    return crun, accepted


@pytest.mark.parametrize("enabled", [False, True])
def test_joint_recommendations_are_normal_trials_inside_b40(orch, enabled):
    # Given: two joint points not covered by the seed-zero baseline, not either witness.
    crun, accepted = joint(orch, POINTS if enabled else ())
    # When: the real native sampler/evaluator runs the unchanged B40 budget.
    orch._continue_accepted_candidate(crun, accepted, run_analysis=False)
    # Then: recommendations follow witnesses and carry formal worker scores, not a private claim.
    assert len(crun.trials) == 40
    assert [t.params for t in crun.trials[:2]] == [w.params for w in accepted.witnesses]
    if enabled:
        assert tuple(t.params for t in crun.trials[2:4]) == POINTS
        assert [t.latency_ms.robust_ms for t in crun.trials[2:4]] == [107.0, 101.0]
        events = orch.store.iter_events()
        queued = [e for e in events if e.type == "RECOMMENDED_CONFIG_QUEUED"]
        assert len(queued) == 2
        assert all(not e.payload.get("reused_measurement") for e in events if e.type == "TRIAL_DONE"
                   and e.payload["trial"]["params"] in [p.model_dump() for p in POINTS])
    else:
        assert all(p not in [t.params for t in crun.trials] for p in POINTS)


@pytest.mark.parametrize(("point", "reason"), [
    ({"x": 5}, "missing_param"), ({"x": 5, "y": 2, "extra": 1}, "unknown_param"),
    ({"x": 100, "y": 2}, "not_in_choices"), ({"x": 4, "y": 99}, "constraint_violated"),
    ({"x": 7, "y": 2}, "duplicate_anchor"), ({"x": "bad", "y": 2}, "type_mismatch"),
])
def test_bad_or_duplicate_recommendation_does_not_reject_candidate(orch, point, reason):
    # Given: a complete valid candidate and one inadmissible or redundant recommendation.
    crun, accepted = joint(orch, (ParamSet(values=point),))
    accepted.space.constraints.append(Constraint(expr="x + y < 150", rationale="fixture"))
    if reason == "constraint_violated":
        accepted.space.constraints.append(Constraint(expr="y < 90", rationale="fixture"))
    # When: native continuation filters requests against its published space.
    orch._continue_accepted_candidate(crun, accepted, run_analysis=False)
    # Then: the candidate still finishes B40 and the skip has an explicit reason.
    skipped = [e.payload["reason"] for e in orch.store.iter_events() if e.type == "RECOMMENDED_CONFIG_SKIPPED"]
    assert skipped == [reason] and len(crun.trials) == 40
    assert crun.candidate.status == "tuned"


@pytest.mark.parametrize("resolved", [False, True])
def test_production_parameterization_uses_only_resolved_output(orch, monkeypatch, resolved):
    # Given: incoming recommendations refer to a key renamed by parameterization.
    crun, _ = joint(orch, POINTS)
    source = crun.source.replace("'y'", "'z'")
    mapped = (ParamSet(values={"x": 97, "z": 83}),) if resolved else ()
    captured = []
    monkeypatch.setattr(orch.deps.parameterizer, "invoke", lambda inputs: captured.append(inputs)
                        or SimpleNamespace(output=proposal(source, mapped),
                                           sandbox=SimpleNamespace(read_output=lambda file: source)))
    monkeypatch.setattr(orch.deps.analyst, "invoke", lambda inputs: SimpleNamespace(
        output=BottleneckReport(summary="fixture")))
    # When: the production pipeline parameterizes, validates and tunes once.
    orch._candidate_pipeline(crun.candidate.candidate_id)
    # Then: missing mappings are not guessed; resolved keys reach actual native measurements.
    assert captured[0].recommended_configs == POINTS
    assert crun.recommended_configs == mapped and len(crun.trials) == 40
    if resolved:
        assert crun.trials[2].params == mapped[0]
    else:
        skipped = [e.payload for e in orch.store.iter_events() if e.type == "RECOMMENDED_CONFIG_SKIPPED"]
        assert len(skipped) == 2 and all(e["reason"] == "mapping_unknown" for e in skipped)


@pytest.mark.parametrize("resolved_x", [80, None, 1000])
def test_expansion_replaces_recommendations_after_prior_best_and_witnesses(orch, monkeypatch, resolved_x):
    # Given: a body-changing expanded space with a different resolved recommendation.
    crun, accepted = joint(orch, POINTS)
    orch.cfg.budgets.space_expansions_per_candidate = 1
    source = crun.source.replace("'x': 7", "'x': 59").replace("PARAMS['x']", "PARAMS['x'] + 1")
    mapped = (ParamSet(values={"x": resolved_x, "y": 81}),) if resolved_x is not None else ()
    output = proposal(source, mapped)
    output.space.params[0].choices.insert(0, -1)
    captured = []
    monkeypatch.setattr(orch.deps.parameterizer, "invoke", lambda inputs: captured.append(inputs)
                        or SimpleNamespace(output=output, sandbox=SimpleNamespace(read_output=lambda file: source)))
    # When: real eligibility admits the second B40.
    orch._continue_accepted_candidate(crun, accepted, run_analysis=False)
    # Then: the new list replaces the old, old best remains first, and no old score is reused on new body.
    assert captured[0].recommended_configs == POINTS and crun.recommended_configs == mapped
    assert len(crun.trials) == 80 and crun.best_ms == 10.0
    assert [t.params.values for t in crun.trials[40:43]] == [
        {"x": 0, "y": 0}, {"x": 59, "y": 2}, {"x": -1, "y": 0}]
    if resolved_x is not None and resolved_x < 100:
        assert crun.trials[43].params == mapped[0]
    else:
        assert not any(e.type == "RECOMMENDED_CONFIG_QUEUED" and e.payload["space_id"] == crun.space.space_id
                       for e in orch.store.iter_events())
    assert crun.trials[40].latency_ms.robust_ms == 110.0


@pytest.mark.parametrize(("pass_intent", "resolved"), [(True, True), (True, False), (False, False)])
def test_bridge_keeps_resolved_metadata_without_reintroducing_disabled_input(
    orch, monkeypatch, prepared, tmp_path, pass_intent, resolved,
):
    # Given: an optional rewriter recommendation and parameterizer-owned mapping.
    shared, _, cfg = prepared
    source = "PARAMS = {'x': 7, 'z': 2}\ndef run(): return PARAMS['x']\n"
    (tmp_path / "child.py").write_text(source)
    mapped = (ParamSet(values={"x": 97, "z": 83}),) if resolved else ()
    captured = []
    monkeypatch.setattr(bridge, "build_orchestrator", lambda *args: orch)
    monkeypatch.setattr(orch.deps.parameterizer, "invoke", lambda inputs: captured.append(inputs)
                        or SimpleNamespace(output=proposal(source, mapped), sandbox=SimpleNamespace(root=tmp_path)))
    incoming = bridge.LegacyProposal(source, "triton", "intent", "H1", BottleneckReport(summary="fixture"), POINTS)
    # When: the actual bridge parameterizes this proposal.
    child = bridge.parameterize_legacy_proposal(shared, bridge.Services(cfg, orch.store, None), incoming,
                                               pass_intent=pass_intent)
    # Then: output mappings alone enter Child; disabling intent also disables incoming rewriter requests.
    assert captured[0].recommended_configs == (POINTS if pass_intent else ())
    assert child.recommended_configs == mapped


def test_retune_input_limits_recommendations_and_delivers_them_to_native(orch, tmp_path):
    # Given: an already parameterized child and at most two source-local recommendations.
    crun, accepted = joint(orch)
    source = tmp_path / "kernel.py"
    space = tmp_path / "space.json"
    source.write_text(crun.source)
    space.write_text(accepted.space.model_dump_json())
    staged = orch.store.run_dir / "inputs"
    staged.mkdir()
    (staged / source.name).write_text(crun.source.replace("return PARAMS", "return 1 + PARAMS"))
    values = dict(task="level3:21", source=source, space=space, reference=orch.task.ref_path,
                  sampler_seed=0, evaluation_seed=0, output=orch.store.run_dir, final_blocks=0,
                  recommended_configs=POINTS)
    with pytest.raises(ValidationError):
        RetuneInputs.model_validate({**values, "recommended_configs": [*POINTS, POINTS[0]]})
    # When: retune consumes its real validated input, without another parameterizer call.
    result = tune_existing(orch, RetuneInputs.model_validate(values))
    # Then: recommendations survive registration and take normal positions inside B40.
    assert result.status == "complete" and len(result.trials) == 40
    assert tuple(t.params for t in result.trials[2:4]) == POINTS


@pytest.mark.parametrize("foreign_basis", [False, True])
def test_duplicate_points_or_foreign_source_do_not_add_extra_anchors(orch, foreign_basis):
    # Given: a repeated point or recommendation whose source basis is not this candidate's source.
    crun, accepted = joint(orch, (POINTS[0], POINTS[0]))
    if foreign_basis:
        crun.recommendations_source_sha = "other-source"
    # When: recommendations are considered without changing the validated witness cache.
    orch._continue_accepted_candidate(crun, accepted, run_analysis=False)
    # Then: no foreign-source request is queued, and repetition queues only one request.
    events = orch.store.iter_events()
    assert sum(e.type == "RECOMMENDED_CONFIG_QUEUED" for e in events) == (0 if foreign_basis else 1)
    assert sum(bool(e.payload.get("reused_measurement")) for e in events if e.type == "TRIAL_DONE") == 2
    assert len(crun.trials) == 40


def test_information_passes_generated_resolved_configs_to_actual_retune(orch, prepared, monkeypatch, tmp_path):
    # Given: the actual bridge and information consumer, with only external agent/worker calls replaced.
    from kernel_optimizer.agents.base import AgentModule
    from kernel_optimizer.agents.runtime import OpencodeServer
    from kernel_optimizer.models.reports import RewriteResult
    from scripts.experiments.c2_information import run_information
    from scripts.experiments.c2_information_inputs import InformationRun
    shared, acquisition, cfg = prepared
    cfg.gpu.compile_screen_enabled = False
    source = "PARAMS = {'x': 7, 'y': 2}\ndef run(): return PARAMS['x']\n"
    sandbox = tmp_path / "model"
    sandbox.mkdir()
    (sandbox / "child.py").write_text(source)
    captured = []
    def invoke(self, inputs):
        captured.append((self.name, inputs))
        outputs = {"analyst": BottleneckReport(summary="fixture"),
                   "rewriter": RewriteResult.model_validate({"candidates": [{"file": "child.py",
                       "backend": "triton", "change_summary": "fixture", "hypothesis_id": "H1",
                       "recommended_configs": [p.model_dump() for p in POINTS]}]}),
                   "parameterizer": proposal(source, POINTS)}
        return SimpleNamespace(output=outputs[self.name],
                               sandbox=SimpleNamespace(root=sandbox, read_output=lambda file: source))
    monkeypatch.setattr(AgentModule, "invoke", invoke)
    monkeypatch.setattr(OpencodeServer, "start", lambda self: "http://unused.invalid")
    monkeypatch.setattr(OpencodeServer, "stop", lambda self: None)
    # When: the real information route generates, parameterizes, validates and retunes its child.
    result = run_information(shared, acquisition, InformationRun(cfg, "G0", tmp_path / "information"))
    # Then: both producer stages and actual RetuneInputs preserve the requests inside native B40.
    assert result.error is None and len(result.child.trials) == 40
    assert tuple(t.params for t in result.child.trials[2:4]) == POINTS
    assert [name for name, _ in captured] == ["analyst", "rewriter", "parameterizer"]
    assert captured[-1][1].recommended_configs == POINTS


def test_production_rewrite_registers_its_own_recommendations(orch, monkeypatch):
    # Given: a tuned parent and a different child with its own suggested points.
    from kernel_optimizer.models.reports import RewriteResult
    parent, accepted = joint(orch)
    orch._continue_accepted_candidate(parent, accepted, run_analysis=False)
    source = parent.source.replace("PARAMS['x']", "PARAMS['x'] * 2")
    output = RewriteResult.model_validate({"candidates": [{"file": "child.py", "backend": "triton",
        "change_summary": "child-only", "hypothesis_id": "H1",
        "recommended_configs": [p.model_dump() for p in POINTS]}]})
    monkeypatch.setattr(orch.deps.rewriter, "invoke", lambda inputs: SimpleNamespace(
        output=output, sandbox=SimpleNamespace(read_output=lambda file: source)))
    captured = []
    monkeypatch.setattr(orch.deps.parameterizer, "invoke", lambda inputs: captured.append(inputs)
                        or SimpleNamespace(output=proposal(source, POINTS),
                                           sandbox=SimpleNamespace(read_output=lambda file: source)))
    monkeypatch.setattr(orch.deps.analyst, "invoke", lambda inputs: SimpleNamespace(
        output=BottleneckReport(summary="fixture")))
    # When: the production rewrite enters registration and the normal candidate pipeline.
    orch._do_rewrite(parent.candidate.family_id, parent)
    # Then: only the child receives its recommendations; parent history is not imported.
    child = next(run for run in orch.runs.values() if run is not parent)
    assert captured[0].recommended_configs == child.recommended_configs == POINTS
    assert tuple(t.params for t in child.trials[2:4]) == POINTS and len(child.trials) == 40
    assert parent.recommended_configs == () and len(parent.trials) == 40
