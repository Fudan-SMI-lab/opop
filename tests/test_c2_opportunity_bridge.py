"""Selected-context bridge tests; real agent seeding with provider transport only faked."""

import json
import sys
from contextlib import chdir, closing
from pathlib import Path

import pytest

from kernel_optimizer.agents.modules import BottleneckAnalystAgent, StructureRewriterAgent
from kernel_optimizer.agents.runtime import OpencodeClient, PromptResult
from kernel_optimizer.conditional.task_response import TaskResponse
from kernel_optimizer.config import AppConfig
from kernel_optimizer.evaluation.task_eval import TaskEvaluation
from kernel_optimizer.models.core import ParamSet
from kernel_optimizer.paramspace.materializer import extract_defaults, materialize
from kernel_optimizer.store.run_store import RunStore
from kernel_optimizer.wiring import Runtime
from scripts.experiments import c2_local_agents as agents
from scripts.experiments.c2_local_inputs import InputError, Shared, stage_inputs


@pytest.fixture
def bridge(tmp_path, monkeypatch):
    shared = Shared.model_validate({
        "task": "level3:21", "state": "off",
        "source": "PARAMS={'x': 1, 'partner': 2}\ndef run(): return PARAMS['x'] * PARAMS['partner']\n",
        "reference_source": "def reference(): return 1\n",
        "parent": {"trial_id": "selected", "candidate_id": "parent", "space_id": "space",
                   "params": {"values": {"x": 7, "partner": 5}}, "status": "complete",
                   "latency_ms": {"mean": 5, "std": 0, "min": 5, "max": 5, "n_samples": 20}},
        "space": {"params": [{"name": "x", "kind": "int", "choices": [1, 3, 7, 9]},
                              {"name": "partner", "kind": "int", "choices": [2, 5, 8]}],
                  "constraints": [{"expr": "x * partner <= 80"}]},
        "trials": [], "semantics": {"training": True}, "device": {},
    })
    shared = shared.model_copy(update={"trials": [shared.parent]})
    response = TaskResponse(candidate_id="parent", axis="x",
        a_params=ParamSet(values={"x": 1, "partner": 5}), b_params=ParamSet(values={"x": 9, "partner": 5}),
        a=TaskEvaluation(score=6.0), b=TaskEvaluation(score=4.0), delta_j=-2.0, gain=2.0, parameter_slope=-0.25)
    common = tmp_path / "common"
    inputs = stage_inputs(shared, common, [response]).model_copy(update={"project_root": common})
    captured, calls, payloads = [], [], {}

    def session(self, directory, title):
        return title

    def prompt(self, session_id, text, **kwargs):
        directory, title = kwargs["directory"], kwargs["schema"]["title"]
        calls.append(title)
        if title in {"BottleneckReport", "RewriteResult"}:
            selected_file = directory / "tuning/selected_params.json"
            response_file = directory / "analysis/conditional_responses.md"
            payloads[title] = {
                "source": (directory / ("candidate/source.py" if title == "BottleneckReport" else "candidate/best.py")).read_text(),
                "selected": json.loads(selected_file.read_text()) if selected_file.exists() else None,
                "responses": response_file.read_text() if response_file.exists() else None,
                "wall_present": (directory / "analysis/resource_walls.md").exists(),
            }
        (directory / "child.py").write_text("import triton\nPARAMS={'x': 2}\n@triton.jit\ndef child(): return 3\n")
        answers = {
            "BottleneckReport": {"summary": "fixture", "hypotheses": []},
            "RewriteResult": {"candidates": [{"file": "child.py", "backend": "triton",
                                               "change_summary": "fixture", "hypothesis_id": "H1"}]},
            "ParameterizationResult": {"file": "child.py", "space": {"params": [
                {"name": "x", "kind": "int", "choices": [1, 2, 4]}]}},
            "TaskRewriteResult": {"candidate_file": "child.py"},
        }
        return PromptResult(text="", structured=answers[title], session_id=session_id)

    def profile(frame, event, arg):
        if event == "call" and frame.f_code in {BottleneckAnalystAgent.seed_sandbox.__code__, StructureRewriterAgent.seed_sandbox.__code__}:
            captured.append(frame.f_locals["inputs"])

    monkeypatch.setattr(OpencodeClient, "create_session", session)
    monkeypatch.setattr(OpencodeClient, "prompt", prompt)
    previous = sys.getprofile()
    with closing(OpencodeClient("http://unused.invalid")) as client:
        runtime = Runtime(AppConfig())
        runtime.client = client
        services = agents.Services(runtime.cfg, RunStore.create(tmp_path, "generation", {}), runtime)
        try:
            sys.setprofile(profile)
            yield shared, inputs, services, captured, calls, payloads
        finally:
            sys.setprofile(previous)


@pytest.mark.parametrize("staging", ["project_root", "absolute_candidate", "legacy_cwd", "missing"])
def test_agent_files_use_selected_point_without_mutating_tunable_inputs(bridge, tmp_path, monkeypatch, staging):
    # Given: defaults deliberately differ in both the probe axis and fixed partner.
    shared, inputs, services, captured, calls, payloads = bridge
    original = shared.model_dump()
    common = inputs.project_root
    expected = (common / "parent.py").read_text()
    decoy = tmp_path / "other-opportunity"
    decoy.mkdir()
    (decoy / "parent.py").write_text("PARAMS={'x': 7, 'partner': 5}\ndef wrong_parent(): return 999\n")
    if staging == "missing":
        inputs = inputs.model_copy(update={"candidate_path": Path("absent.py")})
    elif staging == "absolute_candidate":
        inputs = inputs.model_copy(update={"project_root": decoy, "candidate_path": common / "parent.py"})
    elif staging == "legacy_cwd":
        inputs = inputs.model_copy(update={"project_root": Path(".")})
    materializations = []

    def actual_materialize(source, params):
        materializations.append(params)
        return materialize(source, params)

    monkeypatch.setattr(agents, "materialize", actual_materialize, raising=False)
    # When: the bridge seeds actual agent sandboxes, with an unrelated cwd unless legacy-bound.
    with chdir(common if staging == "legacy_cwd" else decoy):
        agents.generate_legacy_proposal(shared, services, inputs)
    # Then: analyst keeps raw defaults, rewriter gets the actual selected source and params.
    assert payloads["BottleneckReport"]["source"] == shared.source
    assert payloads["RewriteResult"]["source"] == expected
    assert extract_defaults(payloads["RewriteResult"]["source"]) == shared.parent.params.values
    assert all(p["selected"] == shared.parent.params.model_dump() for p in payloads.values())
    assert captured[0].selected_params == captured[1].selected_params == shared.parent.params
    assert captured[1].source_materialized is True
    assert len(materializations) == (1 if staging == "missing" else 0)
    assert calls == ["BottleneckReport", "RewriteResult"] and shared.model_dump() == original
    response = json.loads(payloads["RewriteResult"]["responses"])
    assert response["a_params"]["values"]["partner"] == response["b_params"]["values"]["partner"] == 5
    assert response["a_params"]["values"] != shared.parent.params.values
    assert response["b_params"]["values"] != shared.parent.params.values


@pytest.mark.parametrize("batch", ["empty", "placeholder", "failed", "unknown", "mixed", "observed_not_acquired"])
def test_only_unmeasured_g0_placeholders_are_removed_from_fresh_brief(bridge, batch):
    # Given: synthetic G0 absence versus genuinely supplied failed/unknown responses.
    shared, inputs, services, captured, calls, payloads = bridge
    placeholder = TaskResponse(candidate_id="parent", axis="x", a_params=shared.parent.params, reason="not_acquired")
    unknown = placeholder.model_copy(update={"reason": "probe_budget"})
    failed = placeholder.model_copy(update={"reason": "invalid_endpoint", "a": TaskEvaluation(valid=False, detail="worker failure")})
    supplied = {"empty": [], "placeholder": [placeholder], "failed": [failed], "unknown": [unknown],
                "mixed": [placeholder, failed], "observed_not_acquired": [failed.model_copy(update={"reason": "not_acquired"})]}[batch]
    # When: only the information payload, not group naming, controls the response channel.
    agents.generate_legacy_proposal(shared, services, inputs.model_copy(update={"responses": supplied}))
    # Then: absence does not activate a fresh brief; real negative/unknown evidence survives verbatim.
    expected = supplied[-1].model_dump_json() if batch not in {"empty", "placeholder"} else None
    assert all(p["responses"] == expected and not p["wall_present"] for p in payloads.values())
    assert all(item.conditional_response_text == expected for item in captured)
    assert captured[-1].wall_text is None


@pytest.mark.parametrize("corruption", ["params", "body"])
def test_mismatched_staged_parent_is_not_claimed_materialized(bridge, corruption):
    # Given: a staged file from the wrong point or another opportunity's body.
    shared, inputs, services, captured, calls, payloads = bridge
    path = inputs.project_root / inputs.candidate_path
    source = path.read_text()
    path.write_text(source.replace("7", "9") if corruption == "params" else source.replace("return", "raise"))
    # When / Then: do not send a falsely materialized parent to an agent.
    with pytest.raises(InputError):
        agents.generate_legacy_proposal(shared, services, inputs)
    assert calls == []


@pytest.mark.parametrize("entry", ["legacy", "direct"])
def test_existing_child_callers_remain_compatible(bridge, entry):
    shared, inputs, services, captured, calls, payloads = bridge
    inputs = inputs.model_copy(update={"candidate_path": inputs.project_root / inputs.candidate_path})
    child = agents.legacy_child(shared, services, inputs) if entry == "legacy" else agents.direct_child(inputs, services)
    assert child.path.is_file()
    assert calls == (["BottleneckReport", "RewriteResult", "ParameterizationResult"] if entry == "legacy" else ["TaskRewriteResult"])
