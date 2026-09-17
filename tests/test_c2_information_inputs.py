"""Information masking tests use actual prepared models, not GPU measurements."""

import importlib
from pathlib import Path

import pytest

from kernel_optimizer.conditional.task_response import TaskResponse
from kernel_optimizer.config import AppConfig
from kernel_optimizer.evaluation.task_eval import TaskEvaluation
from kernel_optimizer.models.core import ParamSet
from scripts.experiments.c2_local_costs import Costs
from scripts.experiments.c2_local_inputs import InputError, Responses, Shared, stage_inputs


def information_module(name="c2_information"):
    assert (Path(__file__).parents[1] / "scripts/experiments" / f"{name}.py").is_file(), "information entry is missing"
    return importlib.import_module(f"scripts.experiments.{name}")


@pytest.fixture
def prepared():
    cfg = AppConfig()
    shared = Shared.model_validate({
        "task": "level3:21", "state": "off", "source": "PARAMS={'x': 1}\ndef run(): return 1\n",
        "reference_source": "def reference(): return 1\n", "parent": {
            "trial_id": "parent", "candidate_id": "parent", "space_id": "parent-space",
            "params": {"values": {"x": 1}}, "status": "complete",
            "latency_ms": {"mean": 5, "std": 0, "min": 5, "max": 5, "n_samples": 20},
            "profile": {"n_regs": 31, "shared_bytes": 4096}},
        "space": {"params": [{"name": "x", "kind": "int", "choices": [1, 2]}]},
        "trials": [], "semantics": {"training": True}, "device": cfg.device,
        "evaluation": cfg.evaluation.model_dump(mode="json"),
    })
    shared = shared.model_copy(update={"trials": [shared.parent]})
    response = TaskResponse(
        candidate_id="parent", axis="x", a_params=shared.parent.params, b_params=ParamSet(values={"x": 2}),
        a=TaskEvaluation(score=5.0, metrics={"shared_bytes": 123456.0}, detail="resource bytes 123456"),
        b=TaskEvaluation(score=4.0, metrics={"shared_bytes": 234567.0}, detail="resource profile 234567"),
        delta_j=-1.0, gain=1.0, parameter_slope=-1.0,
        resource_deltas={"shared_bytes": 111111.0}, resource_slopes={"shared_bytes": -0.000009},
        unknown_resources=["resource_error_345678"], resource_status="observed", reason="shared error bytes 456789",
    )
    acquisition = Responses(shared_id=shared.identity(), responses=[response], probe_calls=2,
                            costs=Costs(worker_attempts=2, worker_wall_s=3.0))
    return shared, acquisition, cfg


def test_g1_keeps_j_and_params_but_withholds_every_fresh_resource_channel(prepared):
    # Given: fresh facts in metrics, endpoint details, unknown names, and free-form reason.
    module = information_module("c2_information_inputs")
    shared, acquisition, _ = prepared
    original = acquisition.model_dump()
    # When: each group's agent-facing payload is selected.
    g0, g1, g2 = [module.group_responses(shared, acquisition, group) for group in ("G0", "G1", "G2")]
    # Then: only G1's fresh resource information is withheld, and the acquisition stays unchanged.
    assert g0 == []
    assert g2[0].model_dump() == acquisition.responses[0].model_dump()
    assert g1[0].model_dump(include={"candidate_id", "axis", "a_params", "b_params", "delta_j", "gain", "parameter_slope"}) == g2[0].model_dump(include={"candidate_id", "axis", "a_params", "b_params", "delta_j", "gain", "parameter_slope"})
    for masked, full in ((g1[0].a, g2[0].a), (g1[0].b, g2[0].b)):
        assert masked.score == full.score and masked.valid == full.valid
        assert masked.metrics == {} and masked.detail is None
    assert g1[0].resource_deltas == g1[0].resource_slopes == {}
    assert g1[0].unknown_resources == [] and g1[0].resource_status == "unknown"
    assert g1[0].reason is not None and g1[0].reason != g2[0].reason
    assert acquisition.model_dump() == original


def test_g1_keeps_failure_validity_without_resource_error_details(prepared):
    # Given: an endpoint whose failure detail discloses compiled shared-memory bytes.
    module = information_module("c2_information_inputs")
    shared, acquisition, _ = prepared
    response = acquisition.responses[0].model_copy(update={
        "b": TaskEvaluation(valid=False, detail="requires 234567 shared bytes", metrics={"shared_bytes": 234567.0}),
        "delta_j": None, "gain": None, "parameter_slope": None,
    })
    package = acquisition.model_copy(update={"responses": [response]})
    # When: G1 is rendered from the validated full package.
    masked = module.group_responses(shared, package, "G1")[0]
    # Then: invalidity remains observable without the explanation or fresh resource fields.
    assert not masked.b.valid and masked.b.score is None
    assert masked.b.detail is None and masked.b.metrics == {}
    assert masked.delta_j is None and masked.gain is None


@pytest.mark.parametrize("group", ["G0", "G1", "G2"])
def test_full_envelope_is_validated_even_when_information_will_be_hidden(prepared, group):
    module = information_module("c2_information_inputs")
    shared, acquisition, _ = prepared
    invalid = acquisition.model_copy(update={"shared_id": "another-parent"})
    with pytest.raises(InputError):
        module.group_responses(shared, invalid, group)


def test_ordinary_baseline_files_and_parent_resources_are_identical_for_all_groups(tmp_path, prepared):
    # Given: ordinary resource evidence which must not be masked.
    module = information_module("c2_information_inputs")
    shared, acquisition, _ = prepared
    # When: the existing common-input stager is used for each information group.
    staged = [stage_inputs(shared, tmp_path / group, module.group_responses(shared, acquisition, group))
              for group in ("G0", "G1", "G2")]
    # Then: group labels and fresh measurements never change common evidence.
    assert all(s.model_dump(exclude={"responses"}) == staged[0].model_dump(exclude={"responses"}) for s in staged)
    assert all(s.resources["n_regs"] == 31 for s in staged)
    for name in ("ordinary.json", "stats.json", "device.json", "semantics.json", "evaluation.json", "parent.py", "reference.py"):
        assert len({(tmp_path / group / name).read_bytes() for group in ("G0", "G1", "G2")}) == 1
