"""CPU contract tests; no model or GPU evidence."""

import importlib
import importlib.util
import json
from pathlib import Path

import pytest


def runner_module(name: str):
    qualified = f"scripts.experiments.c2_local_{name}"
    assert (Path(__file__).parents[1] / "scripts" / "experiments" / f"c2_local_{name}.py").is_file(), "C2 runner seam is missing"
    assert importlib.util.find_spec(qualified) is not None, "C2 runner seam is missing"
    return importlib.import_module(qualified)


def test_preparation_keeps_only_ordinary_prefix_when_history_has_probes(tmp_path: Path):
    # Given: an explicit tuned parent, a scan point, and a future faster child.
    inputs = runner_module("inputs")
    history = tmp_path / "history"
    history.mkdir()
    source = "PARAMS = {'tile': 2}\ndef work():\n    return PARAMS['tile']\n"
    parent = history / "candidates" / "parent" / "trials" / "chosen.py"
    parent.parent.mkdir(parents=True)
    parent.write_text(source)
    (history / "manifest.json").write_text(json.dumps({"task": "reference",
        "config": inputs.AppConfig().model_dump(mode="json")}))
    space = tmp_path / "space.json"
    space.write_text(json.dumps({"params": [{"name": "tile", "kind": "int", "choices": [1, 2, 3]}]}))
    reference = tmp_path / "reference.py"
    reference.write_text("def reference(): return 1\n")
    semantics = tmp_path / "semantics.json"
    semantics.write_text('{"training": true}')
    trial = {"trial_id": "chosen", "candidate_id": "parent", "space_id": "space",
             "params": {"values": {"tile": 2}}, "status": "complete",
             "latency_ms": {"mean": 5, "std": 0, "min": 5, "max": 5, "n_samples": 20}}
    events = [
        {"seq": 0, "type": "SPACE_PUBLISHED", "payload": {"space": {
            "space_id": "space", "candidate_id": "parent", "source_sha": "original",
            "domains": [{"name": "tile", "kind": "int", "choices": [1, 2, 3]}]}}},
        {"seq": 1, "type": "TRIAL_DONE", "payload": {"trial": trial}},
        {"seq": 2, "type": "TRIAL_DONE", "payload": {"trial": {**trial, "trial_id": "probe"}}},
        {"seq": 3, "type": "SCAN_POINT_DONE", "payload": {"trial_id": "probe"}},
        {"seq": 5, "type": "TRIAL_DONE", "payload": {"trial": {**trial, "trial_id": "future"}}},
    ]
    (history / "events.jsonl").write_text("\n".join(json.dumps({**e, "ts": e["seq"]}) for e in events))
    spec = inputs.Selection(task="level3:43", historical_run=history, parent_id="parent",
                            parent_trial_id="chosen", parent_source=parent, space=space,
                            cutoff_seq=4, state="off", semantics=semantics)
    # When: shared inputs are prepared without invoking any runtime.
    shared = inputs.prepare(spec, inputs.AppConfig(), reference)
    # Then: tuned values and ordinary prefix survive; scans/future evidence do not.
    assert shared.parent.params.values == {"tile": 2}
    assert [t.trial_id for t in shared.trials] == ["chosen"]
    assert shared.source == source
    space.write_text(json.dumps({"params": [{"name": "tile", "kind": "int", "choices": [2, 3]}]}))
    with pytest.raises(inputs.InputError, match="published space"):
        inputs.prepare(spec, inputs.AppConfig(), reference)


def test_arm_inputs_differ_only_in_responses_when_b_receives_treatment(tmp_path: Path):
    # Given: a baked shared package and current response.
    inputs = runner_module("inputs")
    shared = inputs.Shared.model_validate({
        "task": "level3:43", "state": "off", "source": "PARAMS={'x': 2}\n",
        "reference_source": "reference", "parent": {
            "trial_id": "p", "candidate_id": "parent", "space_id": "s",
            "params": {"values": {"x": 2}}, "status": "complete",
            "latency_ms": {"mean": 5, "std": 0, "min": 5, "max": 5, "n_samples": 20}},
        "space": {"params": [{"name": "x", "kind": "int", "choices": [1, 2]}]},
        "trials": [], "semantics": {"training": True}, "device": {},
    })
    response = inputs.TaskResponse(candidate_id="parent", axis="x", a_params=shared.parent.params)
    # When: both arms receive independently staged inputs.
    a = inputs.stage_inputs(shared, tmp_path / "a", [])
    b = inputs.stage_inputs(shared, tmp_path / "b", [response])
    # Then: serialized agent inputs differ only in treatment, not paths/state.
    assert a.model_dump(exclude={"responses"}) == b.model_dump(exclude={"responses"})
    assert a.responses == [] and b.responses == [response]
    assert not a.project_root.is_absolute() and not a.candidate_path.is_absolute()
    assert set(a.source_paths) >= {Path("ordinary.json"), Path("semantics.json"), Path("device.json")}
    b.space.params.clear()
    assert len(a.space.params) == len(shared.space.params) == 1


def test_selection_rejects_fewer_than_three_final_blocks():
    # Given / When / Then: the experiment boundary refuses an under-sized final protocol.
    inputs = runner_module("inputs")
    with pytest.raises(ValueError):
        inputs.RunOptions(arm="A", path="direct", final_blocks=2)


def test_b_requires_acquisition_envelope_when_responses_are_missing():
    inputs = runner_module("inputs")
    with pytest.raises(ValueError):
        inputs.RunOptions(arm="B", path="direct")


def test_protocol_changes_when_seed_or_model_changes():
    inputs = runner_module("inputs")
    cfg = inputs.AppConfig()
    changed = cfg.model_copy(deep=True)
    changed.run.seed = 71
    assert inputs.protocol(cfg) != inputs.protocol(changed)
    changed = cfg.model_copy(deep=True)
    changed.agents.default_model = "provider/other-model"
    assert inputs.protocol(cfg) != inputs.protocol(changed)
