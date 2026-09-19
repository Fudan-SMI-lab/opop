"""Selected-site preparation follows the handed-off resident binding lifecycle."""

import json
from contextlib import closing
from pathlib import Path
from time import time

from examples.c3_qwen3.model_binding import load_bundle
from examples.c3_qwen3.search_records import EvaluationRequest
from kernel_optimizer.models.core import ParamSet
from tests.c3_search_fakes import search_case
from tests.c3_tiny_backend import Scale


def candidate(directory: Path, parent: Path, sites):
    directory.mkdir()
    data = json.loads(parent.read_text())
    data.update(parent_bundle=load_bundle(parent, {}).bundle_sha256, sites=[
        {"site_id": key, "replacement_callable": "replace"} for key in sites],
        params={"speed": 2}, space={"params": [{"name": "speed", "kind": "int", "choices": [2]}]})
    (directory / "operators.py").write_text("def replace(site, call, params):\n    return call.args[0] + call.args[0]\n")
    path = directory / "bundle.json"
    path.write_text(json.dumps(data))
    return path


def test_new_site_is_registered_only_after_restore_and_only_selected_fixtures_captured(tmp_path):
    # Given: two actual traced modules, but only one is requested initially.
    session, provider, _, _ = search_case(tmp_path)
    backend = session.runner.backend
    backend.model.other = Scale()
    backend.model.named_modules = lambda: iter((("scale", backend.model.scale), ("other", backend.model.other)))
    from examples.c3_qwen3.model_binding import ModelBinding
    session.runner.binding = ModelBinding(backend.model, backend.codec)
    original = backend.forward
    def forward(rows):
        logits = original(rows)
        return tuple(tuple(backend.model.other.forward(value) / 2 for value in row) for row in logits)
    backend.forward = forward
    profile = session.runner.profile(session.goal, session.goal.search_prompt_ids)
    session.set_profile(profile)
    session.begin_opportunity(session.baseline, {}, deadline_unix_s=time() + 60)
    first = candidate(tmp_path / "first", session.baseline, ["first-group"])
    # When: the next cumulative source selects a second site while the first remains bound.
    with session, closing(provider):
        a = session.self_test(EvaluationRequest(bundle=first, params=ParamSet(values={"speed": 2}),
            site_groups={"first-group": ("sca*",)}))
        assert a.valid and set(session.runner.binding.sites) == {"first-group"}
        assert all(key[0] == "first-group" for key in session.runner.binding.fixtures)
        session.begin_opportunity(first, {"first-group": ("scale",)}, deadline_unix_s=time() + 60)
        second = candidate(tmp_path / "second", first, ["first-group", "second-group"])
        (second.parent / "operators.py").write_text("def replace(site, call, params):\n    return 2 * call.args[0]\n")
        b = session.self_test(EvaluationRequest(bundle=second, params=ParamSet(values={"speed": 2}),
            site_groups={"first-group": ("scale",), "second-group": ("other",)}))
    # Then: real registration would raise if restore had not occurred; both cumulative sites execute.
    assert b.valid, b.detail
    assert set(session.runner.binding.sites) == {"first-group", "second-group"}
    assert {key[0] for key in session.runner.binding.fixtures} == {"first-group", "second-group"}
    session.runner.close()


def test_last_admitted_slot_can_prepare_its_selected_site(tmp_path):
    # Given: seven failed requests spent their slots; the eighth is the first valid site request.
    session, provider, _, inputs = search_case(tmp_path)
    session.set_profile(inputs.profile)
    session.begin_opportunity(session.baseline, {}, deadline_unix_s=time() + 60)
    with session, closing(provider):
        for _ in range(7):
            session.self_test(EvaluationRequest(bundle=tmp_path / "missing", params=ParamSet(values={})))
        path = candidate(tmp_path / "candidate", session.baseline, ["scale"])
        # When
        result = session.self_test(EvaluationRequest(bundle=path, params=ParamSet(values={"speed": 2})))
    # Then: preparation belongs to an already-admitted slot, not an extra admission.
    assert result.valid and session.budget.used == 8
    assert session.attempts[-1]["preparation_forward_calls"] > 0
    session.runner.close()


def test_seeded_current_bundle_is_standalone_and_has_all_helpers(tmp_path):
    # Given: a cumulative parent, not only its entry file.
    from kernel_optimizer.agents.sandbox import Sandbox
    from kernel_optimizer.agents.task_rewriter import TaskRewriteInputs, TaskRewriterAgent
    from kernel_optimizer.control.direct_task import TaskSpace
    from kernel_optimizer.tuning.objective import Objective
    from examples.c3_qwen3.search_bundle import baseline_bundle
    parent = baseline_bundle(tmp_path / "parent")
    data = json.loads(parent.read_text())
    data["helpers"] = ["helper.py"]
    parent.write_text(json.dumps(data))
    (parent.parent / "helper.py").write_text("def helper(x): return x + x\n")
    bundle = load_bundle(parent, {})
    inputs = TaskRewriteInputs(project_root=tmp_path, candidate_id="parent", candidate_path=parent.parent / "operators.py",
        goal="fixture", context={}, objective=Objective(direction="minimize"), params=ParamSet(values={}), space=TaskSpace(),
        bundle_document=bundle.document.model_dump(mode="json"),
        bundle_sources={name: text.decode() for name, text in bundle.sources.items()})
    sandbox = Sandbox(tmp_path / "sandbox")
    # When
    TaskRewriterAgent.__new__(TaskRewriterAgent).seed_sandbox(inputs, sandbox)
    # Then: the manifest's relative paths really resolve in the agent workspace.
    staged = load_bundle(sandbox.root / "candidate/bundle/bundle.json", {})
    assert staged.source_hashes == bundle.source_hashes and staged.bundle_sha256 == bundle.bundle_sha256
