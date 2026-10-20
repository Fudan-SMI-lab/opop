import json
from types import SimpleNamespace

import pytest

from kernel_optimizer.models.project import CandidateBundle, FileEdit
from kernel_optimizer.project.replay import transfer_bundle
from test_v6_transactions import setup_project, evaluation


def installer(revisions, parent, name, call="a", **relations):
    code = b"def build(context):\n    return {'a': lambda x: x + 1}\n"
    target = {"id": call, "kind": "function", "target": call+".py", "phase": "run", "parents": []}
    path = f"opop_candidates/{name}.py"
    return CandidateBundle(id=name, parent_revision=parent.id, contract_ref=parent.contract_ref,
        scope="group", implementation="composite", groups=("g"+call,), calls=(call,),
        installer=path, parameters={"tile": 8},
        edits=(FileEdit(path=path, before_sha256=None, content_ref=revisions.store.put_artifact(code)),
               FileEdit(path=path.removesuffix(".py")+".targets.json", before_sha256=None,
                        content_ref=revisions.store.put_artifact(json.dumps([target])))),
        read_set=("call:"+call,),
        write_set=("call:"+call, "file:"+path, "file:"+path.removesuffix(".py")+".targets.json"),
        external_invariants=("same output",),
        **relations)


def commit(revisions, bundle):
    staged = revisions.stage(bundle)
    revisions.evaluate(evaluation(staged, bundle))
    revisions.commit(staged.id)
    return staged


def replay_fixture(tmp_path, *, requires=False, target_installed=False):
    roots = [tmp_path/"origin", tmp_path/"destination"]
    for root in roots:
        root.mkdir()
    source, initial = setup_project(roots[0])
    target, baseline = setup_project(roots[1])
    request = {"entry": "a.py", "files": ["a.py", "b.py"], "assets": {},
               "quality": {"atol": 0}, "continuous_steps": 2, "distributed_backend": None}
    for revisions in (source, target):
        (revisions.store.run_dir/"request.json").write_text(json.dumps(request))
        (revisions.store.run_dir/"prepared.json").write_text(json.dumps({"adapter_sha256": "same-evaluator"}))
    previous = installer(source, initial, "first")
    parent = commit(source, previous)
    final = installer(source, parent, "final", "b" if requires else "a",
                      **({"requires": (previous.id,)} if requires else {"replaces": (previous.id,)}))
    commit(source, final)
    if target_installed:
        baseline = commit(target, installer(target, baseline, "existing", "b"))
    discovery = {"revision": baseline.id,
                 "calls": [{"id": x, "kind": "function", "target": x+".py", "phase": "run", "parents": []}
                           for x in ("a", "b")],
                 "groups": [{"id": "g"+x, "calls": [x]} for x in ("a", "b")]}
    search = SimpleNamespace(revisions=target, store=target.store, output=target.store.run_dir,
        prepared={"adapter_sha256": "same-evaluator"}, request=SimpleNamespace(portable=lambda: request))
    return source, final, search, discovery


def test_final_replacement_can_replay_on_pristine_variant_without_changing_code(tmp_path):
    source, original, search, discovery = replay_fixture(tmp_path)
    with pytest.raises(ValueError, match="explicit identities"):
        transfer_bundle(source.store.run_dir, original.id, search, discovery=discovery)
    replay, record = transfer_bundle(source.store.run_dir, original.id, search, discovery=discovery,
                                     allow_absent_replacements=True)
    assert replay.replaces == () and replay.requires == ()
    assert replay.parameters == original.parameters
    assert search.store.get_artifact(replay.edits[0].content_ref) == source.store.get_artifact(original.edits[0].content_ref)
    assert record["replay"]["omitted_replacements"] == ["first"]
    before = search.revisions.current()
    staged = search.revisions.stage(replay)
    assert staged.bundles == (replay.id,) and search.revisions.current() == before
    with pytest.raises(ValueError, match="passing"):
        search.revisions.commit(staged.id)


def test_pristine_replay_does_not_discard_true_dependencies(tmp_path):
    source, original, search, discovery = replay_fixture(tmp_path, requires=True)
    with pytest.raises(ValueError, match="explicit identities"):
        transfer_bundle(source.store.run_dir, original.id, search, discovery=discovery,
                        allow_absent_replacements=True)


def test_replacement_omission_cannot_remove_an_installed_target_bundle(tmp_path):
    source, original, search, discovery = replay_fixture(tmp_path, target_installed=True)
    with pytest.raises(ValueError, match="empty target"):
        transfer_bundle(source.store.run_dir, original.id, search, discovery=discovery,
                        allow_absent_replacements=True)
