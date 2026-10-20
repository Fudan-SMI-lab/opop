import json

import pytest

from kernel_optimizer.models.project import CandidateBundle, FileEdit
from kernel_optimizer.project.branches import fork_run
from kernel_optimizer.project.transactions import RevisionStore, identity
from kernel_optimizer.project.versioning import coverage_errors, stage_version
from kernel_optimizer.project.experiment import explorable, resume_round
from types import SimpleNamespace
from kernel_optimizer.store.run_store import RunStore
from test_v6_transactions import setup_project, candidate, evaluation


def branch_with_two(tmp_path):
    main, parent = setup_project(tmp_path)
    (main.store.run_dir/"prepared.json").write_text(json.dumps({"status": "prepared", "revision": parent.id}))
    branch_path = tmp_path/"branch"
    fork_run(main.store.run_dir, branch_path)
    branch = RevisionStore(RunStore.open(branch_path))
    bundles = []
    current = branch.current()
    for call in ("a", "b"):
        file = next(f for f in current.files if f.path == call+".py")
        bundle = CandidateBundle(id="change-"+call, parent_revision=current.id, contract_ref=parent.contract_ref,
            scope="group", implementation="kernel", groups=("g"+call,), calls=(call,),
            edits=(FileEdit(path=file.path, before_sha256=file.sha256, content_ref=branch.store.put_artifact(call)),),
            read_set=("call:"+call,), write_set=("call:"+call, "file:"+file.path), external_invariants=("same output",))
        current = branch.stage(bundle)
        branch.evaluate(evaluation(current, bundle))
        branch.commit(current.id)
        bundles.append(bundle)
    return main, parent, branch, bundles


def run_for(bundles):
    return {"ranks": [{"rank": 0, "result": {
        "bundle_hashes": {b.id: identity(b) for b in bundles},
        "coverage": {b.id: {"required": list(b.calls), "counts": {c: 2 for c in b.calls}} for b in bundles}}}]}


def test_complete_snapshot_preserves_bundles_and_needs_fresh_publication(tmp_path):
    main, parent, branch, bundles = branch_with_two(tmp_path)
    incoming = branch.current()
    staged = stage_version(main, branch.store.run_dir)
    assert staged.parent_id == parent.id
    assert staged.files == incoming.files
    assert staged.bundles == tuple(b.id for b in bundles)
    assert main.current() == parent and branch.current() == incoming
    assert all(main.bundle(b.id) == b for b in bundles)
    with pytest.raises(ValueError, match="passing"):
        main.commit(staged.id)
    assert coverage_errors(main, staged, [run_for(bundles), run_for(bundles)]) == []
    second = run_for(bundles)
    second["ranks"][0]["result"]["coverage"][bundles[0].id]["counts"] = {}
    errors = coverage_errors(main, staged, [run_for(bundles), second])
    assert errors == [{"endpoint": 1, "rank": 0, "bundle": bundles[0].id,
                       "reason": "version_bundle_coverage_missing", "calls": ["a"]}]
    # CAS and rollback remain generic for full-version publications.
    path = main.root/"transactions"/(staged.id+".json")
    state = json.loads(path.read_text()); state.update(state="evaluated", accepted=True)
    path.write_text(json.dumps(state))
    main.commit(staged.id)
    assert main.current().bundles == ("change-a", "change-b")
    main.rollback(parent.id, "whole-version rollback")
    assert main.current() == parent


def test_full_version_cannot_cross_frozen_task_or_lose_a_rank(tmp_path):
    main, parent, branch, bundles = branch_with_two(tmp_path)
    staged = stage_version(main, branch.store.run_dir)
    assert coverage_errors(main, staged, [{"ranks": []}])[0]["reason"] == "version_rank_missing"
    wrong = branch.current().model_copy(update={"contract_ref": "different"})
    (branch.root/"revisions"/(wrong.id+".json")).write_text(wrong.model_dump_json())
    with pytest.raises(ValueError, match="anchor and frozen contract"):
        stage_version(main, branch.store.run_dir)


def test_exploration_retains_feasible_losses_but_not_unmeasured_or_mixed_versions():
    result = {"evaluation": {"quality": "pass", "feasible": True}, "measurement": {"framework_source": "same"},
              "decision": {"accepted": False, "reasons": ["objective_not_improved", "paired_gain_not_above_noise"]}}
    assert explorable(result)
    result["measurement"]["framework_source"] = "mixed"
    assert not explorable(result)
    assert not explorable({"decision": {"accepted": False, "reasons": ["no_valid_parameter_candidate"]}})


def test_interrupted_round_reuses_artifact_instead_of_empty_trial_directory(tmp_path):
    (tmp_path/"trials/empty").mkdir(parents=True)
    (tmp_path/"sandboxes/candidate-real-session").mkdir(parents=True)
    search = SimpleNamespace(output=tmp_path, resume_generation=lambda path: ("same_session", path.name))
    assert resume_round(search, {"known_trials": [], "known_sandboxes": []}) == ("same_session", "candidate-real-session")


def test_unprofiled_branch_prepares_discovery_and_recovers_before_generation(tmp_path):
    from kernel_optimizer.project.search import ProjectSearch
    search = ProjectSearch.__new__(ProjectSearch)
    search.output = tmp_path
    parent = SimpleNamespace(id="branch-with-unmeasured-plan")
    search.revisions = SimpleNamespace(current=lambda: parent)
    observed = []
    def discover():
        observed.append(parent.id)
        parent.id = "branch-with-current-plan"
        result = {"revision": parent.id, "groups": ["observed-group"]}
        (tmp_path/"discovery.json").write_text(json.dumps(result))
        return result
    search.discover = discover
    search.store = SimpleNamespace(append=lambda *args: None)
    search.trial = lambda **options: {"discovery": search.current_discovery(), "options": options}
    inflight = {"parent": parent.id, "known_trials": [], "known_sandboxes": [], "round": 3}
    result = resume_round(search, inflight, feedback="retained feedback", trial_options={"allowed_scopes": ("joint",)})
    assert result["discovery"]["groups"] == ["observed-group"]
    assert result["options"] == {"feedback": "retained feedback", "allowed_scopes": ("joint",)}
    assert search.current_discovery() == result["discovery"] and len(observed) == 1
    # Preparation recovery cannot silently start another proposal at a new parent.
    with pytest.raises(ValueError, match="ambiguous or absent"):
        resume_round(search, inflight)
