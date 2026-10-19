import json

import pytest

from kernel_optimizer.project.branches import fork_run
from kernel_optimizer.project.experiment_policy import check_selection, eligible_groups, feedback_view
from kernel_optimizer.project.transactions import RevisionStore
from kernel_optimizer.store.run_store import RunStore
from test_v6_transactions import setup_project, candidate, evaluation


def test_feasible_losing_branch_preserves_source_and_independent_index(tmp_path):
    source, parent = setup_project(tmp_path)
    root = source.store.run_dir
    (root / "prepared.json").write_text(json.dumps({"status": "prepared", "revision": parent.id}))
    (root / "request.json").write_text("{}")
    bundle = candidate(source, parent)
    child = source.stage(bundle)
    assert not source.evaluate(evaluation(child, bundle, objective=3.0))["accepted"]
    source.abort(child.id, "measured slower")
    old_index = (source.store.artifacts_dir / "index.jsonl").read_bytes()
    fork_run(root, tmp_path / "branch", revision_id=child.id)
    branch = RevisionStore(RunStore.open(tmp_path / "branch"))
    assert branch.current().id == child.id
    assert source.current().id == parent.id
    assert (branch.materialize(child, tmp_path / "materialized") / "a.py").read_text() == "value = 3\n"
    branch.store.put_artifact("new branch artifact")
    assert (source.store.artifacts_dir / "index.jsonl").read_bytes() == old_index


def test_failed_quality_is_not_an_exploration_parent(tmp_path):
    source, parent = setup_project(tmp_path)
    (source.store.run_dir / "prepared.json").write_text(json.dumps({"status": "prepared", "revision": parent.id}))
    bundle = candidate(source, parent); child = source.stage(bundle)
    source.evaluate(evaluation(child, bundle, quality="fail"))
    with pytest.raises(ValueError, match="quality-passing"):
        fork_run(source.store.run_dir, tmp_path / "branch", revision_id=child.id)


def test_leaf_group_control_cannot_expand_to_project(tmp_path):
    discovery = {"calls": [{"id": "root", "parents": []}, {"id": "leaf", "parents": ["root"]}],
        "groups": [{"id": "all", "kind": "project", "calls": ["root"]},
                   {"id": "local", "kind": "functional", "calls": ["leaf"]}]}
    eligible = eligible_groups(discovery, leaf_only=True)
    assert [g["id"] for g in eligible] == ["local"]
    selection = {"scope": "group", "groups": ["local"], "affected_calls": ["leaf", "root"],
                 "rationale": "test", "hypothesis": "test"}
    (tmp_path / "selection.json").write_text(json.dumps(selection))
    with pytest.raises(ValueError, match="enclosing"):
        check_selection(tmp_path, discovery, ("group",), eligible)


def test_project_feedback_retains_failure_without_resource_explanation():
    raw = json.dumps({"decision": {"accepted": False}, "project_measurement": {"objective": 2.0},
        "quality_and_coverage": [{"error": "wrong output"}], "resources": {"marker": "private detail"},
        "previous_proposal": {"rationale": "private detail"}, "parameter_search": {"responses": ["private detail"]}})
    value = feedback_view(raw, "project")
    assert "private detail" not in value
    assert "wrong output" in value and json.loads(value)["project_measurement"]["objective"] == 2.0
