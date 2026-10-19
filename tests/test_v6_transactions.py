import json

import pytest

from kernel_optimizer.models.objective import AggregationSpec, ObjectiveSpec, RequiredMetric
from kernel_optimizer.models.project import (
    CallSite, CandidateBundle, ExecutionPlan, FileEdit, GroupRegistry, GroupSpec,
    ProjectContract, ProjectEvaluation,
)
from kernel_optimizer.project.transactions import RevisionStore, identity
from kernel_optimizer.store.run_store import RunStore


def setup_project(tmp_path):
    source = tmp_path / "source"; source.mkdir()
    (source / "a.py").write_text("value = 1\n")
    (source / "b.py").write_text("value = 2\n")
    store = RunStore.create(tmp_path, "run", {})
    revisions = RevisionStore(store)
    objective = ObjectiveSpec(name="latency", unit="ms", direction="minimize",
                              required_metrics=(RequiredMetric(name="latency", unit="ms"),),
                              aggregation=AggregationSpec(kind="sum"))
    contract = ProjectContract(id="task", goal="minimize latency", objective=objective,
                               workload={"batch": 1}, quality={"atol": 0.0}, completion="return",
                               invariants=("same output",), state_protocol="reset each iteration")
    plan = ExecutionPlan(id="p", calls=(CallSite(id="a", phase="run", kind="function", target="a.py"),
                                        CallSite(id="b", phase="run", kind="function", target="b.py")))
    groups = GroupRegistry(id="g", plan_id="p", groups=(GroupSpec(id="ga", calls=("a",), semantics="a"),
                                                      GroupSpec(id="gb", calls=("b",), semantics="b")))
    revision = revisions.initialize(source, ["a.py", "b.py"], contract, plan, groups)
    return revisions, revision


def candidate(revisions, parent, ident="joint"):
    return CandidateBundle(id=ident, parent_revision=parent.id, contract_ref=parent.contract_ref,
        scope="joint", implementation="composite", groups=("ga", "gb"), calls=("a", "b"),
        edits=tuple(FileEdit(path=file.path, before_sha256=file.sha256,
                             content_ref=revisions.store.put_artifact(b"value = 3\n")) for file in parent.files),
        read_set=("file:a.py", "file:b.py"), write_set=("file:a.py", "file:b.py"), external_invariants=("same output",))


def evaluation(revision, bundle, **changes):
    data = dict(id="evaluation", revision_id=revision.id, parent_id=revision.parent_id,
                contract_ref=revision.contract_ref, bundle_id=bundle.id, quality="pass", feasible=True,
                objective=1.0, reference_objective=2.0, required_ranks=(0,),
                rank_bundle_hashes={"0": identity(bundle)}, required_calls=bundle.calls, executed_calls=bundle.calls)
    return ProjectEvaluation(**(data | changes))


def test_joint_install_preserves_parent_and_commit_is_atomic(tmp_path):
    revisions, parent = setup_project(tmp_path)
    bundle = candidate(revisions, parent); staged = revisions.stage(bundle)
    root = revisions.materialize(staged, tmp_path / "staged")
    assert (root / "a.py").read_text() == (root / "b.py").read_text() == "value = 3\n"
    assert revisions.current() == parent
    assert revisions.evaluate(evaluation(staged, bundle))["accepted"]
    revisions.commit(staged.id)
    assert revisions.current().bundles == (bundle.id,)
    assert revisions.rollback(parent.id, "control rollback") == parent


def test_stale_parent_and_missing_rank_coverage_cannot_publish(tmp_path):
    revisions, parent = setup_project(tmp_path)
    bundle = candidate(revisions, parent); staged = revisions.stage(bundle)
    result = revisions.evaluate(evaluation(staged, bundle, executed_calls=("a",)))
    assert "candidate_coverage_missing" in result["reasons"]
    with pytest.raises(ValueError, match="passing"):
        revisions.commit(staged.id)
    result = revisions.evaluate(evaluation(staged, bundle, rank_bundle_hashes={"0": "wrong"}))
    assert "rank_bundle_mismatch" in result["reasons"]
    revisions.evaluate(evaluation(staged, bundle)); revisions.commit(staged.id)
    other = candidate(revisions, parent, "other"); sibling = revisions.stage(other)
    revisions.evaluate(evaluation(sibling, other))
    with pytest.raises(ValueError, match="parent changed"):
        revisions.commit(sibling.id)


def test_recovery_uses_pointer_and_retains_full_bundle_history(tmp_path):
    revisions, parent = setup_project(tmp_path)
    bundle = candidate(revisions, parent); staged = revisions.stage(bundle)
    revisions.evaluate(evaluation(staged, bundle)); revisions.commit(staged.id)
    path = revisions.root / "transactions" / f"{staged.id}.json"
    state = json.loads(path.read_text()); state["state"] = "evaluated"; path.write_text(json.dumps(state))
    assert revisions.recover()[0]["state"] == "committed"
    next_bundle = candidate(revisions, staged, "next"); next_revision = revisions.stage(next_bundle)
    assert next_revision.bundles == ("joint", "next")
    assert revisions.recover()[0]["state"] == "aborted"
    assert revisions.current().id == staged.id


def test_conflicting_edits_and_contract_drift_are_rejected(tmp_path):
    revisions, parent = setup_project(tmp_path); bundle = candidate(revisions, parent)
    with pytest.raises(ValueError, match="overlapping"):
        CandidateBundle(**(bundle.model_dump() | {"edits": bundle.edits + (bundle.edits[0],)}))
    with pytest.raises(ValueError, match="contract"):
        revisions.stage(bundle.model_copy(update={"contract_ref": "different"}))
    with pytest.raises(ValueError, match="stale file"):
        revisions.stage(bundle.model_copy(update={"edits": (bundle.edits[0].model_copy(update={"before_sha256": "stale"}),)}))
