import json

from kernel_optimizer.project.feedback import candidate_artifacts
from kernel_optimizer.project.feedback import route
from kernel_optimizer.store.run_store import RunStore


def test_rejected_implementation_is_available_separately_from_accepted_source(tmp_path):
    store = RunStore.create(tmp_path, "run", {})
    code = b"def build(context):\n    return {}\n"
    ref = store.put_artifact(code)
    bundle = {"id": "rejected", "parent_revision": "parent", "installer": "opop_candidates/x.py",
              "edits": [{"path": "opop_candidates/x.py", "content_ref": ref},
                        {"path": "opop_candidates/x.targets.json", "content_ref": "sha256:"+"a"*64}]}
    inputs = candidate_artifacts(json.dumps({"previous_result": {"bundle": bundle}}), store)
    assert inputs["previous-candidate/candidate.py"] == code
    manifest = json.loads(inputs["previous-candidate/manifest.json"])
    assert [row["status"] for row in manifest["files"]] == ["available", "unavailable"]
    assert manifest["parent_revision"] == "parent"
    assert not any(name.startswith("source/") for name in inputs)


def test_actual_binding_backend_and_setup_reach_the_next_agent():
    setup = [{"bundle": "child", "binding": {"requested": "scoped", "effective": "legacy",
                                          "reason": "unscoped boundary"}, "elapsed_ms": 7.0}]
    raw = {"decision": {"accepted": False}, "candidate": {
        "ranks": [{"rank": 1, "result": {"setup": setup}}]}}
    feedback = json.loads(route(json.dumps(raw), current_revision="parent"))
    assert feedback["binding_execution"]["candidate"][0]["setup"] == setup
    assert feedback["decision"] == raw["decision"]
