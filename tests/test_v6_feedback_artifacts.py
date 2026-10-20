import json

from kernel_optimizer.project.feedback import candidate_artifacts
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
