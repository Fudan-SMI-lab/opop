import hashlib
import json

from kernel_optimizer.project.observation_artifacts import artifacts, feedback_artifacts, ownership


def test_full_diagnostics_remain_available_and_versions_stay_separate():
    before = {"timeline": {"status": "observed", "provenance": {"revision": "parent"},
                           "ranks": [{"rank": 0, "clock": {"status": "unknown"}, "events": [1, 2, 3]}]}}
    after = {"timeline": {"status": "unavailable", "provenance": {"revision": "candidate"},
                          "reason": "native event correlation unavailable"}}
    original = {"decision": {"accepted": False}, "resources": {
        "extra_collectors": {"before": before, "after": after}, "groups": [{"group": "g"}]}}
    files, raw = feedback_artifacts(json.dumps(original))
    indexed = json.loads(raw)
    for arm, expected in (("before", before), ("after", after)):
        ref = indexed["resources"]["extra_collectors"][arm]["timeline"]
        assert json.loads(files[ref["input"]]) == expected["timeline"]
        assert hashlib.sha256(files[ref["input"]].encode()).hexdigest() == ref["sha256"]
    assert indexed["resources"]["groups"] == original["resources"]["groups"]
    assert "events" not in indexed["resources"]["extra_collectors"]["before"]["timeline"]["ranks"][0]
    assert len(files) == 2


def test_shared_parameter_ownership_is_not_timed_collective_attribution():
    discovery = {"revision": "r", "plan": "p", "registry": "g",
                 "calls": [{"id": "a", "kind": "module", "target": "net.left"},
                           {"id": "b", "kind": "module", "target": "net.right"}],
                 "groups": [{"id": "one", "calls": ["a"]}, {"id": "joint", "calls": ["a", "b"]}],
                 "extra_collectors": {"execution_timeline": {"ranks": [{
                     "rank": 1, "gradient_readiness": {"buckets": [{
                         "module": "ddp", "bucket_index": 0,
                         "parameters": ["net.left.weight", "net.right.weight"]}]}}]}}}
    record = ownership(discovery)["buckets"][0]
    assert record["calls"] == ["a", "b"]
    assert record["groups"] == ["joint", "one"]
    assert record["rank"] == 1
    assert record["timed_collective_identity"] == "unknown"
    assert not any(key.endswith("_ms") for key in record)


def test_plugin_names_cannot_collide_or_change_input_directory():
    inputs, manifest = artifacts({"x/y": {"status": "observed"}, "x_y": {"status": "unknown"},
                                 "../y": {"status": "unsupported"}}, "observations/current")
    assert len(inputs) == 3
    assert all(path.startswith("observations/current/") and ".." not in path for path in inputs)
    assert len({row["input"] for row in manifest.values()}) == 3
