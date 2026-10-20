from types import SimpleNamespace

from kernel_optimizer.project.provider_queue import ProviderQueue, retryable_status
from kernel_optimizer.project.recovery import committed_trial
from kernel_optimizer.project.quality_report import quality_and_coverage
from test_v6_transactions import setup_project, candidate, evaluation


def test_truncated_provider_error_keeps_retryable_status():
    assert retryable_status('{"name":"APIError","data":{"statusCode":429,"responseHeaders":{"vary":"Origin, A') == 429
    assert retryable_status('{"statusCode":401}') is None


def test_scheduling_is_activated_by_overload_and_recovers_capacity(tmp_path):
    config = SimpleNamespace(agents=SimpleNamespace(default_model="provider/model"),
                             opencode=SimpleNamespace(launch_cwd=tmp_path))
    queue = ProviderQueue(config)
    with queue.admit():
        assert not queue.state_path.exists()
    first = queue.overloaded(429)
    assert first["temporary_parallel_capacity"] == 1
    queue.succeeded(); queue.succeeded()
    assert queue._read()["capacity"] == 2
    assert queue.overloaded(429)["temporary_parallel_capacity"] == 1


def test_missing_summary_recovers_only_from_an_accepted_ancestor(tmp_path):
    revisions, parent = setup_project(tmp_path)
    bundle = candidate(revisions, parent); child = revisions.stage(bundle)
    revisions.evaluate(evaluation(child, bundle))
    search = SimpleNamespace(revisions=revisions, store=revisions.store)
    assert committed_trial(search, "evaluation") is None
    revisions.commit(child.id)
    result = committed_trial(search, "evaluation")
    assert result["publication"] == "committed"
    assert result["evaluation"]["revision_id"] == child.id
    assert revisions.current().id == child.id


def test_bypassed_parent_is_a_coverage_failure_not_a_numerical_failure():
    run = {"status": "passed", "ranks": [{"rank": 0, "status": "passed", "result": {
        "valid": False, "checks": [{"passed": True}], "math_policy_unchanged": True,
        "coverage": {"old": {"required": ["inner"], "counts": {}},
                     "new": {"required": ["outer"], "counts": {"outer": 2}}}}}]}
    quality, missing = quality_and_coverage([run])
    assert quality == "pass"
    assert missing == [{"endpoint": 0, "rank": 0, "bundle": "old", "calls": ["inner"]}]
