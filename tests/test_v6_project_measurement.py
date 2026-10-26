import statistics

import pytest

from kernel_optimizer.project.measurement import metric, paired_decision, constraints_for_runs


def endpoint(value, throughput=100):
    return {"status": "passed", "ranks": [{"result": {"valid": True,
        "metrics": {"latency": value, "throughput": throughput},
        "samples": [{"latency": value+x} for x in [-0.1, 0, 0.1]]}}]}


def test_pair_reversal_does_not_publish_a_lucky_first_measurement():
    result = paired_decision([{"parent": endpoint(10), "candidate": endpoint(8)},
                              {"parent": endpoint(8), "candidate": endpoint(9)}],
                             {"name": "latency", "direction": "minimize"}, endpoint(10))
    assert result["valid"]
    assert result["paired_gains"] == [2, -1]
    assert result["reasons"] == ["paired_gain_not_above_noise"]


def test_constraint_remains_relative_to_original_not_latest_parent():
    result = paired_decision([{"parent": endpoint(9, 95), "candidate": endpoint(8, 91)},
                              {"parent": endpoint(9, 95), "candidate": endpoint(8, 91)}],
                             {"name": "latency", "direction": "minimize", "constraints": [
                                 {"metric": "throughput", "op": "ge", "baseline_ratio": 0.95}]}, endpoint(10, 100))
    assert result["constraints"][0]["limit"] == 95
    assert result["reasons"] == ["original_baseline_constraint_failed"]


def test_throughput_constraint_checks_the_slowest_participating_rank():
    candidate = endpoint(8, 110)
    candidate["ranks"].extend(endpoint(8, 90)["ranks"])
    anchor = endpoint(10, 100)
    anchor["ranks"].extend(endpoint(10, 100)["ranks"])
    checks = constraints_for_runs([candidate], {"constraints": [{"metric": "throughput", "op": "ge", "baseline_ratio": 0.95}]}, anchor)
    assert checks[0]["values"] == [90]
    assert not checks[0]["passed"]


def test_source_changes_do_not_publish_even_when_all_timings_improve():
    a, b = endpoint(10), endpoint(8)
    a["ranks"][0]["framework_source"] = {"sha256": "old"}
    b["ranks"][0]["framework_source"] = {"sha256": "new"}
    result = paired_decision([{"parent": a, "candidate": b}, {"parent": a, "candidate": b}],
                             {"name": "latency", "direction": "minimize"}, a)
    assert "framework_changed_between_endpoints" in result["reasons"]


def distributed_endpoint(values, *, diagnostic=False):
    return {"status": "passed", "ranks": [{"result": {"valid": True,
        "metrics": {"latency": statistics.median(samples)},
        "samples": [] if diagnostic else [{"latency": value} for value in samples],
        "metric_samples": {"latency": samples} if diagnostic else {}}} for samples in values]}


def test_recorded_ddp_diagnostic_preserves_source_reduction_order():
    run = distributed_endpoint([[35.921772, 38.092142, 38.400203],
                                [42.693940, 34.704439, 35.207955]], diagnostic=True)
    assert metric(run, "latency") == pytest.approx(38.092142)
    assert metric(run, "latency", "max", "ranks_then_samples") == pytest.approx(38.400203)


def test_alternating_slowest_rank_changes_the_paired_decision():
    parent = distributed_endpoint([[30, 10, 10], [10, 30, 10]])
    candidate = distributed_endpoint([[20, 20, 20], [20, 20, 20]])
    pairs = [{"parent": parent, "candidate": candidate}]*2
    objective = {"name": "latency", "direction": "minimize"}
    legacy = paired_decision(pairs, objective, parent)
    aligned = paired_decision(pairs, {**objective, "aggregation_order": "ranks_then_samples"}, parent)
    assert legacy["paired_gains"] == [-10, -10]
    assert legacy["reasons"] == ["paired_gain_not_above_noise"]
    assert aligned["paired_gains"] == [10, 10]
    assert aligned["noise_bounds"] == [0, 0]
    assert aligned["reasons"] == []


def test_constraint_uses_its_own_rank_sample_order():
    anchor = distributed_endpoint([[20]*3, [20]*3])
    candidate = distributed_endpoint([[30, 10, 10], [10, 30, 10]])
    constraint = {"metric": "latency", "op": "le", "baseline_ratio": 1.05}
    assert constraints_for_runs([candidate], {"constraints": [constraint]}, anchor)[0]["passed"]
    check = constraints_for_runs([candidate], {"constraints": [
        {**constraint, "aggregation_order": "ranks_then_samples"}]}, anchor)[0]
    assert check["limit"] == 21
    assert check["values"] == [30]
    assert not check["passed"]


def test_aligned_reduction_does_not_silently_drop_an_unpaired_sample():
    run = distributed_endpoint([[1, 2, 3], [1, 2]])
    with pytest.raises(ValueError, match="aligned rank samples"):
        metric(run, "latency", "max", "ranks_then_samples")


def test_parameter_screening_score_matches_final_endpoint_reduction(monkeypatch):
    from types import SimpleNamespace
    from kernel_optimizer.project.inner_loop import ProjectTaskEvaluator
    from kernel_optimizer.project import candidate_diagnostics

    run = distributed_endpoint([[30, 10, 10], [10, 30, 10]])
    objective = {"name": "latency", "direction": "minimize", "aggregation_order": "ranks_then_samples"}
    template = SimpleNamespace(id="bundle", model_copy=lambda update: SimpleNamespace(**update))
    coordinator = SimpleNamespace(
        request=SimpleNamespace(objective=objective, profiling={}),
        revisions=SimpleNamespace(stage=lambda bundle: "revision"),
        execute=lambda *args: run, prepared={"evaluation": run})
    monkeypatch.setattr(candidate_diagnostics, "should_trace", lambda *args: False)
    result = ProjectTaskEvaluator(coordinator, template).evaluate(None, {}, {})
    assert result.valid
    assert result.score == 30
    assert result.metrics["latency"] == metric(run, "latency", "max", "ranks_then_samples")
