from kernel_optimizer.project.measurement import paired_decision, constraints_for_runs


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
