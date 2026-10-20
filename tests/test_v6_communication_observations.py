from kernel_optimizer.project.communication import available_fields, normalize


def test_observed_metric_name_satisfies_a_required_observation():
    trace = {"traceEvents": [{"cat": "kernel", "ph": "X", "name": "ncclKernel_AllReduce",
                              "ts": 100, "dur": 400, "args": {"correlation": 7}}]}
    observed = normalize(trace, {"kernels": []}, rank=0)
    assert observed["communication_kernel_sum_ms"] == 0.4
    available = available_fields(observed)
    assert {"nccl_kernel_duration", "communication_kernel_sum_ms", "compute_overlap_ms"} <= available
    assert "logical_input_bytes" not in available  # Unknown payload is not zero traffic.
    assert "pcie_bytes" not in available


def test_absent_communication_does_not_turn_default_zeros_into_measurements():
    observed = normalize({"traceEvents": []}, {"kernels": []}, rank=0)
    assert observed["communication_kernel_sum_ms"] == 0
    assert "communication_kernel_sum_ms" not in available_fields(observed)


def test_numeric_availability_uses_observed_values_without_metric_aliases():
    observed = {"status": "observed", "future_measured_field": 0.0, "unknown": None,
                "nonfinite": float("nan"), "flag": True, "description": "present"}
    assert available_fields(observed) == {"future_measured_field"}
