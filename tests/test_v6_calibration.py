from kernel_optimizer.project.calibration import layouts


def test_calibration_uses_observed_sizes_and_keeps_unknown_semantics_explicit():
    row = {"operation": "allreduce", "participants": "[0, 1]", "payload_status": "observed",
           "dtype": "Float", "input_elements": 17, "logical_input_bytes": 68, "process_group": "g",
           "sequence": 3, "calls": ["call"], "kernels": [{"name": "ncclDevKernel_AllReduce_Sum_f32_RING_LL"}]}
    result = layouts({"communication": {"ranks": [{"collectives": [row, {**row, "sequence": 4},
                    {**row, "sequence": 5, "kernels": [{"name": "opaque_custom_collective"}]}]}]}}, 2)
    assert len(result["layouts"]) == 1
    assert result["layouts"][0]["logical_bytes"] == 68
    assert len(result["layouts"][0]["source_collectives"]) == 2
    assert len(result["omitted"]) == 1
    assert layouts({"communication": {"ranks": [{"collectives": [row]}]}}, 4)["layouts"] == []
