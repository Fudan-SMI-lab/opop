from kernel_optimizer.project.nccl_tuning import read_logs


def test_protocol_comes_from_actual_decision_not_kernel_entry_name(tmp_path):
    path = tmp_path/"nccl.log"
    path.write_text(
        "kernel ncclDevKernel_AllReduce_Sum_f32_RING_LL\n"
        "[0] NCCL INFO AllReduce: 28351488 Bytes -> Algo RING proto SIMPLE channel{Lo..Hi}={0..1}\n"
        "[0] NCCL INFO AllReduce: 28351488 Bytes -> Algo RING proto SIMPLE channel{Lo..Hi}={0..1}\n")
    report = read_logs([path])
    assert report["status"] == "observed"
    assert report["decisions"][0]["protocol"] == "SIMPLE"
    assert report["decisions"][0]["occurrences"] == 2
    assert report["decisions"][0]["evidence"]["line"] == 2


def test_unrecognized_log_format_remains_unavailable(tmp_path):
    path = tmp_path/"nccl.log"
    path.write_text("kernel ncclDevKernel_AllReduce_Sum_f32_RING_LL\n")
    report = read_logs([path])
    assert report["status"] == "unavailable" and not report["decisions"]
    assert report["evidence"][0]["sha256"]
