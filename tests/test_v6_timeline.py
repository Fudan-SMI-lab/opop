from copy import deepcopy

from kernel_optimizer.project.timeline import analyze, clock_mapping, cross_rank


def event(name, ts, dur=1, cat="cpu_op", **args):
    return {"name": name, "ts": ts, "dur": dur, "ph": "X", "cat": cat,
            "pid": 1, "tid": 2, "args": args}


def test_native_launch_link_does_not_claim_ready_or_consumer():
    trace = {"traceEvents": [
        event("c10d::allreduce_", 10, 5, **{"External id": 7}),
        event("cudaLaunchKernel", 12, 1, "cuda_runtime", correlation=4, **{"External id": 7}),
        event("compute", 20, 10, "kernel", device=0),
        event("nccl_kernel", 25, 10, "kernel", device=0, correlation=4,
              **{"Seq": 1, "Process Group Name": "0", "Process Group Ranks": "[0, 1]",
                 "Collective name": "allreduce", "dtype": "float", "In msg nelems": 4}),
        event("cudaStreamSynchronize", 30, 10, "cuda_runtime"),
    ]}
    result = analyze(trace, rank=0)
    collective = result["communication"]["collectives"][0]
    assert collective["cpu_ranges"][0]["name"] == "c10d::allreduce_"
    assert collective["launch_links"][0]["launch_to_kernel_start_us"] == 13
    assert collective["gpu_input_ready"]["status"] == "unknown"
    assert collective["consumer_completion"]["status"] == "unknown"
    assert result["devices"]["0"]["compute_overlap_us"] == 5
    assert result["devices"]["0"]["communication_after_last_compute_us"] == 5
    assert len(result["cuda_synchronization"]) == 1


def test_intervals_are_not_merged_across_devices():
    trace = {"traceEvents": [event("compute", 0, 10, "kernel", device=0),
                            event("nccl", 0, 10, "kernel", device=1)]}
    result = analyze(trace, rank=0)
    assert result["devices"]["0"]["compute_overlap_us"] == 0
    assert result["devices"]["1"]["compute_overlap_us"] == 0
    assert result["devices"]["1"]["communication_after_last_compute_us"] is None


def test_clock_brackets_reject_inconsistent_mapping():
    trace = {"traceEvents": [event("a", 10), event("b", 20)]}
    anchors = [{"name": "a", "before_ns": 110000, "after_ns": 112000},
               {"name": "b", "before_ns": 121000, "after_ns": 124000}]
    clock = clock_mapping(trace, anchors, {"boot_id": "boot"})
    assert clock["offset_bounds_us"] == [100, 104]
    anchors[1]["before_ns"] = 130000
    anchors[1]["after_ns"] = 131000
    assert clock_mapping(trace, anchors, {})["status"] == "unknown"


def test_cross_rank_requires_shared_domain_and_unique_submission():
    def report(rank, start):
        return {"rank": rank, "clock": {"status": "observed", "offset_bounds_us": [100, 104],
                                       "domain": {"host": "a", "boot_id": "boot", "clock": "CLOCK_MONOTONIC"}},
                "communication": {"collectives": [{"process_group": "0", "sequence": 1, "operation": "allreduce",
                    "participants": "[0, 1]", "cpu_submission": {"status": "observed", "name": "c10d::allreduce_", "start_us": start}}]}}
    rows = [report(0, 10), report(1, 20)]
    assert cross_rank(rows)["collectives"][0]["cpu_submit_spread_bounds_us"] == [6, 14]
    invalid = deepcopy(rows)
    invalid[1]["clock"]["domain"]["boot_id"] = "other-host-boot"
    assert cross_rank(invalid)["collectives"][0]["status"] == "unknown"
    invalid = deepcopy(rows)
    invalid[1]["communication"]["collectives"][0]["cpu_submission"]["status"] = "unknown"
    assert cross_rank(invalid)["collectives"][0]["status"] == "unknown"
    assert cross_rank(rows[:1])["collectives"][0]["status"] == "unknown"


def test_nested_c10d_wrappers_use_unique_innermost_submission():
    trace = {"traceEvents": [
        event("c10d::allreduce_", 0, 30),
        event("c10d::allreduce_", 10, 10),
        event("cudaLaunchKernel", 12, 1, "cuda_runtime", correlation=4),
        event("nccl", 25, 10, "kernel", device=0, correlation=4,
              **{"Seq": 1, "Process Group Name": "0", "Process Group Ranks": "[0, 1]",
                 "Collective name": "allreduce"})]}
    c = analyze(trace, rank=0)["communication"]["collectives"][0]
    assert len(c["cpu_ranges"]) == 2
    assert c["cpu_submission"]["start_us"] == 10
    trace["traceEvents"].insert(0, event("c10d::different_overlap", 11, 30))
    c = analyze(trace, rank=0)["communication"]["collectives"][0]
    assert c["cpu_submission"]["status"] == "unknown"


def test_repeated_gradient_callbacks_are_kept_without_guessing_microstep():
    trace = {"traceEvents": [event("grad", 10), event("grad", 30)]}
    result = analyze(trace, rank=0, gradients={"grad": ["model.weight"]},
                     inventory={"buckets": [{"parameters": ["model.weight"], "bucket_index": 0}]})
    bucket = result["gradient_readiness"]["buckets"][0]
    assert bucket["cpu_post_accumulate_times_us"]["model.weight"] == [10, 30]
    assert bucket["timed_collective_identity"] == "unknown"
