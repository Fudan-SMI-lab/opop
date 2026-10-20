import json
from kernel_optimizer.project.trace import associate


def test_custom_launch_keeps_bundle_attribution_without_resurrecting_retired_site():
    replacement = "opop_candidate::"+json.dumps({"bundle": "candidate", "call": "retired-call", "phase": "run"}).encode().hex()
    def row(name, ts, dur, cat="user_annotation", args=None):
        return {"name": name, "ph": "X", "pid": 1, "tid": 1, "ts": ts, "dur": dur, "cat": cat, "args": args or {}}
    trace = {"traceEvents": [row("opop::0", 0, 100), row(replacement, 10, 70), row(replacement, 20, 50),
        row("cudaLaunchKernel", 30, 1, "cuda_runtime", {"correlation": 7}),
        row("custom_fused", 40, 3, "kernel", {"correlation": 7, "registers per thread": 48, "shared memory": 128})]}
    result = associate(trace, [{"sequence": 0, "call_id": "root"}], {"root": {"phase": "run", "kind": "project"}})
    assert set(result["call_resources"]) == {"root"}
    measured = result["bundle_resources"]["candidate"]["retired-call"]
    assert measured["kernels"] == 1 and measured["cuda_us"] == 3
    assert measured["launch_configurations"][0]["registers per thread"] == 48
    assert len(result["kernels"][0]["replacement_sites"]) == 1
