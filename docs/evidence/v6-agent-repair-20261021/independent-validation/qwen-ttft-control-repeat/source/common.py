"""Small baseline utilities, deliberately outside the framework's production path."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import time

import torch

ROOT = Path(__file__).resolve().parent


def arguments(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--assets", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--workloads", type=Path, default=ROOT / "workloads.json")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    if args.warmup < 1 or args.repeats < 3:
        parser.error("Require at least one warmup and three clean samples")
    args.output.mkdir(parents=True, exist_ok=True)
    return args


def configure(device=0):
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.manual_seed(20261020)
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision("highest")


def digest(path):
    with Path(path).open("rb") as reader:
        return hashlib.file_digest(reader, "sha256").hexdigest()


def metadata(args, case):
    return {
        "case": case, "created_unix": time.time(), "protocol": "readiness_baseline_not_optimization",
        "python": platform.python_version(), "torch": torch.__version__, "cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(), "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "gpu_inventory": subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid,pci.bus_id,name", "--format=csv,noheader"], text=True).strip().splitlines(),
        "pid": os.getpid(), "warmup": args.warmup, "repeats": args.repeats,
        "source_sha256": {p.name: digest(p) for p in sorted(ROOT.glob("*.py"))},
        "workloads_sha256": digest(args.workloads), "quality_scope": "Reference fidelity; no downstream accuracy/convergence claim",
    }


def save(path, content):
    Path(path).write_text(json.dumps(content, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def summarize(samples):
    result = {}
    for key in samples[0]:
        values = [s[key] for s in samples]
        result[key] = {"median": statistics.median(values), "min": min(values), "max": max(values)}
    return result


def close_check(actual, expected, *, atol, rtol):
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
    return {"passed": True, "max_abs_error": (actual.float() - expected.float()).abs().max().item(), "atol": atol, "rtol": rtol}


def negative_check(reference, *, atol, rtol):
    try:
        torch.testing.assert_close(reference + 1, reference, atol=atol, rtol=rtol)
    except AssertionError:
        return {"corrupted_output_rejected": True}
    raise AssertionError("Correctness control did not reject +1 corruption")


def profile_once(function, output):
    """Diagnostic run separate from all clean timing. Raw traces stay in the run dir."""
    torch.cuda.synchronize()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA], record_shapes=True) as profiler:
        function()
        torch.cuda.synchronize()
    profiler.export_chrome_trace(str(output))
    events = json.loads(Path(output).read_text())["traceEvents"]
    kernels = [e for e in events if e.get("cat") == "kernel" and e.get("ph") == "X"]
    nccl = [e for e in kernels if "nccl" in e.get("name", "").lower()]
    cpu_collectives = [e for e in events if e.get("cat") == "cpu_op" and "allreduce" in e.get("name", "").lower()]
    return {"trace_file": Path(output).name, "trace_sha256": digest(output), "cuda_kernel_count": len(kernels),
            "nccl_kernel_count": len(nccl), "nccl_kernel_sum_ms": sum(e["dur"] for e in nccl) / 1000,
            "cpu_allreduce_event_count": len(cpu_collectives),
            "nccl_kernel_names": sorted({e["name"] for e in nccl}),
            "top_cuda_kernels_by_total_us": sorted(
                ({"name": name, "total_us": sum(e["dur"] for e in kernels if e["name"] == name),
                  "calls": sum(e["name"] == name for e in kernels)} for name in {e["name"] for e in kernels}),
                key=lambda e: e["total_us"], reverse=True)[:15]}
