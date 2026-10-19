"""Optional Nsight Systems collector. Clean timing and Torch profiling stay separate."""
from pathlib import Path
import sqlite3
import subprocess

from kernel_optimizer.runtime.execution import write_json


def command_prefix(tool, *, counters=False):
    result = subprocess.run([tool, "profile", "--help"], capture_output=True, text=True, timeout=20)
    if result.returncode or "--capture-range" not in result.stdout:
        raise ValueError("installed Nsys does not expose the required capture-range interface")
    command = [tool, "profile", "--sample=none", "--cpuctxsw=none", "--trace=cuda,nvtx",
        "--capture-range=cudaProfilerApi", "--capture-range-end=stop", "--wait=all",
        "--output={output}/nsys-rank-{rank}"]
    if "--cuda-graph-trace" in result.stdout:
        command.append("--cuda-graph-trace=node")
    if counters:
        if "--gpu-metrics-devices" not in result.stdout:
            raise ValueError("installed Nsys does not support GPU metric collection")
        command += ["--gpu-metrics-devices=all", "--gpu-metrics-frequency=1000"]
    return command


def capture(adapter, state, operation, synchronize):
    import torch
    handles = []
    for root, model in adapter.models(state).items():
        for path, module in model.named_modules():
            name = root+("."+path if path else "")
            def before(module, args, name=name):
                torch.cuda.nvtx.range_push("opop.module:"+name)
            def after(module, args, result):
                torch.cuda.nvtx.range_pop()
            handles.append(module.register_forward_pre_hook(before))
            handles.append(module.register_forward_hook(after, always_call=True))
    try:
        adapter.reset(state)
        if callable(getattr(adapter, "before_run", None)):
            adapter.before_run(state)
        elif torch.distributed.is_initialized():
            torch.distributed.barrier()
        synchronize()
        torch.cuda.cudart().cudaProfilerStart()
        try:
            with torch.cuda.nvtx.range("opop.project"):
                operation(); synchronize()
        finally:
            torch.cuda.cudart().cudaProfilerStop()
    finally:
        for handle in handles:
            handle.remove()
    return {"valid": True, "collector": "nsys", "capture_boundary": "complete workload after reset and warmup"}


def normalize(path):
    with sqlite3.connect(path.resolve().as_uri()+"?mode=ro", uri=True) as db:
        db.row_factory = sqlite3.Row
        tables = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if "CUPTI_ACTIVITY_KIND_KERNEL" not in tables:
            return {"status": "unavailable", "reason": "no CUDA kernel table in exported trace", "evidence": str(path)}
        columns = {row[1] for row in db.execute("PRAGMA table_info(CUPTI_ACTIVITY_KIND_KERNEL)")}
        if not {"start", "end", "demangledName", "deviceId", "streamId"} <= columns or "StringIds" not in tables:
            return {"status": "unavailable", "reason": "unsupported Nsys SQLite kernel schema", "evidence": str(path)}
        rows = [dict(row) for row in db.execute("SELECT k.*, s.value AS name FROM CUPTI_ACTIVITY_KIND_KERNEL k LEFT JOIN StringIds s ON k.demangledName=s.id")]
        communication = [r for r in rows if "nccl" in (r["name"] or "").lower()]
        memcpy = []
        if "CUPTI_ACTIVITY_KIND_MEMCPY" in tables:
            memcpy = [dict(row) for row in db.execute("SELECT * FROM CUPTI_ACTIVITY_KIND_MEMCPY")]
        metric_tables = sorted(table for table in tables if "GPU_METRICS" in table)
        return {"status": "observed", "evidence": str(path), "cuda_kernels": len(rows),
            "nccl_kernels": len(communication), "nccl_kernel_sum_ms": sum(row["end"]-row["start"] for row in communication)/1e6,
            "kernels": rows, "memcpy": memcpy, "gpu_metric_tables": metric_tables,
            "hardware_traffic_status": "unknown", "hardware_traffic_reason": "Memcpy sizes and NCCL logical payload are not PCIe/NVLink traffic counters",
            "scope": "separate Nsys run; nanosecond SQLite timestamps, no cross-run or cross-rank timestamp subtraction"}


def export(tool, run_directory, ranks):
    result = []
    for rank in ranks:
        report = run_directory/f"nsys-rank-{rank}.nsys-rep"
        database = run_directory/f"nsys-rank-{rank}.sqlite"
        if not report.exists():
            result.append({"status": "unavailable", "reason": "Nsys report missing", "rank": rank})
            continue
        command = [tool, "export", "--type=sqlite", "--output", str(database), str(report)]
        process = subprocess.run(command, capture_output=True, text=True, timeout=120)
        (run_directory/f"nsys-export-{rank}.log").write_text(process.stdout+process.stderr)
        row = normalize(database) if process.returncode == 0 and database.exists() else {
            "status": "unavailable", "reason": "Nsys export failed", "returncode": process.returncode}
        row["rank"] = rank; result.append(row)
    write_json(run_directory/"nsys-normalized.json", result)
    return result
