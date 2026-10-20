"""Manual communication opportunity screening; never an autonomous result.

Compare fresh processes with controls applied before torch/NCCL initialization.
Keep quality checks and clean step time separate from diagnostic NCCL duration.
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
from pathlib import Path
import statistics
import re

from kernel_optimizer.runtime.execution import ExecutionRequest, LocalExecutor, write_json


def run(context):
    import torch
    from kernel_optimizer.project.evaluation import load_adapter, worker_entry
    # This uses all original gradient/parameter references for two continuous steps.
    evaluation = worker_entry(context)
    if not evaluation.get("valid"):
        return {"evaluation": evaluation, "diagnostics": [], "environment": {
            k: v for k, v in os.environ.items() if k.startswith("NCCL_")}}
    p = context["parameters"]
    adapter = load_adapter(Path(p["adapter"]))
    state = adapter.prepare(context, p["request"])
    if hasattr(adapter, "set_mode"):
        adapter.set_mode(state, "profile")

    def synchronize():
        torch.cuda.synchronize()

    def before():
        adapter.reset(state)
        if hasattr(adapter, "before_run"):
            adapter.before_run(state)
        synchronize()

    for _ in range(3):
        before()
        adapter.run(state)
        synchronize()
    diagnostics = []
    for index in range(3):
        before()
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                torch.profiler.ProfilerActivity.CUDA]) as profiler:
            adapter.run(state)
            synchronize()
        path = Path(context["output"]) / f"rank-{context['rank']}-trace-{index}.json"
        profiler.export_chrome_trace(str(path))
        raw = path.read_bytes()
        trace = json.loads(raw)
        kernels = [e for e in trace["traceEvents"]
                   if e.get("cat") == "kernel" and "nccl" in e.get("name", "").lower()]
        diagnostics.append({"nccl_kernel_sum_ms": sum(e.get("dur", 0) for e in kernels) / 1000,
                            "kernels": sorted({e["name"] for e in kernels}), "calls": len(kernels),
                            "trace": str(path.with_suffix(".json.gz"))})
        with gzip.open(path.with_suffix(".json.gz"), "wb") as compressed:
            compressed.write(raw)
        path.unlink()  # same newly generated diagnostic retained losslessly as gzip
    return {"evaluation": evaluation, "diagnostics": diagnostics, "environment": {
        k: v for k, v in os.environ.items() if k.startswith("NCCL_")},
        "caveat": "Summed NCCL device duration includes waiting and overlap; not wire time or end-to-end savings."}


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--arms", type=Path, help="JSON list of {name, env}; explicit manual experiment controls")
    args = parser.parse_args()
    request = json.loads((args.archive / "request.json").read_text())
    request["repeats"] = 15
    prepared = json.loads((args.baseline / "prepared.json").read_text())
    args.output.mkdir(parents=True, exist_ok=False)
    rows = []
    choices = json.loads(args.arms.read_text()) if args.arms else [
        {"name": "auto", "env": {}}, {"name": "Simple", "env": {"NCCL_PROTO": "Simple"}},
        {"name": "LL", "env": {"NCCL_PROTO": "LL"}}]
    # No deliberate weakening of the auto baseline. LL is a diagnostic control.
    for block, order in enumerate((choices, list(reversed(choices)))):
        for choice in order:
            arm = choice["name"]
            if arm == "auto" and "NCCL_PROTO" in os.environ:
                raise ValueError("auto baseline requires an unforced controller environment")
            destination = args.output / f"{block}-{arm}"
            env = {"NCCL_DEBUG": "INFO", "NCCL_DEBUG_SUBSYS": "INIT,ENV,GRAPH,TUNING",
                   "NCCL_DEBUG_FILE": str(destination / "nccl.%h.%p.log")}
            env.update(choice["env"])
            result = LocalExecutor().run(ExecutionRequest(
                entry=Path(__file__), cwd=args.archive, output=destination,
                devices=request["devices"], distributed_backend=request["distributed_backend"],
                lease_dir=Path(request["lease_dir"]), worker_timeout_s=900,
                wait_for_devices=True, env=env,
                parameters={"adapter": str(args.archive / "generated-adapter/opop_adapter.py"),
                            "request": request, "reference_dir": prepared["controls_path"],
                            "action": "evaluate"}))
            row = {"block": block, "arm": arm, "result": result}
            decisions = {}
            for log in destination.glob("nccl.*.log"):
                for line in log.read_text(errors="replace").splitlines():
                    match = re.search(r"(\w+): (\d+) Bytes -> Algo (\S+) proto (\S+) channel\{Lo\.\.Hi\}=\{(\d+)\.\.(\d+)\}", line)
                    if match:
                        key = match.group(0)
                        decisions[key] = decisions.get(key, 0) + 1
            row["tuning_log_decisions"] = decisions
            row["protocol_evidence"] = "NCCL tuning log, full worker scope including preparation/quality. Kernel entry names alone do not identify the selected algorithm/protocol."
            if result["status"] == "passed":
                rank_data = [rank["result"] for rank in result["ranks"]]
                row["quality_passed"] = all(rank["evaluation"]["valid"] for rank in rank_data)
                if row["quality_passed"]:
                    row["max_rank_step_ms"] = statistics.median([
                        max(r["evaluation"]["samples"][i]["step_ms"] for r in rank_data)
                        for i in range(request["repeats"])])
                    row["diagnostic_max_rank_nccl_kernel_sum_ms"] = statistics.median([
                        max(r["diagnostics"][i]["nccl_kernel_sum_ms"] for r in rank_data) for i in range(3)])
                    row["actual_kernels"] = sorted({name for rank in rank_data
                        for diagnostic in rank["diagnostics"] for name in diagnostic["kernels"]})
            rows.append(row)
            write_json(args.output / "result.json", {"scope": "Manual opportunity screening, not autonomous optimization",
                "protocol": "Fresh process per endpoint; original no_sync workload; full two-step quality checks; reversed arm order",
                "runs": rows})
            print(json.dumps({k: v for k, v in row.items() if k != "result"}), flush=True)


if __name__ == "__main__":
    main()
