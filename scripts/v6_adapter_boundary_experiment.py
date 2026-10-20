"""Isolated, developer-started measurement of the stage-2 adapter audit findings.

This script does not edit production adapters or report autonomous optimization.
Reference observations stay in their original stores; only small results and
the exact experimental adapter bytes are written to the output.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from kernel_optimizer.project.request import ProjectRequest
from kernel_optimizer.runtime.execution import ExecutionRequest, LocalExecutor, write_json


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise ValueError("audited adapter bytes no longer match the planned experiment")
    return text.replace(old, new)


def variants(case, original):
    result = {"original": original}
    if case == "resnet-latency":
        result["retain_fixture"] = replace_once(original, '"pixels": pixels,', '"pixels": pixels,\n        "images": images,')
        changed = replace_once(original, '    torch.cuda.reset_peak_memory_stats(device)\n', "")
        changed = replace_once(changed, '    torch.cuda.synchronize(device)\n    state["peak_allocated_bytes"] = int(torch.cuda.max_memory_allocated(device))\n', "")
        result["outer_stats"] = replace_once(changed, 'float(state["peak_allocated_bytes"])', 'float(torch.cuda.max_memory_allocated(state["device"]))')
    elif case == "resnet-memory":
        changed = replace_once(original, '    torch.cuda.synchronize()\n    torch.cuda.reset_peak_memory_stats()\n', "")
        changed = replace_once(changed, '    torch.cuda.synchronize()\n    state["peak_allocated_bytes"] = int(torch.cuda.max_memory_allocated())\n', "")
        result["outer_stats"] = changed
    elif case == "ddp-step":
        result["clear_in_run_only"] = replace_once(original,
            '    state["optimizer"].zero_grad(set_to_none=True)\n', "")
    elif case == "qwen-throughput":
        result["source_timing_metric"] = replace_once(original,
            '"generated_tokens_per_second": work * 1000.0 / elapsed_ms,',
            '"generated_tokens_per_second": float(phase["generated_tokens_per_second"]),')
    else:
        raise ValueError(case)
    return result


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--archive-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cases", nargs="+", default=["resnet-latency", "resnet-memory", "qwen-throughput", "ddp-step"])
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    rows = []
    for case in args.cases:
        source = args.archive_root / case
        prior = args.baseline_root / case
        request = ProjectRequest.model_validate_json((source / "request.json").read_bytes())
        request.repeats = 15
        prepared = json.loads((prior / "prepared.json").read_text())
        reference = prepared["controls_path"]
        original = (source / "generated-adapter/opop_adapter.py").read_text()
        choices = variants(case, original)
        # AB/BA order (three arms: ABC/CBA). Separate fresh worker per endpoint.
        for block, order in enumerate((list(choices), list(reversed(choices)))):
            for arm in order:
                endpoint = args.output / case / f"{block}-{arm}"
                code = endpoint / "source"
                code.mkdir(parents=True)
                adapter = code / "opop_adapter.py"
                adapter.write_text(choices[arm])
                entry = code / "evaluate.py"
                entry.write_text("from kernel_optimizer.project.evaluation import worker_entry as run\n")
                result = LocalExecutor().run(ExecutionRequest(
                    entry=entry, cwd=code, output=endpoint / "evaluation",
                    devices=request.devices, distributed_backend=request.distributed_backend,
                    lease_dir=request.lease_dir, worker_timeout_s=900, wait_for_devices=True,
                    parameters={"adapter": str(adapter), "request": request.portable(),
                                "reference_dir": reference, "action": "evaluate"}))
                row = {"case": case, "block": block, "arm": arm,
                       "adapter_sha256": hashlib.sha256(adapter.read_bytes()).hexdigest(),
                       "reference": reference, "evaluation": result}
                rows.append(row)
                write_json(args.output / "result.json", {
                    "scope": "Developer-started isolated adapter audit; not autonomous optimization",
                    "protocol": "Fresh workers; AB/BA or ABC/CBA; 15 clean samples each; existing full-output reference checks",
                    "runs": rows})
                print(json.dumps({"case": case, "block": block, "arm": arm, "status": result["status"]}), flush=True)


if __name__ == "__main__":
    main()
