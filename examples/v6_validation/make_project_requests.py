"""Translate this public validation matrix into ordinary project requests.

Case definitions belong here; the project coordinator has no model-name routing.
No target groups, candidate implementations or profiling results are supplied.
"""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--deployment", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--leases", type=Path, required=True)
    args = parser.parse_args()
    source = Path(__file__).resolve().parent
    cases = json.loads((source/"workloads.json").read_text())
    deployment = json.loads(args.deployment.read_text())
    units = {"ttft_ms": "ms", "decode_tokens_per_second": "tokens/s", "generated_tokens_per_second": "tokens/s",
             "latency_ms": "ms", "images_per_second": "images/s", "peak_allocated_bytes": "bytes",
             "step_ms": "ms", "diagnostic.max_rank_nccl_kernel_sum_ms": "ms"}
    args.output.mkdir(parents=True, exist_ok=True)
    requests = []
    for name in ("resnet_inference", "qwen_inference", "gpt2_ddp_training"):
        case = cases[name]
        quality = case["correctness"]
        atol = quality.get("logits_atol", quality.get("grad_atol"))
        rtol = quality.get("logits_rtol", quality.get("grad_rtol"))
        for workload in case["workloads"]:
            for goal in workload["goals"]:
                objective = {"name": goal["metric"], "direction": {"min": "minimize", "max": "maximize"}[goal["direction"]],
                             "unit": units[goal["metric"]], "paired_repeats": 2}
                constraints = []
                for label, ratio in goal.get("constraints", {}).items():
                    for suffix, op in (("_min_baseline_ratio", "ge"), ("_max_baseline_ratio", "le")):
                        if label.endswith(suffix):
                            constraints.append({"metric": label.removesuffix(suffix), "baseline_ratio": ratio, "op": op})
                if constraints:
                    objective["constraints"] = constraints
                distributed = workload.get("world_size", 1) > 1
                request = {"source_root": str(source), "entry": name+".py", "files": [name+".py", "common.py", "workloads.json"],
                    "goal": f"{objective['direction']} {goal['metric']} for {case['description']}; preserve the complete frozen workload and source quality checks",
                    "objective": objective, "workload": workload, "quality": {**quality, "atol": atol, "rtol": rtol},
                    "assets": {"assets": deployment["assets"][name]} if name in deployment["assets"] else {},
                    "devices": deployment["dual_gpu_visible_devices" if distributed else "single_gpu_visible_devices"].split(","),
                    "distributed_backend": "nccl" if distributed else None, "lease_dir": str(args.leases.resolve()),
                    "tools": deployment.get("tools", {}), "worker_timeout_s": 900,
                    "notes": ["Preserve source dtype, TF32 policy, seed, real data and completion boundaries.",
                              "Model loading is outside timing. Generated replacement work and synchronization stay inside the declared workload boundary."]}
                if goal["metric"] == "diagnostic.max_rank_nccl_kernel_sum_ms":
                    objective["source"] = {"collector": "torch_communication", "field": "communication_kernel_sum_ms"}
                    objective["rank_aggregation"] = "max"
                    request.update(communication="required", required_observations=["collective_identity", "logical_payload", "cross_rank_collectives"],
                                   profiling={"metric_repeats": 3})
                path = args.output/f"{name}-{workload['id']}-{goal['id']}.json"
                path.write_text(json.dumps(request, indent=2)+"\n")
                requests.append(str(path))
    print(json.dumps({"requests": requests}, indent=2))


if __name__ == "__main__":
    main()
