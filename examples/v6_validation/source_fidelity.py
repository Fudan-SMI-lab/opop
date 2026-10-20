"""Independent source checks for automatically generated validation adapters.

This is a validation-case utility, never a project optimizer or target selector.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

from kernel_optimizer.project.evaluation import compare
from kernel_optimizer.runtime.execution import ExecutionRequest, LocalExecutor, write_json


def ddp_serial_reference(context):
    import torch
    parameters = context["parameters"]
    source = Path(parameters["source"])
    sys.path.insert(0, str(source))
    import gpt2_ddp_training as original
    original.configure(0)
    request = parameters["request"]
    config = json.loads((source/"workloads.json").read_text())["gpt2_ddp_training"]
    workload = request["workload"]
    model = original.make_model(config, 0)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.0, weight_decay=0.0, foreach=False)
    inputs = [tokens for rank in range(workload["world_size"])
              for tokens in original.make_inputs(config, workload, rank, "cuda:0")]
    observations = [torch.load(Path(parameters["controls"])/f"reference-rank-{rank}.pt", map_location="cpu", weights_only=True, mmap=True)
                    for rank in range(workload["world_size"])]
    checks = []
    for step in range(request["continuous_steps"]):
        loss = original.accumulate(model, optimizer, inputs)
        gradients = {name: p.grad.detach().cpu() for name, p in model.named_parameters()}
        optimizer.step()
        weights = {name: p.detach().cpu() for name, p in model.named_parameters()}
        for rank in range(workload["world_size"]):
            actual = observations[rank][step]
            checks.append({"rank": rank, "step": step,
                "gradients": compare(actual["gradients"], gradients, request["quality"]),
                "parameters": compare(actual["parameters"], weights, request["quality"])})
        mean_loss = sum(float(rows[step]["loss"]) for rows in observations)/len(observations)
        checks.append({"step": step, "global_loss": compare(mean_loss, float(loss.detach()), request["quality"])})
        del gradients, weights
    valid = all(check["passed"] for row in checks for name, check in row.items() if name in {"gradients", "parameters", "global_loss"})
    return {"valid": valid, "checks": checks, "method": "original source accumulates the exact global batch serially, for two continuous optimizer updates"}


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--case", choices=["qwen", "resnet", "ddp"], required=True)
    parser.add_argument("--readiness-reference", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prepared = json.loads((args.project/"prepared.json").read_text())
    request = json.loads((args.project/"request.json").read_text())
    controls = Path(prepared["controls_path"])
    if args.case in {"qwen", "resnet"}:
        import torch
        reference = torch.load(args.readiness_reference, map_location="cpu", weights_only=True)
        actual = torch.load(controls/"reference-rank-0.pt", map_location="cpu", weights_only=True)
        expected = ({"tokens": reference["tokens"], "logits": reference["logits"]} if args.case == "qwen"
                    else {"logits": reference["logits"], "top1": reference["logits"].argmax(-1)})
        checks = [compare(row, expected, {"atol": 0, "rtol": 0} if args.case == "qwen" else request["quality"])
                  for row in actual]
        result = {"valid": all(check["passed"] for check in checks), "checks": checks,
                  "reference": str(args.readiness_reference), "reference_sha256": hashlib.sha256(args.readiness_reference.read_bytes()).hexdigest()}
    else:
        source = controls.parent/"reference-source"
        result = LocalExecutor().run(ExecutionRequest(entry=Path(__file__).resolve(), function="ddp_serial_reference",
            cwd=source, output=args.output.with_suffix(".run"), devices=request["devices"][:1],
            lease_dir=Path(request["lease_dir"]), wait_for_devices=True, worker_timeout_s=request["worker_timeout_s"],
            parameters={"request": request, "source": str(source), "controls": str(controls)}))
    write_json(args.output, result)
    print(json.dumps(result))
    if args.case == "ddp":
        return 0 if result["status"] == "passed" and result["ranks"][0]["result"]["valid"] else 1
    return 0 if result["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
