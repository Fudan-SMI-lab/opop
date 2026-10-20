from pathlib import Path
import json,sys
from kernel_optimizer.project.evaluation import compare
def source_reference(context):
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
        with torch.no_grad():
            expected_micro = [(model(input_ids=tokens, labels=tokens, use_cache=False).loss / workload['accumulation_steps']).detach().cpu() for tokens in inputs]
        loss = original.accumulate(model, optimizer, inputs)
        gradients = {name: p.grad.detach().cpu() for name, p in model.named_parameters()}
        optimizer.step()
        weights = {name: p.detach().cpu() for name, p in model.named_parameters()}
        for rank in range(workload["world_size"]):
            actual = observations[rank][step]
            checks.append({"rank": rank, "step": step,
                "gradients": compare(actual["gradients"], gradients, request["quality"]),
                "parameters": compare(actual["parameters"], weights, request["quality"]),
                "micro_losses": compare(actual["micro_losses"], tuple(expected_micro[rank*workload['accumulation_steps']:(rank+1)*workload['accumulation_steps']]), request["quality"])})
        mean_loss = sum(float(rows[step]["loss"]) for rows in observations)/len(observations)
        checks.append({"step": step, "global_loss": compare(mean_loss, float(loss.detach()), request["quality"])})
        del gradients, weights
    valid = all(check["passed"] for row in checks for name, check in row.items() if name in {"gradients", "parameters", "global_loss", "micro_losses"})
    return {"valid": valid, "checks": checks, "method": "original source accumulates the exact global batch serially, for two continuous optimizer updates"}

def audit(context):
    import torch
    result=source_reference(context)
    p=context['parameters'];request=p['request']
    ranks=[torch.load(Path(p['controls'])/f'reference-rank-{rank}.pt',map_location='cpu',weights_only=True,mmap=True) for rank in range(request['workload']['world_size'])]
    checks=[compare(ranks[0][step]['gradients'],ranks[1][step]['gradients'],{'atol':0,'rtol':0}) for step in range(request['continuous_steps'])]
    result.update(replica_gradient_checks=checks,extra_observation_audit='Per-rank normalized micro-loss tuples from independent original-model forwards before each full serial global-batch update')
    result['valid']=result['valid'] and all(x['passed'] for x in checks)
    return result
