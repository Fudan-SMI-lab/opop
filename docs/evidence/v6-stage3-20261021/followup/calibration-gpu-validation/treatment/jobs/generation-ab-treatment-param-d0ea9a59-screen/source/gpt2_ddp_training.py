"""Project input: full GPT-2 architecture training with actual two-rank NCCL.

Weights and token IDs are seeded synthetic fixtures, not pretrained GPT-2 or a
language-quality benchmark. The baseline already uses no_sync accumulation.
"""
from __future__ import annotations

from contextlib import nullcontext
from datetime import timedelta
import gc
import json
import os
import time

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.algorithms.ddp_comm_hooks import default_hooks
from transformers import GPT2Config, GPT2LMHeadModel

from common import arguments, close_check, configure, metadata, negative_check, profile_once, save, summarize


def make_model(config, rank):
    torch.manual_seed(20261020)
    model_config = GPT2Config(**config["model"], resid_pdrop=0.0, embd_pdrop=0.0, attn_pdrop=0.0, use_cache=False, attn_implementation="eager")
    return GPT2LMHeadModel(model_config).float().to(f"cuda:{rank}").train()


def make_inputs(config, workload, source_rank, device):
    generator = torch.Generator().manual_seed(31000 + source_rank)
    return [torch.randint(0, config["model"]["vocab_size"], (workload["micro_batch"], workload["sequence_length"]), generator=generator).to(device)
            for _ in range(workload["accumulation_steps"])]


def accumulate(model, optimizer, inputs):
    optimizer.zero_grad(set_to_none=True)
    losses = []
    for i, tokens in enumerate(inputs):
        context = model.no_sync() if isinstance(model, DDP) and i + 1 < len(inputs) else nullcontext()
        with context:
            loss = model(input_ids=tokens, labels=tokens, use_cache=False).loss / len(inputs)
            loss.backward()
            losses.append(loss.detach())
    return torch.stack(losses).sum()


def main():
    args = arguments(__doc__)
    rank = int(os.environ["LOCAL_RANK"])
    configure(rank)
    dist.init_process_group("nccl", timeout=timedelta(seconds=120), device_id=torch.device("cuda", rank))
    control = dist.new_group(backend="gloo", timeout=timedelta(seconds=120))
    config = json.loads(args.workloads.read_text())["gpt2_ddp_training"]
    workload = config["workloads"][0]
    if dist.get_world_size() != workload["world_size"]:
        raise ValueError("This workload must launch the declared number of ranks")
    raw_model = make_model(config, rank)
    model = DDP(raw_model, device_ids=[rank], broadcast_buffers=False, bucket_cap_mb=25, gradient_as_bucket_view=True)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.0, weight_decay=0.0, foreach=False)
    inputs = make_inputs(config, workload, rank, f"cuda:{rank}")
    initial = {name: value.detach().cpu().clone() for name, value in raw_model.state_dict().items()}
    q = config["correctness"]
    loss = accumulate(model, optimizer, inputs)
    assert torch.isfinite(loss)
    checks = {"finite_loss": True}
    # Reference accumulates the exact global batch serially, without DDP.
    if rank == 0:
        reference = make_model(config, rank)
        ref_optimizer = torch.optim.SGD(reference.parameters(), lr=0.001, momentum=0.0, weight_decay=0.0, foreach=False)
        global_inputs = [x for source_rank in range(dist.get_world_size()) for x in make_inputs(config, workload, source_rank, f"cuda:{rank}")]
        ref_loss = accumulate(reference, ref_optimizer, global_inputs)
        gradient_errors = []
        for actual, expected in zip(raw_model.parameters(), reference.parameters(), strict=True):
            result = close_check(actual.grad, expected.grad, atol=q["grad_atol"], rtol=q["grad_rtol"])
            gradient_errors.append(result["max_abs_error"])
        checks["global_batch_gradient_check"] = {"passed": True, "max_abs_error": max(gradient_errors), "reference_loss": ref_loss.item()}
        checks.update(negative_check(next(reference.parameters()).grad.flatten()[:4096], atol=q["grad_atol"], rtol=q["grad_rtol"]))
        ref_optimizer.step()
    # Check every rank's full gradients, outside all measurements/traces.
    for parameter in raw_model.parameters():
        replica = parameter.grad.detach().clone() if rank == 0 else torch.empty_like(parameter.grad)
        dist.broadcast(replica, src=0)
        torch.testing.assert_close(parameter.grad, replica, atol=0.0, rtol=0.0)
        del replica
    checks["exact_gradient_replica_check"] = True
    optimizer.step()
    if rank == 0:
        errors = [close_check(actual, expected, atol=q["parameter_atol"], rtol=q["parameter_rtol"])["max_abs_error"]
                  for actual, expected in zip(raw_model.parameters(), reference.parameters(), strict=True)]
        checks["parameter_update_check"] = {"passed": True, "max_abs_error": max(errors)}
        del reference, ref_optimizer, global_inputs, actual, expected
    # Release gradients and reference before collecting memory measurements.
    optimizer.zero_grad(set_to_none=True)
    gc.collect()
    torch.cuda.empty_cache()
    dist.barrier(group=control)

    def reset():
        raw_model.load_state_dict(initial)

    def step():
        accumulate(model, optimizer, inputs)
        optimizer.step()
        torch.cuda.synchronize()

    for _ in range(args.warmup):
        reset()
        step()
    samples = []
    for _ in range(args.repeats):
        reset()
        torch.cuda.synchronize()
        dist.barrier(group=control)
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter_ns()
        step()
        elapsed = (time.perf_counter_ns() - start) / 1e6
        samples.append({"step_ms": elapsed, "peak_allocated_bytes": torch.cuda.max_memory_allocated()})
    diagnostics = []
    if args.profile:
        for index in range(3):
            reset()
            dist.barrier(group=control)
            diagnostics.append(profile_once(step, args.output / f"rank-{rank}-step-{index}-trace.json"))
    # Inventory is collected in a separate diagnostic step. It does not alter
    # the reducer used for clean timing or the profiler samples above.
    buckets = []

    def inventory_hook(state, bucket):
        tensor = bucket.buffer()
        state.append({"bucket_index": bucket.index(), "elements": tensor.numel(), "logical_bytes": tensor.numel() * tensor.element_size(), "dtype": str(tensor.dtype)})
        return default_hooks.allreduce_hook(None, bucket)

    model.register_comm_hook(buckets, inventory_hook)
    reset()
    dist.barrier(group=control)
    step()
    rank_record = {"rank": rank, "pid": os.getpid(), "device": str(next(raw_model.parameters()).device),
                   "peer_access": {str(other): torch.cuda.can_device_access_peer(rank, other) for other in range(dist.get_world_size()) if other != rank},
                   "correctness": checks, "samples": samples, "profiles": diagnostics,
                   "separate_hook_inventory": {"buckets": buckets, "allreduce_calls": len(buckets), "logical_bytes_per_rank": sum(b["logical_bytes"] for b in buckets)}}
    save(args.output / f"rank-{rank}.json", rank_record)
    gathered = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, rank_record, group=control)
    if rank == 0:
        combined = []
        token_count = workload["world_size"] * workload["micro_batch"] * workload["sequence_length"] * workload["accumulation_steps"]
        for i in range(args.repeats):
            duration = max(r["samples"][i]["step_ms"] for r in gathered)
            combined.append({"step_ms": duration, "tokens_per_second": token_count * 1000 / duration,
                             "peak_allocated_bytes": max(r["samples"][i]["peak_allocated_bytes"] for r in gathered)})
        record = metadata(args, "gpt2_ddp_training")
        record.update({"workload": workload, "parameter_count": sum(p.numel() for p in raw_model.parameters()), "effective_batch": workload["world_size"] * workload["micro_batch"] * workload["accumulation_steps"], "tokens_per_step": token_count, "samples": combined, "summary": summarize(combined), "ranks": gathered})
        if args.profile:
            values = [max(r["profiles"][i]["nccl_kernel_sum_ms"] for r in gathered) for i in range(3)]
            record["diagnostic"] = {"max_rank_nccl_kernel_sum_ms_samples": values, "max_rank_nccl_kernel_sum_ms": sorted(values)[1], "definition": config["communication_metric_caveat"]}
        save(args.output / "result.json", record)
        print(json.dumps(record["summary"]), flush=True)
    dist.destroy_process_group(control)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
