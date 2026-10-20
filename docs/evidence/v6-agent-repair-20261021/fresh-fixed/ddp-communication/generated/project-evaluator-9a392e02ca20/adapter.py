"""Adapter for the gpt2_ddp_training source: one accumulated two-rank DDP NCCL
training step of GPT-2 124M with seeded synthetic weights and token IDs.

Semantics follow inputs/source/gpt2_ddp_training.py exactly: DDP with
bucket_cap_mb=25, gradient_as_bucket_view=True, broadcast_buffers=False;
no_sync on the first accumulation_steps-1 micro-batches; FP32 with TF32
disabled; SGD lr=0.001, momentum=0.0, weight_decay=0.0, foreach=False.
"""
from __future__ import annotations

from contextlib import nullcontext
from datetime import timedelta

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import GPT2Config, GPT2LMHeadModel

try:
    from kernel_optimizer.project.discovery import phase
except ImportError:
    def phase(name):
        return nullcontext()


MODEL = {"n_layer": 12, "n_embd": 768, "n_head": 12, "vocab_size": 50257, "n_positions": 1024}


def configure(device):
    torch.set_num_threads(4)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    torch.manual_seed(20261020)
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision("highest")


def make_model(device):
    torch.manual_seed(20261020)
    model_config = GPT2Config(**MODEL, resid_pdrop=0.0, embd_pdrop=0.0, attn_pdrop=0.0, use_cache=False, attn_implementation="eager")
    return GPT2LMHeadModel(model_config).float().to(device).train()


def make_inputs(workload, source_rank, device):
    generator = torch.Generator().manual_seed(31000 + source_rank)
    return [torch.randint(0, MODEL["vocab_size"], (workload["micro_batch"], workload["sequence_length"]), generator=generator).to(device)
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
    return losses


def prepare(context, request):
    workload = request["workload"]
    rank = int(context.get("rank", 0))
    world_size = int(context.get("world_size", 1))
    device = context["device"]
    distributed = dist.is_available() and dist.is_initialized()
    if distributed and world_size != workload["world_size"]:
        raise ValueError("This workload must launch the declared number of ranks")
    configure(device)
    device_index = torch.device(device).index
    if device_index is None:
        device_index = torch.cuda.current_device()
    raw_model = make_model(device)
    if distributed:
        model = DDP(raw_model, device_ids=[device_index], broadcast_buffers=False, bucket_cap_mb=25, gradient_as_bucket_view=True)
    else:
        model = raw_model
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.0, weight_decay=0.0, foreach=False)
    inputs = make_inputs(workload, rank, device)
    initial = {name: value.detach().cpu().clone() for name, value in raw_model.state_dict().items()}
    control = dist.new_group(backend="gloo", timeout=timedelta(seconds=120)) if distributed else None
    return {"rank": rank, "world_size": world_size, "device": str(device), "raw_model": raw_model, "model": model,
            "optimizer": optimizer, "inputs": inputs, "initial": initial, "control": control,
            "workload": workload, "mode": "measure"}


def reset(state):
    state["raw_model"].load_state_dict(state["initial"])


def run(state):
    with phase("gradient_accumulation"):
        losses = accumulate(state["model"], state["optimizer"], state["inputs"])
        loss = torch.stack(losses).sum()
    with phase("optimizer_step"):
        state["optimizer"].step()
        torch.cuda.synchronize()
    return {"loss": loss.detach(), "micro_losses": losses}


def observe(state, result):
    raw_model = state["raw_model"]
    gradients = {name: parameter.grad.detach().cpu() for name, parameter in raw_model.named_parameters()}
    parameters = {name: parameter.detach().cpu() for name, parameter in raw_model.named_parameters()}
    return {
        "loss": result["loss"].detach().cpu(),
        "micro_losses": tuple(tensor.detach().cpu() for tensor in result["micro_losses"]),
        "gradients": gradients,
        "parameters": parameters,
    }


def metrics(state, result, elapsed_ms):
    workload = state["workload"]
    elapsed = float(elapsed_ms)
    token_count = workload["world_size"] * workload["micro_batch"] * workload["sequence_length"] * workload["accumulation_steps"]
    return {
        "step_ms": elapsed,
        "tokens_per_second": token_count * 1000.0 / elapsed,
        "tokens_per_step": float(token_count),
        "effective_batch": float(workload["world_size"] * workload["micro_batch"] * workload["accumulation_steps"]),
        "peak_allocated_bytes": float(torch.cuda.max_memory_allocated()),
        "parameter_count": float(sum(parameter.numel() for parameter in state["raw_model"].parameters())),
        "world_size": float(workload["world_size"]),
        "micro_batch": float(workload["micro_batch"]),
        "sequence_length": float(workload["sequence_length"]),
        "accumulation_steps": float(workload["accumulation_steps"]),
        "step_ms_max_baseline_ratio": 1.05,
    }


def models(state):
    return {"model": state["raw_model"]}


def set_mode(state, mode):
    state["mode"] = mode


def before_run(state):
    torch.cuda.synchronize()
    if state.get("control") is not None:
        dist.barrier(group=state["control"])
