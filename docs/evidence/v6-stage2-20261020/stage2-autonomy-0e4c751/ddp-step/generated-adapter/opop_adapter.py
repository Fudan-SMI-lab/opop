"""Adapter for the frozen gpt2_ddp_training 'accumulated_step' workload.

Mirrors inputs/source/gpt2_ddp_training.py and inputs/source/common.py:
seeded synthetic GPT-2 124M FP32 weights, per-rank seeded synthetic token
inputs, no_sync gradient accumulation over 4 micro-steps, DDP gradient
allreduce on the final micro-step, and an SGD update, all under the source's
numerical policies (TF32 disabled, 'highest' float32 matmul precision, weight
seed 20261020). No communication hooks and no profiler in the timed path;
the framework collects NCCL communication diagnostics separately.
"""
from __future__ import annotations

from contextlib import nullcontext
from datetime import timedelta

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import GPT2Config, GPT2LMHeadModel

try:
    from kernel_optimizer.project.discovery import phase as _phase
except Exception:
    _phase = None


def _annotate(name):
    if _phase is None:
        return nullcontext()
    return _phase(name)


# Frozen from inputs/source/workloads.json -> gpt2_ddp_training.model.
MODEL_CONFIG = {"n_layer": 12, "n_embd": 768, "n_head": 12, "vocab_size": 50257, "n_positions": 1024}
WEIGHT_SEED = 20261020
INPUT_SEED_BASE = 31000
SGD_KWARGS = {"lr": 0.001, "momentum": 0.0, "weight_decay": 0.0, "foreach": False}
DDP_KWARGS = {"broadcast_buffers": False, "bucket_cap_mb": 25, "gradient_as_bucket_view": True}


def _configure(device):
    torch.set_num_threads(4)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    torch.manual_seed(WEIGHT_SEED)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision("highest")
    if device.type == "cuda":
        torch.cuda.set_device(device)


def _make_model(device):
    torch.manual_seed(WEIGHT_SEED)
    config = GPT2Config(**MODEL_CONFIG, resid_pdrop=0.0, embd_pdrop=0.0, attn_pdrop=0.0, use_cache=False, attn_implementation="eager")
    return GPT2LMHeadModel(config).float().to(device).train()


def _make_inputs(workload, source_rank, device):
    generator = torch.Generator().manual_seed(INPUT_SEED_BASE + source_rank)
    return [torch.randint(0, MODEL_CONFIG["vocab_size"], (workload["micro_batch"], workload["sequence_length"]), generator=generator).to(device)
            for _ in range(workload["accumulation_steps"])]


def _accumulate(model, optimizer, inputs):
    optimizer.zero_grad(set_to_none=True)
    losses = []
    for i, tokens in enumerate(inputs):
        context = model.no_sync() if isinstance(model, DDP) and i + 1 < len(inputs) else nullcontext()
        with context:
            loss = model(input_ids=tokens, labels=tokens, use_cache=False).loss / len(inputs)
            loss.backward()
            losses.append(loss.detach())
    return torch.stack(losses).sum(), losses


def prepare(context, request):
    device = torch.device(context["device"])
    rank = int(context["rank"])
    world_size = int(context["world_size"])
    workload = request["workload"]
    _configure(device)
    distributed = dist.is_available() and dist.is_initialized()
    if distributed and dist.get_world_size() != workload["world_size"]:
        raise ValueError("This workload must launch the declared number of ranks")
    raw_model = _make_model(device)
    if distributed:
        device_index = device.index if device.index is not None else torch.cuda.current_device()
        model = DDP(raw_model, device_ids=[device_index], **DDP_KWARGS)
        control_group = dist.new_group(backend="gloo", timeout=timedelta(seconds=120))
    else:
        model = raw_model
        control_group = None
    optimizer = torch.optim.SGD(model.parameters(), **SGD_KWARGS)
    inputs = _make_inputs(workload, rank, device)
    initial = {name: value.detach().cpu().clone() for name, value in raw_model.state_dict().items()}
    return {
        "device": device,
        "rank": rank,
        "world_size": world_size,
        "workload": workload,
        "raw_model": raw_model,
        "model": model,
        "optimizer": optimizer,
        "inputs": inputs,
        "initial": initial,
        "distributed": distributed,
        "control_group": control_group,
        "parameter_count": sum(p.numel() for p in raw_model.parameters()),
        "mode": None,
    }


def reset(state):
    state["raw_model"].load_state_dict(state["initial"])


def before_run(state):
    if state["device"].type == "cuda":
        torch.cuda.synchronize()
    if state["distributed"]:
        dist.barrier(group=state["control_group"])
    if state["device"].type == "cuda":
        torch.cuda.reset_peak_memory_stats()


def set_mode(state, mode):
    state["mode"] = mode


def run(state):
    with _annotate("accumulation"):
        loss, micro_losses = _accumulate(state["model"], state["optimizer"], state["inputs"])
    with _annotate("optimizer_step"):
        state["optimizer"].step()
    if state["device"].type == "cuda":
        with _annotate("finalize"):
            torch.cuda.synchronize()
    return {"loss": loss, "micro_losses": micro_losses}


def observe(state, result):
    gradients = {}
    parameters = {}
    for name, parameter in state["raw_model"].named_parameters():
        grad = parameter.grad
        gradients[name] = grad.detach().cpu().clone() if grad is not None else torch.zeros_like(parameter, device="cpu")
        parameters[name] = parameter.detach().cpu().clone()
    return {
        "loss": result["loss"].detach().cpu().clone(),
        "gradients": gradients,
        "parameters": parameters,
    }


def metrics(state, result, elapsed_ms):
    workload = state["workload"]
    elapsed = float(elapsed_ms)
    token_count = workload["world_size"] * workload["micro_batch"] * workload["sequence_length"] * workload["accumulation_steps"]
    project_metrics = {
        "step_ms": elapsed,
        "tokens_per_second": token_count * 1000.0 / elapsed,
        "tokens_per_step": float(token_count),
        "effective_batch": float(workload["world_size"] * workload["micro_batch"] * workload["accumulation_steps"]),
        "parameter_count": float(state["parameter_count"]),
        "loss": float(result["loss"]),
    }
    if state["device"].type == "cuda" and torch.cuda.is_available():
        project_metrics["peak_allocated_bytes"] = float(torch.cuda.max_memory_allocated())
    return project_metrics


def models(state):
    return {"model": state["raw_model"]}
