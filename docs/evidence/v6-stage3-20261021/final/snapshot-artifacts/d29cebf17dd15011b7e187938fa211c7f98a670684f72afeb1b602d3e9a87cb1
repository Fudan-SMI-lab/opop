"""Adapter for the frozen GPT-2 two-rank DDP accumulated training step.

Mirrors inputs/source/gpt2_ddp_training.py for workload 'accumulated_step':
seeded synthetic GPT-2 124M in FP32 with eager attention and no dropout, DDP
with no_sync accumulation for the first accumulation_steps-1 micro-batches, a
final synced micro-batch whose backward performs the NCCL gradient all-reduce,
and an SGD update. The worker owns CUDA binding and the default process group;
this module never initializes or destroys them.
"""
from __future__ import annotations

from contextlib import nullcontext
from datetime import timedelta

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import GPT2Config, GPT2LMHeadModel

try:
    from kernel_optimizer.project.discovery import phase as discovery_phase
except Exception:

    def discovery_phase(name):
        return nullcontext()


MODEL_SEED = 20261020
INPUT_SEED_BASE = 31000
MODEL_CONFIG = {"n_layer": 12, "n_embd": 768, "n_head": 12, "vocab_size": 50257, "n_positions": 1024}
SGD_SETTINGS = {"lr": 0.001, "momentum": 0.0, "weight_decay": 0.0, "foreach": False}


def _configure(device):
    torch.set_num_threads(4)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    torch.manual_seed(MODEL_SEED)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision("highest")


def _build_model(device):
    torch.manual_seed(MODEL_SEED)
    config = GPT2Config(**MODEL_CONFIG, resid_pdrop=0.0, embd_pdrop=0.0, attn_pdrop=0.0, use_cache=False, attn_implementation="eager")
    return GPT2LMHeadModel(config).float().to(device).train()


def _build_inputs(workload, source_rank, device):
    generator = torch.Generator().manual_seed(INPUT_SEED_BASE + source_rank)
    return [torch.randint(0, MODEL_CONFIG["vocab_size"], (workload["micro_batch"], workload["sequence_length"]), generator=generator).to(device)
            for _ in range(workload["accumulation_steps"])]


def prepare(context, request):
    device = torch.device(context["device"])
    rank = int(context["rank"])
    world_size = int(context["world_size"])
    workload = request["workload"]
    distributed = dist.is_available() and dist.is_initialized()
    if distributed and dist.get_world_size() != workload["world_size"]:
        raise ValueError("This workload must launch the declared number of ranks")
    _configure(device)
    raw_model = _build_model(device)
    if distributed:
        device_index = device.index if device.index is not None else torch.cuda.current_device()
        model = DDP(raw_model, device_ids=[device_index], broadcast_buffers=False, bucket_cap_mb=25, gradient_as_bucket_view=True)
        control = dist.new_group(backend="gloo", timeout=timedelta(seconds=120))
    else:
        model = raw_model
        control = None
    optimizer = torch.optim.SGD(model.parameters(), **SGD_SETTINGS)
    inputs = _build_inputs(workload, rank, device)
    initial = {name: value.detach().cpu().clone() for name, value in raw_model.state_dict().items()}
    return {"raw_model": raw_model, "model": model, "optimizer": optimizer, "inputs": inputs,
            "initial": initial, "device": device, "rank": rank, "world_size": world_size,
            "workload": workload, "control": control}


def reset(state):
    state["raw_model"].load_state_dict(state["initial"])


def before_run(state):
    if state["control"] is not None:
        if state["device"].type == "cuda":
            torch.cuda.synchronize()
        dist.barrier(group=state["control"])


def run(state):
    model = state["model"]
    optimizer = state["optimizer"]
    inputs = state["inputs"]
    optimizer.zero_grad(set_to_none=True)
    losses = []
    with discovery_phase("accumulate_microbatches"):
        for i, tokens in enumerate(inputs):
            context = model.no_sync() if isinstance(model, DDP) and i + 1 < len(inputs) else nullcontext()
            with context:
                loss = model(input_ids=tokens, labels=tokens, use_cache=False).loss / len(inputs)
                loss.backward()
                losses.append(loss.detach())
        accumulated_loss = torch.stack(losses).sum()
    with discovery_phase("optimizer_step"):
        optimizer.step()
    if state["device"].type == "cuda":
        torch.cuda.synchronize()
    return {"loss": accumulated_loss, "micro_losses": torch.stack(losses)}


def observe(state, result):
    gradients = {}
    parameters = {}
    for name, parameter in state["raw_model"].named_parameters():
        gradients[name] = parameter.grad.detach().cpu()
        parameters[name] = parameter.detach().cpu()
    return {"loss": result["loss"].detach().cpu(),
            "micro_losses": result["micro_losses"].detach().cpu(),
            "gradients": gradients,
            "parameters": parameters}


def metrics(state, result, elapsed_ms):
    workload = state["workload"]
    token_count = workload["world_size"] * workload["micro_batch"] * workload["sequence_length"] * workload["accumulation_steps"]
    elapsed = float(elapsed_ms)
    values = {"step_ms": elapsed, "tokens_per_second": token_count * 1000.0 / elapsed}
    if state["device"].type == "cuda":
        values["peak_allocated_bytes"] = float(torch.cuda.max_memory_allocated())
    return values


def models(state):
    return {"gpt2": state["raw_model"]}
