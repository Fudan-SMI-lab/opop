"""Adapter for the frozen gpt2_ddp_training 'accumulated_step' workload.

Mirrors inputs/source/gpt2_ddp_training.py exactly: seeded FP32 GPT-2 LM head
model, DDP over NCCL with no_sync gradient accumulation, SGD update, and the
source's timed step boundary (accumulate + optimizer.step + synchronize).
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
except Exception:  # pragma: no cover - fallback when discovery is unavailable
    from contextlib import contextmanager

    @contextmanager
    def _phase(name):
        yield


MODEL_CONFIG = {
    "n_layer": 12,
    "n_embd": 768,
    "n_head": 12,
    "vocab_size": 50257,
    "n_positions": 1024,
}

WEIGHT_SEED = 20261020
INPUT_SEED_BASE = 31000


def _configure(device):
    try:
        torch.set_num_threads(4)
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    torch.manual_seed(WEIGHT_SEED)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision("highest")


def prepare(context, request):
    device = torch.device(context["device"])
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    rank = int(context.get("rank", 0))
    world_size = int(context.get("world_size", 1))
    _configure(device)
    control = None
    if dist.is_available() and dist.is_initialized():
        control = dist.new_group(backend="gloo", timeout=timedelta(seconds=120))
    workload = request["workload"]
    torch.manual_seed(WEIGHT_SEED)
    model_config = GPT2Config(
        **MODEL_CONFIG,
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        attn_pdrop=0.0,
        use_cache=False,
        attn_implementation="eager",
    )
    raw_model = GPT2LMHeadModel(model_config).float().to(device).train()
    model = DDP(
        raw_model,
        device_ids=[device.index],
        broadcast_buffers=False,
        bucket_cap_mb=25,
        gradient_as_bucket_view=True,
    )
    optimizer = torch.optim.SGD(
        model.parameters(), lr=0.001, momentum=0.0, weight_decay=0.0, foreach=False
    )
    generator = torch.Generator().manual_seed(INPUT_SEED_BASE + rank)
    micro_batch = int(workload["micro_batch"])
    sequence_length = int(workload["sequence_length"])
    accumulation_steps = int(workload["accumulation_steps"])
    inputs = [
        torch.randint(
            0,
            MODEL_CONFIG["vocab_size"],
            (micro_batch, sequence_length),
            generator=generator,
        ).to(device)
        for _ in range(accumulation_steps)
    ]
    initial = {
        name: value.detach().cpu().clone()
        for name, value in raw_model.state_dict().items()
    }
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
        "control": control,
        "rng_cpu": torch.get_rng_state(),
        "rng_cuda": torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
        "mode": "measure",
    }


def reset(state):
    state["raw_model"].load_state_dict(state["initial"])
    torch.set_rng_state(state["rng_cpu"])
    if state["rng_cuda"] is not None:
        torch.cuda.set_rng_state(state["rng_cuda"], state["device"])


def before_run(state):
    if state["control"] is not None:
        if state["device"].type == "cuda":
            torch.cuda.synchronize(state["device"])
        dist.barrier(group=state["control"])
    if state["device"].type == "cuda":
        torch.cuda.reset_peak_memory_stats()


def set_mode(state, mode):
    state["mode"] = str(mode)


def run(state):
    model = state["model"]
    optimizer = state["optimizer"]
    inputs = state["inputs"]
    optimizer.zero_grad(set_to_none=True)
    losses = []
    for i, tokens in enumerate(inputs):
        context = (
            model.no_sync()
            if isinstance(model, DDP) and i + 1 < len(inputs)
            else nullcontext()
        )
        with context, _phase("accumulate_micro_step"):
            loss = model(input_ids=tokens, labels=tokens, use_cache=False).loss / len(inputs)
            loss.backward()
            losses.append(loss.detach())
    with _phase("optimizer_step"):
        optimizer.step()
    if state["device"].type == "cuda":
        torch.cuda.synchronize(state["device"])
    return {"loss": torch.stack(losses).sum(), "micro_losses": torch.stack(losses)}


def observe(state, result):
    gradients = {}
    parameters = {}
    for name, parameter in state["raw_model"].named_parameters():
        grad = parameter.grad
        gradients[name] = grad.detach().to("cpu", copy=True) if grad is not None else None
        parameters[name] = parameter.detach().to("cpu", copy=True)
    return {
        "loss": result["loss"].detach().to("cpu", copy=True),
        "gradients": gradients,
        "parameters": parameters,
    }


def metrics(state, result, elapsed_ms):
    workload = state["workload"]
    token_count = (
        int(workload["world_size"])
        * int(workload["micro_batch"])
        * int(workload["sequence_length"])
        * int(workload["accumulation_steps"])
    )
    values = {
        "step_ms": float(elapsed_ms),
        "tokens_per_second": float(token_count * 1000.0 / elapsed_ms),
    }
    if state["device"].type == "cuda":
        values["peak_allocated_bytes"] = float(torch.cuda.max_memory_allocated())
    return values


def models(state):
    return {"model": state["raw_model"]}
