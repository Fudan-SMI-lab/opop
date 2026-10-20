"""Adapter for the gpt2_ddp_training project: one accumulated DDP training step.

Mirrors inputs/source/gpt2_ddp_training.py: a seeded synthetic GPT-2 (FP32,
eager attention, dropout 0) trained with two-rank NCCL DDP using no_sync
gradient accumulation over 4 micro-batches, one SGD update, and
torch.cuda.synchronize() as the completion boundary. The worker supplies the
already-initialized NCCL process group, rank, world_size and bound device.
"""
from __future__ import annotations

from contextlib import contextmanager, nullcontext
from datetime import timedelta

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import GPT2Config, GPT2LMHeadModel

try:
    from kernel_optimizer.project.discovery import phase as _phase
except Exception:  # optional labeling only; never changes execution semantics
    @contextmanager
    def _phase(name):
        yield


MODEL_SEED = 20261020
INPUT_SEED_BASE = 31000
MODEL_CONFIG = {"n_layer": 12, "n_embd": 768, "n_head": 12, "vocab_size": 50257, "n_positions": 1024}
SGD_KWARGS = {"lr": 0.001, "momentum": 0.0, "weight_decay": 0.0, "foreach": False}
BUCKET_CAP_MB = 25


def _configure(device):
    torch.set_num_threads(4)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    torch.manual_seed(MODEL_SEED)
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision("highest")


def _make_model(device):
    torch.manual_seed(MODEL_SEED)
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
            with _phase("microbatch_forward"):
                loss = model(input_ids=tokens, labels=tokens, use_cache=False).loss / len(inputs)
            with _phase("microbatch_backward"):
                loss.backward()
            losses.append(loss.detach())
    return torch.stack(losses).sum(), losses


def prepare(context, request):
    device = torch.device(context["device"])
    rank = int(context["rank"])
    world_size = int(context["world_size"])
    workload = request["workload"]
    if dist.is_available() and dist.is_initialized() and dist.get_world_size() != workload["world_size"]:
        raise ValueError("This workload must launch the declared number of ranks")
    _configure(device)
    raw_model = _make_model(device)
    if device.type == "cuda":
        model = DDP(raw_model, device_ids=[device.index if device.index is not None else rank],
                    broadcast_buffers=False, bucket_cap_mb=BUCKET_CAP_MB, gradient_as_bucket_view=True)
    else:
        model = DDP(raw_model, broadcast_buffers=False, bucket_cap_mb=BUCKET_CAP_MB, gradient_as_bucket_view=True)
    optimizer = torch.optim.SGD(model.parameters(), **SGD_KWARGS)
    inputs = _make_inputs(workload, rank, device)
    initial = {name: value.detach().cpu().clone() for name, value in raw_model.state_dict().items()}
    control_group = None
    if dist.is_available() and dist.is_initialized():
        try:
            control_group = dist.new_group(backend="gloo", timeout=timedelta(seconds=120))
        except Exception:
            control_group = None
    return {
        "raw_model": raw_model,
        "model": model,
        "optimizer": optimizer,
        "inputs": inputs,
        "initial": initial,
        "device": device,
        "rank": rank,
        "world_size": world_size,
        "workload": workload,
        "control_group": control_group,
        "effective_batch": workload["world_size"] * workload["micro_batch"] * workload["accumulation_steps"],
        "tokens_per_step": workload["world_size"] * workload["micro_batch"] * workload["sequence_length"] * workload["accumulation_steps"],
        "parameter_count": sum(p.numel() for p in raw_model.parameters()),
        "micro_losses": [],
        "mode": "measure",
    }


def reset(state):
    state["raw_model"].load_state_dict(state["initial"])
    torch.manual_seed(MODEL_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(MODEL_SEED)
    state["micro_losses"] = []
    # optimizer.zero_grad deliberately stays inside run(): the source clears
    # gradients within the timed accumulate() (gpt2_ddp_training.py:37).


def run(state):
    model = state["model"]
    optimizer = state["optimizer"]
    with _phase("accumulated_step"):
        loss, micro_losses = _accumulate(model, optimizer, state["inputs"])
        state["micro_losses"] = micro_losses
        with _phase("optimizer_step"):
            optimizer.step()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    return {"loss": loss.detach()}


def observe(state, result):
    raw_model = state["raw_model"]
    gradients = {}
    parameters = {}
    for name, parameter in raw_model.named_parameters():
        grad = parameter.grad
        gradients[name] = grad.detach().cpu().clone() if grad is not None else torch.zeros_like(parameter, device="cpu")
        parameters[name] = parameter.detach().cpu().clone()
    return {
        "loss": result["loss"].detach().cpu().clone(),
        "micro_losses": tuple(value.detach().cpu().clone() for value in state["micro_losses"]),
        "gradients": gradients,
        "parameters": parameters,
    }


def metrics(state, result, elapsed_ms):
    step_ms = float(elapsed_ms)
    tokens = int(state["tokens_per_step"])
    workload = state["workload"]
    values = {
        "step_ms": step_ms,
        "tokens_per_second": (tokens * 1000.0 / step_ms) if step_ms > 0.0 else 0.0,
        "loss": float(result["loss"].detach().item()),
        "parameter_count": float(state["parameter_count"]),
        "effective_batch": float(state["effective_batch"]),
        "tokens_per_step": float(tokens),
        "world_size": float(state["world_size"]),
        "micro_batch": float(workload["micro_batch"]),
        "sequence_length": float(workload["sequence_length"]),
        "accumulation_steps": float(workload["accumulation_steps"]),
        "rank": float(state["rank"]),
    }
    if torch.cuda.is_available():
        values["peak_allocated_bytes"] = float(torch.cuda.max_memory_allocated())
    return values


def models(state):
    return {"model": state["raw_model"]}


def set_mode(state, mode):
    state["mode"] = str(mode)


def before_run(state):
    if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
        dist.barrier(group=state.get("control_group"))
