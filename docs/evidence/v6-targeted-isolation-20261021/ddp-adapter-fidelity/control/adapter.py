"""Adapter for the source GPT-2 124M two-rank DDP accumulated training step.

Mirrors inputs/source/gpt2_ddp_training.py: one measured invocation equals one
accumulated DDP training step (no_sync accumulation over the first
accumulation_steps - 1 microbatches, DDP gradient allreduce on the final
microbatch, SGD update, CUDA synchronize).
"""
from __future__ import annotations

from contextlib import contextmanager, nullcontext
from datetime import timedelta
import json
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import GPT2Config, GPT2LMHeadModel

try:
    from kernel_optimizer.project.discovery import phase
except Exception:

    @contextmanager
    def phase(name):
        yield


WEIGHT_SEED = 20261020
TOKEN_SEED_BASE = 31000
SGD_OPTIONS = {"lr": 0.001, "momentum": 0.0, "weight_decay": 0.0, "foreach": False}
BUCKET_CAP_MB = 25
FALLBACK_MODEL_CONFIG = {"n_layer": 12, "n_embd": 768, "n_head": 12, "vocab_size": 50257, "n_positions": 1024}


def _source_case_config(request):
    candidates = [Path(__file__).resolve().parent / "inputs" / "source" / "workloads.json"]
    source_root = request.get("source_root")
    if source_root:
        candidates.append(Path(source_root) / "workloads.json")
    for path in candidates:
        try:
            case = json.loads(path.read_text())["gpt2_ddp_training"]
        except (OSError, TypeError, KeyError, ValueError):
            continue
        if isinstance(case.get("model"), dict):
            return dict(case["model"])
    return dict(FALLBACK_MODEL_CONFIG)


def _configure(device):
    try:
        torch.set_num_threads(4)
    except RuntimeError:
        pass
    try:
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


def _make_model(model_config, device):
    torch.manual_seed(WEIGHT_SEED)
    config = GPT2Config(**model_config, resid_pdrop=0.0, embd_pdrop=0.0, attn_pdrop=0.0,
                        use_cache=False, attn_implementation="eager")
    return GPT2LMHeadModel(config).float().to(device).train()


def _make_inputs(workload, rank, device, vocab_size):
    generator = torch.Generator().manual_seed(TOKEN_SEED_BASE + rank)
    return [torch.randint(0, vocab_size, (workload["micro_batch"], workload["sequence_length"]), generator=generator).to(device)
            for _ in range(workload["accumulation_steps"])]


def _accumulate(model, optimizer, inputs):
    optimizer.zero_grad(set_to_none=True)
    losses = []
    for index, tokens in enumerate(inputs):
        context = model.no_sync() if isinstance(model, DDP) and index + 1 < len(inputs) else nullcontext()
        with context, phase(f"microbatch_{index}"):
            loss = model(input_ids=tokens, labels=tokens, use_cache=False).loss / len(inputs)
            loss.backward()
        losses.append(loss.detach())
    return torch.stack(losses).sum(), losses


def prepare(context, request):
    rank = int(context["rank"])
    world_size = int(context["world_size"])
    device = torch.device(context["device"])
    workload = request["workload"]
    if dist.is_available() and dist.is_initialized() and dist.get_world_size() != workload["world_size"]:
        raise ValueError("This workload must launch the declared number of ranks")
    _configure(device)
    model_config = _source_case_config(request)
    raw_model = _make_model(model_config, device)
    if device.type == "cuda":
        local_index = device.index if device.index is not None else torch.cuda.current_device()
        device_ids = [local_index]
    else:
        device_ids = None
    model = DDP(raw_model, device_ids=device_ids, broadcast_buffers=False,
                bucket_cap_mb=BUCKET_CAP_MB, gradient_as_bucket_view=True)
    optimizer = torch.optim.SGD(model.parameters(), **SGD_OPTIONS)
    inputs = _make_inputs(workload, rank, device, model_config["vocab_size"])
    initial = {name: value.detach().cpu().clone() for name, value in raw_model.state_dict().items()}
    control = None
    if dist.is_available() and dist.is_initialized() and world_size > 1:
        try:
            control = dist.new_group(backend="gloo", timeout=timedelta(seconds=120))
        except (RuntimeError, ValueError):
            control = None
    return {
        "workload": workload,
        "rank": rank,
        "world_size": world_size,
        "device": device,
        "raw_model": raw_model,
        "model": model,
        "optimizer": optimizer,
        "inputs": inputs,
        "initial": initial,
        "control_group": control,
        "mode": "measure",
        "parameter_count": sum(parameter.numel() for parameter in raw_model.parameters()),
    }


def reset(state):
    state["raw_model"].load_state_dict(state["initial"])


def run(state):
    loss, losses = _accumulate(state["model"], state["optimizer"], state["inputs"])
    with phase("optimizer_step"):
        state["optimizer"].step()
    if state["device"].type == "cuda":
        torch.cuda.synchronize()
    return {"loss": loss.detach(), "microbatch_losses": [value.detach() for value in losses]}


def observe(state, result):
    raw_model = state["raw_model"]
    gradients = {}
    parameters = {}
    for name, parameter in raw_model.named_parameters():
        gradients[name] = parameter.grad.detach().cpu().clone() if parameter.grad is not None else None
        parameters[name] = parameter.detach().cpu().clone()
    return {
        "loss": result["loss"].detach().cpu(),
        "microbatch_losses": [value.detach().cpu() for value in result["microbatch_losses"]],
        "gradients": gradients,
        "parameters": parameters,
    }


def metrics(state, result, elapsed_ms):
    workload = state["workload"]
    tokens_per_step = workload["world_size"] * workload["micro_batch"] * workload["sequence_length"] * workload["accumulation_steps"]
    values = {
        "step_ms": float(elapsed_ms),
        "loss": float(result["loss"].detach().cpu()),
        "tokens_per_step": float(tokens_per_step),
        "tokens_per_second": float(tokens_per_step * 1000.0 / elapsed_ms),
        "effective_batch": float(workload["world_size"] * workload["micro_batch"] * workload["accumulation_steps"]),
        "parameter_count": float(state["parameter_count"]),
    }
    values["peak_allocated_bytes"] = float(torch.cuda.max_memory_allocated()) if state["device"].type == "cuda" else 0.0
    return values


def models(state):
    return {"model": state["model"]}


def set_mode(state, mode):
    state["mode"] = mode


def before_run(state):
    control = state.get("control_group")
    if control is not None:
        dist.barrier(group=control)
    elif dist.is_available() and dist.is_initialized() and state["world_size"] > 1:
        dist.barrier()
