from __future__ import annotations

from contextlib import nullcontext
from datetime import timedelta

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import GPT2Config, GPT2LMHeadModel

try:
    from kernel_optimizer.project.discovery import phase as _discovery_phase
except Exception:
    _discovery_phase = None

MODEL_CONFIG = {"n_layer": 12, "n_embd": 768, "n_head": 12, "vocab_size": 50257, "n_positions": 1024}
INIT_SEED = 20261020
INPUT_SEED_BASE = 31000
LEARNING_RATE = 0.001


def _phase(name):
    if _discovery_phase is not None:
        return _discovery_phase(name)
    return nullcontext()


def _configure(device):
    torch.set_num_threads(4)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    torch.manual_seed(INIT_SEED)
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision("highest")


def _make_model(device):
    torch.manual_seed(INIT_SEED)
    config = GPT2Config(
        **MODEL_CONFIG,
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        attn_pdrop=0.0,
        use_cache=False,
        attn_implementation="eager",
    )
    return GPT2LMHeadModel(config).float().to(device).train()


def _make_inputs(rank, workload, device):
    generator = torch.Generator().manual_seed(INPUT_SEED_BASE + rank)
    return [
        torch.randint(
            0,
            MODEL_CONFIG["vocab_size"],
            (workload["micro_batch"], workload["sequence_length"]),
            generator=generator,
        ).to(device)
        for _ in range(workload["accumulation_steps"])
    ]


def prepare(context, request):
    device = torch.device(context["device"])
    rank = int(context["rank"])
    world_size = int(context["world_size"])
    _configure(device)
    workload = request["workload"]
    distributed = dist.is_available() and dist.is_initialized()
    control = None
    if distributed and world_size > 1:
        control = dist.new_group(backend="gloo", timeout=timedelta(seconds=120))
    raw_model = _make_model(device)
    if distributed:
        device_index = device.index if device.index is not None else torch.cuda.current_device()
        model = DDP(
            raw_model,
            device_ids=[device_index],
            broadcast_buffers=False,
            bucket_cap_mb=25,
            gradient_as_bucket_view=True,
        )
    else:
        model = raw_model
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=LEARNING_RATE,
        momentum=0.0,
        weight_decay=0.0,
        foreach=False,
    )
    inputs = _make_inputs(rank, workload, device)
    initial = {name: value.detach().cpu().clone() for name, value in raw_model.state_dict().items()}
    rng_cpu = torch.get_rng_state()
    try:
        rng_cuda = torch.cuda.get_rng_state(device)
    except Exception:
        rng_cuda = None
    return {
        "device": device,
        "rank": rank,
        "world_size": world_size,
        "workload": workload,
        "model": model,
        "raw_model": raw_model,
        "optimizer": optimizer,
        "inputs": inputs,
        "initial": initial,
        "control": control,
        "rng_cpu": rng_cpu,
        "rng_cuda": rng_cuda,
        "parameter_count": sum(p.numel() for p in raw_model.parameters()),
        "tokens_per_step": world_size
        * workload["micro_batch"]
        * workload["sequence_length"]
        * workload["accumulation_steps"],
        "effective_batch": world_size * workload["micro_batch"] * workload["accumulation_steps"],
        "mode": "measure",
    }


def reset(state):
    state["raw_model"].load_state_dict(state["initial"])
    state["optimizer"].state.clear()
    torch.set_rng_state(state["rng_cpu"])
    if state["rng_cuda"] is not None:
        try:
            torch.cuda.set_rng_state(state["rng_cuda"], state["device"])
        except Exception:
            pass


def _accumulate(state):
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
        with context:
            loss = model(input_ids=tokens, labels=tokens, use_cache=False).loss / len(inputs)
            loss.backward()
            losses.append(loss.detach())
    return losses


def run(state):
    with _phase("forward_backward_accumulation"):
        losses = _accumulate(state)
    total = torch.stack(losses).sum()
    with _phase("optimizer_step"):
        state["optimizer"].step()
    torch.cuda.synchronize(state["device"])
    return {"loss": total, "micro_losses": losses}


def observe(state, result):
    raw_model = state["raw_model"]
    gradients = {}
    parameters = {}
    for name, parameter in raw_model.named_parameters():
        gradients[name] = parameter.grad.detach().cpu().clone()
        parameters[name] = parameter.detach().cpu().clone()
    loss = None
    if result is not None and "loss" in result:
        loss = result["loss"].detach().cpu().clone()
    return {"loss": loss, "gradients": gradients, "parameters": parameters}


def metrics(state, result, elapsed_ms):
    elapsed = float(elapsed_ms)
    values = {
        "step_ms": elapsed,
        "tokens_per_second": state["tokens_per_step"] * 1000.0 / elapsed,
        "effective_batch": state["effective_batch"],
        "tokens_per_step": state["tokens_per_step"],
        "parameter_count": state["parameter_count"],
        "world_size": state["world_size"],
        "micro_batch": state["workload"]["micro_batch"],
        "sequence_length": state["workload"]["sequence_length"],
        "accumulation_steps": state["workload"]["accumulation_steps"],
    }
    try:
        values["peak_allocated_bytes"] = float(torch.cuda.max_memory_allocated())
    except Exception:
        pass
    return values


def models(state):
    return {"model": state["raw_model"]}


def before_run(state):
    control = state.get("control")
    if control is not None:
        torch.cuda.synchronize(state["device"])
        dist.barrier(group=control)


def set_mode(state, mode):
    state["mode"] = mode
