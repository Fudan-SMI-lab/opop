"""Adapter for the GPT-2 124M FP32 DDP training project (source: gpt2_ddp_training.py).

Reproduces the frozen workload exactly: seeded synthetic GPT-2 initialization
(torch.manual_seed(20261020)), rank-specific synthetic token IDs
(torch.Generator().manual_seed(31000 + rank)), DDP with no_sync gradient
accumulation over 4 micro-batches, and one SGD(lr=0.001, foreach=False) update
per step. The worker has already initialized the NCCL process group and bound
context['device']; this module never initializes or destroys a process group.
The source's gloo control group is recreated for barriers only, so control
coordination never adds NCCL kernels to communication diagnostics.
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
except Exception:  # optional instrumentation; keep a clean timing path

    @contextmanager
    def _phase(name):
        yield


MODEL_SEED = 20261020
DATA_SEED_BASE = 31000
# Copied verbatim from the source workloads.json "gpt2_ddp_training.model" block.
SOURCE_MODEL_CONFIG = {
    "n_layer": 12,
    "n_embd": 768,
    "n_head": 12,
    "vocab_size": 50257,
    "n_positions": 1024,
}


def _build_model(device):
    torch.manual_seed(MODEL_SEED)
    config = GPT2Config(
        **SOURCE_MODEL_CONFIG,
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        attn_pdrop=0.0,
        use_cache=False,
        attn_implementation="eager",
    )
    return GPT2LMHeadModel(config).float().to(device).train()


def _make_inputs(workload, source_rank, device):
    generator = torch.Generator().manual_seed(DATA_SEED_BASE + source_rank)
    return [
        torch.randint(
            0,
            SOURCE_MODEL_CONFIG["vocab_size"],
            (workload["micro_batch"], workload["sequence_length"]),
            generator=generator,
        ).to(device)
        for _ in range(workload["accumulation_steps"])
    ]


def _accumulate(model, optimizer, inputs):
    """Source accumulate(): zero grads, no_sync micro-steps, final allreduce step."""
    optimizer.zero_grad(set_to_none=True)
    losses = []
    for i, tokens in enumerate(inputs):
        is_final = i + 1 == len(inputs)
        context = model.no_sync() if isinstance(model, DDP) and not is_final else nullcontext()
        with context, _phase("accumulate_micro_step_final_allreduce" if is_final else "accumulate_micro_step_local"):
            loss = model(input_ids=tokens, labels=tokens, use_cache=False).loss / len(inputs)
            loss.backward()
            losses.append(loss.detach())
    return torch.stack(losses).sum(), torch.stack(losses)


def prepare(context, request):
    device = torch.device(context["device"])
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    rank = int(context["rank"])
    world_size = int(context["world_size"])
    workload = request["workload"]
    if world_size != workload["world_size"]:
        raise ValueError("This workload must launch the declared number of ranks")

    # Mirror common.configure(): fixed threads, source seed, strict FP32 numerics.
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

    control = None
    if dist.is_available() and dist.is_initialized():
        control = dist.new_group(backend="gloo", timeout=timedelta(seconds=120))

    raw_model = _build_model(device)
    model = DDP(
        raw_model,
        device_ids=[device.index] if device.type == "cuda" else None,
        broadcast_buffers=False,
        bucket_cap_mb=25,
        gradient_as_bucket_view=True,
    )
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.0, weight_decay=0.0, foreach=False)
    inputs = _make_inputs(workload, rank, device)

    # Immutable starting point for every repetition (CPU-resident snapshots).
    initial = {name: value.detach().cpu().clone() for name, value in raw_model.state_dict().items()}
    initial_optimizer_state = optimizer.state_dict()
    cpu_rng_state = torch.get_rng_state().clone()
    cuda_rng_state = torch.cuda.get_rng_state(device).clone() if device.type == "cuda" else None

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
        "initial_optimizer_state": initial_optimizer_state,
        "cpu_rng_state": cpu_rng_state,
        "cuda_rng_state": cuda_rng_state,
        "control": control,
        "mode": "measure",
    }


def reset(state):
    """Restore weights, optimizer state, gradients and RNG outside timing."""
    state["raw_model"].load_state_dict(state["initial"])
    state["optimizer"].load_state_dict(state["initial_optimizer_state"])
    state["optimizer"].zero_grad(set_to_none=True)
    torch.set_rng_state(state["cpu_rng_state"])
    if state["cuda_rng_state"] is not None:
        torch.cuda.set_rng_state(state["cuda_rng_state"], state["device"])


def run(state):
    model = state["model"]
    optimizer = state["optimizer"]
    inputs = state["inputs"]
    loss, micro_losses = _accumulate(model, optimizer, inputs)
    assert torch.isfinite(loss)
    with _phase("optimizer_step"):
        optimizer.step()
    if state["device"].type == "cuda":
        torch.cuda.synchronize(state["device"])
    return {"loss": loss, "micro_losses": micro_losses}


def observe(state, result):
    raw_model = state["raw_model"]
    gradients = {
        name: parameter.grad.detach().to("cpu", copy=True)
        for name, parameter in raw_model.named_parameters()
    }
    parameters = {
        name: parameter.detach().to("cpu", copy=True)
        for name, parameter in raw_model.named_parameters()
    }
    return {
        "rank": int(state["rank"]),
        "world_size": int(state["world_size"]),
        "loss": result["loss"].detach().to("cpu", copy=True),
        "micro_losses": result["micro_losses"].detach().to("cpu", copy=True),
        "gradients": gradients,
        "parameters": parameters,
    }


def metrics(state, result, elapsed_ms):
    workload = state["workload"]
    tokens_per_step = workload["world_size"] * workload["micro_batch"] * workload["sequence_length"] * workload["accumulation_steps"]
    step_ms = float(elapsed_ms)
    parameter_count = sum(parameter.numel() for parameter in state["raw_model"].parameters())
    values = {
        "step_ms": step_ms,
        "tokens_per_step": float(tokens_per_step),
        "effective_batch": float(workload["world_size"] * workload["micro_batch"] * workload["accumulation_steps"]),
        "tokens_per_second": float(tokens_per_step * 1000.0 / step_ms),
        "parameter_count": float(parameter_count),
        "gradient_logical_bytes_per_rank": float(parameter_count * 4),
        "loss": float(result["loss"].item()),
        "finite_loss": 1.0,
        "step_ms_max_baseline_ratio": 1.05,
    }
    if state["device"].type == "cuda":
        values["peak_allocated_bytes"] = float(torch.cuda.max_memory_allocated())
    return values


def models(state):
    return {"model": state["raw_model"]}


def set_mode(state, mode):
    state["mode"] = mode


def before_run(state):
    """Source-defined pre-step coordination, outside the timer."""
    if state["device"].type == "cuda":
        torch.cuda.synchronize(state["device"])
    if state["control"] is not None:
        dist.barrier(group=state["control"])
    if state["device"].type == "cuda":
        torch.cuda.reset_peak_memory_stats()
