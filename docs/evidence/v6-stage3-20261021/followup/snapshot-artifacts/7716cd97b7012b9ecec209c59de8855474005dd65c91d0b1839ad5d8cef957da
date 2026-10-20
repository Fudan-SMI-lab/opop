"""Adapter for the frozen gpt2_ddp_training "accumulated_step" workload.

Mirrors inputs/source/gpt2_ddp_training.py and inputs/source/common.py:
seeded synthetic GPT-2 (12L/768d, FP32, eager attention, dropout 0), two-rank
DDP over NCCL with no_sync gradient accumulation, SGD update, and the source's
gloo control barrier. The diagnostic objective
(diagnostic.max_rank_nccl_kernel_sum_ms) is collected by the framework's
torch_communication collector in separate diagnostic runs; metrics() reports
only the source's clean project metrics.
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
except Exception:  # discovery is optional; semantics must not depend on it
    from contextlib import contextmanager

    @contextmanager
    def _phase(name):
        yield


# Frozen source constants (inputs/source/workloads.json -> "gpt2_ddp_training").
MODEL_CONFIG = {"n_layer": 12, "n_embd": 768, "n_head": 12, "vocab_size": 50257, "n_positions": 1024}
SEED = 20261020
INPUT_SEED_BASE = 31000
LR = 0.001


def _device_index(device):
    return device.index if device.index is not None else torch.cuda.current_device()


def prepare(context, request):
    workload = request["workload"]
    rank = int(context["rank"])
    world_size = int(context["world_size"])
    device = torch.device(context["device"])
    if world_size != int(workload["world_size"]):
        raise ValueError("This workload must launch the declared number of ranks")

    # Numerical policies from common.configure().
    torch.set_num_threads(4)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    torch.manual_seed(SEED)
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision("highest")

    # Source's out-of-band control group, used only for pre-step barriers.
    control = None
    if dist.is_available() and dist.is_initialized():
        control = dist.new_group(backend="gloo", timeout=timedelta(seconds=120))

    torch.manual_seed(SEED)
    model_config = GPT2Config(**MODEL_CONFIG, resid_pdrop=0.0, embd_pdrop=0.0, attn_pdrop=0.0, use_cache=False, attn_implementation="eager")
    raw_model = GPT2LMHeadModel(model_config).float().to(device).train()
    model = DDP(raw_model, device_ids=[_device_index(device)], broadcast_buffers=False, bucket_cap_mb=25, gradient_as_bucket_view=True)
    optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.0, weight_decay=0.0, foreach=False)

    generator = torch.Generator().manual_seed(INPUT_SEED_BASE + rank)
    inputs = [torch.randint(0, MODEL_CONFIG["vocab_size"], (workload["micro_batch"], workload["sequence_length"]), generator=generator).to(device)
              for _ in range(workload["accumulation_steps"])]

    # Initial CPU snapshot for exact repetition of the workload.
    initial = {name: value.detach().cpu().clone() for name, value in raw_model.state_dict().items()}

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
        "control": control,
        "mode": "measure",
    }


def set_mode(state, mode):
    state["mode"] = mode


def before_run(state):
    # Source-defined pre-step coordination, outside the measured boundary.
    torch.cuda.synchronize(state["device"])
    control = state.get("control")
    if control is not None:
        dist.barrier(group=control)
    torch.cuda.reset_peak_memory_stats(state["device"])


def reset(state):
    state["raw_model"].load_state_dict(state["initial"])


def run(state):
    model = state["model"]
    optimizer = state["optimizer"]
    inputs = state["inputs"]
    optimizer.zero_grad(set_to_none=True)
    losses = []
    with _phase("accumulate"):
        for i, tokens in enumerate(inputs):
            context = model.no_sync() if isinstance(model, DDP) and i + 1 < len(inputs) else nullcontext()
            with context:
                loss = model(input_ids=tokens, labels=tokens, use_cache=False).loss / len(inputs)
                loss.backward()
                losses.append(loss.detach())
        total_loss = torch.stack(losses).sum()
    with _phase("optimizer_step"):
        optimizer.step()
    torch.cuda.synchronize(state["device"])
    return {"loss": total_loss}


def observe(state, result):
    raw_model = state["raw_model"]
    gradients = {}
    parameters = {}
    for name, parameter in raw_model.named_parameters():
        gradient = parameter.grad
        if gradient is None:
            gradient = torch.zeros_like(parameter)
        gradients[name] = gradient.detach().cpu().clone()
        parameters[name] = parameter.detach().cpu().clone()
    loss = result["loss"].detach().cpu().clone()
    return {"loss": loss, "gradients": gradients, "parameters": parameters}


def metrics(state, result, elapsed_ms):
    workload = state["workload"]
    elapsed_ms = float(elapsed_ms)
    tokens_per_step = workload["world_size"] * workload["micro_batch"] * workload["sequence_length"] * workload["accumulation_steps"]
    return {
        "step_ms": elapsed_ms,
        "tokens_per_second": tokens_per_step * 1000.0 / elapsed_ms,
        "peak_allocated_bytes": float(torch.cuda.max_memory_allocated(state["device"])),
        "effective_batch": float(workload["world_size"] * workload["micro_batch"] * workload["accumulation_steps"]),
        "tokens_per_step": float(tokens_per_step),
        "parameter_count": float(sum(parameter.numel() for parameter in state["raw_model"].parameters())),
    }


def models(state):
    return {"model": state["raw_model"]}
