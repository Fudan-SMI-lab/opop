"""Adapter for the frozen gpt2_ddp_training accumulated_step workload.

Source reference: gpt2_ddp_training.py (GPT-2 124M FP32, two-rank NCCL DDP,
deterministic seeded synthetic weights/tokens, SGD with momentum 0, no_sync
gradient accumulation). Semantics preserved exactly: see understanding.json.
"""
from __future__ import annotations

from contextlib import nullcontext
from datetime import timedelta

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import GPT2Config, GPT2LMHeadModel

try:  # optional discovery annotations; worker environment ships kernel_optimizer
    from kernel_optimizer.project.discovery import phase as _phase
except Exception:  # pragma: no cover - keep the adapter importable anywhere
    from contextlib import contextmanager

    @contextmanager
    def _phase(name):  # noqa: D401 - inert fallback
        yield


# Frozen constants from the source workloads.json "gpt2_ddp_training" section.
MODEL_CONFIG = {
    "n_layer": 12,
    "n_embd": 768,
    "n_head": 12,
    "vocab_size": 50257,
    "n_positions": 1024,
}
INIT_SEED = 20261020
INPUT_SEED_BASE = 31000
SGD_PARAMS = {"lr": 0.001, "momentum": 0.0, "weight_decay": 0.0}
CONTROL_TIMEOUT = timedelta(seconds=600)


def _configure(device: torch.device) -> None:
    """Mirror common.configure(rank): thread counts, seeds, TF32-off policy."""
    torch.set_num_threads(4)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass  # already initialized earlier in this process
    torch.manual_seed(INIT_SEED)
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision("highest")


def _make_model(device: torch.device) -> GPT2LMHeadModel:
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


def _make_inputs(workload: dict, source_rank: int, device: torch.device) -> list:
    generator = torch.Generator().manual_seed(INPUT_SEED_BASE + source_rank)
    return [
        torch.randint(
            0,
            MODEL_CONFIG["vocab_size"],
            (workload["micro_batch"], workload["sequence_length"]),
            generator=generator,
        ).to(device)
        for _ in range(workload["accumulation_steps"])
    ]


def prepare(context: dict, request: dict) -> dict:
    rank = int(context["rank"])
    world_size = int(context["world_size"])
    device = torch.device(context["device"])

    _configure(device)

    workload = request["workload"]
    if world_size != workload["world_size"]:
        raise ValueError("This workload must launch the declared number of ranks")

    raw_model = _make_model(device)
    device_ids = [device.index if device.index is not None else torch.cuda.current_device()]
    model = DDP(
        raw_model,
        device_ids=device_ids,
        broadcast_buffers=False,
        bucket_cap_mb=25,
        gradient_as_bucket_view=True,
    )
    optimizer = torch.optim.SGD(model.parameters(), foreach=False, **SGD_PARAMS)
    inputs = _make_inputs(workload, rank, device)
    initial = {name: value.detach().cpu().clone() for name, value in raw_model.state_dict().items()}

    control = None
    if dist.is_available() and dist.is_initialized() and world_size > 1:
        control = dist.new_group(backend="gloo", timeout=CONTROL_TIMEOUT)

    return {
        "raw_model": raw_model,
        "model": model,
        "optimizer": optimizer,
        "inputs": inputs,
        "initial": initial,
        "control": control,
        "rank": rank,
        "world_size": world_size,
        "device": device,
        "workload": workload,
    }


def reset(state: dict) -> None:
    # Source reset(): reload the pre-step weights. Gradients are intentionally
    # NOT cleared here; the source clears them with optimizer.zero_grad inside
    # the timed accumulate(). SGD(momentum=0) holds no optimizer state.
    state["raw_model"].load_state_dict(state["initial"])


def before_run(state: dict) -> None:
    control = state.get("control")
    if control is not None:
        dist.barrier(group=control)


def run(state: dict) -> dict:
    model = state["model"]
    optimizer = state["optimizer"]
    inputs = state["inputs"]

    with _phase("step"):
        with _phase("accumulate"):
            optimizer.zero_grad(set_to_none=True)
            losses = []
            for i, tokens in enumerate(inputs):
                context = (
                    model.no_sync()
                    if isinstance(model, DDP) and i + 1 < len(inputs)
                    else nullcontext()
                )
                with context:
                    with _phase("micro_step"):
                        loss = model(input_ids=tokens, labels=tokens, use_cache=False).loss / len(inputs)
                        loss.backward()
                losses.append(loss.detach())
            total = torch.stack(losses).sum()
        with _phase("optimizer_step"):
            optimizer.step()
        torch.cuda.synchronize()

    assert torch.isfinite(total)
    return {"loss": total, "micro_losses": losses}


def observe(state: dict, result: dict) -> dict:
    raw_model = state["raw_model"]
    gradients = {}
    parameters = {}
    for name, parameter in raw_model.named_parameters():
        grad = parameter.grad
        gradients[name] = (
            grad.detach().cpu().clone() if grad is not None else torch.zeros_like(parameter).cpu()
        )
        parameters[name] = parameter.detach().cpu().clone()
    return {
        "loss": result["loss"].detach().cpu(),
        "gradients": gradients,
        "parameters": parameters,
    }


def metrics(state: dict, result: dict, elapsed_ms: float) -> dict:
    workload = state["workload"]
    tokens_per_step = (
        workload["world_size"]
        * workload["micro_batch"]
        * workload["sequence_length"]
        * workload["accumulation_steps"]
    )
    return {
        "step_ms": float(elapsed_ms),
        "loss": float(result["loss"].item()),
        "tokens_per_step": float(tokens_per_step),
        "effective_batch": float(
            workload["world_size"] * workload["micro_batch"] * workload["accumulation_steps"]
        ),
        "tokens_per_second": tokens_per_step * 1000.0 / float(elapsed_ms),
        "parameter_count": float(sum(p.numel() for p in state["raw_model"].parameters())),
        "micro_batch": float(workload["micro_batch"]),
        "sequence_length": float(workload["sequence_length"]),
        "accumulation_steps": float(workload["accumulation_steps"]),
        "world_size": float(workload["world_size"]),
    }


def models(state: dict) -> dict:
    return {"model": state["raw_model"]}
