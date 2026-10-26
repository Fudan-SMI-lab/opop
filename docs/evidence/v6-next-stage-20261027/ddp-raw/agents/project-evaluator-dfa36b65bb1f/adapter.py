"""Baseline adapter for the GPT-2 DDP accumulated-training-step project.

Reproduces the semantics of gpt2_ddp_training.py (source root in
inputs/source) for one frozen workload:

* configure(): FP32, TF32 disabled, float32 matmul precision "highest",
  deterministic seeds (model init torch.manual_seed(20261020), token
  fixtures Generator seed 31000 + rank).
* One measured invocation = step() at gpt2_ddp_training.py:102-105:
  accumulate() (optimizer.zero_grad(set_to_none=True) followed by all
  accumulation_steps microbatches of forward+backward, DDP no_sync() on
  every microbatch except the last so gradients are all-reduced once),
  optimizer.step(), torch.cuda.synchronize().
* reset() = restore the initial seeded weights from a CPU snapshot
  (gpt2_ddp_training.py:99-100). Gradients are cleared inside the timed
  invocation by accumulate(), exactly as in the source.

The worker has already bound CUDA and initialized the NCCL process group;
this module must not create another default process group.
"""

from __future__ import annotations

import copy
import json
from contextlib import nullcontext
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import GPT2Config, GPT2LMHeadModel

try:  # optional phase annotation, best effort
    from kernel_optimizer.project.discovery import phase as _phase
except Exception:  # pragma: no cover - annotation is optional
    from contextlib import contextmanager

    @contextmanager
    def _phase(name):  # noqa: D401 - silent fallback
        yield


MODEL_SEED = 20261020
INPUT_SEED_BASE = 31000
FALLBACK_MODEL_CONFIG = {
    "n_layer": 12,
    "n_embd": 768,
    "n_head": 12,
    "vocab_size": 50257,
    "n_positions": 1024,
}


def _load_model_config(request):
    """Model architecture comes from the source workloads.json, not the workload."""
    candidates = [Path(__file__).resolve().parent / "inputs" / "source" / "workloads.json"]
    source_root = request.get("source_root")
    if source_root:
        candidates.append(Path(source_root) / "workloads.json")
    for candidate in candidates:
        try:
            if candidate.exists():
                config = json.loads(candidate.read_text())
                model = config["gpt2_ddp_training"]["model"]
                if model:
                    return dict(model)
        except Exception:
            continue
    return dict(FALLBACK_MODEL_CONFIG)


def _configure(device):
    """Numerical policies and seed from common.configure()."""
    torch.set_num_threads(4)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass  # already initialized in this process
    torch.manual_seed(MODEL_SEED)
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision("highest")


def _make_model(model_config, device):
    torch.manual_seed(MODEL_SEED)
    config = GPT2Config(
        **model_config,
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        attn_pdrop=0.0,
        use_cache=False,
        attn_implementation="eager",
    )
    return GPT2LMHeadModel(config).float().to(device).train()


def _make_inputs(model_config, workload, source_rank, device):
    generator = torch.Generator().manual_seed(INPUT_SEED_BASE + source_rank)
    return [
        torch.randint(
            0,
            model_config["vocab_size"],
            (workload["micro_batch"], workload["sequence_length"]),
            generator=generator,
        ).to(device)
        for _ in range(workload["accumulation_steps"])
    ]


def _accumulate(model, optimizer, inputs):
    """gpt2_ddp_training.accumulate: gradient clearing lives inside the step."""
    optimizer.zero_grad(set_to_none=True)
    losses = []
    for index, tokens in enumerate(inputs):
        context = model.no_sync() if isinstance(model, DDP) and index + 1 < len(inputs) else nullcontext()
        with context:
            with _phase("microbatch_forward_backward"):
                loss = model(input_ids=tokens, labels=tokens, use_cache=False).loss / len(inputs)
                loss.backward()
            losses.append(loss.detach())
    return torch.stack(losses).sum()


def prepare(context, request):
    device = torch.device(context["device"])
    rank = int(context["rank"])
    world_size = int(context["world_size"])

    workload = request["workload"]
    if world_size != workload["world_size"]:
        raise ValueError("This workload must launch the declared number of ranks")

    model_config = _load_model_config(request)
    _configure(device)

    raw_model = _make_model(model_config, device)
    local_index = device.index if device.index is not None else torch.cuda.current_device()
    model = DDP(
        raw_model,
        device_ids=[local_index],
        broadcast_buffers=False,
        bucket_cap_mb=25,
        gradient_as_bucket_view=True,
    )
    optimizer = torch.optim.SGD(
        model.parameters(), lr=0.001, momentum=0.0, weight_decay=0.0, foreach=False
    )
    inputs = _make_inputs(model_config, workload, rank, device)
    initial = {name: value.detach().cpu().clone() for name, value in raw_model.state_dict().items()}

    control = None
    if dist.is_available() and dist.is_initialized() and world_size > 1:
        control = dist.new_group(backend="gloo", timeout=timedelta(seconds=120))

    state = {
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
        "parameter_count": sum(p.numel() for p in raw_model.parameters()),
        "optimizer_snapshot": copy.deepcopy(optimizer.state_dict()),
        "rng_cpu": torch.get_rng_state(),
        "rng_cuda": torch.cuda.get_rng_state(device),
        "mode": "measure",
    }
    return state


def reset(state):
    """Restore the source's starting point (weights from the CPU snapshot)."""
    torch.set_rng_state(state["rng_cpu"])
    torch.cuda.set_rng_state(state["rng_cuda"], state["device"])
    state["raw_model"].load_state_dict(state["initial"])
    try:
        state["optimizer"].load_state_dict(copy.deepcopy(state["optimizer_snapshot"]))
    except Exception:
        pass  # SGD momentum=0 keeps no persistent state


def before_run(state):
    """Source-defined coordination: CUDA sync + control barrier outside timing."""
    torch.cuda.synchronize(state["device"])
    control = state.get("control")
    if control is not None:
        dist.barrier(group=control)


def run(state):
    """One full accumulated training step (gpt2_ddp_training.step)."""
    model = state["model"]
    optimizer = state["optimizer"]
    with _phase("accumulated_training_step"):
        loss = _accumulate(model, optimizer, state["inputs"])
        with _phase("optimizer_step"):
            optimizer.step()
        torch.cuda.synchronize(state["device"])
    return {"loss": loss.detach()}


def observe(state, result):
    """Loss, every gradient and every updated parameter (detached tensors)."""
    raw_model = state["raw_model"]
    gradients = {}
    parameters = {}
    for name, parameter in raw_model.named_parameters():
        if parameter.grad is not None:
            gradients[name] = parameter.grad.detach().clone()
        parameters[name] = parameter.detach().clone()
    return {
        "loss": result["loss"].detach().clone(),
        "gradients": gradients,
        "parameters": parameters,
    }


def metrics(state, result, elapsed_ms):
    """Clean project metrics from the source's own step sample semantics."""
    workload = state["workload"]
    elapsed_ms = float(elapsed_ms)
    tokens_per_step = (
        workload["world_size"] * workload["micro_batch"] * workload["sequence_length"] * workload["accumulation_steps"]
    )
    return {
        "step_ms": elapsed_ms,
        "tokens_per_second": tokens_per_step * 1000.0 / elapsed_ms,
        "tokens_per_step": float(tokens_per_step),
        "effective_batch": float(workload["world_size"] * workload["micro_batch"] * workload["accumulation_steps"]),
        "parameter_count": float(state["parameter_count"]),
        "peak_allocated_bytes": float(torch.cuda.max_memory_allocated(state["device"])),
        "loss_value": float(result["loss"].item()),
    }


def models(state):
    return {"gpt2": state["raw_model"]}


def set_mode(state, mode):
    state["mode"] = mode
