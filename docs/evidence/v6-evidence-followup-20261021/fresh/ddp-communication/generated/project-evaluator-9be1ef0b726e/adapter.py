from __future__ import annotations

from contextlib import contextmanager, nullcontext
from datetime import timedelta

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


MODEL_CONFIG = {"n_layer": 12, "n_embd": 768, "n_head": 12, "vocab_size": 50257, "n_positions": 1024}
WEIGHT_SEED = 20261020
INPUT_SEED_BASE = 31000
SGD_LR = 0.001
DDP_BUCKET_CAP_MB = 25
CONTROL_TIMEOUT_S = 120


def _configure(device):
    torch.set_num_threads(4)
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


def _make_model(device):
    torch.manual_seed(WEIGHT_SEED)
    config = GPT2Config(**MODEL_CONFIG, resid_pdrop=0.0, embd_pdrop=0.0, attn_pdrop=0.0, use_cache=False, attn_implementation="eager")
    return GPT2LMHeadModel(config).float().to(device).train()


def _make_inputs(workload, source_rank, device):
    generator = torch.Generator().manual_seed(INPUT_SEED_BASE + source_rank)
    shape = (workload["micro_batch"], workload["sequence_length"])
    return [torch.randint(0, MODEL_CONFIG["vocab_size"], shape, generator=generator).to(device) for _ in range(workload["accumulation_steps"])]


def _accumulate(state):
    model = state["model"]
    optimizer = state["optimizer"]
    inputs = state["inputs"]
    optimizer.zero_grad(set_to_none=True)
    losses = []
    for index, tokens in enumerate(inputs):
        final = index + 1 == len(inputs)
        context = model.no_sync() if isinstance(model, DDP) and not final else nullcontext()
        with context:
            with phase("microbatch_forward"):
                loss = model(input_ids=tokens, labels=tokens, use_cache=False).loss / len(inputs)
            with phase("synchronized_backward" if final else "backward"):
                loss.backward()
        losses.append(loss.detach())
    stacked = torch.stack(losses)
    return stacked.sum(), stacked


def prepare(context, request):
    device = torch.device(context["device"])
    distributed = dist.is_available() and dist.is_initialized()
    if distributed:
        rank = int(context.get("rank", dist.get_rank()))
        world_size = int(context.get("world_size", dist.get_world_size()))
    else:
        rank = int(context.get("rank", 0))
        world_size = int(context.get("world_size", 1))
    workload = request["workload"]
    if distributed and world_size != workload["world_size"]:
        raise ValueError("This workload must launch the declared number of ranks")
    _configure(device)
    raw_model = _make_model(device)
    control = None
    if distributed:
        local_index = device.index if device.index is not None else torch.cuda.current_device()
        model = DDP(raw_model, device_ids=[local_index], broadcast_buffers=False, bucket_cap_mb=DDP_BUCKET_CAP_MB, gradient_as_bucket_view=True)
        inputs = _make_inputs(workload, rank, device)
        control = dist.new_group(backend="gloo", timeout=timedelta(seconds=CONTROL_TIMEOUT_S))
    else:
        model = raw_model
        inputs = [tokens for source_rank in range(workload["world_size"]) for tokens in _make_inputs(workload, source_rank, device)]
    optimizer = torch.optim.SGD(model.parameters(), lr=SGD_LR, momentum=0.0, weight_decay=0.0, foreach=False)
    initial = {name: value.detach().cpu().clone() for name, value in raw_model.state_dict().items()}
    return {
        "request": request,
        "workload": workload,
        "device": device,
        "rank": rank,
        "world_size": world_size,
        "distributed": distributed,
        "raw_model": raw_model,
        "model": model,
        "optimizer": optimizer,
        "inputs": inputs,
        "initial": initial,
        "control": control,
        "parameter_count": sum(parameter.numel() for parameter in raw_model.parameters()),
        "mode": None,
    }


def reset(state):
    state["raw_model"].load_state_dict(state["initial"])


def before_run(state):
    if state["distributed"]:
        dist.barrier(group=state["control"])


def run(state):
    loss, microbatch_losses = _accumulate(state)
    with phase("optimizer_step"):
        state["optimizer"].step()
    if state["device"].type == "cuda":
        torch.cuda.synchronize()
    return {"loss": loss, "microbatch_losses": microbatch_losses}


def set_mode(state, mode):
    state["mode"] = mode


def observe(state, result):
    raw_model = state["raw_model"]
    return {
        "loss": result["loss"].detach().clone(),
        "microbatch_losses": result["microbatch_losses"].detach().clone(),
        "gradients": {name: parameter.grad.detach().clone() for name, parameter in raw_model.named_parameters()},
        "parameters": {name: parameter.detach().clone() for name, parameter in raw_model.named_parameters()},
    }


def metrics(state, result, elapsed_ms):
    workload = state["workload"]
    tokens_per_step = workload["world_size"] * workload["micro_batch"] * workload["sequence_length"] * workload["accumulation_steps"]
    step_ms = float(elapsed_ms)
    peak = float(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0.0
    return {
        "step_ms": step_ms,
        "tokens_per_step": float(tokens_per_step),
        "effective_batch": float(workload["world_size"] * workload["micro_batch"] * workload["accumulation_steps"]),
        "tokens_per_second": float(tokens_per_step) * 1000.0 / step_ms if step_ms > 0.0 else 0.0,
        "peak_allocated_bytes": peak,
        "parameter_count": float(state["parameter_count"]),
        "world_size": float(state["world_size"]),
        "micro_batch": float(workload["micro_batch"]),
        "sequence_length": float(workload["sequence_length"]),
        "accumulation_steps": float(workload["accumulation_steps"]),
    }


def models(state):
    return {"gpt2": state["raw_model"]}
