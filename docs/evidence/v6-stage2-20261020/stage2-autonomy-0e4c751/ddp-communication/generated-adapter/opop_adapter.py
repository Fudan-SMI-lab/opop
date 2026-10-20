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

MODEL_SEED = 20261020
INPUT_SEED_BASE = 31000
MODEL_CONFIG = {"n_layer": 12, "n_embd": 768, "n_head": 12, "vocab_size": 50257, "n_positions": 1024}


def _annotate(name):
    if _discovery_phase is None:
        return nullcontext()
    try:
        return _discovery_phase(name)
    except Exception:
        return nullcontext()


def _is_cuda(device):
    if isinstance(device, torch.device):
        return device.type == "cuda"
    return str(device).startswith("cuda")


def _device_index(device):
    if isinstance(device, torch.device):
        return device.index if device.index is not None else torch.cuda.current_device()
    text = str(device)
    if ":" in text:
        return int(text.rsplit(":", 1)[1])
    return torch.cuda.current_device()


def _synchronize(state):
    if _is_cuda(state["device"]):
        torch.cuda.synchronize()


def _make_model(device):
    torch.manual_seed(MODEL_SEED)
    config = GPT2Config(**MODEL_CONFIG, resid_pdrop=0.0, embd_pdrop=0.0, attn_pdrop=0.0, use_cache=False, attn_implementation="eager")
    return GPT2LMHeadModel(config).float().to(device).train()


def _make_inputs(workload, rank, device):
    generator = torch.Generator().manual_seed(INPUT_SEED_BASE + rank)
    return [torch.randint(0, MODEL_CONFIG["vocab_size"], (workload["micro_batch"], workload["sequence_length"]), generator=generator).to(device)
            for _ in range(workload["accumulation_steps"])]


def _accumulate(state):
    model = state["model"]
    optimizer = state["optimizer"]
    inputs = state["inputs"]
    optimizer.zero_grad(set_to_none=True)
    losses = []
    for index, tokens in enumerate(inputs):
        context = model.no_sync() if isinstance(model, DDP) and index + 1 < len(inputs) else nullcontext()
        with context, _annotate("forward_backward"):
            loss = model(input_ids=tokens, labels=tokens, use_cache=False).loss / len(inputs)
            loss.backward()
            losses.append(loss.detach())
    return torch.stack(losses).sum()


def prepare(context, request):
    device = context["device"]
    rank = int(context["rank"])
    world_size = int(context["world_size"])
    workload = request["workload"]
    if world_size != int(workload["world_size"]):
        raise ValueError("This workload must launch the declared number of ranks")
    torch.set_num_threads(4)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    torch.manual_seed(MODEL_SEED)
    if _is_cuda(device):
        torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision("highest")
    control = None
    if dist.is_available() and dist.is_initialized() and world_size > 1:
        try:
            control = dist.new_group(backend="gloo", timeout=timedelta(seconds=120))
        except Exception:
            control = None
    raw_model = _make_model(device)
    model = DDP(raw_model, device_ids=[_device_index(device)], broadcast_buffers=False, bucket_cap_mb=25, gradient_as_bucket_view=True)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.0, weight_decay=0.0, foreach=False)
    inputs = _make_inputs(workload, rank, device)
    initial = {name: value.detach().cpu().clone() for name, value in raw_model.state_dict().items()}
    return {"raw_model": raw_model, "model": model, "optimizer": optimizer, "inputs": inputs, "initial": initial,
            "control": control, "workload": workload, "rank": rank, "world_size": world_size, "device": device,
            "mode": "measure"}


def reset(state):
    state["raw_model"].load_state_dict(state["initial"])
    state["optimizer"].zero_grad(set_to_none=True)
    torch.manual_seed(MODEL_SEED)


def run(state):
    loss = _accumulate(state)
    with _annotate("optimizer_step"):
        state["optimizer"].step()
    _synchronize(state)
    return {"loss": loss.detach()}


def observe(state, result):
    gradients = {}
    parameters = {}
    for name, parameter in state["raw_model"].named_parameters():
        gradient = parameter.grad if parameter.grad is not None else torch.zeros_like(parameter)
        gradients[name] = gradient.detach().to("cpu").clone()
        parameters[name] = parameter.detach().to("cpu").clone()
    return {"loss": result["loss"].detach().to("cpu").clone(), "gradients": gradients, "parameters": parameters}


def metrics(state, result, elapsed_ms):
    workload = state["workload"]
    step_ms = float(elapsed_ms)
    tokens_per_step = int(workload["world_size"]) * int(workload["micro_batch"]) * int(workload["sequence_length"]) * int(workload["accumulation_steps"])
    peak = float(torch.cuda.max_memory_allocated()) if torch.cuda.is_initialized() else 0.0
    return {
        "step_ms": step_ms,
        "tokens_per_step": float(tokens_per_step),
        "tokens_per_second": tokens_per_step * 1000.0 / step_ms,
        "effective_batch": float(int(workload["world_size"]) * int(workload["micro_batch"]) * int(workload["accumulation_steps"])),
        "parameter_count": float(sum(parameter.numel() for parameter in state["raw_model"].parameters())),
        "peak_allocated_bytes": peak,
    }


def before_run(state):
    if not (dist.is_available() and dist.is_initialized()):
        return
    _synchronize(state)
    if state["world_size"] > 1 and state["control"] is not None:
        dist.barrier(group=state["control"])
    if torch.cuda.is_initialized():
        torch.cuda.reset_peak_memory_stats()


def set_mode(state, mode):
    state["mode"] = mode


def models(state):
    return {"model": state["raw_model"]}
