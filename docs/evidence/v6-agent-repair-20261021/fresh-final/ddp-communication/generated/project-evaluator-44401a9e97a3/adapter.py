from __future__ import annotations

from contextlib import nullcontext
from datetime import timedelta

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import GPT2Config, GPT2LMHeadModel

try:
    from kernel_optimizer.project.discovery import phase as _phase
except Exception:
    from contextlib import contextmanager

    @contextmanager
    def _phase(name):
        yield


MODEL_CONFIG = {"n_layer": 12, "n_embd": 768, "n_head": 12, "vocab_size": 50257, "n_positions": 1024}
WEIGHT_SEED = 20261020
INPUT_SEED_BASE = 31000


def _configure(device_index):
    torch.set_num_threads(4)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    torch.manual_seed(WEIGHT_SEED)
    torch.cuda.set_device(device_index)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision("highest")


def _make_model(device):
    torch.manual_seed(WEIGHT_SEED)
    config = GPT2Config(
        **MODEL_CONFIG,
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        attn_pdrop=0.0,
        use_cache=False,
        attn_implementation="eager",
    )
    return GPT2LMHeadModel(config).float().to(device).train()


def _make_inputs(vocab_size, micro_batch, sequence_length, accumulation_steps, source_rank, device):
    generator = torch.Generator().manual_seed(INPUT_SEED_BASE + source_rank)
    return [
        torch.randint(0, vocab_size, (micro_batch, sequence_length), generator=generator).to(device)
        for _ in range(accumulation_steps)
    ]


def _accumulate(model, optimizer, inputs):
    optimizer.zero_grad(set_to_none=True)
    losses = []
    for i, tokens in enumerate(inputs):
        context = model.no_sync() if isinstance(model, DDP) and i + 1 < len(inputs) else nullcontext()
        with context, _phase(f"micro_batch_{i + 1}_of_{len(inputs)}"):
            loss = model(input_ids=tokens, labels=tokens, use_cache=False).loss / len(inputs)
            loss.backward()
            losses.append(loss.detach())
    return torch.stack(losses).sum()


def prepare(context, request):
    workload = request["workload"]
    device = torch.device(context["device"])
    device_index = device.index if device.index is not None else torch.cuda.current_device()
    rank = int(context["rank"])
    world_size = int(context["world_size"])
    _configure(device_index)
    if world_size != workload["world_size"]:
        raise ValueError("This workload must launch the declared number of ranks")
    raw_model = _make_model(device)
    model = DDP(
        raw_model,
        device_ids=[device_index],
        broadcast_buffers=False,
        bucket_cap_mb=25,
        gradient_as_bucket_view=True,
    )
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.0, weight_decay=0.0, foreach=False)
    inputs = _make_inputs(
        MODEL_CONFIG["vocab_size"],
        workload["micro_batch"],
        workload["sequence_length"],
        workload["accumulation_steps"],
        rank,
        device,
    )
    initial = {name: value.detach().cpu().clone() for name, value in raw_model.state_dict().items()}
    initial_optimizer = optimizer.state_dict()
    control = None
    if dist.is_available() and dist.is_initialized():
        control = dist.new_group(backend="gloo", timeout=timedelta(seconds=120))
    state = {
        "request": request,
        "workload": workload,
        "device": device,
        "rank": rank,
        "world_size": world_size,
        "raw_model": raw_model,
        "model": model,
        "optimizer": optimizer,
        "inputs": inputs,
        "initial": initial,
        "initial_optimizer": initial_optimizer,
        "control": control,
        "effective_batch": workload["world_size"] * workload["micro_batch"] * workload["accumulation_steps"],
        "tokens_per_step": workload["world_size"]
        * workload["micro_batch"]
        * workload["sequence_length"]
        * workload["accumulation_steps"],
        "parameter_count": sum(p.numel() for p in raw_model.parameters()),
        "mode": None,
    }
    return state


def reset(state):
    state["raw_model"].load_state_dict(state["initial"])
    state["optimizer"].load_state_dict(state["initial_optimizer"])
    torch.manual_seed(WEIGHT_SEED)


def before_run(state):
    control = state.get("control")
    if control is not None:
        dist.barrier(group=control)


def set_mode(state, mode):
    state["mode"] = mode


def run(state):
    loss = _accumulate(state["model"], state["optimizer"], state["inputs"])
    with _phase("optimizer_step"):
        state["optimizer"].step()
    torch.cuda.synchronize()
    return {"loss": loss.detach()}


def observe(state, result):
    raw_model = state["raw_model"]
    gradients = {
        name: parameter.grad.detach().to("cpu").clone()
        for name, parameter in raw_model.named_parameters()
    }
    parameters = {name: tensor.detach().to("cpu").clone() for name, tensor in raw_model.state_dict().items()}
    return {
        "loss": result["loss"].detach().to("cpu").clone(),
        "gradients": gradients,
        "parameters": parameters,
    }


def metrics(state, result, elapsed_ms):
    step_ms = float(elapsed_ms)
    values = {
        "step_ms": step_ms,
        "tokens_per_second": state["tokens_per_step"] * 1000.0 / step_ms,
        "effective_batch": float(state["effective_batch"]),
        "tokens_per_step": float(state["tokens_per_step"]),
        "parameter_count": float(state["parameter_count"]),
    }
    if torch.cuda.is_available():
        values["peak_allocated_bytes"] = float(torch.cuda.max_memory_allocated())
    return values


def models(state):
    return {"model": state["raw_model"]}
