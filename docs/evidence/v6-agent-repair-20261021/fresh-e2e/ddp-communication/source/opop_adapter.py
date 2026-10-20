"""Adapter for the gpt2_ddp_training.py accumulated DDP training step.

Mirrors the source: FP32 GPT-2 with seed-20261020 initialization, per-rank
seeded token fixtures (generator seed 31000 + rank), DDP(broadcast_buffers=False,
bucket_cap_mb=25, gradient_as_bucket_view=True), no_sync on the first
accumulation_steps - 1 micro-batches with one gradient allreduce on the last,
SGD(lr=0.001, momentum=0, weight_decay=0, foreach=False), zero_grad inside the
timed step and a cuda synchronize at the step completion point.
"""
from contextlib import contextmanager, nullcontext
from datetime import timedelta

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import GPT2Config, GPT2LMHeadModel

try:
    from kernel_optimizer.project.discovery import phase as _phase
except Exception:

    @contextmanager
    def _phase(name):
        yield


INIT_SEED = 20261020
INPUT_SEED_BASE = 31000


def _configure(device):
    torch.set_num_threads(4)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    torch.manual_seed(INIT_SEED)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision("highest")


def prepare(context, request):
    workload = request["workload"]
    model_config = dict(workload["model"])
    device = torch.device(context["device"])
    rank = int(context["rank"])
    world_size = int(context["world_size"])
    if dist.is_available() and dist.is_initialized() and dist.get_world_size() != world_size:
        raise ValueError("This workload must launch the declared number of ranks")
    _configure(device)
    torch.manual_seed(INIT_SEED)
    config = GPT2Config(
        **model_config,
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        attn_pdrop=0.0,
        use_cache=False,
        attn_implementation="eager",
    )
    raw_model = GPT2LMHeadModel(config).float().to(device).train()
    if dist.is_available() and dist.is_initialized():
        index = device.index if device.index is not None else torch.cuda.current_device()
        model = DDP(
            raw_model,
            device_ids=None if device.type != "cuda" else [index],
            broadcast_buffers=False,
            bucket_cap_mb=25,
            gradient_as_bucket_view=True,
        )
    else:
        model = raw_model
    optimizer = torch.optim.SGD(
        model.parameters(), lr=0.001, momentum=0.0, weight_decay=0.0, foreach=False
    )
    generator = torch.Generator().manual_seed(INPUT_SEED_BASE + rank)
    shape = (int(workload["micro_batch"]), int(workload["sequence_length"]))
    accumulation_steps = int(workload["accumulation_steps"])
    inputs = [
        torch.randint(0, int(model_config["vocab_size"]), shape, generator=generator).to(device)
        for _ in range(accumulation_steps)
    ]
    initial = {name: value.detach().cpu().clone() for name, value in raw_model.state_dict().items()}
    control = None
    if world_size > 1 and dist.is_available() and dist.is_initialized():
        try:
            control = dist.new_group(backend="gloo", timeout=timedelta(seconds=120))
        except Exception:
            control = None
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
        "control": control,
    }


def reset(state):
    state["raw_model"].load_state_dict(state["initial"])


def run(state):
    model = state["model"]
    optimizer = state["optimizer"]
    inputs = state["inputs"]
    count = len(inputs)
    losses = []
    with _phase("accumulate"):
        optimizer.zero_grad(set_to_none=True)
        for index, tokens in enumerate(inputs):
            context = (
                model.no_sync() if isinstance(model, DDP) and index + 1 < count else nullcontext()
            )
            with context:
                loss = model(input_ids=tokens, labels=tokens, use_cache=False).loss / count
                loss.backward()
                losses.append(loss.detach())
    total = torch.stack(losses).sum()
    with _phase("optimizer_step"):
        optimizer.step()
    if state["device"].type == "cuda":
        torch.cuda.synchronize()
    return {"loss": total}


def observe(state, result):
    raw_model = state["raw_model"]
    gradients = {}
    parameters = {}
    for name, parameter in raw_model.named_parameters():
        gradients[name] = parameter.grad.detach().cpu().clone()
        parameters[name] = parameter.detach().cpu().clone()
    return {
        "loss": result["loss"].detach().cpu().clone(),
        "gradients": gradients,
        "parameters": parameters,
    }


def metrics(state, result, elapsed_ms):
    workload = state["workload"]
    tokens_per_step = (
        int(workload["world_size"])
        * int(workload["micro_batch"])
        * int(workload["sequence_length"])
        * int(workload["accumulation_steps"])
    )
    values = {
        "step_ms": float(elapsed_ms),
        "tokens_per_second": tokens_per_step * 1000.0 / float(elapsed_ms),
        "loss": float(result["loss"].detach().cpu()),
        "effective_batch": float(
            int(workload["world_size"])
            * int(workload["micro_batch"])
            * int(workload["accumulation_steps"])
        ),
        "tokens_per_step": float(tokens_per_step),
        "rank": float(state["rank"]),
        "world_size": float(state["world_size"]),
    }
    if state["device"].type == "cuda" and torch.cuda.is_available():
        values["peak_allocated_bytes"] = float(torch.cuda.max_memory_allocated())
    return values


def models(state):
    return {"model": state["raw_model"]}


def before_run(state):
    control = state.get("control")
    if control is not None:
        dist.barrier(group=control)
