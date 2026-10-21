"""Baseline adapter for the gpt2_ddp_training source project (accumulated_step).

Reproduces the source's measured invocation verbatim: one accumulated DDP
training step (zero_grad inside the timed region, no_sync accumulation over
accumulation_steps microbatches, final-microbatch gradient allreduce, SGD
update, cuda synchronize) on the seeded synthetic GPT-2 124M FP32 model.
Model/reference construction, correctness checks, the cross-rank gradient
replica broadcast, the bucket inventory hook and the separate NCCL profiler
diagnostics remain outside run(); the framework collects observations and the
communication objective in its own diagnostic runs.
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
    from kernel_optimizer.project.discovery import phase as _phase
except Exception:  # annotation is optional; never block execution on it
    @contextmanager
    def _phase(name):
        yield


WEIGHT_SEED = 20261020
INPUT_SEED_BASE = 31000
CASE_KEY = "gpt2_ddp_training"
# Frozen architecture from the source workloads.json (gpt2_ddp_training.model);
# used only when that file is not reachable. Workload dimensions are always
# taken from request["workload"], never from these constants.
DEFAULT_MODEL_CONFIG = {
    "n_layer": 12,
    "n_embd": 768,
    "n_head": 12,
    "vocab_size": 50257,
    "n_positions": 1024,
}


def _model_config(request):
    candidates = [Path(__file__).resolve().parent / "inputs" / "source" / "workloads.json"]
    source_root = request.get("source_root")
    if source_root:
        candidates.append(Path(source_root) / "workloads.json")
    for candidate in candidates:
        try:
            if candidate.is_file():
                data = json.loads(candidate.read_text())
                model = data[CASE_KEY].get("model")
                if model:
                    return model
        except Exception:
            continue
    return dict(DEFAULT_MODEL_CONFIG)


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


def _make_model(config, device):
    torch.manual_seed(WEIGHT_SEED)
    model_config = GPT2Config(
        **config,
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        attn_pdrop=0.0,
        use_cache=False,
        attn_implementation="eager",
    )
    return GPT2LMHeadModel(model_config).float().to(device).train()


def _make_inputs(vocab_size, workload, source_rank, device):
    generator = torch.Generator().manual_seed(INPUT_SEED_BASE + source_rank)
    return [
        torch.randint(
            0,
            vocab_size,
            (workload["micro_batch"], workload["sequence_length"]),
            generator=generator,
        ).to(device)
        for _ in range(workload["accumulation_steps"])
    ]


def _accumulate(model, optimizer, inputs):
    optimizer.zero_grad(set_to_none=True)
    losses = []
    for i, tokens in enumerate(inputs):
        context = model.no_sync() if isinstance(model, DDP) and i + 1 < len(inputs) else nullcontext()
        with context:
            loss = model(input_ids=tokens, labels=tokens, use_cache=False).loss / len(inputs)
            loss.backward()
            losses.append(loss.detach())
    return torch.stack(losses).sum()


def prepare(context, request):
    device = torch.device(context["device"])
    if device.index is None:
        device = torch.device(device.type, int(context["rank"]))
    rank = int(context["rank"])
    workload = request["workload"]
    if not (dist.is_available() and dist.is_initialized()):
        raise RuntimeError("gpt2_ddp_training requires the worker-initialized process group")
    _configure(device)
    if dist.get_world_size() != workload["world_size"]:
        raise ValueError("This workload must launch the declared number of ranks")
    config = _model_config(request)
    raw_model = _make_model(config, device)
    model = DDP(
        raw_model,
        device_ids=[device.index],
        broadcast_buffers=False,
        bucket_cap_mb=25,
        gradient_as_bucket_view=True,
    )
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.0, weight_decay=0.0, foreach=False)
    inputs = _make_inputs(config["vocab_size"], workload, rank, device)
    initial = {name: value.detach().cpu().clone() for name, value in raw_model.state_dict().items()}
    control = dist.new_group(backend="gloo", timeout=timedelta(seconds=120))
    return {
        "device": device,
        "rank": rank,
        "world_size": int(context["world_size"]),
        "workload": workload,
        "raw_model": raw_model,
        "model": model,
        "optimizer": optimizer,
        "inputs": inputs,
        "initial": initial,
        "control": control,
        "parameter_count": sum(p.numel() for p in raw_model.parameters()),
        "mode": "measure",
    }


def set_mode(state, mode):
    state["mode"] = mode


def reset(state):
    state["raw_model"].load_state_dict(state["initial"])


def before_run(state):
    control = state.get("control")
    if control is not None:
        dist.barrier(group=control)


def run(state):
    with _phase("accumulate"):
        loss = _accumulate(state["model"], state["optimizer"], state["inputs"])
    with _phase("optimizer_step"):
        state["optimizer"].step()
        torch.cuda.synchronize()
    return {"loss": loss}


def observe(state, result):
    gradients = {}
    parameters = {}
    for name, parameter in state["raw_model"].named_parameters():
        gradients[name] = parameter.grad.detach().cpu()
        parameters[name] = parameter.detach().cpu()
    return {"loss": result["loss"].detach().cpu(), "gradients": gradients, "parameters": parameters}


def metrics(state, result, elapsed_ms):
    workload = state["workload"]
    token_count = (
        workload["world_size"]
        * workload["micro_batch"]
        * workload["sequence_length"]
        * workload["accumulation_steps"]
    )
    step_ms = float(elapsed_ms)
    values = {
        "step_ms": step_ms,
        "tokens_per_second": token_count * 1000.0 / step_ms,
        "effective_batch": workload["world_size"] * workload["micro_batch"] * workload["accumulation_steps"],
        "tokens_per_step": token_count,
        "parameter_count": state["parameter_count"],
    }
    if torch.cuda.is_available():
        values["peak_allocated_bytes"] = float(torch.cuda.max_memory_allocated())
    return values


def models(state):
    return {"model": state["raw_model"]}
