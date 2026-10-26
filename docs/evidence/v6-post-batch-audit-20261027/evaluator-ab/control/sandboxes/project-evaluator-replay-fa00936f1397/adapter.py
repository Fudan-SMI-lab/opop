from contextlib import nullcontext
from datetime import timedelta

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import GPT2Config, GPT2LMHeadModel

from kernel_optimizer.project.discovery import phase

ARCH = {"n_layer": 12, "n_embd": 768, "n_head": 12, "vocab_size": 50257, "n_positions": 1024}
MODEL_SEED = 20261020
INPUT_SEED_BASE = 31000


def _make_model(device):
    torch.manual_seed(MODEL_SEED)
    model_config = GPT2Config(**ARCH, resid_pdrop=0.0, embd_pdrop=0.0, attn_pdrop=0.0, use_cache=False, attn_implementation="eager")
    return GPT2LMHeadModel(model_config).float().to(device).train()


def _make_inputs(workload, source_rank, device):
    generator = torch.Generator().manual_seed(INPUT_SEED_BASE + source_rank)
    return [torch.randint(0, ARCH["vocab_size"], (int(workload["micro_batch"]), int(workload["sequence_length"])), generator=generator).to(device)
            for _ in range(int(workload["accumulation_steps"]))]


def prepare(context, request):
    device = torch.device(context["device"])
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
    if device.type == "cuda":
        torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision("highest")
    raw_model = _make_model(device)
    if device.type == "cuda":
        device_index = device.index if device.index is not None else torch.cuda.current_device()
    else:
        device_index = None
    if device_index is not None:
        model = DDP(raw_model, device_ids=[device_index], broadcast_buffers=False, bucket_cap_mb=25, gradient_as_bucket_view=True)
    else:
        model = DDP(raw_model, broadcast_buffers=False, bucket_cap_mb=25, gradient_as_bucket_view=True)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.0, weight_decay=0.0, foreach=False)
    inputs = _make_inputs(workload, rank, device)
    initial = {name: value.detach().cpu().clone() for name, value in raw_model.state_dict().items()}
    control = None
    if dist.is_available() and dist.is_initialized() and world_size > 1:
        control = dist.new_group(backend="gloo", timeout=timedelta(seconds=120))
    return {"raw_model": raw_model, "model": model, "optimizer": optimizer, "inputs": inputs,
            "initial": initial, "workload": workload, "device": device, "rank": rank,
            "world_size": world_size, "control": control}


def reset(state):
    state["raw_model"].load_state_dict(state["initial"])


def run(state):
    model = state["model"]
    optimizer = state["optimizer"]
    inputs = state["inputs"]
    optimizer.zero_grad(set_to_none=True)
    losses = []
    with phase("forward_backward"):
        for i, tokens in enumerate(inputs):
            context = model.no_sync() if isinstance(model, DDP) and i + 1 < len(inputs) else nullcontext()
            with context:
                loss = model(input_ids=tokens, labels=tokens, use_cache=False).loss / len(inputs)
                loss.backward()
                losses.append(loss.detach())
    total = torch.stack(losses).sum()
    with phase("optimizer_step"):
        optimizer.step()
    if state["device"].type == "cuda":
        torch.cuda.synchronize()
    return {"loss": total, "losses": list(losses)}


def observe(state, result):
    raw_model = state["raw_model"]
    gradients = {}
    parameters = {}
    for name, parameter in raw_model.named_parameters():
        parameters[name] = parameter.detach().cpu().clone()
        if parameter.grad is not None:
            gradients[name] = parameter.grad.detach().cpu().clone()
    return {
        "loss": result["loss"].detach().cpu().clone(),
        "loss_per_microbatch": [loss.detach().cpu().clone() for loss in result["losses"]],
        "gradients": gradients,
        "parameters": parameters,
    }


def metrics(state, result, elapsed_ms):
    workload = state["workload"]
    tokens_per_step = (int(workload["world_size"]) * int(workload["micro_batch"])
                       * int(workload["sequence_length"]) * int(workload["accumulation_steps"]))
    values = {
        "step_ms": float(elapsed_ms),
        "tokens_per_step": float(tokens_per_step),
        "effective_batch": float(int(workload["world_size"]) * int(workload["micro_batch"]) * int(workload["accumulation_steps"])),
        "tokens_per_second": tokens_per_step * 1000.0 / float(elapsed_ms),
        "parameter_count": float(sum(parameter.numel() for parameter in state["raw_model"].parameters())),
    }
    if state["device"].type == "cuda":
        values["peak_allocated_bytes"] = float(torch.cuda.max_memory_allocated())
    return values


def models(state):
    return {"gpt2": state["raw_model"]}


def before_run(state):
    control = state.get("control")
    if control is not None:
        dist.barrier(group=control)
