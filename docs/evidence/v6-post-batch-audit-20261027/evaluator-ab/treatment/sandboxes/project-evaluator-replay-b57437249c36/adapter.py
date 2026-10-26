"""Baseline adapter reproducing gpt2_ddp_training.py (GPT-2 124M FP32 DDP accumulated step)."""
from __future__ import annotations

from contextlib import contextmanager, nullcontext

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


MODEL_CONFIG = {"n_layer": 12, "n_embd": 768, "n_head": 12, "vocab_size": 50257, "n_positions": 1024}
MODEL_SEED = 20261020
INPUT_SEED_BASE = 31000


def _configure(device):
    torch.set_num_threads(4)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    torch.manual_seed(MODEL_SEED)
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision("highest")


def _local_device_index(device):
    resolved = torch.device(device)
    return resolved.index if resolved.index is not None else torch.cuda.current_device()


def _make_model(device):
    torch.manual_seed(MODEL_SEED)
    model_config = GPT2Config(**MODEL_CONFIG, resid_pdrop=0.0, embd_pdrop=0.0, attn_pdrop=0.0, use_cache=False, attn_implementation="eager")
    return GPT2LMHeadModel(model_config).float().to(device).train()


def _make_inputs(workload, source_rank, device):
    generator = torch.Generator().manual_seed(INPUT_SEED_BASE + source_rank)
    return [torch.randint(0, MODEL_CONFIG["vocab_size"], (workload["micro_batch"], workload["sequence_length"]), generator=generator).to(device)
            for _ in range(workload["accumulation_steps"])]


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
    device = context["device"]
    rank = int(context["rank"])
    workload = request["workload"]
    _configure(device)
    if dist.is_available() and dist.is_initialized() and dist.get_world_size() != workload["world_size"]:
        raise ValueError("This workload must launch the declared number of ranks")
    raw_model = _make_model(device)
    model = DDP(raw_model, device_ids=[_local_device_index(device)], broadcast_buffers=False, bucket_cap_mb=25, gradient_as_bucket_view=True)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.0, weight_decay=0.0, foreach=False)
    inputs = _make_inputs(workload, rank, device)
    initial = {name: value.detach().cpu().clone() for name, value in raw_model.state_dict().items()}
    control = None
    if dist.is_available() and dist.is_initialized():
        try:
            control = dist.new_group(backend="gloo")
        except Exception:
            control = None
    return {
        "device": device,
        "rank": rank,
        "world_size": int(context["world_size"]),
        "workload": workload,
        "quality": request["quality"],
        "raw_model": raw_model,
        "model": model,
        "optimizer": optimizer,
        "inputs": inputs,
        "initial": initial,
        "control": control,
        "parameter_count": sum(p.numel() for p in raw_model.parameters()),
        "effective_batch": workload["world_size"] * workload["micro_batch"] * workload["accumulation_steps"],
        "tokens_per_step": workload["world_size"] * workload["micro_batch"] * workload["sequence_length"] * workload["accumulation_steps"],
        "mode": None,
    }


def set_mode(state, mode):
    state["mode"] = mode


def reset(state):
    state["raw_model"].load_state_dict(state["initial"])


def before_run(state):
    torch.cuda.synchronize()
    if dist.is_available() and dist.is_initialized():
        dist.barrier(group=state.get("control"))


def run(state):
    with _phase("gradient_accumulation"):
        loss = _accumulate(state["model"], state["optimizer"], state["inputs"])
    with _phase("optimizer_step"):
        state["optimizer"].step()
    torch.cuda.synchronize()
    return {"loss": loss.detach()}


def observe(state, result):
    raw_model = state["raw_model"]
    rank = state["rank"]
    quality = state["quality"]
    loss = result["loss"].detach().cpu().clone()
    gradients = {}
    replica_max_abs_diff = 0.0
    distributed = dist.is_available() and dist.is_initialized()
    for name, parameter in raw_model.named_parameters():
        grad = parameter.grad
        if grad is None:
            raise RuntimeError(f"Missing gradient for parameter {name}")
        if distributed:
            replica = grad.detach().clone() if rank == 0 else torch.empty_like(grad)
            dist.broadcast(replica, src=0)
            replica_max_abs_diff = max(replica_max_abs_diff, (grad.detach() - replica).abs().max().item())
            torch.testing.assert_close(grad, replica, atol=0.0, rtol=0.0)
            del replica
        gradients[name] = grad.detach().cpu().clone()
    finite_loss = bool(torch.isfinite(loss).item())
    probe = next(iter(gradients.values())).flatten()[:4096]
    try:
        torch.testing.assert_close(probe + 1, probe, atol=quality["grad_atol"], rtol=quality["grad_rtol"])
        corrupted_output_rejected = False
    except AssertionError:
        corrupted_output_rejected = True
    if not corrupted_output_rejected:
        raise AssertionError("Correctness control did not reject +1 corruption")
    if not finite_loss:
        raise AssertionError("Training loss is not finite")
    parameters = {name: parameter.detach().cpu().clone() for name, parameter in raw_model.named_parameters()}
    return {
        "loss": loss,
        "gradients": gradients,
        "parameters": parameters,
        "checks": {
            "finite_loss": 1 if finite_loss else 0,
            "exact_gradient_replica_check": 1 if replica_max_abs_diff == 0.0 else 0,
            "gradient_replica_max_abs_diff": replica_max_abs_diff,
            "corrupted_output_rejected": 1 if corrupted_output_rejected else 0,
        },
    }


def metrics(state, result, elapsed_ms):
    elapsed = float(elapsed_ms)
    return {
        "step_ms": elapsed,
        "tokens_per_second": state["tokens_per_step"] * 1000.0 / elapsed,
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "loss": float(result["loss"].detach().cpu()),
        "parameter_count": state["parameter_count"],
        "effective_batch": state["effective_batch"],
        "tokens_per_step": state["tokens_per_step"],
    }


def models(state):
    return {"gpt2": state["raw_model"]}
