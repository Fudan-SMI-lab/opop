"""Baseline adapter for the gpt2_ddp_training source project.

Reproduces the source's DDP-accumulated GPT-2 training step: seeded synthetic
FP32 GPT-2 weights, per-rank sharded synthetic microbatches, no_sync
accumulation with one real NCCL gradient synchronization per step, SGD update,
and the source's full correctness checks captured in observe() outside the
timed run(): exact cross-rank gradient equality (atol=0, rtol=0 via rank-0
broadcast), serial global-batch reference comparison, post-step parameter
comparison and the +1-corruption negative control.

Repair versus the previous revision: (1) the state restore around the serial
reference now uses full state_dict snapshots. named_parameters() deduplicates
GPT-2's tied lm_head/wte weight, and restoring through a parameters-only
snapshot reloaded a stale tied key over the shared storage, corrupting wte
for subsequent continuous steps (the previous run's failing 38597376-element
tensor). (2) The serial reference for a continuous step k>1 is anchored to
the observed post-step-(k-1) parameters, i.e. the exact parameter point the
distributed run started that step from, instead of a serial trajectory
accumulated from the initial weights, so every step is compared single step
exactly like the source's one-shot check without compounding tolerated
first-step differences through the loss curvature. The reference reuses
raw_model on rank 0 (no second GPU model) and restores parameters and
gradients afterwards.
"""
from __future__ import annotations

from contextlib import nullcontext
from datetime import timedelta
import json
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import GPT2Config, GPT2LMHeadModel

try:
    from kernel_optimizer.project.discovery import phase as _phase
except Exception:  # pragma: no cover - diagnostic annotation only
    _phase = None

WEIGHT_SEED = 20261020
INPUT_SEED_BASE = 31000
FALLBACK_MODEL_SPEC = {
    "n_layer": 12,
    "n_embd": 768,
    "n_head": 12,
    "vocab_size": 50257,
    "n_positions": 1024,
}
SGD_LR = 0.001
CONTROL_TIMEOUT_S = 600.0


def _maybe_phase(name):
    if _phase is None:
        return nullcontext()
    return _phase(name)


def _configure(device):
    torch.set_num_threads(4)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    torch.manual_seed(WEIGHT_SEED)
    if device.type == "cuda":
        torch.cuda.set_device(device if device.index is not None else torch.cuda.current_device())
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision("highest")


def _model_spec(request):
    spec = None
    source_root = request.get("source_root")
    if source_root:
        path = Path(source_root) / "workloads.json"
        try:
            if path.exists():
                spec = json.loads(path.read_text())["gpt2_ddp_training"]["model"]
        except (OSError, KeyError, ValueError):
            spec = None
    if not isinstance(spec, dict) or not spec:
        spec = dict(FALLBACK_MODEL_SPEC)
    return {key: spec[key] for key in FALLBACK_MODEL_SPEC}


def _make_model(model_spec, device):
    torch.manual_seed(WEIGHT_SEED)
    config = GPT2Config(
        **model_spec,
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        attn_pdrop=0.0,
        use_cache=False,
        attn_implementation="eager",
    )
    return GPT2LMHeadModel(config).float().to(device).train()


def _make_cpu_inputs(model_spec, workload, source_rank):
    generator = torch.Generator().manual_seed(INPUT_SEED_BASE + source_rank)
    shape = (int(workload["micro_batch"]), int(workload["sequence_length"]))
    return [
        torch.randint(0, int(model_spec["vocab_size"]), shape, generator=generator)
        for _ in range(int(workload["accumulation_steps"]))
    ]


def _accumulate(model, optimizer, inputs):
    optimizer.zero_grad(set_to_none=True)
    losses = []
    for i, tokens in enumerate(inputs):
        context = model.no_sync() if isinstance(model, DDP) and i + 1 < len(inputs) else nullcontext()
        with context, _maybe_phase(f"microbatch_{i}_forward_backward"):
            loss = model(input_ids=tokens, labels=tokens, use_cache=False).loss / len(inputs)
            loss.backward()
            losses.append(loss.detach())
    return torch.stack(losses).sum()


def prepare(context, request):
    workload = request["workload"]
    rank = int(context["rank"])
    world_size = int(context["world_size"])
    device = torch.device(context["device"])
    if world_size != int(workload["world_size"]):
        raise ValueError("This workload must launch the declared number of ranks")
    _configure(device)
    dist_ready = dist.is_available() and dist.is_initialized()
    if world_size > 1 and not dist_ready:
        raise RuntimeError("Distributed workload requires the worker-initialized process group")
    if dist_ready and dist.get_world_size() != int(workload["world_size"]):
        raise ValueError("This workload must launch the declared number of ranks")
    model_spec = _model_spec(request)
    raw_model = _make_model(model_spec, device)
    if world_size > 1:
        device_ids = None
        if device.type == "cuda":
            local_index = device.index if device.index is not None else torch.cuda.current_device()
            device_ids = [local_index]
        model = DDP(
            raw_model,
            device_ids=device_ids,
            broadcast_buffers=False,
            bucket_cap_mb=25,
            gradient_as_bucket_view=True,
        )
    else:
        model = raw_model
    optimizer = torch.optim.SGD(
        model.parameters(), lr=SGD_LR, momentum=0.0, weight_decay=0.0, foreach=False
    )
    cpu_inputs_by_rank = {r: _make_cpu_inputs(model_spec, workload, r) for r in range(world_size)}
    inputs = [tensor.to(device) for tensor in cpu_inputs_by_rank[rank]]
    initial = {name: value.detach().cpu().clone() for name, value in raw_model.state_dict().items()}
    control = None
    if dist_ready and world_size > 1:
        control = dist.new_group(backend="gloo", timeout=timedelta(seconds=CONTROL_TIMEOUT_S))
    cpu_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state(device) if device.type == "cuda" else None
    return {
        "rank": rank,
        "world_size": world_size,
        "device": device,
        "workload": workload,
        "model_spec": model_spec,
        "quality": dict(request.get("quality") or {}),
        "raw_model": raw_model,
        "model": model,
        "optimizer": optimizer,
        "inputs": inputs,
        "cpu_inputs_by_rank": cpu_inputs_by_rank,
        "initial": initial,
        "control": control,
        "mode": "quality",
        "run_index": 0,
        "anchor_params": None,
        "cpu_rng": cpu_rng,
        "cuda_rng": cuda_rng,
    }


def reset(state):
    state["raw_model"].load_state_dict(state["initial"])
    state["optimizer"].state.clear()
    torch.set_rng_state(state["cpu_rng"])
    if state["cuda_rng"] is not None:
        torch.cuda.set_rng_state(state["cuda_rng"], state["device"])
    state["run_index"] = 0
    state["anchor_params"] = None


def set_mode(state, mode):
    state["mode"] = str(mode)


def before_run(state):
    if state.get("control") is not None:
        if state["device"].type == "cuda":
            torch.cuda.synchronize(state["device"])
        dist.barrier(group=state["control"])


def run(state):
    state["run_index"] += 1
    with _maybe_phase("train_step"):
        loss = _accumulate(state["model"], state["optimizer"], state["inputs"])
        with _maybe_phase("optimizer_step"):
            state["optimizer"].step()
        if state["device"].type == "cuda":
            torch.cuda.synchronize(state["device"])
    return {"accumulated_loss": loss.detach()}


def _cross_rank_exact_gradient_check(state):
    if not (dist.is_available() and dist.is_initialized()) or state["world_size"] <= 1:
        return 0
    count = 0
    for parameter in state["raw_model"].parameters():
        replica = (
            parameter.grad.detach().clone()
            if state["rank"] == 0
            else torch.empty_like(parameter.grad)
        )
        dist.broadcast(replica, src=0)
        torch.testing.assert_close(parameter.grad, replica, atol=0.0, rtol=0.0)
        count += 1
        del replica
    return count


def _serial_global_batch_reference(state, anchor):
    """Serially process the exact global batch from `anchor` without DDP.

    Mirrors the source's rank-0 reference (gpt2_ddp_training.py:67-79): every
    rank's microbatches in rank order, each loss divided by the global
    microbatch count, gradients accumulated, then the same SGD update. The
    model is restored to its observed post-run state before returning.
    `anchor` and the stash are full state_dict snapshots: named_parameters()
    deduplicates the tied lm_head/wte weight, and reloading a stale tied key
    would silently overwrite the shared storage.
    """
    raw_model = state["raw_model"]
    optimizer = state["optimizer"]
    device = state["device"]
    named = list(raw_model.named_parameters())
    lr = float(optimizer.param_groups[0]["lr"])
    stash_grads = {name: p.grad.detach().cpu().clone() for name, p in named}
    stash_state = {name: value.detach().cpu().clone() for name, value in raw_model.state_dict().items()}
    global_inputs = [
        tensor
        for source_rank in sorted(state["cpu_inputs_by_rank"])
        for tensor in state["cpu_inputs_by_rank"][source_rank]
    ]
    raw_model.load_state_dict(anchor)
    optimizer.zero_grad(set_to_none=True)
    losses = []
    with _maybe_phase("serial_global_batch_reference"):
        for tokens_cpu in global_inputs:
            tokens = tokens_cpu.to(device)
            loss = raw_model(input_ids=tokens, labels=tokens, use_cache=False).loss / len(global_inputs)
            loss.backward()
            losses.append(loss.detach())
        reference_loss = torch.stack(losses).sum()
        reference_grads = {name: p.grad.detach().cpu().clone() for name, p in named}
        with torch.no_grad():
            for _, parameter in named:
                parameter.add_(parameter.grad, alpha=-lr)
        reference_params = {name: p.detach().cpu().clone() for name, p in named}
    raw_model.load_state_dict(stash_state)
    for name, parameter in named:
        parameter.grad = stash_grads[name].to(device)
    first_name = named[0][0]
    return {
        "grads": reference_grads,
        "params": reference_params,
        "loss": reference_loss,
        "slice": reference_grads[first_name].flatten()[:4096].clone(),
    }


def _negative_control(reference, *, atol, rtol):
    try:
        torch.testing.assert_close(reference + 1, reference, atol=atol, rtol=rtol)
    except AssertionError:
        return True
    raise AssertionError("Correctness control did not reject +1 corruption")


def observe(state, result):
    raw_model = state["raw_model"]
    loss = result.get("accumulated_loss") if isinstance(result, dict) else result
    if not isinstance(loss, torch.Tensor):
        raise RuntimeError("run() result does not carry the accumulated loss")
    loss_cpu = loss.detach().float().cpu()
    if not bool(torch.isfinite(loss_cpu).all()):
        raise AssertionError("Accumulated loss is not finite")
    named = list(raw_model.named_parameters())
    missing = [name for name, p in named if p.grad is None]
    if missing:
        raise RuntimeError(f"Missing gradients after run(): {missing[:4]}")
    run_index = int(state["run_index"])
    grads_cpu = {name: p.grad.detach().float().cpu().clone() for name, p in named}
    params_cpu = {name: p.detach().float().cpu().clone() for name, p in named}
    state_snapshot = {name: value.detach().cpu().clone() for name, value in raw_model.state_dict().items()}
    checks = {
        "finite_loss": True,
        "gradient_tensor_count": len(named),
        "parameter_tensor_count": len(named),
        "run_index": run_index,
    }
    comparisons = _cross_rank_exact_gradient_check(state)
    if comparisons:
        checks["exact_gradient_replica_check"] = True
        checks["cross_rank_tensor_comparisons"] = comparisons
    if state["rank"] == 0:
        quality = state["quality"]
        grad_atol = float(quality.get("grad_atol", quality.get("atol", 1e-06)))
        grad_rtol = float(quality.get("grad_rtol", quality.get("rtol", 1e-04)))
        param_atol = float(quality.get("parameter_atol", quality.get("atol", 1e-06)))
        param_rtol = float(quality.get("parameter_rtol", quality.get("rtol", 1e-04)))
        anchor = state["initial"] if run_index <= 1 else state.get("anchor_params")
        if anchor is None:
            raise RuntimeError("continuous step is missing its anchored start parameters")
        reference = _serial_global_batch_reference(state, anchor)
        grad_errors = []
        for name, _ in named:
            torch.testing.assert_close(
                grads_cpu[name], reference["grads"][name], atol=grad_atol, rtol=grad_rtol
            )
            grad_errors.append((grads_cpu[name] - reference["grads"][name]).abs().max().item())
        param_errors = []
        for name, _ in named:
            torch.testing.assert_close(
                params_cpu[name], reference["params"][name], atol=param_atol, rtol=param_rtol
            )
            param_errors.append((params_cpu[name] - reference["params"][name]).abs().max().item())
        checks["global_batch_gradient_check"] = {
            "passed": True,
            "max_abs_error": max(grad_errors),
            "reference_loss": float(reference["loss"]),
        }
        checks["parameter_update_check"] = {
            "passed": True,
            "max_abs_error": max(param_errors),
        }
        checks["corrupted_output_rejected"] = _negative_control(
            reference["slice"], atol=grad_atol, rtol=grad_rtol
        )
        state["anchor_params"] = state_snapshot
    return {
        "accumulated_loss": loss_cpu,
        "finite_loss": True,
        "gradients": grads_cpu,
        "parameters": params_cpu,
        "checks": checks,
    }


def metrics(state, result, elapsed_ms):
    workload = state["workload"]
    token_count = (
        int(workload["world_size"])
        * int(workload["micro_batch"])
        * int(workload["sequence_length"])
        * int(workload["accumulation_steps"])
    )
    elapsed = max(float(elapsed_ms), 1e-09)
    values = {
        "step_ms": float(elapsed_ms),
        "tokens_per_second": token_count * 1000.0 / elapsed,
        "tokens_per_step": float(token_count),
        "effective_batch": float(
            int(workload["world_size"]) * int(workload["micro_batch"]) * int(workload["accumulation_steps"])
        ),
        "parameter_count": float(sum(p.numel() for p in state["raw_model"].parameters())),
        "micro_batch": float(workload["micro_batch"]),
        "sequence_length": float(workload["sequence_length"]),
        "accumulation_steps": float(workload["accumulation_steps"]),
        "world_size": float(state["world_size"]),
    }
    if state["device"].type == "cuda":
        values["peak_allocated_bytes"] = float(torch.cuda.max_memory_allocated(state["device"]))
    return values


def models(state):
    return {"gpt2": state["raw_model"]}
