"""Execution-policy candidate: one fused NCCL gradient allreduce per step.

Replaces project.invoke with the same accumulated DDP training step, except
gradient synchronization packs every parameter gradient into a single fp32
buffer, divides by world_size and issues exactly one NCCL SUM allreduce on the
default group (device-synchronized and rank-aligned first), then writes the
synchronized values back. The two observed DDP bucket allreduce sites are
eliminated; the synchronized gradients are element-wise identical to the
default reducer's divide-by-world-then-SUM arithmetic.
"""
from __future__ import annotations

from contextlib import nullcontext

import torch
import torch.distributed as dist

try:
    from kernel_optimizer.project.discovery import phase as _phase
except Exception:
    _phase = None

CALL_ID = "5d7d6bb13691b02f533f62b0"
_SCRATCH = {}


def _maybe_phase(name):
    if _phase is None:
        return nullcontext()
    return _phase(name)


def _copy_many(dsts, srcs):
    if not dsts:
        return
    try:
        torch._foreach_copy_(dsts, srcs)
        return
    except (AttributeError, RuntimeError, TypeError, ValueError):
        pass
    for dst, src in zip(dsts, srcs):
        dst.copy_(src)


def _fused_gradient_synchronization(state, params, device, world_size):
    total = 0
    for parameter in params:
        total += parameter.numel()
    if total <= 0:
        raise RuntimeError("fused synchronization found no gradient elements")
    key = (str(device), total)
    scratch = _SCRATCH.get(key)
    if scratch is None or scratch.device != device or scratch.dtype != torch.float32:
        scratch = torch.empty(total, dtype=torch.float32, device=device)
        _SCRATCH[key] = scratch
    pack_dsts = []
    pack_srcs = []
    offset = 0
    for parameter in params:
        grad = parameter.grad
        if grad is None:
            raise RuntimeError(
                f"parameter of shape {tuple(parameter.shape)} finished backward without a gradient"
            )
        count = parameter.numel()
        pack_dsts.append(torch.narrow(scratch, 0, offset, count))
        pack_srcs.append(grad.detach().reshape(-1))
        offset += count
    _copy_many(pack_dsts, pack_srcs)
    scratch.div_(world_size)
    torch.cuda.synchronize(device)
    control = state.get("control")
    if control is not None:
        dist.barrier(group=control)
    dist.all_reduce(scratch, op=dist.ReduceOp.SUM)
    unpack_dsts = []
    unpack_srcs = []
    offset = 0
    for parameter in params:
        grad = parameter.grad
        count = parameter.numel()
        unpack_dsts.append(grad)
        unpack_srcs.append(torch.narrow(scratch, 0, offset, count).view(grad.shape))
        offset += count
    _copy_many(unpack_dsts, unpack_srcs)


def _fused_supported(state):
    device = state.get("device")
    if device is None or device.type != "cuda":
        return False
    world_size = state.get("world_size")
    if not isinstance(world_size, int) or world_size <= 1:
        return False
    model = state.get("model")
    if not hasattr(model, "no_sync"):
        return False
    raw_model = state.get("raw_model")
    if raw_model is None:
        return False
    if not state.get("inputs"):
        return False
    if not hasattr(state.get("optimizer"), "step"):
        return False
    params = [p for p in raw_model.parameters() if p.requires_grad]
    if not params:
        return False
    for parameter in params:
        if parameter.dtype != torch.float32 or parameter.device.type != "cuda":
            return False
    if not (dist.is_available() and dist.is_initialized()):
        return False
    if dist.get_world_size() != world_size:
        return False
    if dist.get_backend() != "nccl":
        return False
    return True


def _build_invoke(parent_run, record_route):
    def invoke(state):
        try:
            supported = _fused_supported(state)
        except Exception:
            supported = False
        if supported is not True:
            record_route(CALL_ID, "invoke:unsupported-state:parent-fallback")
            return parent_run(state)
        record_route(CALL_ID, "invoke:fused-single-allreduce")
        device = state["device"]
        world_size = int(state["world_size"])
        model = state["model"]
        optimizer = state["optimizer"]
        inputs = state["inputs"]
        raw_model = state["raw_model"]
        params = [p for p in raw_model.parameters() if p.requires_grad]
        microbatches = len(inputs)
        state["run_index"] = int(state.get("run_index", 0)) + 1
        losses = []
        with _maybe_phase("train_step"):
            optimizer.zero_grad(set_to_none=True)
            for index, tokens in enumerate(inputs):
                with model.no_sync(), _maybe_phase(f"microbatch_{index}_forward_backward"):
                    loss = model(input_ids=tokens, labels=tokens, use_cache=False).loss / microbatches
                    loss.backward()
                    losses.append(loss.detach())
            accumulated = torch.stack(losses).sum()
            _fused_gradient_synchronization(state, params, device, world_size)
            with _maybe_phase("optimizer_step"):
                optimizer.step()
            if device.type == "cuda":
                torch.cuda.synchronize(device)
        return {"accumulated_loss": accumulated.detach()}

    return invoke


def build(context):
    targets = context.get("targets") or {}
    references = context.get("references") or {}
    record_route = context.get("record_route", lambda *_: None)
    spec = targets.get(CALL_ID)
    if not isinstance(spec, dict):
        return {}
    if spec.get("kind") != "project" or spec.get("target") != "project.invoke":
        return {}
    parent_run = references.get(CALL_ID)
    if not callable(parent_run):
        return {}
    return {CALL_ID: _build_invoke(parent_run, record_route)}
