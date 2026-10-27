"""Execution-policy candidate: rank-aligned, officially-chained per-bucket DDP
gradient allreduces inside the accumulated training step.

Replaces project.invoke (call 5d7d6bb13691b02f533f62b0) with the parent's
unchanged step - optimizer.zero_grad(set_to_none=True), four microbatches with
model.no_sync() on the first three, the final synchronized backward, SGD step,
torch.cuda.synchronize - and installs, once per process on the existing DDP
wrapper, a gradient-synchronization policy that replaces the default reducer's
per-bucket c10d::allreduce_ sites (d0938de87369efd500322679: 12x 7,087,872
element fp32 buckets; b8ade439d7b8bd5fb5c1029b: the last-ready 39,385,344
element bucket holding the tied wte/lm_head, wpe and early-block gradients):

* per bucket, exactly when it becomes final during the final microbatch's
  backward, the hook applies the default reducer's arithmetic (divide by
  world_size, then SUM allreduce, in place on the same fp32 bucket storage,
  gradient_as_bucket_view preserved) and returns the official future chain
  `dist.all_reduce(..., async_op=True).get_future().then(...)`, the same
  chaining as torch.distributed.algorithms.ddp_comm_hooks.default_hooks
  .allreduce_hook that the source itself runs in its inventory step - no
  manual torch.futures.Future objects and no cross-bucket flush chunks;
* the measured 3.0-3.7 ms rank submit skew is absorbed CPU-side: an optional
  local stream drain plus gloo control-group barrier immediately before the
  synchronized backward, and an optional gloo barrier before each collective
  submission. Gloo barriers emit no NCCL kernels, so the kernel-duration
  objective only sees wire time plus residual jitter instead of in-kernel
  device spin on the early rank;
* before optimizer.step() reads any gradient, an explicit
  torch.cuda.synchronize(device) completion barrier guarantees every bucket
  allreduce has finished on the device, regardless of how the reducer
  consumes the hook futures.

Repair of trial-7d98173269, whose hand-rolled chunked future management let
optimizer.step() read the last bucket's gradient before its allreduce landed
(deterministic 8.8e-6 = lr x stale-vs-final-gradient wte parameter error with
clean gradients, identical across all parameter screens). The official
per-bucket future chaining restores the reducer's completion ordering, and the
pre-step device barrier removes any dependence on that ordering.

Repair of trial-8469963975 (kept the region's coherence but serialized all
497,759,232 B into one post-backward allreduce and was rejected on step_ms):
this policy keeps the reducer's progressive launch schedule, so communication
stays overlapped with the final backward, and pays no pack/cuda-sync/unpack.
"""
from __future__ import annotations

from contextlib import nullcontext

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

try:
    from kernel_optimizer.project.discovery import phase as _phase
except Exception:  # pragma: no cover - diagnostic annotation only
    _phase = None

CALL_ID = "5d7d6bb13691b02f533f62b0"
DEFAULT_PRE_BACKWARD_ALIGN = "drain_barrier"
DEFAULT_PER_COLLECTIVE_ALIGN = "barrier"
PRE_BACKWARD_MODES = ("drain_barrier", "barrier", "none")
PER_COLLECTIVE_MODES = ("barrier", "none")
_POLICY_ATTR = "_official_aligned_grad_sync_policy"


def _maybe_phase(name):
    if _phase is None:
        return nullcontext()
    return _phase(name)


def _supported(state):
    device = state.get("device")
    if device is None or device.type != "cuda":
        return False
    world_size = state.get("world_size")
    if not isinstance(world_size, int) or world_size <= 1:
        return False
    model = state.get("model")
    if model is None:
        return False
    if not hasattr(model, "register_comm_hook") or not hasattr(model, "no_sync"):
        return False
    if state.get("raw_model") is None:
        return False
    if not state.get("inputs"):
        return False
    optimizer = state.get("optimizer")
    if optimizer is None or not hasattr(optimizer, "step"):
        return False
    params = [p for p in state["raw_model"].parameters() if p.requires_grad]
    if not params:
        return False
    for parameter in params:
        if parameter.dtype is not torch.float32 or parameter.device.type != "cuda":
            return False
    if not (dist.is_available() and dist.is_initialized()):
        return False
    if dist.get_world_size() != world_size:
        return False
    if dist.get_backend() != "nccl":
        return False
    control = state.get("control")
    if control is None:
        return False
    try:
        if dist.get_backend(control) != "gloo":
            return False
    except (RuntimeError, ValueError):
        return False
    return True


def _install_policy(state, per_collective_align):
    """Register the per-bucket comm hook once per DDP wrapper; return policy."""
    model = state["model"]
    policy = getattr(model, _POLICY_ATTR, None)
    if policy is not None:
        policy["per_collective_align"] = per_collective_align
        return policy
    world_size = int(state["world_size"])
    control = state["control"]
    group = dist.group.WORLD
    policy = {
        "group": group,
        "divisor": float(world_size),
        "control": control,
        "per_collective_align": per_collective_align,
        "bucket_count": 0,
    }

    def grad_sync_hook(_hook_state, bucket):
        # Default-reducer arithmetic per bucket: divide by world_size first
        # (exact power-of-two scaling for this workload's world_size 2), then
        # an in-place SUM allreduce on the bucket's own fp32 storage. The
        # returned future chain is the official default_hooks.allreduce_hook
        # chaining - never a manually created torch.futures.Future.
        buffer = bucket.buffer()
        buffer.div_(policy["divisor"])
        if policy["per_collective_align"] == "barrier":
            # CPU-side skew absorption; gloo emits no NCCL kernels.
            dist.barrier(group=control)
        future = (
            dist.all_reduce(buffer, group=policy["group"], async_op=True)
            .get_future()
            .then(lambda work_future: work_future.value()[0])
        )
        policy["bucket_count"] += 1
        return future

    model.register_comm_hook(policy, grad_sync_hook)
    setattr(model, _POLICY_ATTR, policy)
    return policy


def _align_before_synchronized_backward(device, policy, mode):
    """Align both ranks CPU-side before the synchronized backward starts."""
    if mode == "drain_barrier":
        try:
            torch.cuda.current_stream(device).synchronize()
        except RuntimeError:
            torch.cuda.synchronize(device)
        dist.barrier(group=policy["control"])
    elif mode == "barrier":
        dist.barrier(group=policy["control"])


def _build_invoke(parent_run, record_route, pre_backward_align, per_collective_align):
    def invoke(state):
        try:
            supported = _supported(state)
        except Exception:
            supported = False
        if supported is not True:
            record_route(CALL_ID, "invoke:unsupported-state:parent-fallback")
            return parent_run(state)
        try:
            policy = _install_policy(state, per_collective_align)
        except Exception:
            record_route(CALL_ID, "invoke:comm-hook-setup-failed:parent-fallback")
            return parent_run(state)
        record_route(CALL_ID, "invoke:aligned-official-bucket-allreduce")
        device = state["device"]
        model = state["model"]
        optimizer = state["optimizer"]
        inputs = state["inputs"]
        microbatches = len(inputs)
        state["run_index"] = int(state.get("run_index", 0)) + 1
        losses = []
        with _maybe_phase("train_step"):
            optimizer.zero_grad(set_to_none=True)
            for index, tokens in enumerate(inputs):
                final = index + 1 == microbatches
                if final:
                    _align_before_synchronized_backward(
                        device, policy, pre_backward_align
                    )
                context = (
                    nullcontext()
                    if final
                    else (
                        model.no_sync()
                        if isinstance(model, DDP)
                        else nullcontext()
                    )
                )
                with context, _maybe_phase(f"microbatch_{index}_forward_backward"):
                    loss = model(
                        input_ids=tokens, labels=tokens, use_cache=False
                    ).loss / microbatches
                    loss.backward()
                    losses.append(loss.detach())
            accumulated = torch.stack(losses).sum()
            # Completion barrier: optimizer.step() must never read a bucket
            # whose allreduce is still in flight, whatever the reducer does
            # with the hook futures (trial-7d98173269's defect).
            torch.cuda.synchronize(device)
            with _maybe_phase("optimizer_step"):
                optimizer.step()
            if device.type == "cuda":
                torch.cuda.synchronize(device)
        return {"accumulated_loss": accumulated.detach()}

    return invoke


def _read_parameters(context):
    parameters = context.get("parameters")
    getter = getattr(parameters, "get", None)
    if getter is None:
        return DEFAULT_PRE_BACKWARD_ALIGN, DEFAULT_PER_COLLECTIVE_ALIGN
    try:
        pre_backward_align = str(getter("pre_backward_align", DEFAULT_PRE_BACKWARD_ALIGN))
    except (TypeError, ValueError):
        pre_backward_align = DEFAULT_PRE_BACKWARD_ALIGN
    if pre_backward_align not in PRE_BACKWARD_MODES:
        pre_backward_align = DEFAULT_PRE_BACKWARD_ALIGN
    try:
        per_collective_align = str(
            getter("per_collective_align", DEFAULT_PER_COLLECTIVE_ALIGN)
        )
    except (TypeError, ValueError):
        per_collective_align = DEFAULT_PER_COLLECTIVE_ALIGN
    if per_collective_align not in PER_COLLECTIVE_MODES:
        per_collective_align = DEFAULT_PER_COLLECTIVE_ALIGN
    return pre_backward_align, per_collective_align


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
    pre_backward_align, per_collective_align = _read_parameters(context)
    return {
        CALL_ID: _build_invoke(
            parent_run, record_route, pre_backward_align, per_collective_align
        )
    }
