"""Execution-policy candidate: rank-aligned progressive DDP bucket allreduces.

Replaces project.invoke with the accumulated DDP training step of the parent
(zero_grad, four microbatches with no_sync on the first three, the final
synchronized backward, SGD step, cuda synchronize) and installs, once per
process on the existing DDP wrapper, a gradient-synchronization policy that
replaces the default reducer's per-bucket c10d::allreduce_ sites:

* the DDP comm hook fires progressively, exactly when each 25 MiB gradient
  bucket becomes final during the final microbatch's backward, preserving the
  baseline's measured overlap of communication under remaining backward
  compute (first collective launch a few ms into that backward);
* ready buckets are flushed in small fused chunks: one ProcessGroupNCCL
  grouped allreduce over the chunk's bucket buffers, in place, on the same
  fp32 bucket storage (gradient_as_bucket_view), with the default reducer's
  divide-by-world_size applied first - element-wise the same SUM of
  commutative fp32 adds, so synchronized gradients are bitwise identical to
  the baseline for world_size 2 regardless of chunk fusion;
* before each NCCL submission (and once before the synchronized backward,
  after draining the local stream), ranks align on the existing gloo control
  group, which blocks CPU-side only and emits no NCCL kernels, so the
  measured 3-4 ms rank-arrival skew is absorbed as CPU wait on the early
  rank instead of as in-kernel device spin.

The two observed allreduce sites (d0938de87369efd500322679,
b8ade439d7b8bd5fb5c1029b) are eliminated; their consumers (optimizer.step and
the harness gradient checks) read p.grad, which now receives bit-identical
synchronized values from this policy.

Repair of trial-8469963975, which reached the communication floor with one
fully serialized post-backward allreduce but exceeded the step_ms constraint
by giving up the reducer's overlap. This candidate keeps the parent's launch
schedule and alignment cost CPU-side, restoring that overlap.
"""
from __future__ import annotations

from contextlib import nullcontext

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

try:
    from kernel_optimizer.project.discovery import phase as _phase
except Exception:
    _phase = None

CALL_ID = "5d7d6bb13691b02f533f62b0"
SMALL_BUCKET_ELEMENTS = 7_087_872
DEFAULT_BUCKETS_PER_FLUSH = 4
DEFAULT_FLUSH_ALIGNMENT = "barrier"
_POLICY_ATTR = "_aligned_grad_sync_policy"


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


def _install_policy(state, buckets_per_flush, flush_alignment):
    """Register the comm hook once per DDP wrapper; return its policy."""
    model = state["model"]
    policy = getattr(model, _POLICY_ATTR, None)
    if policy is not None:
        return policy
    options = dist.AllreduceOptions()
    options.reduceOp = dist.ReduceOp.SUM
    options.asyncOp = True
    policy = {
        "group": dist.group.WORLD,
        "options": options,
        "divisor": float(int(state["world_size"])),
        "control": state["control"],
        "flush_target_elements": max(1, int(buckets_per_flush)) * SMALL_BUCKET_ELEMENTS,
        "flush_alignment": str(flush_alignment),
        "pending": [],
        "futures": [],
        "pending_elements": 0,
        "seen_last": False,
    }

    def flush():
        buffers = policy["pending"]
        futures = policy["futures"]
        policy["pending"] = []
        policy["futures"] = []
        policy["pending_elements"] = 0
        if not buffers:
            return
        try:
            divisor = policy["divisor"]
            try:
                torch._foreach_div_(buffers, divisor)
            except (AttributeError, RuntimeError, TypeError, ValueError):
                for buffer in buffers:
                    buffer.div_(divisor)
            if policy["flush_alignment"] == "barrier" and policy["control"] is not None:
                dist.barrier(group=policy["control"])
            work = policy["group"].allreduce(buffers, policy["options"])
            work_future = work.get_future()
        except BaseException as exc:
            for future in futures:
                if not future.done():
                    future.set_exception(exc)
            raise

        def finish(work_future, buffers=buffers, futures=futures):
            try:
                values = work_future.value()
            except BaseException as exc:
                for future in futures:
                    if not future.done():
                        future.set_exception(exc)
                raise
            for future, value in zip(futures, values):
                if not future.done():
                    future.set_result(value)
            return buffers[0]

        work_future.then(finish)

    def grad_sync_hook(_hook_state, bucket):
        buffer = bucket.buffer()
        future = torch.futures.Future()
        policy["pending"].append(buffer)
        policy["futures"].append(future)
        policy["pending_elements"] += buffer.numel()
        try:
            if bucket.is_last():
                policy["seen_last"] = True
            if (
                policy["seen_last"]
                or policy["pending_elements"] >= policy["flush_target_elements"]
            ):
                flush()
        except BaseException as exc:
            policy["seen_last"] = True
            stranded = policy["futures"]
            policy["pending"] = []
            policy["futures"] = []
            policy["pending_elements"] = 0
            for future in stranded:
                if not future.done():
                    future.set_exception(exc)
            raise
        return future

    model.register_comm_hook(None, grad_sync_hook)
    setattr(model, _POLICY_ATTR, policy)
    return policy


def _align_before_synchronized_backward(device, policy):
    """Drain the local stream, then align ranks CPU-side on the gloo group.

    Both ranks enter the synchronized backward with an empty local queue at
    the same wall time, so the progressive chunk allreduces below start
    back-to-back on both GPUs instead of spinning on rank-arrival skew.
    The gloo barrier emits no NCCL kernels.
    """
    control = policy.get("control")
    if control is None:
        return
    try:
        torch.cuda.current_stream(device).synchronize()
    except RuntimeError:
        torch.cuda.synchronize(device)
    dist.barrier(group=control)


def _build_invoke(parent_run, record_route, buckets_per_flush, flush_alignment):
    def invoke(state):
        try:
            supported = _supported(state)
        except Exception:
            supported = False
        if supported is not True:
            record_route(CALL_ID, "invoke:unsupported-state:parent-fallback")
            return parent_run(state)
        try:
            policy = _install_policy(state, buckets_per_flush, flush_alignment)
        except Exception:
            record_route(CALL_ID, "invoke:comm-hook-setup-failed:parent-fallback")
            return parent_run(state)
        record_route(CALL_ID, "invoke:aligned-progressive-bucket-allreduce")
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
                    _align_before_synchronized_backward(device, policy)
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
        return DEFAULT_BUCKETS_PER_FLUSH, DEFAULT_FLUSH_ALIGNMENT
    try:
        buckets_per_flush = int(
            getter("buckets_per_flush", DEFAULT_BUCKETS_PER_FLUSH)
        )
    except (TypeError, ValueError):
        buckets_per_flush = DEFAULT_BUCKETS_PER_FLUSH
    buckets_per_flush = max(1, buckets_per_flush)
    try:
        flush_alignment = str(getter("flush_alignment", DEFAULT_FLUSH_ALIGNMENT))
    except (TypeError, ValueError):
        flush_alignment = DEFAULT_FLUSH_ALIGNMENT
    if flush_alignment not in ("barrier", "none"):
        flush_alignment = DEFAULT_FLUSH_ALIGNMENT
    return buckets_per_flush, flush_alignment


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
    buckets_per_flush, flush_alignment = _read_parameters(context)
    return {
        CALL_ID: _build_invoke(
            parent_run, record_route, buckets_per_flush, flush_alignment
        )
    }
