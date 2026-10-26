"""SIMPLE-protocol gradient allreduce for the two DDP reducer c10d::allreduce_ sites.

Joint replacement for the dispatch sites that jointly cover all 13 per-step
gradient-bucket SUM allreduces on process group 0:

  d0938de87369efd500322679   12 x 7,087,872-element fp32 buckets (28,351,488 B)
  b8ade439d7b8bd5fb5c1029b    1 x 39,385,344-element fp32 bucket (157,541,376 B)

Mechanism: at build time both ranks create one extra NCCL process group over all
ranks with a bound device_id, with NCCL_PROTO=SIMPLE scoped around the group's
communicator initialization (set, forced by a 4-byte warmup allreduce that makes
init unconditional, then restored). The warmed nccl_tuning collector on this
exact GPU pair logs NCCL's own choice of Algo RING proto SIMPLE for precisely
these payload sizes (28,351,488 B and 157,541,376 B), while every in-step kernel
observed for the default communicator is ncclDevKernel_AllReduce_Sum_f32_RING_LL;
the dedicated communicator pins the bandwidth-oriented protocol for these
bandwidth-bound payloads. The warmup call also captures the group's dispatcher
custom-class handle (its boxed form as a torch.classes.c10d.ProcessGroup
ScriptObject) through a transient TorchDispatchMode, so no setup work, extra
dispatches or extra collectives remain inside any invocation.

Each replacement invocation performs exactly ONE c10d::allreduce_ execution: the
parent reference called with the caller's own tensors, ReduceOp, sparse_indices,
async_op and timeout, with only the ProcessGroup argument replaced by the fast
group. Nothing is deferred, fused, grouped, staged or eliminated, and the
returned (tensors, Work) contract, in-place bucket reduction, stream ordering
and gradient_as_bucket_view aliasing are inherited unchanged from the parent op
path. Because the reduction stays a 2-rank SUM over the same world with the
same inputs, every output element is a single commutative fp32 addition
(a[i] + b[i]) regardless of NCCL protocol, so reduced gradients are bitwise
identical to the parent path: the exact cross-rank replica check, the
serial-reference gradient check and the parameter-update check are unaffected.

Fallbacks: signature-incompatible calls (non-fp32, non-contiguous, multi-tensor,
sparse, foreign device), single-rank runs, and any symmetric setup or capture
failure run exact parent passthrough on the caller's own process group, with
distinct route labels retained for diagnosis. Synchronous callers (async_op
false) always take the passthrough, executed as an async parent call whose Work
is waited before returning, because a direct Python op call with async_op=false
does not execute the collective in this torch build.
"""

from __future__ import annotations

import os
from datetime import timedelta

import torch
import torch.distributed as dist

try:  # available in the measured worker; guarded for exotic inspection envs
    from torch.utils._python_dispatch import TorchDispatchMode
except Exception:  # pragma: no cover - CPU inspection only
    TorchDispatchMode = None

FAST_PROTO = "SIMPLE"
GROUP_TIMEOUT_S = 600.0
ALLREDUCE_TARGET = "c10d.allreduce_.default"
KNOWN_CALL_IDS = (
    "d0938de87369efd500322679",
    "b8ade439d7b8bd5fb5c1029b",
)


def _make_capture_mode():
    """Transient dispatch mode recording the fast group's boxed process group."""

    if TorchDispatchMode is None:
        return None

    class _FastGroupCapture(TorchDispatchMode):
        def __init__(self):
            super().__init__()
            self.captured = None

        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            if self.captured is None and func is torch.ops.c10d.allreduce_.default and len(args) > 1:
                self.captured = args[1]
            return func(*args, **(kwargs or {}))

    return _FastGroupCapture()


def _warm_and_capture(group, warm_tensor, opts):
    """One tiny allreduce on the fast group; return its boxed handle or None.

    The pybind ``group.allreduce`` path routes through the c10d::allreduce_
    dispatcher op, which boxes the group as __torch__.torch.classes.c10d.ProcessGroup;
    the transient mode records that box so fast-path calls can invoke the parent
    op with the fast group substituted, without per-invocation dispatch dances.
    """

    mode = _make_capture_mode()
    try:
        if mode is None:
            work = group.allreduce([warm_tensor], opts)
        else:
            with mode:
                work = group.allreduce([warm_tensor], opts)
        if work is not None:
            work.wait()
    except Exception:  # noqa: BLE001 - reported as capture/setup failure by caller
        return None
    return mode.captured if mode is not None else None


def _setup_fast_group(state):
    """Create the SIMPLE-protocol NCCL group and its dispatcher handle.

    Returns (group, handle, reason); raises nothing so every rank keeps a
    coherent fallback category instead of a half-installed group.
    """

    device = state.get("device") if isinstance(state, dict) else None
    try:
        if not (torch.cuda.is_available() and dist.is_available() and dist.is_initialized()):
            raise RuntimeError("setup-failed:cuda-or-distributed-unavailable")
        if not isinstance(device, torch.device) or device.type != "cuda":
            raise RuntimeError("setup-failed:adapter-state-device-unavailable")
        if int(dist.get_world_size()) < 2:
            raise RuntimeError("setup-failed:single-rank-nothing-to-reduce")
        device_index = device.index if device.index is not None else torch.cuda.current_device()

        opts = dist.AllreduceOptions()
        opts.reduceOp = dist.ReduceOp.SUM
        opts.asyncOp = True

        previous = os.environ.get("NCCL_PROTO")
        os.environ["NCCL_PROTO"] = FAST_PROTO
        try:
            group = dist.new_group(
                backend="nccl",
                timeout=timedelta(seconds=GROUP_TIMEOUT_S),
                device_id=torch.device("cuda", device_index),
            )
            # Force communicator initialization and one warm kernel under the
            # forced protocol so later lazy init cannot re-read the restored env;
            # capture the group's dispatcher handle in the same call.
            warm = torch.ones(1, dtype=torch.float32, device=device)
            handle = _warm_and_capture(group, warm, opts)
            torch.cuda.synchronize(device)
        finally:
            if previous is None:
                os.environ.pop("NCCL_PROTO", None)
            else:
                os.environ["NCCL_PROTO"] = previous
        if handle is None:
            return group, None, "parent-passthrough-no-capture"
        return group, handle, "fast-pg-ready"
    except Exception as exc:  # noqa: BLE001 - concise setup failure category
        return None, None, "setup-failed:" + type(exc).__name__


def build(context):
    state = context.get("state") or {}
    references = context.get("references") or {}
    targets = context.get("targets") or {}
    record_route = context.get("record_route", lambda *_: None)

    def route(call_id, label):
        try:
            record_route(call_id, label)
        except Exception:
            pass

    fast_pg, fast_so, reason = _setup_fast_group(state)
    box = {"active": fast_pg is not None and fast_so is not None, "handle": fast_so, "reason": reason}

    device = state.get("device") if isinstance(state, dict) else None
    if isinstance(device, torch.device) and device.index is not None:
        expected_index = device.index
    else:
        expected_index = None

    def make_replacement(call_id):
        parent = references.get(call_id)

        def passthrough(tensors, process_group, reduce_op, sparse_indices, async_op, timeout, **kwargs):
            """Exact parent semantics on the caller's own group.

            Synchronous originals execute as an async parent call whose Work is
            waited before returning: a direct Python op call with async_op=false
            does not run the collective in this torch build.
            """

            if async_op:
                return parent(tensors, process_group, reduce_op, sparse_indices, async_op, timeout, **kwargs)
            result = parent(tensors, process_group, reduce_op, sparse_indices, True, timeout, **kwargs)
            try:
                work = result[1] if isinstance(result, tuple) else result
                if work is not None and hasattr(work, "wait"):
                    work.wait()
            except Exception:  # noqa: BLE001 - result already holds parent contract
                pass
            return result

        def repl(tensors, process_group, reduce_op, sparse_indices=None, async_op=True, timeout=-1, **kwargs):
            if parent is None:
                raise RuntimeError("c10d.allreduce_ parent reference unavailable")

            if not box["active"]:
                route(call_id, box["reason"])
                return passthrough(tensors, process_group, reduce_op, sparse_indices, async_op, timeout, **kwargs)

            try:
                tlist = list(tensors)
            except TypeError:
                route(call_id, "parent-passthrough-signature")
                return passthrough(tensors, process_group, reduce_op, sparse_indices, async_op, timeout, **kwargs)

            tensor = tlist[0] if len(tlist) == 1 else None
            index = tensor.device.index if tensor is not None and tensor.is_cuda else None
            expected = expected_index if expected_index is not None else torch.cuda.current_device()
            compatible = (
                sparse_indices is None
                and bool(async_op)
                and tensor is not None
                and tensor.is_cuda
                and tensor.dtype is torch.float32
                and tensor.is_contiguous()
                and index == expected
            )
            if not compatible:
                route(call_id, "parent-passthrough-signature")
                return passthrough(tensors, process_group, reduce_op, sparse_indices, async_op, timeout, **kwargs)

            route(call_id, "fastpg-simple-allreduce")
            return parent(tlist, box["handle"], reduce_op, sparse_indices, async_op, timeout, **kwargs)

        return repl

    selected = []
    for call_id, spec in (targets or {}).items():
        if isinstance(spec, dict) and spec.get("target") == ALLREDUCE_TARGET and call_id in references:
            selected.append(call_id)
    if not selected:
        selected = [call_id for call_id in KNOWN_CALL_IDS if call_id in references]

    return {call_id: make_replacement(call_id) for call_id in selected}
