"""Rank alignment before the synchronized final backward, at the project boundary.

Repair of trial-b04d1d014e: its joint functional replacements at the two
c10d::allreduce_ sites measured the alignment mechanism working (objective
median max-rank 63.542 ms vs the 74.676 ms baseline) but breached step_ms
(256.978 ms vs the 138.660 ms limit) because, under the request's frozen
binding_backend=legacy, unscoped functional sites take the global
__torch_dispatch__ path over every operator of the invocation (binding_cost
collector: parent arm 139.58/139.86 ms vs binding-only passthrough
253.08/231.30 ms, ~92-113 ms of pure binding cost; python_runtime pstats:
7563 intercepted ops per step). This candidate keeps the validated mechanism
and eliminates functional dispatch entirely: the single project.invoke call
5d7d6bb13691b02f533f62b0 - the only call parent of all 13 reducer-issued
c10d::allreduce_ dispatches, installed as a direct callable with zero
functional sites - is replaced by a faithful re-execution of adapter.run
(same optimizer.zero_grad(set_to_none=True) inside the timed step, same
no_sync accumulation loop over state['inputs'], same loss scaling by
len(inputs), same phase markers, same SGD step and final
torch.cuda.synchronize, same return structure, recomputed from live state on
every invocation) plus exactly one addition: a single gloo barrier on the
adapter's existing state['control'] group, issued from the main thread
immediately before the final microbatch's loss.backward() - the last safe
host point before the collective burst, with the autograd engine idle, unlike
the failed engine-thread placement. The barrier emits no NCCL kernel and
touches no tensors, so bytes, DDP flags, bucket layout, algorithm, transport,
op order and arithmetic are unchanged and losses, gradients (including the
bit-exact cross-rank replica property) and post-update parameters match the
baseline. The execution_timeline evidence being traded: the leading rank
submits the burst 6.25-7.29 ms (bounds) before the peer and its first NCCL
kernel runs 9.702 ms against the 3.275-3.675 ms peer/synchronized floor
(collective_calibration floor 12x3.593 + 19.642 = ~62.8 ms per rank); the
barrier converts that device spin-wait into a bounded host wait that the
trailing rank - which paces the 132.06 ms baseline max-rank wall - never
pays. Missing control group or unexpected state falls back to the exact
parent adapter.run with a setup-failure label; a failed barrier permanently
disables the alignment and serves the plain parent step semantics.
"""
from __future__ import annotations

from contextlib import contextmanager, nullcontext

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

try:
    from kernel_optimizer.project.discovery import phase
except Exception:

    @contextmanager
    def phase(name):
        yield


PROJECT_CALL_ID = "5d7d6bb13691b02f533f62b0"

_BOX = {
    "parent": None,
    "control": None,
    "distributed": False,
    "disabled": False,
    "reason": "installed",
    "alignments": 0,
    "invocations": 0,
    "record": lambda *_, **__: None,
}


def _barrier_label(box):
    if box["disabled"]:
        return "alignment_disabled_parent_step"
    if not box["distributed"] or box["control"] is None:
        return "setup_no_control_group_parent_step"
    try:
        dist.barrier(group=box["control"])
    except Exception as exc:
        box["disabled"] = True
        box["reason"] = "barrier_failed_" + type(exc).__name__
        return box["reason"]
    box["alignments"] += 1
    return "aligned_final_backward_barrier"


def _project_step(state):
    box = _BOX
    box["invocations"] += 1
    try:
        model = state["model"]
        optimizer = state["optimizer"]
        inputs = state["inputs"]
        device = state["device"]
    except (KeyError, AttributeError, TypeError):
        box["record"](PROJECT_CALL_ID, "unsupported_state_parent_run")
        return box["parent"](state)
    optimizer.zero_grad(set_to_none=True)
    losses = []
    total = len(inputs)
    for index, tokens in enumerate(inputs):
        final = index + 1 == total
        context = model.no_sync() if isinstance(model, DDP) and not final else nullcontext()
        with context:
            with phase("microbatch_forward"):
                loss = model(input_ids=tokens, labels=tokens, use_cache=False).loss / total
            with phase("synchronized_backward" if final else "backward"):
                if final:
                    box["record"](PROJECT_CALL_ID, _barrier_label(box))
                loss.backward()
        losses.append(loss.detach())
    stacked = torch.stack(losses)
    with phase("optimizer_step"):
        optimizer.step()
    if device.type == "cuda":
        torch.cuda.synchronize()
    return {"loss": stacked.sum(), "microbatch_losses": stacked}


def build(context):
    references = context.get("references", {})
    parent = references.get(PROJECT_CALL_ID)
    if not callable(parent):
        raise RuntimeError("project_alignment_replacement_missing_parent_reference")
    state = context.get("state") or {}
    control = state.get("control") if hasattr(state, "get") else None
    distributed = bool(state.get("distributed")) if hasattr(state, "get") else False
    record = context.get("record_route", lambda *_: None)
    _BOX.update(
        parent=parent,
        control=control,
        distributed=bool(distributed) and dist.is_available() and dist.is_initialized(),
        disabled=False,
        reason="installed",
        alignments=0,
        invocations=0,
        record=record if callable(record) else (lambda *_, **__: None),
    )
    return {PROJECT_CALL_ID: _project_step}


__all__ = ["build"]
