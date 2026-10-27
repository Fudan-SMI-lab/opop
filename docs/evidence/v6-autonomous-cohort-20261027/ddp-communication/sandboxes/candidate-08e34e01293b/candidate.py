"""Candidate: rank re-synchronization immediately before the final DDP backward.

Replaces the observed ``project.invoke`` site (call 5d7d6bb13691b02f533f62b0,
group 3c57c8731ec33ed76cbb, phase ``run``) with the adapter's accumulated
training step reproduced verbatim, plus exactly one additional synchronization:

* identical ``optimizer.zero_grad(set_to_none=True)``;
* identical microbatch loop with the parent ``no_sync`` schedule (all but the
  final microbatch inside ``model.no_sync()``);
* identical forward call (``model(input_ids=tokens, labels=tokens,
  use_cache=False).loss / total``), identical ``loss.backward()`` and
  ``losses.append(loss.detach())`` bookkeeping;
* identical ``step_loss = torch.stack(losses).sum()`` inside the
  ``gradient_accumulation`` phase and ``optimizer.step()`` +
  ``torch.cuda.synchronize()`` inside the ``optimizer_step`` phase, using the
  same ``kernel_optimizer.project.discovery.phase`` contexts as the parent;
* identical return structure ``{"loss": step_loss, "microbatch_losses":
  torch.stack(losses)}``.

The single insertion is ``dist.barrier(group=state["control_group"])`` placed
after the final microbatch's forward and immediately before its ``backward()``
-- the last host-side point before the 13 DDP allreduce launches. The barrier
runs on the gloo control group that ``prepare()`` already created and
``before_run()`` already uses, so it launches no NCCL kernel, registers no
communication hook and leaves the reducer, bucket layout, collective count,
payloads, gradients and optimizer state byte-identical to the parent.

Measured motivation (selection de4d95e2, execution_timeline / cross_rank):
the early rank's CPU submits all 13 allreduces 12.5-12.9 ms before the late
rank's; the offset is already established at the first collective (seq 21)
and stays flat through seq 33, i.e. it is accumulated before the final backward
begins, not during the collective window and not wire time. The early rank's
first ``ncclDevKernel_AllReduce`` consequently spins 13.7-14.6 ms in-kernel
while the late rank pays the near-wire floor (35.35-35.41 ms kernel sum vs
~34.4 ms synthetic calibration floor). Resetting the inter-rank submission
offset at the last point before the allreduce launches removes that spin from
whichever rank is early. The previous joint candidate (host-lean bitwise
identical block path) showed that symmetric host speedups do not reduce the
*relative* offset; this candidate removes the offset itself.
"""

from contextlib import contextmanager, nullcontext

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

try:
    from kernel_optimizer.project.discovery import phase as _discovery_phase
except Exception:  # pragma: no cover - worker always provides the package
    _discovery_phase = None

_ROUTE_BARRIER = "distributed:gloo_control_barrier_before_final_backward"
_ROUTE_NO_BARRIER = "fallback:no_control_group_parent_step"
_ROUTE_SETUP_GUARD = "fallback:setup_guard_parent_step"


@contextmanager
def _phase(name):
    if _discovery_phase is None:
        yield
    else:
        with _discovery_phase(name):
            yield


def _make_replacement(call_id, state, parent, record):
    """Build the project.invoke replacement for one declared site.

    Returns a labelled parent passthrough when the build-time structural guard
    fails, so a setup failure stays observable instead of silently running the
    parent everywhere.
    """
    try:
        model = state.get("model") if isinstance(state, dict) else None
        optimizer = state.get("optimizer") if isinstance(state, dict) else None
        inputs = state.get("inputs") if isinstance(state, dict) else None
        ready = (
            callable(model)
            and callable(getattr(optimizer, "step", None))
            and isinstance(inputs, (list, tuple))
            and len(inputs) > 0
        )
    except Exception:
        ready = False

    if not ready:
        def setup_guard_failed(invocation_state, *args, **kwargs):
            record(call_id, _ROUTE_SETUP_GUARD)
            return parent(invocation_state, *args, **kwargs)

        return setup_guard_failed

    def run_resynchronized_step(invocation_state, *args, **kwargs):
        try:
            control = invocation_state.get("control_group")
            world_size = int(invocation_state.get("world_size") or 1)
        except Exception:
            control = None
            world_size = 1
        if (control is None or world_size < 2
                or not (dist.is_available() and dist.is_initialized())):
            record(call_id, _ROUTE_NO_BARRIER)
            return parent(invocation_state, *args, **kwargs)

        model = invocation_state["model"]
        optimizer = invocation_state["optimizer"]
        inputs = invocation_state["inputs"]
        total = len(inputs)
        optimizer.zero_grad(set_to_none=True)
        losses = []
        with _phase("gradient_accumulation"):
            for index, tokens in enumerate(inputs):
                final = index + 1 >= total
                context = model.no_sync() if isinstance(model, DDP) and not final else nullcontext()
                with context:
                    loss = model(input_ids=tokens, labels=tokens, use_cache=False).loss / total
                    if final:
                        # Last host-side point before the 13 DDP allreduce
                        # launches: reset the accumulated inter-rank CPU
                        # submission offset without touching any numeric work.
                        record(call_id, _ROUTE_BARRIER)
                        dist.barrier(group=control)
                    loss.backward()
                    losses.append(loss.detach())
            step_loss = torch.stack(losses).sum()
        with _phase("optimizer_step"):
            optimizer.step()
            torch.cuda.synchronize()
        return {"loss": step_loss, "microbatch_losses": torch.stack(losses)}

    return run_resynchronized_step


def build(context):
    targets = context.get("targets") or {}
    references = context.get("references") or {}
    state = context.get("state") or {}
    record = context.get("record_route") or (lambda *_: None)
    replacements = {}
    for call_id, spec in targets.items():
        try:
            kind = spec.get("kind")
        except Exception:
            continue
        if kind != "project":
            continue
        parent = references.get(call_id)
        if not callable(parent):
            continue
        replacements[call_id] = _make_replacement(call_id, state, parent, record)
    return replacements
