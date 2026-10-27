"""Candidate: route the DDP gradient allreduces onto a candidate-created NCCL
process group with more channels, preserving the accepted control barrier.

Replaces the observed ``project.invoke`` site (call 5d7d6bb13691b02f533f62b0,
group 3c57c8731ec33ed76cbb, phase ``run``) with the accumulated training step
reproduced verbatim from ``adapter.run``, including the accepted bundle's one
insertion: a gloo control-group barrier after the final microbatch's forward
and immediately before its backward (the last host-side point before the 13
DDP allreduce launches).  This candidate explicitly replaces the installed
bundle ``trial-ee9a17b054-param-286bf06b`` and preserves its behavior itself.

The optimization is communicator placement.  The default process group's NCCL
communicator was created in ``prepare()`` (DDP construction broadcast), before
``build()``, so channel/protocol environment variables can only act on a NEW
communicator.  At build time this candidate:

* saves the current ``NCCL_MAX_CHANNELS`` / ``NCCL_PROTO`` environment values,
* raises them for this worker process only (``nccl_max_channels`` from the
  parameter space, default 8; ``nccl_proto`` default ``LL`` to retain the
  measured LL advantage of ~15.2 GB/s over default SIMPLE's ~14.3 GB/s at the
  28 MB bucket size),
* creates a new nccl process group with eager ``device_id`` initialization and
  warms it with one tiny allreduce plus a device synchronize, so the
  communicator and its channels exist under those settings,
* restores the previous environment exactly, leaving the default group's
  communicator untouched,
* wires the new group into DDP via
  ``model.register_comm_hook(new_group, default_hooks.allreduce_hook)`` --
  torch's documented behavior-preserving div-then-SUM hook, the same hook the
  source's own bucket-inventory diagnostic uses.

Every invocation then runs the parent's exact step (identical
``optimizer.zero_grad(set_to_none=True)``, identical ``no_sync`` schedule,
identical forward/backward/loss bookkeeping, identical ``optimizer.step()`` +
``torch.cuda.synchronize()`` and the same discovery phase contexts), so the
13 bucket allreduces (12 x 28,351,488 B + 1 x 157,541,376 B, fp32 SUM) still
execute with identical bucket timing, payloads, gradients and parameter
updates -- only on the new communicator's channels.
"""

import os
from contextlib import contextmanager, nullcontext
from datetime import timedelta

import torch
import torch.distributed as dist
from torch.distributed.algorithms.ddp_comm_hooks import default_hooks as _ddp_default_hooks
from torch.nn.parallel import DistributedDataParallel as DDP

try:
    from kernel_optimizer.project.discovery import phase as _discovery_phase
except Exception:  # pragma: no cover - worker always provides the package
    _discovery_phase = None

_ROUTE_HOOK_STEP = "distributed:new_nccl_pg_comm_hook_step"
_ROUTE_SETUP_GUARD = "fallback:setup_guard_parent_step"
_ROUTE_PG_SETUP = "fallback:pg_setup_failed_parent_step"

_PG_TIMEOUT = timedelta(seconds=300)


@contextmanager
def _phase(name):
    if _discovery_phase is None:
        yield
    else:
        with _discovery_phase(name):
            yield


def _restore_env(saved):
    for name, value in saved.items():
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value


def _create_hook_process_group(state, channels, proto):
    """Create and warm a new nccl group while channel/protocol env is raised.

    The environment is restored before returning, so only communicators
    initialized inside this window (the new group, eagerly initialized via
    ``device_id`` and materialized by the warmup allreduce) observe the
    raised values.
    """
    device = state.get("device")
    if not isinstance(device, torch.device):
        device = torch.device("cuda", torch.cuda.current_device())
    changes = {"NCCL_MAX_CHANNELS": str(int(channels))}
    if proto and str(proto).lower() != "auto":
        changes["NCCL_PROTO"] = str(proto)
    saved = {name: os.environ.get(name) for name in changes}
    for name, value in changes.items():
        os.environ[name] = value
    try:
        group = dist.new_group(backend="nccl", timeout=_PG_TIMEOUT, device_id=device)
        warm = torch.zeros(1, device=device)
        dist.all_reduce(warm, group=group)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        if not bool(torch.isfinite(warm).all()):
            raise RuntimeError("warmup allreduce on the new process group was not finite")
        del warm
    finally:
        _restore_env(saved)
    return group


def _make_replacement(call_id, state, parent, record, channels, proto):
    """Build the project.invoke replacement for one declared site.

    Structural setup failure and new-group/hook setup failure stay observable
    through distinct fallback labels instead of silently running the parent
    everywhere.
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

    group = None
    if (
        dist.is_available()
        and dist.is_initialized()
        and int(state.get("world_size") or 1) >= 2
        and isinstance(model, DDP)
    ):
        try:
            group = _create_hook_process_group(state, channels, proto)
            # Documented behavior-preserving wiring: the process group is the
            # hook state, exactly as in torch's register_comm_hook example.
            model.register_comm_hook(group, _ddp_default_hooks.allreduce_hook)
        except Exception:
            group = None

    if group is None:
        def no_new_group_parent_step(invocation_state, *args, **kwargs):
            record(call_id, _ROUTE_PG_SETUP)
            return parent(invocation_state, *args, **kwargs)

        return no_new_group_parent_step

    def run_step_on_new_group(invocation_state, *args, **kwargs):
        record(call_id, _ROUTE_HOOK_STEP)
        model = invocation_state["model"]
        optimizer = invocation_state["optimizer"]
        inputs = invocation_state["inputs"]
        control = invocation_state.get("control_group")
        total = len(inputs)
        optimizer.zero_grad(set_to_none=True)
        losses = []
        with _phase("gradient_accumulation"):
            for index, tokens in enumerate(inputs):
                final = index + 1 >= total
                context = model.no_sync() if isinstance(model, DDP) and not final else nullcontext()
                with context:
                    loss = model(input_ids=tokens, labels=tokens, use_cache=False).loss / total
                    if final and control is not None and dist.is_available() and dist.is_initialized():
                        # Preserved accepted-bundle behavior: reset the
                        # inter-rank CPU submission offset at the last
                        # host-side point before the 13 DDP allreduce launches.
                        dist.barrier(group=control)
                    loss.backward()
                    losses.append(loss.detach())
            step_loss = torch.stack(losses).sum()
        with _phase("optimizer_step"):
            optimizer.step()
            torch.cuda.synchronize()
        return {"loss": step_loss, "microbatch_losses": torch.stack(losses)}

    return run_step_on_new_group


def build(context):
    targets = context.get("targets") or {}
    references = context.get("references") or {}
    state = context.get("state") or {}
    record = context.get("record_route", lambda *_: None)
    parameters = context.get("parameters") or {}
    try:
        channels = int(parameters.get("nccl_max_channels", 8))
    except Exception:
        channels = 8
    try:
        proto = str(parameters.get("nccl_proto", "LL"))
    except Exception:
        proto = "LL"

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
        replacements[call_id] = _make_replacement(call_id, state, parent, record, channels, proto)
    return replacements
