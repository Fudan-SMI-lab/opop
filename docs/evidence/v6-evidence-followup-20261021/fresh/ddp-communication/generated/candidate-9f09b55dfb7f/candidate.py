"""Whole-step CUDA Graph execution plan for the accumulated DDP training step.

Replaces project.invoke. The parent step (optimizer.zero_grad(set_to_none=True),
three no_sync microbatch forward/backward passes, the synchronized final backward
containing the 13 DDP bucket allreduces, the SGD update) runs unchanged through
the parent on the first invocation. During the second invocation - a quality or
warmup invocation that is never part of the timed repeats - the same step is
captured into a CUDA graph on a side stream from the current state, replayed
once, and verified against a freshly executed eager step (parameters, every
gradient, losses). Verified graphs are replayed for every later invocation, so
the full device workload still executes on current inputs each time while the
thousands of per-op host dispatches that caused the measured cross-rank
submission skew disappear. Capture decisions are agreed rank-symmetrically over
the adapter's gloo control group; any failure restores the eager post-step state
and falls back to the unmodified parent, first for a prefix graph (no_sync
microbatches only, eager synchronized final backward), then for plain eager
execution. Storage stability of parameters, inputs and gradient buffers is
guarded by pointer checks on every replay.
"""
from __future__ import annotations

from contextlib import contextmanager, nullcontext

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

try:
    from kernel_optimizer.project.discovery import phase as _phase
except Exception:

    @contextmanager
    def _phase(name):
        yield


CALL_ID = "5d7d6bb13691b02f533f62b0"
CAPTURE_INVOCATION = 2

_LOSS_ATOL = 1e-6
_LOSS_RTOL = 1e-5
_PARAM_ATOL = 1e-6
_PARAM_RTOL = 1e-5
_GRAD_ATOL = 1e-5
_GRAD_RTOL = 1e-4

_BOX = {
    "n": 0,
    "mode": "eager",
    "graph": None,
    "static": None,
    "ptrs": None,
    "reason": "not_attempted",
    "parent": None,
    "record": lambda label: None,
}


def _parent(state):
    parent = _BOX["parent"]
    if parent is None:
        raise RuntimeError("project.invoke parent reference is unavailable")
    return parent(state)


def _named(state):
    return list(state["raw_model"].named_parameters())


def _run_step(state, sink, *, stop_before_final=False):
    model = state["model"]
    optimizer = state["optimizer"]
    inputs = state["inputs"]
    total = len(inputs)
    optimizer.zero_grad(set_to_none=True)
    losses = []
    for index, tokens in enumerate(inputs):
        final = index + 1 == total
        if stop_before_final and final:
            break
        context = model.no_sync() if isinstance(model, DDP) and not final else nullcontext()
        with context:
            with _phase("microbatch_forward"):
                loss = model(input_ids=tokens, labels=tokens, use_cache=False).loss / total
            with _phase("synchronized_backward" if final else "backward"):
                loss.backward()
        losses.append(loss.detach())
    sink.append(losses)


def _finish_step(state):
    with _phase("optimizer_step"):
        state["optimizer"].step()


def _final_microbatch(state):
    model = state["model"]
    inputs = state["inputs"]
    tokens = inputs[-1]
    total = len(inputs)
    with _phase("microbatch_forward"):
        loss = model(input_ids=tokens, labels=tokens, use_cache=False).loss / total
    with _phase("synchronized_backward"):
        loss.backward()
    return loss.detach()


def _warmup_on_side_stream(state):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        _run_step(state, [])
        _finish_step(state)
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()


def _restore_params(named, reference):
    with torch.no_grad():
        for name, parameter in named:
            parameter.copy_(reference[name])


def _restore_grads(named, reference):
    with torch.no_grad():
        for name, parameter in named:
            value = reference[name]
            if parameter.grad is None:
                parameter.grad = value.clone()
            else:
                parameter.grad.copy_(value)


def _pointers(state, named):
    params = tuple(parameter.data_ptr() for _, parameter in named)
    inputs = tuple(tensor.data_ptr() for tensor in state["inputs"])
    grads = tuple(parameter.grad.data_ptr() if parameter.grad is not None else -1 for _, parameter in named)
    return (params, inputs, grads)


def _verify(named, after, grads_reference, losses, eager_losses):
    try:
        for index, value in enumerate(losses):
            if not torch.isfinite(value).all().item():
                return False
            if not value.allclose(eager_losses[index], atol=_LOSS_ATOL, rtol=_LOSS_RTOL):
                return False
        for name, parameter in named:
            if not torch.isfinite(parameter).all().item():
                return False
            if not parameter.allclose(after[name], atol=_PARAM_ATOL, rtol=_PARAM_RTOL):
                return False
            gradient = parameter.grad
            if gradient is None or not gradient.allclose(grads_reference[name], atol=_GRAD_ATOL, rtol=_GRAD_RTOL):
                return False
        return True
    except Exception:
        return False


def _agree(flag, state):
    if not state.get("distributed") or state.get("control") is None:
        return bool(flag)
    try:
        gathered = [None] * int(state["world_size"])
        dist.all_gather_object(gathered, bool(flag), group=state["control"])
        return all(item is True for item in gathered)
    except Exception:
        return False


def _discard(graph):
    try:
        graph.reset()
    except Exception:
        pass


def _outputs(losses):
    stacked = torch.stack(list(losses))
    return {"loss": stacked.sum(), "microbatch_losses": stacked}


def _ineligible(state):
    if not torch.cuda.is_available():
        return "cuda_unavailable"
    device = state.get("device")
    if device is None or device.type != "cuda":
        return "not_cuda_device"
    if not state.get("inputs"):
        return "no_inputs"
    if not hasattr(state.get("model"), "parameters") or not hasattr(state.get("raw_model"), "named_parameters"):
        return "no_model"
    if not hasattr(torch.cuda, "CUDAGraph") or not hasattr(torch.cuda, "graph"):
        return "graph_api_unavailable"
    if state.get("distributed"):
        if tuple(torch.cuda.nccl.version()) < (2, 9, 6):
            return "nccl_capture_unsupported"
    if not _named(state):
        return "no_parameters"
    return ""


def _attempt_tier1(state, named, ctx):
    record = _BOX["record"]
    problem = ""
    graph = None
    static = None
    try:
        _warmup_on_side_stream(state)
        _restore_params(named, ctx["before"])
        graph = torch.cuda.CUDAGraph()
        sink = []
        with torch.cuda.graph(graph):
            _run_step(state, sink)
            _finish_step(state)
        static = list(sink[0])
        if len(static) != len(state["inputs"]):
            raise RuntimeError("captured_loss_count_mismatch")
    except Exception as exc:
        problem = "capture_failed_" + type(exc).__name__
        if graph is not None:
            _discard(graph)
        graph = None
        static = None
    if not _agree(not problem, state):
        _restore_params(named, ctx["after"])
        _restore_grads(named, ctx["grads"])
        return None, problem or "peer_capture_failed"
    problem = ""
    try:
        graph.replay()
        torch.cuda.synchronize()
        if not _verify(named, ctx["after"], ctx["grads"], static, ctx["eager"]):
            raise RuntimeError("verify_mismatch")
    except Exception as exc:
        problem = "verify_failed_" + type(exc).__name__
    if not _agree(not problem, state):
        _discard(graph)
        _restore_params(named, ctx["after"])
        _restore_grads(named, ctx["grads"])
        return None, problem or "peer_verify_failed"
    _BOX.update(mode="tier1", graph=graph, static=static, ptrs=_pointers(state, named), reason="graph_active")
    record("graph_capture_verified_replay")
    return static, ""


def _attempt_tier2(state, named, ctx):
    record = _BOX["record"]
    problem = ""
    graph = None
    prefix = None
    final = None
    try:
        _warmup_on_side_stream(state)
        _restore_params(named, ctx["before"])
        graph = torch.cuda.CUDAGraph()
        sink = []
        with torch.cuda.graph(graph):
            _run_step(state, sink, stop_before_final=True)
        prefix = list(sink[0])
        if len(prefix) != len(state["inputs"]) - 1:
            raise RuntimeError("captured_loss_count_mismatch")
        graph.replay()
        final = _final_microbatch(state)
        _finish_step(state)
        torch.cuda.synchronize()
        if not _verify(named, ctx["after"], ctx["grads"], [*prefix, final], ctx["eager"]):
            raise RuntimeError("verify_mismatch")
    except Exception as exc:
        problem = "prefix_failed_" + type(exc).__name__
        if graph is not None:
            _discard(graph)
        graph = None
        prefix = None
        final = None
    if not _agree(not problem, state):
        _restore_params(named, ctx["after"])
        _restore_grads(named, ctx["grads"])
        return None, problem or "peer_prefix_failed"
    _BOX.update(mode="tier2", graph=graph, static=prefix, ptrs=_pointers(state, named), reason="prefix_graph_active")
    record("prefix_graph_verified_replay")
    return [*prefix, final], ""


def _setup(state):
    record = _BOX["record"]
    named = _named(state)
    problem = _ineligible(state)
    if not _agree(not problem, state):
        _BOX["reason"] = "setup_skipped_" + (problem or "peer_ineligible")
        record(_BOX["reason"])
        return _parent(state)
    before = {name: parameter.detach().clone() for name, parameter in named}
    eager_result = _parent(state)
    eager_loss = eager_result["loss"].detach().clone()
    eager_losses = eager_result["microbatch_losses"].detach().clone()
    ctx = {
        "before": before,
        "after": {},
        "grads": {},
        "eager": [eager_losses[index] for index in range(eager_losses.numel())],
    }
    try:
        ctx["grads"] = {name: parameter.grad.detach().clone() for name, parameter in named}
        ctx["after"] = {name: parameter.detach().clone() for name, parameter in named}
        static, _ = _attempt_tier1(state, named, ctx)
        if static is not None:
            return _outputs(static)
        if len(state["inputs"]) >= 2:
            served, _ = _attempt_tier2(state, named, ctx)
            if served is not None:
                return _outputs(served)
        _BOX["mode"] = "eager"
        _BOX["reason"] = "capture_unavailable"
        record("setup_fallback_eager")
    except Exception as exc:
        if ctx["after"]:
            _restore_params(named, ctx["after"])
        if ctx["grads"]:
            _restore_grads(named, ctx["grads"])
        _BOX["mode"] = "eager"
        _BOX["reason"] = "setup_exception_" + type(exc).__name__
        record(_BOX["reason"])
    return {"loss": eager_loss, "microbatch_losses": eager_losses}


def _serve_tier1(state):
    record = _BOX["record"]
    named = _named(state)
    if _pointers(state, named) != _BOX["ptrs"]:
        _BOX.update(mode="eager", reason="graph_stale_buffers")
        record("graph_stale_buffers")
        return _parent(state)
    try:
        _BOX["graph"].replay()
        torch.cuda.synchronize()
    except Exception as exc:
        _BOX.update(mode="eager", reason="replay_failed_" + type(exc).__name__)
        record(_BOX["reason"])
        return _parent(state)
    record("graph_replay")
    return _outputs(_BOX["static"])


def _serve_tier2(state):
    record = _BOX["record"]
    named = _named(state)
    if _pointers(state, named) != _BOX["ptrs"]:
        _BOX.update(mode="eager", reason="graph_stale_buffers")
        record("graph_stale_buffers")
        return _parent(state)
    try:
        _BOX["graph"].replay()
        final = _final_microbatch(state)
        _finish_step(state)
        torch.cuda.synchronize()
    except Exception as exc:
        _BOX.update(mode="eager", reason="serve_failed_" + type(exc).__name__)
        record(_BOX["reason"])
        return _parent(state)
    record("prefix_replay_eager_final")
    return _outputs([*_BOX["static"], final])


def _replacement(state):
    _BOX["n"] += 1
    invocation = _BOX["n"]
    mode = _BOX["mode"]
    if mode == "tier1":
        return _serve_tier1(state)
    if mode == "tier2":
        return _serve_tier2(state)
    if invocation == CAPTURE_INVOCATION and _BOX["reason"] == "not_attempted":
        return _setup(state)
    record = _BOX["record"]
    record("eager_parent_warmup" if invocation < CAPTURE_INVOCATION else "eager_parent")
    return _parent(state)


def build(context):
    references = context.get("references", {})
    _BOX["parent"] = references.get(CALL_ID)
    raw_record = context.get("record_route")

    def record(label):
        try:
            if callable(raw_record):
                raw_record(CALL_ID, label)
        except Exception:
            pass

    _BOX["record"] = record
    return {CALL_ID: _replacement}


__all__ = ["build"]
