"""Whole-region CUDA-Graph capture-once/replay replacement for the frozen
ResNet-18 classifier forward (group eca3643289778aea5af3, module call
f2cf62caf57ad09f5d2f36aa = models(state)['model'] under phase('forward')).

The first real invocation lazily captures the accepted parent forward taken
from context['references'] inside the live dispatch context, on a
non-default stream after side-stream eager warmups, then replays the graph
for that invocation; every subsequent invocation replays it. Replay
re-executes the identical captured kernels against the current contents of
the persistent pixels storage, returns a fresh logits clone in a fresh
output object, and never caches fixture outputs. Any non-matching call
routes to the eager parent with distinct labels: 'cuda_graph_capture',
'cuda_graph_replay', 'parent_signature_or_state_fallback',
'setup_failure_parent_forward', 'replay_runtime_failure_parent_forward'.
"""

from __future__ import annotations

import torch

CALL_ID = "f2cf62caf57ad09f5d2f36aa"

_ROUTE_CAPTURE = "cuda_graph_capture"
_ROUTE_REPLAY = "cuda_graph_replay"
_ROUTE_FALLBACK = "parent_signature_or_state_fallback"
_ROUTE_SETUP_FAILURE = "setup_failure_parent_forward"
_ROUTE_RUNTIME_FAILURE = "replay_runtime_failure_parent_forward"

_FP32 = torch.float32
_WARMUP_CALLS = 3


def _identity_records(model):
    identity = []
    training_sensitive = []
    for path, mod in model.named_modules():
        for name, ref in mod._parameters.items():
            identity.append((mod._parameters, name, ref))
        if mod._buffers:
            training_sensitive.append(mod)
            for name, ref in mod._buffers.items():
                identity.append((mod._buffers, name, ref))
        if path:
            parent_path, _, attr = path.rpartition(".")
            parent = model.get_submodule(parent_path) if parent_path else model
            identity.append((parent._modules, attr, mod))
    return tuple(identity), tuple(training_sensitive)


def _match_input(args, kwargs):
    n = len(args)
    keys = set(kwargs)
    optional = ("labels", "output_hidden_states", "return_dict")
    if n == 1 and not keys:
        return args[0], True
    if n == 0 and keys == {"pixel_values"}:
        return kwargs["pixel_values"], True
    if (
        n == 0
        and "pixel_values" in keys
        and keys <= {"pixel_values", *optional}
        and all(kwargs.get(k) is None for k in optional)
    ):
        return kwargs["pixel_values"], True
    return None, False


def build(context):
    record_route = context.get("record_route", lambda *_: None)
    references = context.get("references") or {}
    modules = context.get("modules") or {}
    parent = references.get(CALL_ID)
    model = modules.get(CALL_ID)

    if not callable(parent):
        return {}

    if not isinstance(model, torch.nn.Module):
        def _setup_failure_forward(*args, **kwargs):
            record_route(CALL_ID, _ROUTE_SETUP_FAILURE)
            return parent(*args, **kwargs)

        return {CALL_ID: _setup_failure_forward}

    identity, training_sensitive = _identity_records(model)

    box = {
        "graph": None,
        "logits": None,
        "out_cls": None,
        "ptr": 0,
        "shape": None,
        "device": None,
        "failed": False,
    }

    def _state_ok():
        if model.training:
            return False
        for mod in training_sensitive:
            if mod.training:
                return False
        for d, k, r in identity:
            if d.get(k) is not r:
                return False
        return True

    def _input_ok(pv):
        return (
            torch.is_tensor(pv)
            and pv.dtype is _FP32
            and pv.is_cuda
            and pv.device == box["device"]
            and tuple(pv.shape) == box["shape"]
            and pv.data_ptr() == box["ptr"]
        )

    def _capture(args, kwargs, pv):
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(_WARMUP_CALLS):
                parent(*args, **kwargs)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = parent(*args, **kwargs)
        logits = out.logits
        out_cls = out.__class__
        out_cls(loss=None, logits=logits, hidden_states=None)
        box["graph"] = graph
        box["logits"] = logits
        box["out_cls"] = out_cls
        box["ptr"] = pv.data_ptr()
        box["shape"] = tuple(pv.shape)
        box["device"] = pv.device

    def _forward(*args, **kwargs):
        pv, matched = _match_input(args, kwargs)
        if not matched:
            record_route(CALL_ID, _ROUTE_FALLBACK)
            return parent(*args, **kwargs)
        if box["failed"]:
            record_route(CALL_ID, _ROUTE_SETUP_FAILURE)
            return parent(*args, **kwargs)
        graph = box["graph"]
        if graph is None:
            if not (
                torch.is_tensor(pv)
                and pv.dim() == 4
                and pv.is_cuda
                and pv.dtype is _FP32
                and not model.training
            ):
                record_route(CALL_ID, _ROUTE_FALLBACK)
                return parent(*args, **kwargs)
            try:
                _capture(args, kwargs, pv)
                graph = box["graph"]
            except Exception:
                box["failed"] = True
                record_route(CALL_ID, _ROUTE_SETUP_FAILURE)
                return parent(*args, **kwargs)
            record_route(CALL_ID, _ROUTE_CAPTURE)
        elif not (_input_ok(pv) and _state_ok()):
            record_route(CALL_ID, _ROUTE_FALLBACK)
            return parent(*args, **kwargs)
        else:
            record_route(CALL_ID, _ROUTE_REPLAY)
        try:
            graph.replay()
            return box["out_cls"](
                loss=None, logits=box["logits"].clone(), hidden_states=None
            )
        except Exception:
            box["failed"] = True
            record_route(CALL_ID, _ROUTE_RUNTIME_FAILURE)
            return parent(*args, **kwargs)

    return {CALL_ID: _forward}
