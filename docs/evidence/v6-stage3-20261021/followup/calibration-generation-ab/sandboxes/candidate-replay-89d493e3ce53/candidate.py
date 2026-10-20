from __future__ import annotations

import itertools
import threading
from datetime import timedelta

import torch
import torch.distributed as dist

CALL_BUCKETS = "d0938de87369efd500322679"
CALL_EMBEDDING = "b8ade439d7b8bd5fb5c1029b"

_DEFAULT_EXPECTED = {
    CALL_BUCKETS: {7087872},
    CALL_EMBEDDING: {39385344},
}

_EVENT_LOCK = threading.Lock()
_EVENT_COUNTER = itertools.count(1)

_GROUP_LOCK = threading.Lock()
_GROUP_STATE = {"tried": False, "group": None}


def _expected_numel(targets, call_id):
    sizes = set()
    spec = targets.get(call_id)
    if isinstance(spec, dict):
        for signature in spec.get("signatures") or []:
            args = (signature or {}).get("args") or []
            if args and isinstance(args[0], (list, tuple)) and args[0]:
                head = args[0][0]
                if isinstance(head, dict):
                    shape = head.get("shape")
                    if shape and len(shape) == 1:
                        try:
                            sizes.add(int(shape[0]))
                        except (TypeError, ValueError):
                            continue
    return sizes


def _tensor_list(args, kwargs):
    if args:
        return args[0]
    return kwargs.get("tensors")


def _sparse_indices_is_none(args, kwargs):
    if "sparse_indices" in kwargs:
        return kwargs["sparse_indices"] is None
    if len(args) >= 4:
        return args[3] is None
    return True


def _signature_matches(tensors, expected_sizes):
    try:
        if not isinstance(tensors, (list, tuple)) or len(tensors) != 1:
            return False
        tensor = tensors[0]
        if not isinstance(tensor, torch.Tensor):
            return False
        if tensor.dtype is not torch.float32 or not tensor.is_cuda:
            return False
        if tensor.requires_grad or tensor.dim() != 1:
            return False
        if tuple(tensor.stride()) != (1,):
            return False
        return tensor.numel() in expected_sizes
    except Exception:
        return False


def _next_event_index():
    with _EVENT_LOCK:
        return next(_EVENT_COUNTER)


def _rendezvous_group(state):
    control = state.get("control") if isinstance(state, dict) else None
    if control is not None:
        return control
    with _GROUP_LOCK:
        if not _GROUP_STATE["tried"]:
            _GROUP_STATE["tried"] = True
            try:
                if dist.is_available() and dist.is_initialized():
                    _GROUP_STATE["group"] = dist.new_group(backend="gloo", timeout=timedelta(seconds=120))
            except Exception:
                _GROUP_STATE["group"] = None
        return _GROUP_STATE["group"]


def build(context):
    state = context.get("state") or {}
    targets = context.get("targets") or {}
    references = context.get("references") or {}
    record_route = context.get("record_route", lambda *_: None)
    parameters = context.get("parameters") or {}
    try:
        stride = int(parameters.get("rendezvous_stride", 1))
    except (TypeError, ValueError):
        stride = 1
    if stride not in (1, 2, 3):
        stride = 1

    replacements = {}
    for call_id in (CALL_BUCKETS, CALL_EMBEDDING):
        parent = references.get(call_id)
        if parent is None:
            continue
        expected_sizes = _expected_numel(targets, call_id) or _DEFAULT_EXPECTED.get(call_id, set())

        def replacement(*args, _parent=parent, _call_id=call_id, _expected=expected_sizes, **kwargs):
            tensors = _tensor_list(args, kwargs)
            if not _signature_matches(tensors, _expected) or not _sparse_indices_is_none(args, kwargs):
                record_route(_call_id, "fallback:unsupported_signature")
                return _parent(*args, **kwargs)
            if _next_event_index() % stride != 0:
                record_route(_call_id, "parent_allreduce_stride_skip")
                return _parent(*args, **kwargs)
            group = _rendezvous_group(state)
            if group is None:
                record_route(_call_id, "fallback:setup_no_rendezvous_group")
                return _parent(*args, **kwargs)
            dist.barrier(group=group, async_op=False)
            record_route(_call_id, "gloo_rendezvous_then_parent_allreduce")
            return _parent(*args, **kwargs)

        replacements[call_id] = replacement
    return replacements
