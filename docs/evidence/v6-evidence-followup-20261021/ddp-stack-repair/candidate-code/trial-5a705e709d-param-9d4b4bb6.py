from __future__ import annotations

import os
import threading
from datetime import timedelta

import torch
import torch.distributed as dist

DECLARED_CALLS = ("d0938de87369efd500322679", "b8ade439d7b8bd5fb5c1029b")

_NCHANNELS_CHOICES = (0, 4, 8, 16)
_PROTO_CHOICES = ("default", "Simple", "LL128")
_DEFAULT_NCHANNELS = 8
_DEFAULT_PROTO = "Simple"
_SETUP_TIMEOUT_S = 300.0

_ProcessGroup = dist.distributed_c10d.ProcessGroup
_lock = threading.Lock()
_channel = {"status": "cold", "boxed": None, "group": None, "saved_env": None}


def _parameter_value(parameters, name, legal, default):
    value = parameters.get(name, default) if isinstance(parameters, dict) else default
    try:
        if isinstance(default, int) and not isinstance(default, bool):
            value = int(value)
        else:
            value = str(value)
    except (TypeError, ValueError):
        return default
    return value if value in legal else default


def _tuning(parameters):
    nchannels = _parameter_value(parameters, "nchannels", _NCHANNELS_CHOICES, _DEFAULT_NCHANNELS)
    proto = _parameter_value(parameters, "proto", _PROTO_CHOICES, _DEFAULT_PROTO)
    return nchannels, proto


def _restore_env(saved):
    for key, previous in saved.items():
        if previous is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = previous


def _setup_channel(tensor, nchannels, proto, record, call_id):
    updates = {}
    if nchannels > 0:
        updates["NCCL_NCHANNELS"] = str(int(nchannels))
    if proto != "default":
        updates["NCCL_PROTO"] = str(proto)
    saved = {}
    try:
        if not (dist.is_available() and dist.is_initialized()):
            raise RuntimeError("torch.distributed is not initialized")
        if not tensor.is_cuda:
            raise RuntimeError("bucket tensor is not on a cuda device")
        for key, value in updates.items():
            saved[key] = os.environ.get(key)
            os.environ[key] = value
        index = tensor.device.index
        if index is None:
            index = torch.cuda.current_device()
        device = torch.device("cuda", index)
        group = dist.new_group(backend="nccl", timeout=timedelta(seconds=_SETUP_TIMEOUT_S), device_id=device)
        boxed = _ProcessGroup.boxed(group)
        if _ProcessGroup.unbox(boxed) is not group:
            raise RuntimeError("boxed process group handle failed to round-trip")
        _channel.update(status="ready", boxed=boxed, group=group, saved_env=saved)
        record(call_id, "setup:created_tuned_nccl_group")
        return boxed
    except Exception as exc:
        _channel["status"] = "setup_failed"
        _restore_env(saved)
        record(call_id, "setup:failed:" + type(exc).__name__)
        return None


def _acquire_channel(tensors, nchannels, proto, record, call_id):
    with _lock:
        status = _channel["status"]
        if status == "ready":
            return _channel["boxed"]
        if status != "cold":
            return None
        try:
            if not (isinstance(tensors, (list, tuple)) and tensors and isinstance(tensors[0], torch.Tensor)):
                _channel["status"] = "setup_failed"
                record(call_id, "setup:failed:unsupported_bucket_layout")
                return None
            return _setup_channel(tensors[0], nchannels, proto, record, call_id)
        except Exception as exc:
            _channel["status"] = "setup_failed"
            record(call_id, "setup:failed:" + type(exc).__name__)
            return None


def _release_env_after_use():
    with _lock:
        saved = _channel.get("saved_env")
        if saved:
            _restore_env(saved)
            _channel["saved_env"] = None


def _plausible_bucket(tensors, sparse_indices, extra, kwargs):
    return (
        isinstance(tensors, (list, tuple))
        and len(tensors) == 1
        and isinstance(tensors[0], torch.Tensor)
        and tensors[0].is_cuda
        and tensors[0].dtype == torch.float32
        and tensors[0].numel() > 0
        and sparse_indices is None
        and not extra
        and not kwargs
    )


def _make_replacement(call_id, reference, record, nchannels, proto):
    def bucket_allreduce_(tensors, process_group, reduce_op, sparse_indices=None, *extra, **kwargs):
        if not _plausible_bucket(tensors, sparse_indices, extra, kwargs):
            record(call_id, "fallback:parent:unsupported_signature")
            return reference(tensors, process_group, reduce_op, sparse_indices, *extra, **kwargs)
        boxed = _acquire_channel(tensors, nchannels, proto, record, call_id)
        if boxed is None:
            record(call_id, "fallback:parent:" + _channel["status"])
            return reference(tensors, process_group, reduce_op, sparse_indices, *extra, **kwargs)
        try:
            result = reference(list(tensors), boxed, reduce_op, sparse_indices, *extra, **kwargs)
        except Exception as exc:
            with _lock:
                if _channel["status"] == "ready":
                    _channel["status"] = "runtime_failed"
                saved = _channel.get("saved_env")
                if saved:
                    _restore_env(saved)
                    _channel["saved_env"] = None
            record(call_id, "runtime_failed:" + type(exc).__name__)
            record(call_id, "fallback:parent:runtime_failed")
            return reference(tensors, process_group, reduce_op, sparse_indices, *extra, **kwargs)
        _release_env_after_use()
        record(call_id, "tuned_nccl_group")
        return result

    return bucket_allreduce_


def build(context):
    context = context or {}
    record = context.get("record_route") or (lambda *_: None)
    references = context.get("references") or {}
    nchannels, proto = _tuning(context.get("parameters"))
    replacements = {}
    for call_id in DECLARED_CALLS:
        reference = references.get(call_id)
        if reference is None:
            continue
        replacements[call_id] = _make_replacement(call_id, reference, record, nchannels, proto)
    return replacements
