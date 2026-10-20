"""Prefill CUDA-graph execution policy for the frozen Qwen3-4B 'interactive' workload.

The selected site is the prefill-phase call of the root ``Qwen3ForCausalLM``
module (call ``900c23e99c808d2c9bda215a``, target ``model``). The replacement
lazily captures the unchanged parent forward into a CUDA graph during an actual
adapter invocation, then replays it on every later invocation after copying the
current input ids into the static input buffer. The recorded kernel sequence is
exactly the parent's, so outputs remain bitwise identical while the per-kernel
host launch/dispatch gaps inside prefill collapse into a single graph replay.
Decode runs the untouched parent path and consumes the returned cache wrapper
that rebinds the graph's static KV buffers on each invocation.
"""
from __future__ import annotations

import threading

import torch

PREFILL_MODEL_CALL = "900c23e99c808d2c9bda215a"

_WARMUP_FOR_CAPTURE = 2

_ALLOWED_KWARGS = frozenset(
    (
        "input_ids",
        "attention_mask",
        "position_ids",
        "past_key_values",
        "inputs_embeds",
        "labels",
        "use_cache",
        "logits_to_keep",
    )
)


def _profiler_active() -> bool:
    try:
        return bool(torch.autograd._profiler_enabled())
    except Exception:
        return False


class _PrefillGraphEngine:
    """Lazy CUDA-graph wrapper around the parent Qwen3ForCausalLM prefill call."""

    def __init__(self, call_id, parent, record):
        self._call_id = call_id
        self._parent = parent
        self._record = record
        self._lock = threading.Lock()
        self._state = "idle"
        self._graph = None
        self._static_ids = None
        self._output = None
        self._pool_pairs = None
        self._wrapper_layers = None
        self._shape = None

    def execute(self, args, kwargs):
        input_ids = self._compatible_input(args, kwargs)
        if input_ids is None:
            self._record(self._call_id, "fallback_signature")
            return self._parent(*args, **kwargs)
        if self._state == "disabled":
            self._record(self._call_id, "fallback_parent_disabled")
            return self._parent(*args, **kwargs)
        try:
            if self._state != "ready":
                self._setup(input_ids)
                if self._state != "ready":
                    self._record(self._call_id, "passthrough_parent_after_setup")
                    return self._parent(*args, **kwargs)
            self._static_ids.copy_(input_ids)
            self._graph.replay()
            self._rebind_cache()
            self._record(self._call_id, "cuda_graph_replay")
            return self._output
        except Exception:
            self._state = "disabled"
            self._graph = None
            self._output = None
            self._static_ids = None
            self._pool_pairs = None
            self._wrapper_layers = None
            self._record(self._call_id, "runtime_failure_fallback_parent")
            return self._parent(*args, **kwargs)

    def _compatible_input(self, args, kwargs):
        if args or set(kwargs) - _ALLOWED_KWARGS:
            return None
        if (
            kwargs.get("attention_mask") is not None
            or kwargs.get("position_ids") is not None
            or kwargs.get("past_key_values") is not None
            or kwargs.get("inputs_embeds") is not None
            or kwargs.get("labels") is not None
        ):
            return None
        if kwargs.get("use_cache") is not True:
            return None
        logits_to_keep = kwargs.get("logits_to_keep")
        if isinstance(logits_to_keep, bool) or logits_to_keep != 1:
            return None
        input_ids = kwargs.get("input_ids")
        if not isinstance(input_ids, torch.Tensor):
            return None
        if input_ids.device.type != "cuda" or input_ids.dtype != torch.int64:
            return None
        if input_ids.dim() != 2 or input_ids.requires_grad:
            return None
        shape = tuple(input_ids.shape)
        if self._shape is not None and shape != self._shape:
            return None
        return input_ids

    def _setup(self, input_ids):
        with self._lock:
            if self._state in ("ready", "disabled"):
                return
            if _profiler_active():
                self._state = "disabled"
                self._record(self._call_id, "setup_failure_profiler_active")
                return
            try:
                self._capture(input_ids)
            except Exception:
                self._state = "disabled"
                self._graph = None
                self._output = None
                self._static_ids = None
                self._pool_pairs = None
                self._wrapper_layers = None
                self._record(self._call_id, "setup_failure_capture")
                return
            self._state = "ready"
            self._record(self._call_id, "cuda_graph_captured")

    def _capture(self, input_ids):
        device = input_ids.device
        static_ids = input_ids.detach().clone()
        stream = torch.cuda.Stream(device=device)
        current = torch.cuda.current_stream(device)
        stream.wait_stream(current)
        with torch.cuda.stream(stream):
            for _ in range(_WARMUP_FOR_CAPTURE):
                self._parent(input_ids=static_ids, use_cache=True, logits_to_keep=1)
        current.wait_stream(stream)
        torch.cuda.synchronize(device)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            output = self._parent(input_ids=static_ids, use_cache=True, logits_to_keep=1)
        past = output.past_key_values
        if past is None or not getattr(past, "layers", None):
            raise RuntimeError("parent prefill returned no cache layers")
        pool_pairs = tuple((layer.keys, layer.values) for layer in past.layers)
        if any(keys is None or values is None for keys, values in pool_pairs):
            raise RuntimeError("incomplete cache layers after prefill capture")
        self._graph = graph
        self._static_ids = static_ids
        self._shape = tuple(static_ids.shape)
        self._output = output
        self._pool_pairs = pool_pairs
        self._wrap_cache()

    def _wrap_cache(self):
        from transformers.cache_utils import Cache, DynamicCache, DynamicLayer

        wrapper_layers = []
        for keys, values in self._pool_pairs:
            layer = DynamicLayer()
            layer.keys = keys
            layer.values = values
            layer.is_initialized = True
            wrapper_layers.append(layer)
        wrapper = DynamicCache.__new__(DynamicCache)
        Cache.__init__(wrapper, layers=wrapper_layers)
        self._wrapper_layers = wrapper_layers
        self._output.past_key_values = wrapper

    def _rebind_cache(self):
        for layer, (keys, values) in zip(self._wrapper_layers, self._pool_pairs):
            layer.keys = keys
            layer.values = values
            layer.is_initialized = True


def build(context):
    references = context["references"]
    record = context.get("record_route", lambda *_: None)
    parent = references[PREFILL_MODEL_CALL]
    engine = _PrefillGraphEngine(PREFILL_MODEL_CALL, parent, record)

    def prefill_model_forward(*args, **kwargs):
        return engine.execute(args, kwargs)

    return {PREFILL_MODEL_CALL: prefill_model_forward}
