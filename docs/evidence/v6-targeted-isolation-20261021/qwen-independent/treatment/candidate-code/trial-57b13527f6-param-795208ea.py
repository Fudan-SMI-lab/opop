"""Candidate for the qwen-ttft region selection (groups 7f49ae8c64c12e322632,
92e2f6fb9310a942405a): lazy CUDA Graph capture/replay of the Qwen3Model
prefill boundary.

Returned replacement: call 2441f10600ba127de5cb753c (model.model, phase
prefill, input_ids [1, 256], past_key_values None, use_cache True). The decode
call site of the same physical module (e7df98b905ed8aa688d5227a) gets no
wrapper at all, so its 31 KV-cached invocations keep exact parent semantics
and dispatch cost.

On the first matching prefill invocation (real adapter context, inference
mode, eval) the replacement warms up and captures the CURRENT parent callable
from context['references'] on a side stream, then replays the graph on every
subsequent matching invocation after copying the live input_ids into the
static input buffer. Every invocation re-executes the full prefill workload
on its actual inputs; no fixture output is cached. Each replay returns a
fresh DynamicCache sharing the graph's static k/v tensors, so downstream
decode appends (torch.cat) behave exactly like the parent's cache. Any
signature/phase/grad mismatch, positional args, or setup failure falls back
to the parent with a concise route label.
"""

from __future__ import annotations

import torch

PREFILL_CALL_ID = "2441f10600ba127de5cb753c"
_EXPECTED_SHAPE = (1, 256)
_WARMUP_ITERATIONS = 3
_NONE_KWARGS = ("attention_mask", "position_ids", "past_key_values", "inputs_embeds")
_ALLOWED_KEYS = frozenset({"input_ids", "use_cache", *_NONE_KWARGS})


def build(context):
    references = context.get("references") or {}
    modules = context.get("modules") or {}
    parent = references.get(PREFILL_CALL_ID)
    module = modules.get(PREFILL_CALL_ID)
    record_route = context.get("record_route") or (lambda *_: None)
    if parent is None or module is None:
        return {}

    from transformers import DynamicCache
    from transformers.cache_utils import Cache as LayeredCache, DynamicLayer
    from transformers.modeling_outputs import BaseModelOutputWithPast

    config = getattr(module, "config", None)
    holder = {}

    def _call_parent(kwargs):
        return parent(**kwargs)

    def _new_cache(num_layers):
        if config is not None:
            cache = DynamicCache(config=config)
            if len(cache.layers) == num_layers:
                return cache
        return LayeredCache(layers=[DynamicLayer() for _ in range(num_layers)])

    def _signature_matches(kwargs):
        input_ids = kwargs.get("input_ids")
        if not isinstance(input_ids, torch.Tensor):
            return False
        if tuple(input_ids.shape) != _EXPECTED_SHAPE:
            return False
        if input_ids.dtype != torch.long or input_ids.device.type != "cuda":
            return False
        if kwargs.get("use_cache") is not True:
            return False
        if any(kwargs.get(key) is not None for key in _NONE_KWARGS):
            return False
        if set(kwargs) - _ALLOWED_KEYS:
            return False
        return True

    def _capture(kwargs):
        static_kwargs = dict(kwargs)
        static_input = kwargs["input_ids"].detach().clone()
        static_kwargs["input_ids"] = static_input

        current = torch.cuda.current_stream()
        side = torch.cuda.Stream()
        side.wait_stream(current)
        with torch.cuda.stream(side):
            for _ in range(_WARMUP_ITERATIONS):
                _call_parent(static_kwargs)
        current.wait_stream(side)
        torch.cuda.synchronize()

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = _call_parent(static_kwargs)

        cache = captured.past_key_values
        if cache is None or captured.last_hidden_state is None:
            raise RuntimeError("parent returned no hidden state or cache")
        static_keys = [layer.keys for layer in cache.layers]
        static_values = [layer.values for layer in cache.layers]
        if not static_keys or any(key is None for key in static_keys):
            raise RuntimeError("parent cache layers incomplete")

        holder["graph"] = graph
        holder["static_input"] = static_input
        holder["static_hidden"] = captured.last_hidden_state
        holder["static_keys"] = static_keys
        holder["static_values"] = static_values
        holder["template"] = {
            key: value for key, value in kwargs.items() if key != "input_ids"
        }
        holder["device"] = static_input.device

    def _replay_result():
        cache = _new_cache(len(holder["static_keys"]))
        for layer, key, value in zip(
            cache.layers, holder["static_keys"], holder["static_values"]
        ):
            layer.keys = key
            layer.values = value
            layer.is_initialized = True
            layer.dtype = key.dtype
            layer.device = key.device
        return BaseModelOutputWithPast(
            last_hidden_state=holder["static_hidden"],
            past_key_values=cache,
        )

    def _execution_context_ok():
        return (
            not torch.is_grad_enabled()
            and torch.is_inference_mode_enabled()
            and not module.training
        )

    def _replay_compatible(kwargs):
        if not _execution_context_ok():
            return False
        if not _signature_matches(kwargs):
            return False
        if kwargs["input_ids"].device != holder["device"]:
            return False
        template = holder["template"]
        if set(kwargs) - {"input_ids"} != set(template):
            return False
        return all(kwargs[key] == value for key, value in template.items())

    def replacement(*args, **kwargs):
        if args:
            record_route(PREFILL_CALL_ID, "parent-fallback:positional-args")
            return parent(*args, **kwargs)
        if "graph" not in holder:
            if holder.get("setup_error"):
                record_route(
                    PREFILL_CALL_ID, "parent-fallback:" + holder["setup_error"]
                )
                return _call_parent(kwargs)
            if not _signature_matches(kwargs):
                record_route(PREFILL_CALL_ID, "parent-fallback:unsupported-signature")
                return _call_parent(kwargs)
            if not _execution_context_ok():
                record_route(PREFILL_CALL_ID, "parent-fallback:execution-context")
                return _call_parent(kwargs)
            try:
                _capture(kwargs)
            except Exception as error:
                holder["setup_error"] = "setup-failure:" + type(error).__name__
                record_route(
                    PREFILL_CALL_ID, "parent-fallback:" + holder["setup_error"]
                )
                return _call_parent(kwargs)
            record_route(PREFILL_CALL_ID, "cuda-graph-capture")
        elif not _replay_compatible(kwargs):
            record_route(PREFILL_CALL_ID, "parent-fallback:unsupported-signature")
            return _call_parent(kwargs)
        holder["static_input"].copy_(kwargs["input_ids"])
        holder["graph"].replay()
        record_route(PREFILL_CALL_ID, "cuda-graph-replay")
        return _replay_result()

    return {PREFILL_CALL_ID: replacement}
