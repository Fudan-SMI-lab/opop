"""TTFT candidate: CUDA Graph replay of the prefill-phase Qwen3-4B forward.

Selected region (inputs/selection.json, scope "region"): the prefill subtree of
transformers.models.qwen3.modeling_qwen3.Qwen3ForCausalLM.  The replacement is
installed only at the region root, the prefill-phase module boundary of target
"qwen3_4b" (call id resolved from context["targets"]); every inner prefill call
site of the region is subsumed by the replayed graph.

Per-invocation route on the graph path:
  1. copy the current input_ids into a small static buffer (2 KB),
  2. replay the CUDA graph captured from the accepted parent forward
     (context["references"][call_id]) - all prefill kernels re-execute on the
     current input contents, nothing is precomputed,
  3. clone the static logits and the static per-layer K/V tensors into fresh
     per-invocation tensors and wrap them in a new DynamicCache so the decode
     steps (which keep running the unmodified parent) consume a cache object
     with parent semantics and no cross-invocation aliasing.

Lazy setup on the first eligible invocation (inside the adapter's real prefill
phase, so the accepted parent semantics are what gets captured): three eager
parent warmup runs on a side stream, graph capture, one validation replay and a
bitwise comparison of the logits and every layer's K/V against the eager warmup
output.  Signature mismatches, grad-enabled or foreign-capture contexts, setup
failures and runtime failures each fall back to the parent with a distinct
route label.
"""

from __future__ import annotations

import torch
from transformers.cache_utils import DynamicCache, DynamicLayer
from transformers.modeling_outputs import CausalLMOutputWithPast

_WARMUP_ITERS = 3
_MAX_GRAPHS = 4

_LABEL_REPLAY = "cuda_graph_replay"
_LABEL_CAPTURE = "cuda_graph_capture"
_LABEL_SETUP_FAILURE = "parent_fallback_setup_failure"
_LABEL_SIGNATURE = "parent_fallback_signature"
_LABEL_CONTEXT = "parent_fallback_context"
_LABEL_RUNTIME = "parent_fallback_runtime_failure"
_LABEL_LIMIT = "parent_fallback_graph_limit"


def _find_region_root(targets):
    """Return the prefill-phase Qwen3ForCausalLM module call id, if observed."""
    fallback = None
    for call_id, spec in targets.items():
        if not isinstance(spec, dict):
            continue
        if spec.get("kind") != "module" or spec.get("phase") != "prefill":
            continue
        semantics = str(spec.get("semantics") or "")
        if semantics.endswith("Qwen3ForCausalLM"):
            return call_id
        if fallback is None and "ForCausalLM" in semantics:
            fallback = call_id
    return fallback


def build(context):
    targets = context.get("targets") or {}
    references = context.get("references") or {}
    modules = context.get("modules") or {}
    record_route = context.get("record_route")
    if not callable(record_route):
        record_route = lambda *_: None

    call_id = _find_region_root(targets)
    if call_id is None or call_id not in references:
        return {}
    parent = references[call_id]
    module = modules.get(call_id)
    config = getattr(module, "config", None)

    state = {
        "graphs": {},
        "setup_failed": False,
        "failure": None,
        "foreach_ok": hasattr(torch, "_foreach_copy_"),
    }

    def _new_cache(keys, values):
        cache = None
        layers = []
        if config is not None:
            try:
                candidate_cache = DynamicCache(config=config)
                if len(candidate_cache.layers) == len(keys):
                    cache = candidate_cache
                    layers = candidate_cache.layers
            except Exception:
                cache = None
        if cache is None:
            cache = DynamicCache()
            for _ in keys:
                layers.append(DynamicLayer())
            cache.layers.extend(layers)
        for layer, key, value in zip(layers, keys, values):
            layer.keys = key
            layer.values = value
            layer.is_initialized = True
            layer.dtype = key.dtype
            layer.device = key.device
        return cache

    def _setup(input_ids):
        device = input_ids.device
        static_input = torch.empty_like(input_ids)
        side_stream = torch.cuda.Stream(device=device)
        side_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side_stream):
            static_input.copy_(input_ids)
            warm = None
            for _ in range(_WARMUP_ITERS):
                warm = parent(input_ids=static_input, use_cache=True, logits_to_keep=1)
        torch.cuda.current_stream().wait_stream(side_stream)

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = parent(input_ids=static_input, use_cache=True, logits_to_keep=1)
        static_logits = captured.logits
        captured_cache = captured.past_key_values
        if static_logits is None or captured_cache is None:
            raise RuntimeError("parent forward returned no logits or no past_key_values")
        static_keys = [layer.keys for layer in captured_cache.layers]
        static_values = [layer.values for layer in captured_cache.layers]
        if not static_keys or any(t is None for t in static_keys + static_values):
            raise RuntimeError("captured cache layers are incomplete")

        graph.replay()
        torch.cuda.synchronize()
        if not torch.equal(static_logits, warm.logits):
            raise RuntimeError("replay logits differ bitwise from eager parent output")
        for static_key, static_value, warm_layer in zip(static_keys, static_values, warm.past_key_values.layers):
            if not (torch.equal(static_key, warm_layer.keys) and torch.equal(static_value, warm_layer.values)):
                raise RuntimeError("replay K/V differ bitwise from eager parent output")

        return {
            "static_input": static_input,
            "graph": graph,
            "static_logits": static_logits,
            "static_keys": static_keys,
            "static_values": static_values,
        }

    def _output_from_static(entry):
        logits = entry["static_logits"].clone()
        keys = [torch.empty_like(t) for t in entry["static_keys"]]
        values = [torch.empty_like(t) for t in entry["static_values"]]
        if state["foreach_ok"]:
            torch._foreach_copy_(keys + values, entry["static_keys"] + entry["static_values"])
        else:
            for dst, src in zip(keys + values, entry["static_keys"] + entry["static_values"]):
                dst.copy_(src)
        cache = _new_cache(keys, values)
        return CausalLMOutputWithPast(
            loss=None,
            logits=logits,
            past_key_values=cache,
            hidden_states=None,
            attentions=None,
        )

    def prefill_forward(
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        logits_to_keep=0,
        **kwargs,
    ):
        call_kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "past_key_values": past_key_values,
            "inputs_embeds": inputs_embeds,
            "labels": labels,
            "use_cache": use_cache,
            "logits_to_keep": logits_to_keep,
        }
        call_kwargs.update(kwargs)

        tensor_ok = isinstance(input_ids, torch.Tensor) and input_ids.is_cuda
        if (
            not tensor_ok
            or torch.is_grad_enabled()
            or torch.cuda.is_current_stream_capturing()
        ):
            record_route(call_id, _LABEL_CONTEXT)
            return parent(**call_kwargs)
        if not (
            input_ids.ndim == 2
            and attention_mask is None
            and position_ids is None
            and past_key_values is None
            and inputs_embeds is None
            and labels is None
            and use_cache is True
            and isinstance(logits_to_keep, int)
            and logits_to_keep == 1
            and not kwargs
        ):
            record_route(call_id, _LABEL_SIGNATURE)
            return parent(**call_kwargs)

        key = (str(input_ids.device), tuple(input_ids.shape), input_ids.dtype)
        entry = state["graphs"].get(key)
        if entry is None:
            if state["setup_failed"]:
                record_route(call_id, _LABEL_SETUP_FAILURE)
                return parent(**call_kwargs)
            if len(state["graphs"]) >= _MAX_GRAPHS:
                record_route(call_id, _LABEL_LIMIT)
                return parent(**call_kwargs)
            try:
                entry = _setup(input_ids)
            except Exception as exc:  # keep a concise setup-failure category
                state["setup_failed"] = True
                state["failure"] = repr(exc)[:300]
                record_route(call_id, _LABEL_SETUP_FAILURE)
                return parent(**call_kwargs)
            state["graphs"][key] = entry
            try:
                output = _output_from_static(entry)
            except Exception as exc:
                state["graphs"].pop(key, None)
                state["setup_failed"] = True
                state["failure"] = repr(exc)[:300]
                record_route(call_id, _LABEL_RUNTIME)
                return parent(**call_kwargs)
            record_route(call_id, _LABEL_CAPTURE)
            return output

        try:
            entry["static_input"].copy_(input_ids)
            entry["graph"].replay()
            output = _output_from_static(entry)
        except Exception as exc:
            state["graphs"].pop(key, None)
            state["setup_failed"] = True
            state["failure"] = repr(exc)[:300]
            record_route(call_id, _LABEL_RUNTIME)
            return parent(**call_kwargs)
        record_route(call_id, _LABEL_REPLAY)
        return output

    return {call_id: prefill_forward}
