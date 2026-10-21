"""TTFT candidate (repair of trial-9221b5ebdd): CUDA Graph replay of the prefill-phase
Qwen3-4B forward.

Selected region (inputs/selection.json, scope "region"): the prefill subtree of
transformers.models.qwen3.modeling_qwen3.Qwen3ForCausalLM.  The replacement is
installed only at the region root, the prefill-phase module boundary of target
"qwen3_4b" (call id resolved from context["targets"]); every inner prefill call
site of the region is subsumed by the replayed graph and is listed in
proposal.json eliminated_calls.

Per-invocation route on the graph path:
  1. copy the current input_ids into a small static buffer (2 KB),
  2. replay the CUDA graph captured from the accepted parent forward
     (context["references"][call_id]) - all prefill kernels re-execute on the
     current input contents, nothing is precomputed,
  3. clone the static logits and the static per-layer K/V tensors into fresh
     per-invocation tensors and wrap them in a new DynamicCache so the decode
     steps (which keep running the unmodified parent) consume a cache object
     with parent semantics and no cross-invocation aliasing.

Repair of the previous failure (trial-9221b5ebdd, parent_fallback_setup_failure
at its own bitwise gate): capture succeeded and the graph replayed, but the
route was permanently disabled because replay logits did not satisfy
torch.equal against a separately-run eager output (candidate_diagnostics pins
the raise at candidate.py:135).  cublasLt/SDPA kernels re-selected on the
capture stream can differ at ULP level from the eager-stream kernels even
though both are valid computations of the parent forward, and the eager parent
itself is deterministic, so a bitwise cross-stream gate is unsatisfiable by
design.  This candidate validates the captured graph with the project's actual
quality semantics instead:
  - logits within torch.testing.assert_close tolerance atol=rtol=0.02
    (request.json quality), and
  - identical argmax token (the exact-greedy torch.equal requirement), and
  - per-layer K/V within the same tolerance, and
  - one eager parent decode step fed a cache cloned from the graph statics
    matching the same step fed the eager prefill cache (validates the consumer
    path the 31 decode steps use, which the previous trial never executed).
Only a semantic divergence disables the route; a benign ULP-level difference no
longer does.

Lazy setup runs inside the adapter's real prefill phase on the first eligible
invocation (three eager parent warmups on a side stream, graph capture, one
eager reference on the current stream, one validation replay, the decode
consumer probe), so the accepted parent semantics are what gets captured.
Signature mismatches, grad-enabled or foreign-capture contexts, setup failures
and runtime failures each fall back to the parent with a distinct route label.
"""

from __future__ import annotations

import torch
from transformers.cache_utils import DynamicCache, DynamicLayer
from transformers.modeling_outputs import CausalLMOutputWithPast

WARMUP_ITERS = 3
MAX_GRAPHS = 2

# Project quality thresholds (request.json quality: logits_atol/logits_rtol and
# require_exact_greedy_tokens).  The graph route must satisfy the same
# semantics, not bitwise equality.
ATOL = 0.02
RTOL = 0.02

LABEL_REPLAY = "cuda_graph_replay"
LABEL_CAPTURE = "cuda_graph_capture"
LABEL_SETUP_FAILURE = "parent_fallback_setup_failure"
LABEL_SIGNATURE = "parent_fallback_signature"
LABEL_CONTEXT = "parent_fallback_context"
LABEL_RUNTIME = "parent_fallback_runtime_failure"
LABEL_LIMIT = "parent_fallback_graph_limit"


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

    def _close(a, b):
        return bool(torch.allclose(a.float(), b.float(), atol=ATOL, rtol=RTOL))

    def _last_argmax(logits):
        return logits[:, -1].argmax(dim=-1, keepdim=True)

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
                layer = DynamicLayer()
                cache.layers.append(layer)
                layers.append(layer)
        for layer, key, value in zip(layers, keys, values):
            layer.keys = key
            layer.values = value
            layer.is_initialized = True
            layer.dtype = key.dtype
            layer.device = key.device
        return cache

    def _clone_kv(entry):
        keys = [torch.empty_like(t) for t in entry["static_keys"]]
        values = [torch.empty_like(t) for t in entry["static_values"]]
        if state["foreach_ok"]:
            torch._foreach_copy_(keys + values, entry["static_keys"] + entry["static_values"])
        else:
            for dst, src in zip(keys + values, entry["static_keys"] + entry["static_values"]):
                dst.copy_(src)
        return keys, values

    def _output_from_static(entry):
        logits = entry["static_logits"].clone()
        keys, values = _clone_kv(entry)
        cache = _new_cache(keys, values)
        return CausalLMOutputWithPast(
            loss=None,
            logits=logits,
            past_key_values=cache,
            hidden_states=None,
            attentions=None,
        )

    def _setup(input_ids):
        device = input_ids.device
        with torch.cuda.device(device):
            static_input = torch.empty_like(input_ids)
            side_stream = torch.cuda.Stream(device=device)
            side_stream.wait_stream(torch.cuda.current_stream(device))
            with torch.cuda.stream(side_stream):
                static_input.copy_(input_ids)
                warm = None
                for _ in range(WARMUP_ITERS):
                    warm = parent(input_ids=static_input, use_cache=True, logits_to_keep=1)
            torch.cuda.current_stream(device).wait_stream(side_stream)

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

            # Eager reference on the current stream after capture, then one
            # validation replay: the buffers below hold the replay result.
            eager = parent(input_ids=static_input, use_cache=True, logits_to_keep=1)
            graph.replay()
            torch.cuda.synchronize(device)

            # Project-quality validation replaces the failed bitwise gate.
            if not _close(static_logits, eager.logits):
                raise RuntimeError("replay logits differ from eager parent beyond project tolerance")
            token_replay = _last_argmax(static_logits)
            token_eager = _last_argmax(eager.logits)
            if not torch.equal(token_replay, token_eager):
                raise RuntimeError("replay argmax token differs from eager parent")
            for static_key, static_value, eager_layer in zip(
                static_keys, static_values, eager.past_key_values.layers
            ):
                if not (_close(static_key, eager_layer.keys) and _close(static_value, eager_layer.values)):
                    raise RuntimeError("replay K/V differ from eager parent beyond project tolerance")

            # Consumer probe: one parent decode step on a cache cloned from the
            # graph statics must match the same step on the eager prefill cache.
            probe_keys, probe_values = _clone_kv(
                {"static_keys": static_keys, "static_values": static_values}
            )
            probe_cache = _new_cache(probe_keys, probe_values)
            probe_token = token_replay.detach().clone()
            decode_replay = parent(
                input_ids=probe_token, past_key_values=probe_cache,
                use_cache=True, logits_to_keep=1,
            )
            decode_eager = parent(
                input_ids=probe_token, past_key_values=eager.past_key_values,
                use_cache=True, logits_to_keep=1,
            )
            if not _close(decode_replay.logits, decode_eager.logits):
                raise RuntimeError("decode logits from graph cache differ from eager cache")
            if not torch.equal(_last_argmax(decode_replay.logits), _last_argmax(decode_eager.logits)):
                raise RuntimeError("decode argmax token from graph cache differs from eager cache")
            torch.cuda.synchronize(device)

            del warm, eager, captured, captured_cache, probe_cache
            return {
                "static_input": static_input,
                "graph": graph,
                "static_logits": static_logits,
                "static_keys": static_keys,
                "static_values": static_values,
            }

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
            record_route(call_id, LABEL_CONTEXT)
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
            record_route(call_id, LABEL_SIGNATURE)
            return parent(**call_kwargs)

        key = (str(input_ids.device), tuple(input_ids.shape), input_ids.dtype)
        entry = state["graphs"].get(key)
        if entry is None:
            if state["setup_failed"]:
                record_route(call_id, LABEL_SETUP_FAILURE)
                return parent(**call_kwargs)
            if len(state["graphs"]) >= MAX_GRAPHS:
                record_route(call_id, LABEL_LIMIT)
                return parent(**call_kwargs)
            try:
                entry = _setup(input_ids)
            except Exception as exc:  # keep a concise setup-failure category
                state["setup_failed"] = True
                state["failure"] = repr(exc)[:300]
                record_route(call_id, LABEL_SETUP_FAILURE)
                return parent(**call_kwargs)
            state["graphs"][key] = entry
            try:
                output = _output_from_static(entry)
            except Exception as exc:
                state["graphs"].pop(key, None)
                state["setup_failed"] = True
                state["failure"] = repr(exc)[:300]
                record_route(call_id, LABEL_RUNTIME)
                return parent(**call_kwargs)
            record_route(call_id, LABEL_CAPTURE)
            return output

        try:
            entry["static_input"].copy_(input_ids)
            entry["graph"].replay()
            output = _output_from_static(entry)
        except Exception as exc:
            state["graphs"].pop(key, None)
            state["setup_failed"] = True
            state["failure"] = repr(exc)[:300]
            record_route(call_id, LABEL_RUNTIME)
            return parent(**call_kwargs)
        record_route(call_id, LABEL_REPLAY)
        return output

    return {call_id: prefill_forward}
