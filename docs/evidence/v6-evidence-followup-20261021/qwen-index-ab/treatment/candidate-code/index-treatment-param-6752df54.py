"""TTFT candidate (repair of trial-9221b5ebdd): CUDA Graph replay of the prefill-phase
Qwen3-4B forward for the region root call (phase "prefill", Qwen3ForCausalLM).

Previous attempt failed because setup validated the first replay with bitwise
torch.equal against the eager parent output; benign bf16 kernel-selection drift
under graph capture tripped it, latched a permanent parent fallback and the
optimization never ran. This repair keeps capture/replay identical but validates
against the framework's actual quality semantics instead of bitwise identity:
bounded closeness (atol=rtol=0.015, inside the frozen 0.02 quality threshold),
exact equality of the greedy first-token argmax, exact per-layer K/V closeness,
and one full end-to-end 31-step decode generation from both the replayed and the
eager caches whose stacked tokens must satisfy torch.equal and whose per-step
logits must stay inside the same tolerance. Any gate failure keeps the concise
setup-failure fallback category and the parent serves every later call.

Per-invocation graph path: copy current input_ids into a static buffer, replay
the graph captured from the accepted parent forward (all prefill kernels re-run
on current contents; nothing is precomputed), clone the static logits and
per-layer K/V into fresh tensors and wrap them in a new DynamicCache so decode
steps keep running the unmodified parent with parent cache semantics and no
cross-invocation aliasing. Setup is lazy inside the first eligible invocation
(real adapter prefill phase, inference_mode): side-stream warmups, eager
reference, capture, validation replay plus the two validation generations.
"""

from __future__ import annotations

import torch

try:
    from transformers.cache_utils import DynamicCache, DynamicLayer
    from transformers.modeling_outputs import CausalLMOutputWithPast

    _CACHE_API_OK = True
except Exception:  # pragma: no cover - transformers always present on the worker
    _CACHE_API_OK = False

_WARMUP_ITERS = 3
_MAX_GRAPHS = 4
_ATOL = 0.015
_RTOL = 0.015

_LABEL_REPLAY = "cuda_graph_replay"
_LABEL_CAPTURE = "cuda_graph_capture"
_LABEL_SETUP_FAILURE = "parent_fallback_setup_failure"
_LABEL_SIGNATURE = "parent_fallback_signature"
_LABEL_CONTEXT = "parent_fallback_context"
_LABEL_RUNTIME = "parent_fallback_runtime_failure"
_LABEL_LIMIT = "parent_fallback_graph_limit"


def _find_region_root(targets):
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
    if not _CACHE_API_OK:
        return {}
    targets = context.get("targets") or {}
    references = context.get("references") or {}
    modules = context.get("modules") or {}
    record_route = context.get("record_route", lambda *_: None)
    if not callable(record_route):
        record_route = lambda *_: None

    call_id = _find_region_root(targets)
    if call_id is None or call_id not in references:
        return {}
    parent = references[call_id]
    module = modules.get(call_id)
    config = getattr(module, "config", None)

    adapter_state = context.get("state") or {}
    try:
        decode_steps = int(adapter_state.get("new_tokens", 32)) - 1
    except (TypeError, ValueError):
        decode_steps = 31
    decode_steps = max(0, decode_steps)

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
            layers = [DynamicLayer() for _ in keys]
            cache = DynamicCache()
            cache.layers.extend(layers)
        for layer, key, value in zip(layers, keys, values):
            layer.keys = key
            layer.values = value
            layer.is_initialized = True
            layer.dtype = key.dtype
            layer.device = key.device
        return cache

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

    def _check_close(actual, expected, what):
        if actual.shape != expected.shape:
            raise RuntimeError(f"{what} shape differs from eager parent output")
        if not torch.allclose(actual, expected, rtol=_RTOL, atol=_ATOL):
            diff = (actual.float() - expected.float()).abs().max().item()
            raise RuntimeError(
                f"{what} exceed validation tolerance (max abs diff {diff:.6g})"
            )

    def _generate(token, cache, steps):
        tokens = [token]
        logits = []
        for _ in range(steps):
            out = parent(input_ids=token, past_key_values=cache, use_cache=True, logits_to_keep=1)
            cache = out.past_key_values
            token = out.logits[:, -1].argmax(dim=-1, keepdim=True)
            tokens.append(token)
            logits.append(out.logits[:, -1])
        stacked = torch.stack(logits, dim=1) if logits else None
        return torch.cat(tokens, dim=1), stacked

    def _setup(input_ids):
        device = input_ids.device
        static_input = torch.empty_like(input_ids)
        side_stream = torch.cuda.Stream(device=device)
        side_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side_stream):
            static_input.copy_(input_ids)
            for _ in range(_WARMUP_ITERS):
                parent(input_ids=static_input, use_cache=True, logits_to_keep=1)
        torch.cuda.current_stream().wait_stream(side_stream)

        ref = parent(input_ids=input_ids, use_cache=True, logits_to_keep=1)
        if ref.logits is None or ref.past_key_values is None:
            raise RuntimeError("eager parent forward returned no logits or no past_key_values")

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = parent(input_ids=static_input, use_cache=True, logits_to_keep=1)
        static_logits = captured.logits
        captured_cache = captured.past_key_values
        if static_logits is None or captured_cache is None:
            raise RuntimeError("captured parent forward returned no logits or no past_key_values")
        static_keys = [layer.keys for layer in captured_cache.layers]
        static_values = [layer.values for layer in captured_cache.layers]
        if not static_keys or any(t is None for t in static_keys + static_values):
            raise RuntimeError("captured cache layers are incomplete")

        graph.replay()
        torch.cuda.synchronize()

        _check_close(static_logits, ref.logits, "replay logits")
        if not torch.equal(
            static_logits[:, -1].argmax(dim=-1), ref.logits[:, -1].argmax(dim=-1)
        ):
            raise RuntimeError("replay changes the greedy first token")
        ref_layers = ref.past_key_values.layers
        if len(ref_layers) != len(static_keys):
            raise RuntimeError("captured cache depth differs from eager parent cache")
        for idx, (static_key, static_value) in enumerate(zip(static_keys, static_values)):
            _check_close(static_key, ref_layers[idx].keys, f"replay layer {idx} keys")
            _check_close(static_value, ref_layers[idx].values, f"replay layer {idx} values")

        entry = {
            "static_input": static_input,
            "graph": graph,
            "static_logits": static_logits,
            "static_keys": static_keys,
            "static_values": static_values,
        }

        if decode_steps > 0:
            replay_out = _output_from_static(entry)
            tokens_replay, logits_replay = _generate(
                replay_out.logits[:, -1].argmax(dim=-1, keepdim=True),
                replay_out.past_key_values,
                decode_steps,
            )
            tokens_eager, logits_eager = _generate(
                ref.logits[:, -1].argmax(dim=-1, keepdim=True),
                ref.past_key_values,
                decode_steps,
            )
            if not torch.equal(tokens_replay, tokens_eager):
                raise RuntimeError("replay changes greedy tokens over the full generation")
            _check_close(logits_replay, logits_eager, "replay decode logits")
        return entry

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
            except Exception as exc:
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
