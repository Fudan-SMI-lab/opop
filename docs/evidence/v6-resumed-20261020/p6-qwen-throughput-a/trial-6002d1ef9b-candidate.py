"""Decode-body CUDA-graph replay with eagerly produced per-step logits.

Repair of the defect recorded in inputs/previous-feedback.txt (trial-82eb37adf6,
decision no_valid_parameter_candidate): both paired quality repeats failed on
output.logits only (10,431,056 / 38,895,616 mismatched, max abs 4.0, per-step
argmax exact), the deterministic signature of the adapter holding views of
CUDA-graph pool memory that later replays rewrote. This candidate makes that
aliasing impossible by construction: one CUDA graph per absolute decode
position captures only the Qwen3Model decoder body (embedding, 36 decoder
layers with genuine DynamicCache torch.cat appends, final RMSNorm) into one
shared private pool captured in replay order, and the last-position lm_head
projection runs eagerly outside every graph on the graph's static hidden-state
output, so output.logits is a freshly allocated eager tensor on every
invocation. Step 0 of each run binds that run's eager prefill K/V into the
static base buffers with device copies; every subsequent replay re-materializes
the run's own KV chain through the pool. Any guard mismatch falls back to the
pristine parent with the original arguments; capture or replay failure latches
to the pristine parent permanently.
"""
from __future__ import annotations

import torch
from transformers.cache_utils import DynamicCache
from transformers.modeling_outputs import CausalLMOutputWithPast

CALL_ID = "323bcbe006b343648b346af9"
_MARKER = "_opop_body_graph_marker"
_EXPECTED_KWARGS = frozenset(
    {"input_ids", "past_key_values", "use_cache", "logits_to_keep"}
)


class _GuardFailure(Exception):
    pass


def _seed_cache(config, keys, values):
    cache = DynamicCache(config=config)
    for layer, key, value in zip(cache.layers, keys, values):
        layer.keys = key
        layer.values = value
        layer.is_initialized = True
        layer.dtype = key.dtype
        layer.device = key.device
    return cache


def build(context):
    modules = context.get("modules") or {}
    module = modules.get(CALL_ID)
    references = context.get("references") or {}
    parent = references.get(CALL_ID)
    if module is None or parent is None or not callable(parent):
        return {}
    inner = getattr(module, "model", None)
    lm_head = getattr(module, "lm_head", None)
    config = getattr(module, "config", None)
    if inner is None or lm_head is None or config is None:
        return {}

    record_route = context.get("record_route")
    if not callable(record_route):

        def record_route(*_args, **_kwargs):
            return None

    st = {
        "inner": inner,
        "lm_head": lm_head,
        "config": config,
        "parent": parent,
        "record_route": record_route,
        "broken": False,
        "warmed": False,
        "pool": None,
        "graphs": [],
        "hidden": [],
        "markers": [],
        "static_token": None,
        "static_keys": None,
        "static_values": None,
        "work_cache": None,
        "base_len": None,
    }

    def _fallback(*args, **kwargs):
        record_route(CALL_ID, "parent_fallback")
        return parent(*args, **kwargs)

    def _seed_from_static():
        return _seed_cache(config, st["static_keys"], st["static_values"])

    def _snapshot_marker():
        keys = [layer.keys for layer in st["work_cache"].layers]
        values = [layer.values for layer in st["work_cache"].layers]
        marker = _seed_cache(config, keys, values)
        setattr(marker, _MARKER, True)
        return marker

    def _bind_run(cache, input_ids):
        layers = cache.layers
        if config.num_hidden_layers != len(layers):
            raise _GuardFailure("cache layer count")
        first = layers[0].keys
        if first is None or layers[0].values is None or first.dim() != 4:
            raise _GuardFailure("cache geometry")
        length = first.shape[-2]
        for layer in layers:
            if layer.keys is None or layer.values is None:
                raise _GuardFailure("uninitialized cache layer")
            if layer.keys.shape != first.shape or layer.keys.dtype != first.dtype:
                raise _GuardFailure("nonuniform cache")
        if st["base_len"] is None:
            if length <= 0 or input_ids.shape[0] <= 0:
                raise _GuardFailure("empty bind")
            st["base_len"] = length
            st["static_keys"] = [
                torch.empty(layer.keys.shape, dtype=layer.keys.dtype, device=layer.keys.device)
                for layer in layers
            ]
            st["static_values"] = [
                torch.empty(layer.values.shape, dtype=layer.values.dtype, device=layer.values.device)
                for layer in layers
            ]
            st["static_token"] = torch.empty(
                input_ids.shape, dtype=input_ids.dtype, device=input_ids.device
            )
            st["work_cache"] = _seed_from_static()
        elif st["base_len"] != length:
            raise _GuardFailure("cache length changed")
        for i, layer in enumerate(layers):
            static_key = st["static_keys"][i]
            static_value = st["static_values"][i]
            if static_key.shape != layer.keys.shape or static_key.dtype != layer.keys.dtype:
                raise _GuardFailure("bind shape mismatch")
            static_key.copy_(layer.keys, non_blocking=True)
            static_value.copy_(layer.values, non_blocking=True)

    def _capture(step):
        if st["pool"] is None:
            st["pool"] = torch.cuda.graph_pool_handle()
        if not st["warmed"]:
            st["warmed"] = True
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                for _ in range(3):
                    inner(
                        input_ids=st["static_token"],
                        past_key_values=_seed_from_static(),
                        use_cache=True,
                    )
            torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, pool=st["pool"]):
            out = inner(
                input_ids=st["static_token"],
                past_key_values=st["work_cache"],
                use_cache=True,
            )
        st["graphs"].append(graph)
        st["hidden"].append(out.last_hidden_state)
        st["markers"].append(_snapshot_marker())

    def replacement(*args, **kwargs):
        try:
            if st["broken"]:
                return _fallback(*args, **kwargs)
            if args or frozenset(kwargs) != _EXPECTED_KWARGS:
                raise _GuardFailure("signature")
            input_ids = kwargs["input_ids"]
            passed_cache = kwargs["past_key_values"]
            if kwargs["use_cache"] is not True or kwargs["logits_to_keep"] != 1:
                raise _GuardFailure("flags")
            if (
                not isinstance(input_ids, torch.Tensor)
                or input_ids.dim() != 2
                or input_ids.shape[1] != 1
                or input_ids.dtype != torch.int64
                or input_ids.device.type != "cuda"
            ):
                raise _GuardFailure("input_ids")
            if type(passed_cache) is not DynamicCache:
                raise _GuardFailure("cache type")
            length = passed_cache.get_seq_length()
            is_marker = getattr(passed_cache, _MARKER, False) is True
            if st["base_len"] is None:
                if is_marker or length <= 0:
                    raise _GuardFailure("first bind")
                _bind_run(passed_cache, input_ids)
                step = 0
            else:
                step = length - st["base_len"]
                if (
                    step < 0
                    or st["static_token"].shape[0] != input_ids.shape[0]
                    or st["static_token"].dtype != input_ids.dtype
                ):
                    raise _GuardFailure("step geometry")
                if step == 0:
                    if is_marker:
                        raise _GuardFailure("marker at base length")
                    _bind_run(passed_cache, input_ids)
                elif not is_marker:
                    raise _GuardFailure("foreign mid-run cache")
            st["static_token"].copy_(input_ids, non_blocking=True)
            if step < len(st["graphs"]):
                st["graphs"][step].replay()
                record_route(CALL_ID, "cuda_graph_replay_body")
            elif step == len(st["graphs"]):
                _capture(step)
                record_route(CALL_ID, "cuda_graph_capture_body")
            else:
                raise _GuardFailure("graph chain gap")
            hidden = st["hidden"][step]
            logits = lm_head(hidden[:, -1:, :])
            return CausalLMOutputWithPast(
                loss=None,
                logits=logits,
                past_key_values=st["markers"][step],
            )
        except _GuardFailure:
            return _fallback(*args, **kwargs)
        except Exception:
            st["broken"] = True
            return _fallback(*args, **kwargs)

    return {CALL_ID: replacement}
