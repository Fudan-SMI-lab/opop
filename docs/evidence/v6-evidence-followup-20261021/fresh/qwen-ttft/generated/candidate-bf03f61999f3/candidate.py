"""Repair candidate for the qwen-ttft region selection (groups
7f49ae8c64c12e322632 prefill, 92e2f6fb9310a942405a decode): fidelity-gated
CUDA Graph capture/replay of the Qwen3Model prefill boundary.

Previous trial-57b13527f6 was rejected on QUALITY only: its captured graph
executed 36 fmha_cutlassF kernels (SDPA backend substitution during stream
capture) while the eager parent executes 36 pytorch_flash::flash_fwd_splitkv
kernels, and the substituted prefill K/V drifted through all 31 decode steps
past atol=rtol=0.02 (41.5% of per-step logits mismatched, max abs 5.0625).

This repair keeps the accepted boundary and adds a hard numerical gate:

1. Warmup and capture both run under torch.nn.attention.sdpa_kernel with the
   exclusive allowlist [FLASH_ATTENTION], so the recorded kernel sequence is
   the eager flash selection, not the cutlassF substitute. If flash is
   unusable during capture in torch 2.13.0+cu129, the restricted dispatch
   raises instead of silently substituting a different backend.
2. Before replay is ever enabled, one eager parent reference forward on the
   same static input (unforced dispatch, i.e. exactly what the workload would
   otherwise run) must satisfy torch.equal on last_hidden_state and on every
   layer's keys and values against the replay outputs, for two consecutive
   replays. Any mismatch, dispatch refusal or setup exception permanently
   routes every later call to the parent with a concise route label, so the
   late-decode divergence cannot recur.
3. Each replay returns a fresh DynamicCache aliasing the verified static K/V
   tensors, so decode torch.cat appends and get_seq_length()==256 behave
   exactly like the parent's cache; decode invocations are delegated
   unchanged to the parent (site e7df98b905ed8aa688d5227a).

The workload executes on current inputs on every invocation: live input_ids
are copied into the static buffer before each replay. No fixture output is
cached. Setup is lazy inside the first matching quality invocation only.
"""

from __future__ import annotations

import torch

PREFILL_CALL_ID = "2441f10600ba127de5cb753c"
DECODE_CALL_ID = "e7df98b905ed8aa688d5227a"

_EXPECTED_SHAPE = (1, 256)
_WARMUP_ITERATIONS = 3
_NONE_KWARGS = ("attention_mask", "position_ids", "past_key_values", "inputs_embeds")
_ALLOWED_KEYS = frozenset({"input_ids", "use_cache", *_NONE_KWARGS})


def build(context):
    references = context.get("references") or {}
    modules = context.get("modules") or {}
    parent = references.get(PREFILL_CALL_ID)
    decode_parent = references.get(DECODE_CALL_ID)
    module = modules.get(PREFILL_CALL_ID)
    record_route = context.get("record_route") or (lambda *_: None)
    if parent is None or module is None:
        return {}

    from torch.nn.attention import SDPBackend, sdpa_kernel
    from transformers import DynamicCache
    from transformers.cache_utils import Cache as LayeredCache, DynamicLayer
    from transformers.modeling_outputs import BaseModelOutputWithPast

    config = getattr(module, "config", None)

    def _flash_only():
        return sdpa_kernel([SDPBackend.FLASH_ATTENTION])

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

    def _layer_tensors(cache):
        keys = [layer.keys for layer in cache.layers]
        values = [layer.values for layer in cache.layers]
        return keys, values

    def _capture(kwargs):
        static_kwargs = dict(kwargs)
        static_input = kwargs["input_ids"].detach().clone()
        static_kwargs["input_ids"] = static_input

        current = torch.cuda.current_stream()
        side = torch.cuda.Stream()
        side.wait_stream(current)
        with torch.cuda.stream(side), _flash_only():
            for _ in range(_WARMUP_ITERATIONS):
                _call_parent(static_kwargs)
        current.wait_stream(side)
        torch.cuda.synchronize()

        graph = torch.cuda.CUDAGraph()
        with _flash_only(), torch.cuda.graph(graph):
            captured = _call_parent(static_kwargs)

        cache = captured.past_key_values
        if cache is None or captured.last_hidden_state is None:
            raise RuntimeError("parent returned no hidden state or cache")
        static_keys, static_values = _layer_tensors(cache)
        if not static_keys or any(key is None for key in static_keys):
            raise RuntimeError("parent cache layers incomplete")
        if any(value is None for value in static_values):
            raise RuntimeError("parent cache layers incomplete")

        holder["graph"] = graph
        holder["static_input"] = static_input
        holder["static_kwargs"] = static_kwargs
        holder["static_hidden"] = captured.last_hidden_state
        holder["static_keys"] = static_keys
        holder["static_values"] = static_values
        holder["template"] = {
            key: value for key, value in kwargs.items() if key != "input_ids"
        }
        holder["device"] = static_input.device

    def _replay_matches(reference):
        ref_keys, ref_values = _layer_tensors(reference.past_key_values)
        if len(ref_keys) != len(holder["static_keys"]):
            return False
        for key, value in zip(ref_keys, ref_values):
            if key is None or value is None:
                return False
        holder["graph"].replay()
        torch.cuda.synchronize()
        if not torch.equal(holder["static_hidden"], reference.last_hidden_state):
            return False
        for captured_key, ref_key, captured_value, ref_value in zip(
            holder["static_keys"], ref_keys, holder["static_values"], ref_values
        ):
            if not torch.equal(captured_key, ref_key):
                return False
            if not torch.equal(captured_value, ref_value):
                return False
        return True

    def _verify():
        reference = _call_parent(holder["static_kwargs"])
        if reference.past_key_values is None or reference.last_hidden_state is None:
            return False
        # Two consecutive replays must each be bit-equal to the eager parent.
        return _replay_matches(reference) and _replay_matches(reference)

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

    def _setup_error_label(error):
        if "No available kernel" in str(error):
            return "setup-failure:flash-dispatch-refused"
        return "setup-failure:" + type(error).__name__

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
                verified = _verify()
            except Exception as error:
                holder["setup_error"] = _setup_error_label(error)
                record_route(
                    PREFILL_CALL_ID, "parent-fallback:" + holder["setup_error"]
                )
                return _call_parent(kwargs)
            if not verified:
                holder["setup_error"] = "replay-fidelity-mismatch"
                record_route(
                    PREFILL_CALL_ID, "parent-fallback:replay-fidelity-mismatch"
                )
                return _call_parent(kwargs)
            record_route(PREFILL_CALL_ID, "cuda-graph-capture-verified")
        elif not _replay_compatible(kwargs):
            record_route(PREFILL_CALL_ID, "parent-fallback:unsupported-signature")
            return _call_parent(kwargs)
        holder["static_input"].copy_(kwargs["input_ids"])
        holder["graph"].replay()
        record_route(PREFILL_CALL_ID, "cuda-graph-replay")
        return _replay_result()

    def decode_replacement(*args, **kwargs):
        record_route(DECODE_CALL_ID, "parent-delegate")
        return decode_parent(*args, **kwargs)

    replacements = {PREFILL_CALL_ID: replacement}
    if decode_parent is not None:
        replacements[DECODE_CALL_ID] = decode_replacement
    return replacements
