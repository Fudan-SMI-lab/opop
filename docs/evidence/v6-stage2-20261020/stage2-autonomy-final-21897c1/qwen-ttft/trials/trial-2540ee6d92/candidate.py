from __future__ import annotations

import torch

PREFILL_CALL_ID = "900c23e99c808d2c9bda215a"
PREFILL_INPUT_SHAPE = (1, 256)
AMBIENT_WARMUP_FORWARDS = 2
CAPTURE_WARMUP_FORWARDS = 3
CAPTURE_ATTEMPTS = 2
_EXPECTED_KWARGS = {"input_ids", "use_cache", "logits_to_keep"}


def build(context):
    references = context.get("references") or {}
    modules = context.get("modules") or {}
    record_route = context.get("record_route") or (lambda *_: None)

    parent = references.get(PREFILL_CALL_ID)
    module = modules.get(PREFILL_CALL_ID)
    if parent is None or module is None:
        return {}

    config = getattr(module, "config", None)
    num_layers = int(getattr(config, "num_hidden_layers", 0) or 0)
    vocab_size = int(getattr(config, "vocab_size", 0) or 0)
    if num_layers <= 0 or vocab_size <= 0 or not torch.cuda.is_available():
        return {}

    box = {
        "state": "cold",
        "fail_label": None,
        "graph": None,
        "static_input": None,
        "static_logits": None,
        "static_keys": None,
        "static_values": None,
    }

    def _signature_ok(kwargs):
        if set(kwargs) != _EXPECTED_KWARGS:
            return False
        input_ids = kwargs["input_ids"]
        if not isinstance(input_ids, torch.Tensor):
            return False
        if tuple(input_ids.shape) != PREFILL_INPUT_SHAPE:
            return False
        if input_ids.dtype != torch.int64 or not input_ids.is_cuda:
            return False
        return kwargs["use_cache"] is True and kwargs["logits_to_keep"] == 1

    def _cache_layers(cache):
        layers = getattr(cache, "layers", None)
        if layers is None or len(layers) != num_layers:
            raise RuntimeError("cache does not expose the expected per-layer state")
        return layers

    def _validate_captured(logits, keys, values, prompt_len):
        if not isinstance(logits, torch.Tensor) or tuple(logits.shape) != (1, 1, vocab_size):
            raise RuntimeError("captured logits shape mismatch")
        if logits.dtype != torch.bfloat16 or not logits.is_cuda:
            raise RuntimeError("captured logits dtype/device mismatch")
        if len(keys) != num_layers or len(values) != num_layers:
            raise RuntimeError("captured cache layer count mismatch")
        for tensor in keys + values:
            if not isinstance(tensor, torch.Tensor) or not tensor.is_cuda:
                raise RuntimeError("captured cache tensor is not a CUDA tensor")
            if tensor.dtype != torch.bfloat16 or tensor.dim() != 4:
                raise RuntimeError("captured cache tensor dtype/rank mismatch")
            shape = tuple(tensor.shape)
            if shape[0] != 1 or shape[2] != prompt_len:
                raise RuntimeError("captured cache tensor sequence mismatch")

    def _setup(input_ids):
        device = input_ids.device
        prompt_len = int(input_ids.shape[1])

        for _ in range(AMBIENT_WARMUP_FORWARDS):
            parent(input_ids=input_ids, use_cache=True, logits_to_keep=1)
        torch.cuda.synchronize(device)

        static_input = torch.empty_like(input_ids)
        static_input.copy_(input_ids)

        capture_stream = torch.cuda.Stream(device=device)
        capture_stream.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(capture_stream):
            for _ in range(CAPTURE_WARMUP_FORWARDS):
                parent(input_ids=static_input, use_cache=True, logits_to_keep=1)
        torch.cuda.current_stream(device).wait_stream(capture_stream)
        torch.cuda.synchronize(device)

        eager = None
        for _attempt in range(CAPTURE_ATTEMPTS):
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=capture_stream):
                captured = parent(input_ids=static_input, use_cache=True, logits_to_keep=1)
            captured_layers = _cache_layers(captured.past_key_values)
            keys = [layer.keys for layer in captured_layers]
            values = [layer.values for layer in captured_layers]
            _validate_captured(captured.logits, keys, values, prompt_len)

            eager = parent(input_ids=input_ids, use_cache=True, logits_to_keep=1)
            static_input.copy_(input_ids)
            graph.replay()
            torch.cuda.synchronize(device)

            bit_exact = torch.equal(eager.logits, captured.logits)
            if bit_exact:
                eager_layers = _cache_layers(eager.past_key_values)
                for eager_layer, key, value in zip(eager_layers, keys, values):
                    if not torch.equal(eager_layer.keys, key):
                        bit_exact = False
                        break
                    if not torch.equal(eager_layer.values, value):
                        bit_exact = False
                        break
            if bit_exact:
                box["graph"] = graph
                box["static_input"] = static_input
                box["static_logits"] = captured.logits
                box["static_keys"] = keys
                box["static_values"] = values
                return eager, True
            del graph, captured, captured_layers, keys, values

        return eager, False

    def _fresh_cache(keys, values):
        from transformers.cache_utils import DynamicCache

        cache = DynamicCache(config=config)
        layers = cache.layers
        if len(layers) != len(keys):
            raise RuntimeError("fresh cache layer count mismatch")
        for layer, key, value in zip(layers, keys, values):
            layer.dtype = key.dtype
            layer.device = key.device
            layer.keys = key
            layer.values = value
            layer.is_initialized = True
        return cache

    def replacement(*args, **kwargs):
        if args or not _signature_ok(kwargs):
            record_route(PREFILL_CALL_ID, "fallback:signature")
            return parent(*args, **kwargs)
        if box["state"] == "failed":
            record_route(PREFILL_CALL_ID, box["fail_label"])
            return parent(**kwargs)
        if box["state"] == "cold":
            try:
                eager, accepted = _setup(kwargs["input_ids"])
            except Exception as exc:
                box["state"] = "failed"
                box["fail_label"] = "fallback:setup_failure:" + type(exc).__name__
                record_route(PREFILL_CALL_ID, box["fail_label"])
                return parent(**kwargs)
            if not accepted:
                box["state"] = "failed"
                box["fail_label"] = "fallback:bit_exactness_gate"
                record_route(PREFILL_CALL_ID, box["fail_label"])
                return eager
            box["state"] = "ready"
            record_route(PREFILL_CALL_ID, "cuda_graph_captured")
            return eager
        try:
            box["static_input"].copy_(kwargs["input_ids"])
            box["graph"].replay()
            logits = box["static_logits"].clone()
            keys = [key.clone() for key in box["static_keys"]]
            values = [value.clone() for value in box["static_values"]]
            cache = _fresh_cache(keys, values)
        except Exception as exc:
            box["state"] = "failed"
            box["fail_label"] = "fallback:runtime_failure:" + type(exc).__name__
            record_route(PREFILL_CALL_ID, box["fail_label"])
            return parent(**kwargs)
        record_route(PREFILL_CALL_ID, "cuda_graph_replay")
        from transformers.modeling_outputs import CausalLMOutputWithPast

        return CausalLMOutputWithPast(
            loss=None,
            logits=logits,
            past_key_values=cache,
            hidden_states=None,
            attentions=None,
        )

    return {PREFILL_CALL_ID: replacement}
