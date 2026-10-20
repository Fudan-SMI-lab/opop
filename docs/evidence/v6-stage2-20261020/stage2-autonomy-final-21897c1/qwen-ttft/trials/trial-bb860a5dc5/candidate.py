from __future__ import annotations

import torch

PREFILL_CALL_ID = "900c23e99c808d2c9bda215a"
PREFILL_INPUT_SHAPE = (1, 256)
CAPTURE_WARMUP_FORWARDS = 3
_FORBIDDEN_KWARGS = (
    "attention_mask",
    "position_ids",
    "past_key_values",
    "inputs_embeds",
    "labels",
    "cache_position",
)
_FLAG_KWARGS = ("output_attentions", "output_hidden_states")


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
        "graph": None,
        "static_input": None,
        "static_logits": None,
        "static_keys": None,
        "static_values": None,
    }

    def _signature_ok(kwargs):
        input_ids = kwargs.get("input_ids")
        if not isinstance(input_ids, torch.Tensor):
            return False
        if tuple(input_ids.shape) != PREFILL_INPUT_SHAPE:
            return False
        if input_ids.dtype != torch.int64 or not input_ids.is_cuda:
            return False
        if kwargs.get("use_cache") is not True or kwargs.get("logits_to_keep") != 1:
            return False
        for name in _FORBIDDEN_KWARGS:
            if kwargs.get(name) is not None:
                return False
        for name in _FLAG_KWARGS:
            if kwargs.get(name):
                return False
        return True

    def _validate_captured(logits, keys, values, prompt_len):
        if not isinstance(logits, torch.Tensor) or tuple(logits.shape) != (1, 1, vocab_size):
            raise RuntimeError("captured logits shape mismatch")
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

    def _capture(input_ids):
        device = input_ids.device
        prompt_len = int(input_ids.shape[1])
        static_input = torch.empty_like(input_ids)
        static_input.copy_(input_ids)

        side_stream = torch.cuda.Stream(device=device)
        side_stream.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(side_stream):
            for _ in range(CAPTURE_WARMUP_FORWARDS):
                parent(input_ids=static_input, use_cache=True, logits_to_keep=1)
        torch.cuda.current_stream(device).wait_stream(side_stream)
        torch.cuda.synchronize(device)

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = parent(input_ids=static_input, use_cache=True, logits_to_keep=1)

        cache = captured.past_key_values
        layers = getattr(cache, "layers", None)
        if layers is None:
            raise RuntimeError("parent returned a cache without layers")
        keys = [layer.keys for layer in layers]
        values = [layer.values for layer in layers]
        _validate_captured(captured.logits, keys, values, prompt_len)

        box["graph"] = graph
        box["static_input"] = static_input
        box["static_logits"] = captured.logits
        box["static_keys"] = keys
        box["static_values"] = values
        box["state"] = "ready"

    def _fresh_cache():
        from transformers.cache_utils import DynamicCache

        cache = DynamicCache(config=config)
        layers = cache.layers
        if len(layers) != num_layers:
            raise RuntimeError("fresh cache layer count mismatch")
        for layer, key, value in zip(layers, box["static_keys"], box["static_values"]):
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
            record_route(PREFILL_CALL_ID, "fallback:setup_or_runtime_failure")
            return parent(**kwargs)
        if box["state"] == "cold":
            try:
                _capture(kwargs["input_ids"])
                record_route(PREFILL_CALL_ID, "cuda_graph_captured")
            except Exception as exc:
                box["state"] = "failed"
                record_route(PREFILL_CALL_ID, "fallback:setup_failure:" + type(exc).__name__)
                return parent(**kwargs)
        try:
            box["static_input"].copy_(kwargs["input_ids"])
            box["graph"].replay()
            logits = box["static_logits"].clone()
            cache = _fresh_cache()
        except Exception as exc:
            box["state"] = "failed"
            record_route(PREFILL_CALL_ID, "fallback:runtime_failure:" + type(exc).__name__)
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
