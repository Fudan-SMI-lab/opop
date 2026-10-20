from __future__ import annotations

import torch
from transformers import DynamicCache
from transformers.cache_utils import DynamicLayer
from transformers.modeling_outputs import CausalLMOutputWithPast

CALL_ID = "323bcbe006b343648b346af9"
_MARKER = "_opop_decode_graph_step"

_ALLOWED_KWARGS = {
    "input_ids",
    "attention_mask",
    "position_ids",
    "past_key_values",
    "inputs_embeds",
    "labels",
    "use_cache",
    "logits_to_keep",
    "cache_position",
    "output_attentions",
    "output_hidden_states",
    "return_dict",
}
_NONE_ONLY_KWARGS = (
    "attention_mask",
    "position_ids",
    "inputs_embeds",
    "labels",
    "cache_position",
    "output_attentions",
    "output_hidden_states",
    "return_dict",
)


class _Fallback(Exception):
    pass


class _Step:
    __slots__ = ("graph", "logits", "cache")

    def __init__(self, graph, logits, cache):
        self.graph = graph
        self.logits = logits
        self.cache = cache


class _DecodeGraphRunner:
    def __init__(self, context):
        self.parent = context["references"][CALL_ID]
        self.state = context["state"]
        self.num_layers = None
        models = context.get("models") or {}
        model = models.get("model")
        if model is not None:
            config = getattr(model, "config", None)
            if config is not None and hasattr(config, "get_text_config"):
                try:
                    config = config.get_text_config()
                except Exception:
                    pass
            if config is not None and getattr(config, "num_hidden_layers", None) is not None:
                self.num_layers = int(config.num_hidden_layers)
        self.batch = int(self.state["batch"])
        self.prompt_tokens = int(self.state["prompt_tokens"])
        self.decode_steps = int(self.state["new_tokens"]) - 1
        self.failed = False
        self.dtype = None
        self.device = None
        self.static_input = None
        self.static_k = None
        self.static_v = None
        self.pool = None
        self.capture_stream = None
        self.steps = []

    def __call__(self, *args, **kwargs):
        try:
            return self._run(args, kwargs)
        except _Fallback:
            return self._parent(args, kwargs)
        except Exception:
            self.failed = True
            return self._parent(args, kwargs)

    def _run(self, args, kwargs):
        if self.failed or args:
            raise _Fallback()
        if kwargs.get("use_cache") is not True or kwargs.get("logits_to_keep") != 1:
            raise _Fallback()
        for key, value in kwargs.items():
            if key not in _ALLOWED_KWARGS and value is not None:
                raise _Fallback()
        for key in _NONE_ONLY_KWARGS:
            if kwargs.get(key) is not None:
                raise _Fallback()
        input_ids = kwargs.get("input_ids")
        if not isinstance(input_ids, torch.Tensor):
            raise _Fallback()
        if input_ids.dim() != 2 or tuple(input_ids.shape) != (self.batch, 1):
            raise _Fallback()
        if input_ids.dtype != torch.int64 or input_ids.device.type != "cuda":
            raise _Fallback()
        cache = kwargs.get("past_key_values")
        if not isinstance(cache, DynamicCache):
            raise _Fallback()
        marker = getattr(cache, _MARKER, None)
        if isinstance(marker, int) and not isinstance(marker, bool):
            if marker < 0 or marker + 1 >= self.decode_steps:
                raise _Fallback()
            if marker >= len(self.steps) or self.steps[marker].cache is not cache:
                raise _Fallback()
            index = marker + 1
        else:
            if not self._load_prefill(cache):
                raise _Fallback()
            index = 0
        self.static_input.copy_(input_ids, non_blocking=True)
        if index == len(self.steps):
            step = self._capture(index)
        else:
            step = self.steps[index]
            step.graph.replay()
        logits = step.logits
        if self.state.get("mode") == "quality":
            logits = logits.clone()
        return CausalLMOutputWithPast(loss=None, logits=logits, past_key_values=step.cache)

    def _load_prefill(self, cache):
        if getattr(cache, "offloading", False):
            return False
        layers = getattr(cache, "layers", None)
        if not isinstance(layers, list) or not layers:
            return False
        if self.num_layers is not None and len(layers) != self.num_layers:
            return False
        pairs = []
        for layer in layers:
            key = getattr(layer, "keys", None)
            value = getattr(layer, "values", None)
            if not isinstance(key, torch.Tensor) or not isinstance(value, torch.Tensor):
                return False
            if key.dim() != 4 or tuple(key.shape) != tuple(value.shape):
                return False
            if key.shape[0] != self.batch or int(key.shape[2]) != self.prompt_tokens:
                return False
            if key.device.type != "cuda" or not key.is_contiguous() or not value.is_contiguous():
                return False
            if self.dtype is None:
                self.dtype = key.dtype
            elif key.dtype != self.dtype:
                return False
            pairs.append((key, value))
        if self.static_k is None:
            self.num_layers = len(pairs)
            self.device = pairs[0][0].device
            self.static_k = [torch.empty_like(key) for key, _ in pairs]
            self.static_v = [torch.empty_like(value) for _, value in pairs]
            self.static_input = torch.zeros((self.batch, 1), dtype=torch.int64, device=self.device)
        else:
            for (key, _), static_key in zip(pairs, self.static_k):
                if tuple(key.shape) != tuple(static_key.shape):
                    return False
        for (key, value), static_key, static_value in zip(pairs, self.static_k, self.static_v):
            static_key.copy_(key, non_blocking=True)
            static_value.copy_(value, non_blocking=True)
        return True

    def _capture(self, index):
        if index == 0:
            base = list(zip(self.static_k, self.static_v))
        else:
            previous = self.steps[index - 1]
            base = [(layer.keys, layer.values) for layer in previous.cache.layers]
        if self.capture_stream is None:
            self.capture_stream = torch.cuda.Stream(device=self.device)
        if self.pool is None:
            self.pool = torch.cuda.graph_pool_handle()
        stream = self.capture_stream
        current = torch.cuda.current_stream()
        stream.wait_stream(current)
        with torch.cuda.stream(stream):
            for _ in range(3 if index == 0 else 1):
                self.parent(
                    input_ids=self.static_input,
                    past_key_values=self._make_cache(base),
                    use_cache=True,
                    logits_to_keep=1,
                )
        current.wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        capture_cache = self._make_cache(base)
        with torch.cuda.graph(graph, pool=self.pool, stream=stream):
            output = self.parent(
                input_ids=self.static_input,
                past_key_values=capture_cache,
                use_cache=True,
                logits_to_keep=1,
            )
        setattr(capture_cache, _MARKER, index)
        graph.replay()
        step = _Step(graph, output.logits, capture_cache)
        self.steps.append(step)
        return step

    @staticmethod
    def _make_cache(pairs):
        cache = DynamicCache()
        for key, value in pairs:
            layer = DynamicLayer()
            layer.keys = key
            layer.values = value
            layer.is_initialized = True
            layer.dtype = key.dtype
            layer.device = key.device
            cache.layers.append(layer)
        return cache

    def _parent(self, args, kwargs):
        cache = kwargs.get("past_key_values")
        if isinstance(getattr(cache, _MARKER, None), int) and not isinstance(getattr(cache, _MARKER, None), bool):
            kwargs = dict(kwargs)
            kwargs["past_key_values"] = self._clone_cache(cache)
        return self.parent(*args, **kwargs)

    @staticmethod
    def _clone_cache(cache):
        cloned = DynamicCache()
        for layer in cache.layers:
            key = layer.keys
            value = layer.values
            new_layer = DynamicLayer()
            new_layer.keys = key.clone()
            new_layer.values = value.clone()
            new_layer.is_initialized = True
            new_layer.dtype = key.dtype
            new_layer.device = key.device
            cloned.layers.append(new_layer)
        return cloned


def build(context):
    return {CALL_ID: _DecodeGraphRunner(context)}
