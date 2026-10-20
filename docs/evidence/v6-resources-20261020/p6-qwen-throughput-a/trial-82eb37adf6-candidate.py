"""Whole-decode-step CUDA-graph replay for the frozen Qwen3-4B offline workload.

Repair of trial-eeb66b190e (inputs/previous-feedback.txt): that trial replays
per-position CUDA graphs around the pristine decode forward but returned the
static graph-pool logits buffer to the adapter, whose quality path appends
detached views of output.logits[:, -1] and stacks them only after the final
decode step, so every stacked row read the last replay's contents (26.8%
mismatched elements, max abs error 4.0 on both paired repeats). This candidate
keeps the identical execution policy and changes exactly that observation
path: the per-step logits are unconditionally materialized into freshly
allocated contiguous memory on every invocation, never a view of pool memory
and never mode-dependent. The KV data path is not rewritten: the pristine
parent forward, including its genuine DynamicCache torch.cat appends, is what
gets captured, and the only fallback is the pristine parent itself.
"""

from __future__ import annotations

import torch
from transformers.cache_utils import DynamicCache
from transformers.modeling_outputs import CausalLMOutputWithPast

CALL_ID = "323bcbe006b343648b346af9"
_EXPECTED_KWARGS = frozenset(
    ("input_ids", "past_key_values", "use_cache", "logits_to_keep")
)


class _DecodeGraphEngine:
    def __init__(self, model, parent, prompt_tokens, n_steps, route):
        self._model = model
        self._parent = parent
        self._prompt = int(prompt_tokens)
        self._n_steps = int(n_steps)
        self._route = route
        self._device = next(model.parameters()).device
        self._n_layers = int(getattr(model.config, "num_hidden_layers", 0))
        self._static_ids = None
        self._static_k = None
        self._static_v = None
        self._marker = None
        self._pool = None
        self._graphs = []
        self._logits_bufs = []
        self._kv_history = []
        self._last_good = None
        self._step = 0
        self._disabled = False
        self._warmup_stream = None

    def _fallback(self, args, kwargs):
        self._route(CALL_ID, "fallback_parent")
        if self._marker is not None and kwargs.get("past_key_values") is self._marker:
            kwargs = dict(kwargs)
            kwargs["past_key_values"] = self._clone_cache(self._marker)
        return self._parent(*args, **kwargs)

    def _clone_cache(self, cache):
        clone = DynamicCache(config=self._model.config)
        for i, layer in enumerate(cache.layers):
            dst = clone.layers[i]
            if layer.keys is not None and layer.values is not None:
                dst.keys = layer.keys.detach().clone()
                dst.values = layer.values.detach().clone()
                dst.is_initialized = True
        return clone

    def _restore_last_good(self):
        if self._marker is not None and self._last_good is not None:
            for i, layer in enumerate(self._marker.layers):
                if i < len(self._last_good):
                    layer.keys, layer.values = self._last_good[i]

    def _begin_run(self, layers, input_ids):
        n = len(layers)
        if self._static_k is None:
            if not input_ids.is_cuda or input_ids.device != layers[0].keys.device:
                return False
            for i in range(n):
                k, v = layers[i].keys, layers[i].values
                if not torch.is_tensor(k) or not torch.is_tensor(v):
                    return False
                if not k.is_cuda or k.device != self._device:
                    return False
                if k.dim() != 4 or v.dim() != 4 or v.shape != k.shape:
                    return False
            self._static_ids = torch.empty_like(input_ids)
            self._static_k = [torch.empty_like(layers[i].keys) for i in range(n)]
            self._static_v = [torch.empty_like(layers[i].values) for i in range(n)]
            self._marker = DynamicCache(config=self._model.config)
            if len(self._marker.layers) != n:
                return False
        else:
            if input_ids.shape != self._static_ids.shape:
                return False
            if input_ids.dtype != self._static_ids.dtype:
                return False
            for i in range(n):
                k, v = layers[i].keys, layers[i].values
                sk, sv = self._static_k[i], self._static_v[i]
                if k.shape != sk.shape or k.dtype != sk.dtype or k.device != sk.device:
                    return False
                if v.shape != sv.shape or v.dtype != sv.dtype or v.device != sv.device:
                    return False
        for i in range(n):
            self._static_k[i].copy_(layers[i].keys)
            self._static_v[i].copy_(layers[i].values)
        self._static_ids.copy_(input_ids)
        for i, layer in enumerate(self._marker.layers):
            layer.keys = self._static_k[i]
            layer.values = self._static_v[i]
            layer.is_initialized = True
        self._last_good = list(zip(self._static_k, self._static_v))
        self._step = 0
        return True

    def _capture(self, input_ids):
        s = self._step
        if self._pool is None:
            self._pool = torch.cuda.graph_pool_handle()
        if self._warmup_stream is None:
            self._warmup_stream = torch.cuda.Stream(device=self._device)
        self._static_ids.copy_(input_ids)
        saved = [(layer.keys, layer.values) for layer in self._marker.layers]
        iters = 2 if s == 0 else 1
        current = torch.cuda.current_stream(self._device)
        ws = self._warmup_stream
        ws.wait_stream(current)
        with torch.cuda.stream(ws):
            for _ in range(iters):
                self._parent(
                    input_ids=self._static_ids,
                    past_key_values=self._marker,
                    use_cache=True,
                    logits_to_keep=1,
                )
        current.wait_stream(ws)
        for i, layer in enumerate(self._marker.layers):
            layer.keys, layer.values = saved[i]
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, pool=self._pool):
            out = self._parent(
                input_ids=self._static_ids,
                past_key_values=self._marker,
                use_cache=True,
                logits_to_keep=1,
            )
        graph.replay()
        torch.cuda.synchronize(self._device)
        self._graphs.append(graph)
        self._logits_bufs.append(out.logits)
        self._kv_history.append([(layer.keys, layer.values) for layer in self._marker.layers])
        return out.logits

    def _guarded(self, args, kwargs):
        if self._disabled:
            return False, None
        if args:
            return False, None
        if set(kwargs.keys()) != _EXPECTED_KWARGS:
            return False, None
        input_ids = kwargs["input_ids"]
        past = kwargs["past_key_values"]
        if kwargs["use_cache"] is not True or kwargs["logits_to_keep"] != 1:
            return False, None
        if not torch.is_tensor(input_ids):
            return False, None
        if input_ids.dim() != 2 or input_ids.shape[1] != 1 or input_ids.dtype != torch.int64:
            return False, None
        if not isinstance(past, DynamicCache):
            return False, None
        layers = past.layers
        if len(layers) != self._n_layers:
            return False, None

        if past is not self._marker:
            k0 = layers[0].keys
            if not torch.is_tensor(k0) or k0.dim() != 4 or k0.shape[-2] != self._prompt:
                return False, None
            if not self._begin_run(layers, input_ids):
                return False, None
        else:
            if self._marker.layers[0].keys.shape[-2] != self._prompt + self._step:
                return False, None
            if input_ids.shape != self._static_ids.shape:
                return False, None

        if self._step >= self._n_steps:
            return False, None

        if self._step == len(self._graphs):
            logits_buf = self._capture(input_ids)
            label = "cuda_graph_capture"
        else:
            self._static_ids.copy_(input_ids)
            self._graphs[self._step].replay()
            logits_buf = self._logits_bufs[self._step]
            label = "cuda_graph_replay"

        refs = self._kv_history[self._step]
        for i, layer in enumerate(self._marker.layers):
            layer.keys, layer.values = refs[i]
        logits = logits_buf.clone()
        self._last_good = list(refs)
        self._step += 1
        self._route(CALL_ID, label)
        return True, CausalLMOutputWithPast(logits=logits, past_key_values=self._marker)

    def step(self, args, kwargs):
        try:
            ok, result = self._guarded(args, kwargs)
        except Exception:
            self._disabled = True
            self._restore_last_good()
            return self._fallback(args, kwargs)
        if not ok:
            return self._fallback(args, kwargs)
        return result


def build(context):
    state = context["state"]
    model = context["modules"].get(CALL_ID)
    parent = context["references"].get(CALL_ID)
    if model is None or parent is None:
        return {}
    route = context.get("record_route", lambda *_: None)

    prompt_tokens = int(state.get("prompt_tokens", 256))
    new_tokens = int(state.get("new_tokens", 32))
    engine = _DecodeGraphEngine(
        model,
        parent,
        prompt_tokens=prompt_tokens,
        n_steps=max(new_tokens - 1, 0),
        route=route,
    )

    def _decode_step(*args, **kwargs):
        return engine.step(args, kwargs)

    return {CALL_ID: _decode_step}
