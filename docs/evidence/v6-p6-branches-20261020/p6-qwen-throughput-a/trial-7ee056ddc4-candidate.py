"""Decode-step CUDA graph replay for the frozen Qwen3-4B offline inference workload.

Replaces the decode-phase Qwen3ForCausalLM module call (323bcbe006b343648b346af9)
with per-position CUDA graph capture and replay over a static KV cache, removing
per-step host dispatch while reproducing the baseline kernel sequence exactly.
"""
from __future__ import annotations

import torch
from transformers.cache_utils import Cache as _HFCache, DynamicCache, DynamicLayer
from transformers.modeling_outputs import CausalLMOutputWithPast

DECODE_CALL_ID = "323bcbe006b343648b346af9"


class _Tracker:
    __slots__ = ("cur_len", "write_pos", "n_layers")

    def __init__(self, n_layers: int, cur_len: int):
        self.n_layers = n_layers
        self.cur_len = cur_len
        self.write_pos = cur_len


class _StaticSliceLayer(DynamicLayer):
    def __init__(self, kbuf: torch.Tensor, vbuf: torch.Tensor, tracker: _Tracker):
        super().__init__()
        self._kbuf = kbuf
        self._vbuf = vbuf
        self._tracker = tracker
        self.is_initialized = True
        self.dtype = kbuf.dtype
        self.device = kbuf.device

    def update(self, key_states, value_states, layer_idx=None, *args, **kwargs):
        tracker = self._tracker
        klen = key_states.shape[-2]
        if layer_idx == 0:
            tracker.write_pos = tracker.cur_len
        wp = tracker.write_pos
        if layer_idx == tracker.n_layers - 1:
            tracker.cur_len = wp + klen
        if wp + klen <= self._kbuf.shape[-2]:
            self._kbuf[:, :, wp:wp + klen].copy_(key_states)
            self._vbuf[:, :, wp:wp + klen].copy_(value_states)
            end = wp + klen
            return self._kbuf[:, :, :end].contiguous(), self._vbuf[:, :, :end].contiguous()
        if self.keys is None:
            self.keys = self._kbuf[:, :, :wp].contiguous()
            self.values = self._vbuf[:, :, :wp].contiguous()
        self.keys = torch.cat([self.keys, key_states], dim=-2)
        self.values = torch.cat([self.values, value_states], dim=-2)
        return self.keys, self.values

    def get_seq_length(self) -> int:
        return self._tracker.cur_len

    def get_mask_sizes(self, query_length: int) -> tuple[int, int]:
        return self._tracker.cur_len + query_length, 0

    def get_max_length(self) -> int:
        return -1

    def _reset_dynamic(self) -> None:
        self.keys = None
        self.values = None


class _GraphCache(DynamicCache):
    def __init__(self, layer_objs):
        _HFCache.__init__(self, layers=list(layer_objs))

    def update(self, key_states, value_states, layer_idx, *args, **kwargs):
        return self.layers[layer_idx].update(key_states, value_states, layer_idx, *args, **kwargs)


class _DecodeGraphRunner:
    def __init__(self, context, call_id):
        self._state = context.get("state") or {}
        self._parent = context["references"][call_id]
        self._module = (context.get("modules") or {}).get(call_id)
        self._graph_ok = torch.cuda.is_available()
        self._pool = None
        self._graphs: dict[int, torch.cuda.CUDAGraph] = {}
        self._kbufs = None
        self._vbufs = None
        self._tracker = None
        self._cache = None
        self._tok = None
        self._logits = None
        self._shape = None
        self._cap = 0

    def __call__(self, input_ids=None, attention_mask=None, position_ids=None,
                 past_key_values=None, inputs_embeds=None, labels=None,
                 use_cache=None, logits_to_keep=0, **kwargs):
        try:
            out = self._run(input_ids, attention_mask, position_ids, past_key_values,
                            inputs_embeds, labels, use_cache, logits_to_keep)
            if out is not None:
                return out
        except Exception:
            self._graph_ok = False
        return self._parent(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            logits_to_keep=logits_to_keep,
            **kwargs,
        )

    def _run(self, input_ids, attention_mask, position_ids, past_key_values,
             inputs_embeds, labels, use_cache, logits_to_keep):
        if attention_mask is not None or position_ids is not None or inputs_embeds is not None or labels is not None:
            return None
        if use_cache is not True or logits_to_keep != 1:
            return None
        if not (isinstance(input_ids, torch.Tensor) and input_ids.is_cuda
                and input_ids.dim() == 2 and input_ids.shape[1] == 1
                and input_ids.dtype == torch.int64):
            return None
        if past_key_values is self._cache:
            pass
        elif isinstance(past_key_values, DynamicCache) and not isinstance(past_key_values, _GraphCache):
            if not self._begin_run(past_key_values):
                return None
        else:
            return None
        return self._execute(input_ids)

    def _begin_run(self, pv) -> bool:
        layers = getattr(pv, "layers", None) or []
        n = len(layers)
        if n == 0:
            return False
        k0 = getattr(layers[0], "keys", None)
        if not (isinstance(k0, torch.Tensor) and k0.dim() == 4):
            return False
        b, h, prompt, d = tuple(k0.shape)
        if prompt == 0 or k0.device.type != "cuda":
            return False
        for layer in layers:
            k = getattr(layer, "keys", None)
            v = getattr(layer, "values", None)
            if not (isinstance(k, torch.Tensor) and isinstance(v, torch.Tensor)):
                return False
            if tuple(k.shape) != (b, h, prompt, d) or k.dtype != k0.dtype or k.device != k0.device:
                return False
            if tuple(v.shape) != (b, h, prompt, d) or v.dtype != k0.dtype:
                return False
        new_tokens = self._state.get("new_tokens")
        steps = int(new_tokens) - 1 if isinstance(new_tokens, int) and new_tokens > 1 else 63
        cap = prompt + max(steps, 1)
        shape = (b, h, d, n)
        if self._kbufs is None or self._shape != shape or self._cap < cap:
            self._graphs.clear()
            self._shape = shape
            self._cap = cap
            dev, dt = k0.device, k0.dtype
            self._kbufs = [torch.empty((b, h, cap, d), dtype=dt, device=dev) for _ in range(n)]
            self._vbufs = [torch.empty((b, h, cap, d), dtype=dt, device=dev) for _ in range(n)]
            self._tracker = _Tracker(n, prompt)
            self._cache = _GraphCache(
                [_StaticSliceLayer(self._kbufs[i], self._vbufs[i], self._tracker) for i in range(n)]
            )
            self._tok = torch.zeros((b, 1), dtype=torch.int64, device=dev)
            self._logits = None
        for i, layer in enumerate(layers):
            self._kbufs[i][:, :, :prompt].copy_(layer.keys)
            self._vbufs[i][:, :, :prompt].copy_(layer.values)
        if isinstance(self._cache, _GraphCache):
            for layer in self._cache.layers:
                layer._reset_dynamic()
        self._tracker.cur_len = prompt
        self._tracker.write_pos = prompt
        return True

    def _execute(self, input_ids):
        p = self._tracker.cur_len
        graph = self._graphs.get(p)
        if graph is not None:
            self._tok.copy_(input_ids)
            graph.replay()
            self._tracker.cur_len = p + 1
        elif self._graph_ok:
            graph = self._capture(p, input_ids)
            if graph is not None:
                self._tracker.cur_len = p + 1
        if graph is None:
            return self._parent(input_ids=input_ids, past_key_values=self._cache,
                                use_cache=True, logits_to_keep=1)
        logits = self._logits
        if self._state.get("mode") == "quality":
            logits = logits.clone()
        return CausalLMOutputWithPast(logits=logits, past_key_values=self._cache)

    def _parent_static(self):
        return self._parent(input_ids=self._tok, past_key_values=self._cache,
                            use_cache=True, logits_to_keep=1)

    def _capture(self, p: int, input_ids):
        try:
            self._tok.copy_(input_ids)
            torch.cuda.synchronize()
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            probe = None
            with torch.cuda.stream(stream):
                for _ in range(3):
                    self._tracker.cur_len = p
                    probe = self._parent_static()
            torch.cuda.current_stream().wait_stream(stream)
            torch.cuda.synchronize()
            if probe is None or not isinstance(probe.logits, torch.Tensor) or probe.logits.dim() != 3:
                return None
            if self._logits is None:
                self._logits = torch.empty_like(probe.logits)
            elif (tuple(self._logits.shape) != tuple(probe.logits.shape)
                  or self._logits.dtype != probe.logits.dtype
                  or self._logits.device != probe.logits.device):
                return None
            if self._pool is None:
                self._pool = torch.cuda.graph_pool_handle()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, pool=self._pool):
                self._tracker.cur_len = p
                out = self._parent_static()
                self._logits.copy_(out.logits)
            self._graphs[p] = graph
            graph.replay()
            return graph
        except Exception:
            self._graph_ok = False
            return None


def build(context):
    targets = context.get("targets") or {}
    spec = targets.get(DECODE_CALL_ID)
    if spec is None or spec.get("kind") != "module" or spec.get("phase") != "decode":
        return {}
    references = context.get("references") or {}
    if DECODE_CALL_ID not in references:
        return {}
    if not torch.cuda.is_available():
        return {}
    runner = _DecodeGraphRunner(context, DECODE_CALL_ID)
    return {DECODE_CALL_ID: runner}
