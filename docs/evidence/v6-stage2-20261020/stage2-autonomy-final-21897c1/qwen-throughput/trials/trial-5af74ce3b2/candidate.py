"""CUDA-graph decode step for the decode-phase Qwen3ForCausalLM boundary.

Replaces call 323bcbe006b343648b346af9 (group 3e293c00b1fb98fbe29f): the
decode-phase forward of the root Qwen3ForCausalLM for the frozen batch-1 /
256-prompt / 32-token greedy workload. The parent forward itself (via
context.references) is captured once inside the adapter's decode phase over a
StaticCache seeded from each run's prefill DynamicCache; every decode step
then copies the current token into a static buffer and replays the graph, so
the identical eager kernels execute with per-step host dispatch and launch
gaps removed. Signature-incompatible calls, setup failures and runtime
failures fall back to the parent reference.
"""
from __future__ import annotations

import weakref

import torch

CALL_ID = "323bcbe006b343648b346af9"
MAX_CACHE_LEN = 288
DECODE_STEPS = 31
WARMUP_STEPS = 3


def build(context):
    from transformers.cache_utils import DynamicCache, StaticCache
    from transformers.modeling_outputs import CausalLMOutputWithPast

    record_route = context.get("record_route", lambda *_: None)
    module = context["modules"][CALL_ID]
    reference = context["references"][CALL_ID]
    config = module.config
    num_layers = int(config.num_hidden_layers)
    kv_heads = int(config.num_key_value_heads)
    head_dim = int(getattr(config, "head_dim", 0) or (config.hidden_size // config.num_attention_heads))
    dtype = torch.bfloat16
    cuda_ok = torch.cuda.is_available()

    state = {"graph": None, "cache": None, "input_ids": None, "logits": None,
             "run": None, "ready": False, "broken": None}

    def reset_counters(value):
        for layer in state["cache"].layers:
            layer.cumulative_length.fill_(value)

    def parent_step():
        return reference(input_ids=state["input_ids"], past_key_values=state["cache"],
                         use_cache=True, logits_to_keep=1)

    def capture_graph(prompt_len):
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(WARMUP_STEPS):
                reset_counters(prompt_len)
                parent_step()
        torch.cuda.current_stream().wait_stream(side)
        reset_counters(prompt_len)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            state["logits"] = parent_step().logits
        state["graph"] = graph

    def compatible(past):
        if not isinstance(past, DynamicCache) or len(past.layers) != num_layers:
            return None
        lengths = set()
        for layer in past.layers:
            keys = layer.keys
            if (keys.shape[0] != 1 or keys.shape[1] != kv_heads or keys.shape[-1] != head_dim
                    or keys.dtype != dtype or layer.values.dtype != dtype or not keys.is_cuda):
                return None
            lengths.add(int(keys.shape[-2]))
        return lengths.pop() if len(lengths) == 1 else None

    def decode_forward(*args, **kwargs):
        expected = {"input_ids", "past_key_values", "use_cache", "logits_to_keep"}
        if (args or set(kwargs) != expected or kwargs["use_cache"] is not True
                or kwargs["logits_to_keep"] != 1 or module.training):
            record_route(CALL_ID, "fallback:unsupported-signature")
            return reference(*args, **kwargs)
        input_ids = kwargs["input_ids"]
        past = kwargs["past_key_values"]
        if not (isinstance(input_ids, torch.Tensor) and tuple(input_ids.shape) == (1, 1)
                and input_ids.dtype == torch.int64 and input_ids.is_cuda):
            record_route(CALL_ID, "fallback:unsupported-signature")
            return reference(*args, **kwargs)
        if state["broken"] is not None:
            record_route(CALL_ID, "fallback:setup-failure:" + state["broken"])
            return reference(*args, **kwargs)
        run = state["run"]
        if run is None or run() is not past:
            state["run"] = weakref.ref(past)
            state["ready"] = False
            prompt_len = compatible(past)
            if prompt_len is not None and prompt_len + DECODE_STEPS <= MAX_CACHE_LEN and cuda_ok:
                try:
                    if state["cache"] is None:
                        state["cache"] = StaticCache(config=config, max_cache_len=MAX_CACHE_LEN)
                        state["cache"].early_initialization(
                            batch_size=1, num_heads=kv_heads, head_dim=head_dim,
                            dtype=dtype, device=input_ids.device)
                        state["input_ids"] = torch.zeros((1, 1), dtype=torch.int64,
                                                         device=input_ids.device)
                    if state["input_ids"].device != input_ids.device:
                        record_route(CALL_ID, "fallback:parent-eager")
                        return reference(*args, **kwargs)
                    for dst, src in zip(state["cache"].layers, past.layers):
                        dst.keys[:, :, :prompt_len].copy_(src.keys)
                        dst.values[:, :, :prompt_len].copy_(src.values)
                    reset_counters(prompt_len)
                    state["input_ids"].copy_(input_ids)
                    if state["graph"] is None:
                        capture_graph(prompt_len)
                        record_route(CALL_ID, "cuda_graph:captured")
                    else:
                        record_route(CALL_ID, "cuda_graph:reseeded")
                    state["ready"] = True
                except Exception as exc:
                    state["graph"] = None
                    state["broken"] = type(exc).__name__
                    record_route(CALL_ID, "fallback:setup-failure:" + type(exc).__name__)
                    return reference(*args, **kwargs)
        if not state["ready"] or state["input_ids"].device != input_ids.device:
            record_route(CALL_ID, "fallback:parent-eager")
            return reference(*args, **kwargs)
        try:
            state["input_ids"].copy_(input_ids)
            state["graph"].replay()
        except Exception as exc:
            state["broken"] = type(exc).__name__
            record_route(CALL_ID, "fallback:runtime-failure:" + type(exc).__name__)
            return reference(*args, **kwargs)
        record_route(CALL_ID, "cuda_graph:replay")
        return CausalLMOutputWithPast(logits=state["logits"].clone(), past_key_values=past)

    return {CALL_ID: decode_forward}
