"""Whole-decode-step replacement executing the UNCHANGED parent forward on a
StaticCache window, captured once into a CUDA graph and replayed per step.

Target: the decode-phase Qwen3ForCausalLM module boundary (call
323bcbe006b343648b346af9, group 3e293c00b1fb98fbe29f). Repair of trial
3c2ed50501, whose hand-written composite diverged at the first graph-produced
step (max abs error 1.65625); this candidate never re-implements decode math.
Every kernel is the parent's own: the replacement calls the bound parent
forward with a preallocated transformers StaticCache (capacity
prompt_tokens+new_tokens+8) whose per-layer cumulative_length is a device
scalar updated in-place, so cache_position, position_ids and the causal mask
are derived on device from live cache state inside the captured region and
stay value-dynamic across replays.

Lifecycle: the static window is allocated lazily on the first compatible
decode call; the first decode step of every invocation re-seeds the window
from the invocation's own prefill DynamicCache (per-layer slice copies plus a
device position reset), so the workload always executes on current inputs.
Steps 0..1 of the first invocation run the parent eagerly on the static cache;
at step 2 the parent step is warmed up on a side stream, captured into a CUDA
graph, and verified by comparing the replayed logits against the eager output
on identical inputs (torch.equal). On success the replay is that step's
execution; every later decode step of every invocation copies the incoming
token into the static input buffer and replays. Any signature mismatch, setup
failure, capacity problem, capture/verify failure or runtime error falls back
to the unchanged parent on the original arguments; the runtime fallback first
rewinds the device cache positions so cache state stays consistent with the
returned cache object.
"""

from __future__ import annotations

import torch
from transformers.cache_utils import DynamicCache, StaticCache
from transformers.modeling_outputs import CausalLMOutputWithPast

DECODE_CALL_ID = "323bcbe006b343648b346af9"
EAGER_STEPS_BEFORE_CAPTURE = 2
CAPACITY_SLACK = 8
WARMUP_ITERS = 2

_ALLOWED_KWARGS = {"input_ids", "past_key_values", "use_cache", "logits_to_keep"}


def build(context):
    record_route = context.get("record_route") or (lambda *_: None)
    module = (context.get("modules") or {}).get(DECODE_CALL_ID)
    parent = (context.get("references") or {}).get(DECODE_CALL_ID)
    if module is None or parent is None:
        return {}

    state = context.get("state") or {}
    try:
        cfg = module.config
        n_layers = int(cfg.num_hidden_layers)
        n_kv = int(cfg.num_key_value_heads)
        n_heads = int(cfg.num_attention_heads)
        head_dim = int(getattr(cfg, "head_dim", 0) or (cfg.hidden_size // n_heads))
        base = module.model
        embed = base.embed_tokens
        dtype = embed.weight.dtype
        device = embed.weight.device
        decoder_layers = list(base.layers)
        if len(decoder_layers) != n_layers:
            raise ValueError("layer count mismatch")
        # Probe every submodule the parent decode path touches, at build time.
        for layer in decoder_layers:
            attn = layer.self_attn
            _ = (attn.q_proj, attn.k_proj, attn.v_proj, attn.o_proj, attn.q_norm, attn.k_norm)
            _ = (layer.mlp, layer.input_layernorm, layer.post_attention_layernorm)
        _ = (base.norm, base.rotary_emb, module.lm_head)
    except Exception:
        return {}

    prompt_tokens = int(state.get("prompt_tokens") or 0)
    new_tokens = int(state.get("new_tokens") or 0)
    if prompt_tokens <= 0 or new_tokens <= 1:
        # Nothing to accelerate: prefill-only workloads never reach the decode site.
        return {}
    capacity = prompt_tokens + new_tokens + CAPACITY_SLACK

    st = {
        "ready": False,
        "init_failed": False,
        "capture_failed": False,
        "graph": None,
        "static_out": None,
        "cache": None,
        "input_ids": None,
        "pos": -1,
        "steps": 0,
        "capacity": capacity,
        "device": None,
    }

    def _reset_positions(p):
        # Rewind every layer's device-side cumulative_length scalar; KV slots
        # beyond p are masked out by the parent's causal mask until rewritten.
        for lyr in st["cache"].layers:
            lyr.cumulative_length.fill_(p)

    def _init(input_ids):
        dev = input_ids.device
        cache = StaticCache(config=cfg, max_cache_len=st["capacity"])
        # Allocate the full window now (zeros) and move cumulative_length
        # scalars to the device so no allocation happens inside the graph.
        cache.early_initialization(
            batch_size=1, num_heads=n_kv, head_dim=head_dim, dtype=dtype, device=dev
        )
        for lyr in cache.layers:
            if not lyr.is_initialized:
                raise RuntimeError("static layer not initialized")
        st["cache"] = cache
        st["input_ids"] = torch.zeros((1, 1), dtype=torch.int64, device=dev)
        st["device"] = dev
        st["ready"] = True

    def _import(past):
        if not isinstance(past, DynamicCache):
            return False
        layers = getattr(past, "layers", None)
        if layers is None or len(layers) != n_layers:
            return False
        length = int(past.get_seq_length())
        if length <= 0 or length >= st["capacity"]:
            return False
        for lyr in layers:
            keys = getattr(lyr, "keys", None)
            values = getattr(lyr, "values", None)
            if keys is None or values is None or not getattr(lyr, "is_initialized", False):
                return False
            if tuple(keys.shape) != (1, n_kv, length, head_dim) or keys.dtype != dtype:
                return False
            if tuple(values.shape) != (1, n_kv, length, head_dim) or values.dtype != dtype:
                return False
            if keys.device != st["device"] or values.device != st["device"]:
                return False
        static_layers = st["cache"].layers
        for i, lyr in enumerate(layers):
            static_layers[i].keys[:, :, :length].copy_(lyr.keys)
            static_layers[i].values[:, :, :length].copy_(lyr.values)
        _reset_positions(length)
        st["pos"] = length
        return True

    def _step():
        # The unchanged parent decode forward on the static buffers.
        return parent(
            input_ids=st["input_ids"],
            past_key_values=st["cache"],
            use_cache=True,
            logits_to_keep=1,
        )

    def _rewind_and_sync():
        try:
            torch.cuda.synchronize()
        except Exception:
            pass
        if st["ready"] and st["cache"] is not None and st["pos"] >= 0:
            try:
                _reset_positions(st["pos"])
            except Exception:
                pass

    def _try_capture():
        """Warm up, capture and verify the parent step. Positions are left at
        pos on every return path so the caller executes exactly one step."""
        p = st["pos"]
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        warm = None
        with torch.cuda.stream(stream):
            for _ in range(WARMUP_ITERS):
                _reset_positions(p)
                warm = _step()
        torch.cuda.current_stream().wait_stream(stream)
        _reset_positions(p)
        graph = torch.cuda.CUDAGraph()
        try:
            with torch.cuda.graph(graph, capture_error_mode="thread_local"):
                out = _step()
        except Exception:
            _rewind_and_sync()
            st["capture_failed"] = True
            record_route(DECODE_CALL_ID, "fallback_capture")
            return None, None
        # Verify the replayed graph reproduces the eager parent on the same
        # inputs before it is ever trusted (repair guard for the previous
        # graph-path divergence).
        _reset_positions(p)
        graph.replay()
        ok = False
        try:
            ok = torch.equal(out.logits, warm.logits)
        except Exception:
            ok = False
        _reset_positions(p)
        if not ok:
            st["capture_failed"] = True
            record_route(DECODE_CALL_ID, "fallback_verify")
            return None, None
        return graph, out

    def _cuda_ready():
        return st["device"] is not None and st["device"].type == "cuda" and torch.cuda.is_available()

    def _run(*args, **kwargs):
        if args or not _ALLOWED_KWARGS.issuperset(kwargs):
            record_route(DECODE_CALL_ID, "fallback_signature")
            return parent(*args, **kwargs)
        input_ids = kwargs.get("input_ids")
        past = kwargs.get("past_key_values")
        use_cache = kwargs.get("use_cache")
        logits_to_keep = kwargs.get("logits_to_keep")
        if not (
            isinstance(input_ids, torch.Tensor)
            and input_ids.ndim == 2
            and tuple(input_ids.shape) == (1, 1)
            and input_ids.dtype == torch.int64
            and use_cache is True
            and type(logits_to_keep) is int
            and logits_to_keep == 1
        ):
            record_route(DECODE_CALL_ID, "fallback_signature")
            return parent(*args, **kwargs)

        continuing = past is st["cache"]
        if not continuing and not isinstance(past, DynamicCache):
            record_route(DECODE_CALL_ID, "fallback_signature")
            return parent(*args, **kwargs)

        if not st["ready"]:
            if st["init_failed"]:
                record_route(DECODE_CALL_ID, "fallback_setup")
                return parent(*args, **kwargs)
            try:
                _init(input_ids)
            except Exception:
                st["init_failed"] = True
                record_route(DECODE_CALL_ID, "fallback_setup")
                return parent(*args, **kwargs)

        if input_ids.device != st["device"]:
            record_route(DECODE_CALL_ID, "fallback_signature")
            return parent(*args, **kwargs)

        if not continuing:
            try:
                imported = _import(past)
            except Exception:
                imported = False
            if not imported:
                record_route(DECODE_CALL_ID, "fallback_signature")
                return parent(*args, **kwargs)
            st["steps"] = 0

        pos = st["pos"]
        if pos < 0 or pos + 1 > st["capacity"]:
            record_route(DECODE_CALL_ID, "fallback_capacity")
            result = parent(*args, **kwargs)
            if past is st["cache"]:
                st["pos"] = pos + 1
                st["steps"] += 1
            return result

        st["input_ids"].copy_(input_ids)
        try:
            if st["graph"] is not None:
                st["graph"].replay()
                logits = st["static_out"].logits.clone()
                route = "cuda_graph_replay"
            elif (
                st["steps"] >= EAGER_STEPS_BEFORE_CAPTURE
                and not st["capture_failed"]
                and _cuda_ready()
            ):
                graph, out = _try_capture()
                if graph is not None:
                    st["graph"] = graph
                    st["static_out"] = out
                    graph.replay()
                    logits = out.logits.clone()
                    route = "cuda_graph_capture"
                else:
                    result = _step()
                    logits = result.logits.clone()
                    route = "eager_static_parent"
            else:
                result = _step()
                logits = result.logits.clone()
                route = "eager_static_parent"
        except Exception:
            # Repair state so the cache object we may have returned earlier
            # stays consistent, then run the unchanged parent for this step.
            _rewind_and_sync()
            st["graph"] = None
            st["static_out"] = None
            st["capture_failed"] = True
            record_route(DECODE_CALL_ID, "fallback_runtime")
            result = parent(*args, **kwargs)
            st["pos"] = pos + 1
            st["steps"] += 1
            return result
        record_route(DECODE_CALL_ID, route)

        st["pos"] = pos + 1
        st["steps"] += 1
        return CausalLMOutputWithPast(
            loss=None,
            logits=logits,
            past_key_values=st["cache"],
            hidden_states=None,
            attentions=None,
        )

    def replacement(*args, **kwargs):
        try:
            return _run(*args, **kwargs)
        except Exception:
            record_route(DECODE_CALL_ID, "fallback_runtime")
            return parent(*args, **kwargs)

    return {DECODE_CALL_ID: replacement}
