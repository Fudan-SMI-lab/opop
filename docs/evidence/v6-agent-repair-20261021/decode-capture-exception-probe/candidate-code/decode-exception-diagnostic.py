"""Whole-decode-step replacement running the unchanged parent Qwen3 decode
forward on a preallocated in-place KV window, captured into per-length CUDA
graphs and replayed per step.

Repair of trial trial-87dde5b43d at the decode-phase Qwen3ForCausalLM module
boundary (call 323bcbe006b343648b346af9, group 3e293c00b1fb98fbe29f). The
measured feedback records two concrete failures. First, returning a
StaticCache changed the live call interface, so the framework's observed-
signature guard routed 60 of 62 decode calls to the parent and the graph
never ran. Second, the StaticCache path itself attends over the padded
capacity with a bool mask and repeat_kv instead of the DynamicCache path's
unmasked GQA sdpa over the exact valid length, which diverged far beyond the
0.02 logits tolerance (23.3% of elements, max abs error 1.0). The developer
diagnosis measured that keeping static storage but attending over the valid
KV prefix with the original attention restores two full generations to zero
error.

This candidate therefore presents the parent a DynamicCache whose layers
write in place into a preallocated window and expose only the valid-length
prefix, so every kernel, shape and mask decision matches the reference path,
and it always returns the caller's own DynamicCache object: imported from
that invocation's prefill cache at the first decode step, re-synced from the
window at generation end and before any parent fallback, so the observed
signature never changes and mid-run fallbacks stay correct. One CUDA graph
is captured per decode attention length on the first compatible invocation,
each verified with torch.equal against the eager parent on identical inputs
before it is trusted; every later step is one device token copy plus one
graph replay plus a logits clone. Any signature mismatch, setup, capture or
verify failure, capacity problem or runtime error repairs the caller cache
from the window and falls back to the unchanged parent.
"""

from __future__ import annotations

import torch
from transformers.cache_utils import DynamicCache, DynamicLayer
from transformers.modeling_outputs import CausalLMOutputWithPast

DECODE_CALL_ID = "323bcbe006b343648b346af9"
WARMUP_ITERS = 2
CAPACITY_SLACK = 8

_ALLOWED_KWARGS = {"input_ids", "past_key_values", "use_cache", "logits_to_keep"}


class _WindowLayer(DynamicLayer):
    """DynamicLayer-compatible view over one preallocated KV window slot pair.

    update() copies the new key/value states in place at the current write
    position and returns contiguous valid-length prefix views, so the parent
    attention sees exactly the tensors a DynamicCache would have produced
    (same shape, stride, dtype and valid length, no padding, no mask).
    """

    def __init__(self, keys_window, values_window):
        self._keys_window = keys_window
        self._values_window = values_window
        self.dtype = keys_window.dtype
        self.device = keys_window.device
        self._length = 0
        self.keys = keys_window[:, :, :0]
        self.values = values_window[:, :, :0]
        self.is_initialized = True

    def lazy_initialization(self, key_states, value_states):
        self.is_initialized = True

    def update(self, key_states, value_states, *args, **kwargs):
        added = key_states.shape[-2]
        end = self._length + added
        if end > self._keys_window.shape[-2]:
            raise RuntimeError("kv window capacity exceeded")
        self._keys_window[:, :, self._length:end].copy_(key_states)
        self._values_window[:, :, self._length:end].copy_(value_states)
        self._length = end
        self.keys = self._keys_window[:, :, :end]
        self.values = self._values_window[:, :, :end]
        return self.keys, self.values

    def get_seq_length(self):
        return self._length

    def get_mask_sizes(self, query_length):
        return self._length + query_length, 0

    def get_max_length(self):
        return self._keys_window.shape[-2]

    def set_seq_length(self, length):
        self._length = length
        self.keys = self._keys_window[:, :, :length]
        self.values = self._values_window[:, :, :length]


def build(context):
    record_route = context.get("record_route") or (lambda *_: None)
    modules = context.get("modules") or {}
    references = context.get("references") or {}
    module = modules.get(DECODE_CALL_ID)
    parent = references.get(DECODE_CALL_ID)
    if module is None or parent is None:
        return {}

    state = context.get("state") or {}
    prompt_tokens = int(state.get("prompt_tokens") or 0)
    new_tokens = int(state.get("new_tokens") or 0)
    if prompt_tokens <= 0 or new_tokens <= 1:
        return {}
    steps = new_tokens - 1

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
        for layer in decoder_layers:
            attn = layer.self_attn
            _ = (attn.q_proj, attn.k_proj, attn.v_proj, attn.o_proj, attn.q_norm, attn.k_norm)
            _ = (layer.mlp, layer.input_layernorm, layer.post_attention_layernorm)
        _ = (base.norm, base.rotary_emb, module.lm_head)
    except Exception:
        return {}

    capacity = prompt_tokens + new_tokens + CAPACITY_SLACK

    st = {
        "capacity": capacity,
        "steps": steps,
        "dtype": dtype,
        "n_layers": n_layers,
        "n_kv": n_kv,
        "head_dim": head_dim,
        "device": None,
        "ready": False,
        "init_failed": False,
        "capture_failed": False,
        "eager_failed": False,
        "invocation_broken": False,
        "graphs": None,
        "graph_outs": None,
        "window": None,
        "input": None,
        "pos": 0,
        "caller_cache": None,
        "sync_target": None,
    }

    def _cuda_ready():
        return st["device"] is not None and st["device"].type == "cuda" and torch.cuda.is_available()

    def _set_lengths(n):
        for window_layer in st["window"].layers:
            window_layer.set_seq_length(n)

    def _init(input_ids):
        dev = input_ids.device
        cache = DynamicCache(config=cfg)
        layers = []
        for _ in range(n_layers):
            keys = torch.zeros((1, n_kv, capacity, head_dim), dtype=dtype, device=dev)
            values = torch.zeros((1, n_kv, capacity, head_dim), dtype=dtype, device=dev)
            layers.append(_WindowLayer(keys, values))
        cache.layers = layers
        st["window"] = cache
        st["input"] = torch.zeros((1, 1), dtype=torch.int64, device=dev)
        st["device"] = dev
        st["ready"] = True

    def _import(past):
        layers = getattr(past, "layers", None)
        if layers is None or len(layers) != n_layers:
            return False
        length = int(past.get_seq_length())
        if length <= 0 or length + steps > capacity:
            return False
        for caller_layer, window_layer in zip(layers, st["window"].layers):
            keys = getattr(caller_layer, "keys", None)
            values = getattr(caller_layer, "values", None)
            if not isinstance(keys, torch.Tensor) or not isinstance(values, torch.Tensor):
                return False
            if not getattr(caller_layer, "is_initialized", False):
                return False
            if tuple(keys.shape) != (1, n_kv, length, head_dim) or keys.dtype != dtype:
                return False
            if tuple(values.shape) != (1, n_kv, length, head_dim) or values.dtype != dtype:
                return False
            if keys.device != st["device"] or values.device != st["device"]:
                return False
            window_layer._keys_window[:, :, :length].copy_(keys)
            window_layer._values_window[:, :, :length].copy_(values)
        _set_lengths(length)
        st["pos"] = length
        st["sync_target"] = length + steps
        return True

    def _parent_step():
        return parent(
            input_ids=st["input"],
            past_key_values=st["window"],
            use_cache=True,
            logits_to_keep=1,
        )

    def _sync_caller_cache():
        caller = st["caller_cache"]
        if caller is None or st["window"] is None:
            return
        layers = getattr(caller, "layers", None)
        if layers is None or len(layers) != n_layers:
            return
        n = st["pos"]
        for caller_layer, window_layer in zip(layers, st["window"].layers):
            if not getattr(caller_layer, "is_initialized", False):
                caller_layer.lazy_initialization(window_layer.keys, window_layer.values)
            caller_layer.keys = window_layer._keys_window[:, :, :n].clone()
            caller_layer.values = window_layer._values_window[:, :, :n].clone()

    def _capture_all():
        base_length = st["pos"]
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(WARMUP_ITERS):
                _set_lengths(base_length)
                _parent_step()
        torch.cuda.current_stream().wait_stream(side)
        graphs = {}
        outs = {}
        for target in range(base_length + 1, base_length + steps + 1):
            graph = torch.cuda.CUDAGraph()
            _set_lengths(target - 1)
            with torch.cuda.graph(graph, capture_error_mode="thread_local"):
                out = _parent_step()
            _set_lengths(target - 1)
            graph.replay()
            _set_lengths(target - 1)
            eager = _parent_step()
            _set_lengths(target - 1)
            if not torch.equal(out.logits, eager.logits):
                raise RuntimeError("captured step diverged from eager parent")
            graphs[target] = graph
            outs[target] = out
        _set_lengths(base_length)
        st["graphs"] = graphs
        st["graph_outs"] = outs
        return True

    def _disable_graphs():
        st["graphs"] = None
        st["graph_outs"] = None

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
            and isinstance(past, DynamicCache)
        ):
            record_route(DECODE_CALL_ID, "fallback_signature")
            return parent(*args, **kwargs)

        continuing = past is st["caller_cache"]
        if continuing and st["invocation_broken"]:
            record_route(DECODE_CALL_ID, "fallback_runtime")
            return parent(*args, **kwargs)

        if not continuing:
            if st["eager_failed"] or (st["init_failed"] and not st["ready"]):
                record_route(DECODE_CALL_ID, "fallback_setup")
                return parent(*args, **kwargs)
            if not st["ready"]:
                try:
                    _init(input_ids)
                except Exception:
                    st["init_failed"] = True
                    record_route(DECODE_CALL_ID, "fallback_setup")
                    return parent(*args, **kwargs)
            if input_ids.device != st["device"]:
                record_route(DECODE_CALL_ID, "fallback_signature")
                return parent(*args, **kwargs)
            imported = False
            try:
                imported = _import(past)
            except Exception:
                imported = False
            if not imported:
                st["caller_cache"] = None
                record_route(DECODE_CALL_ID, "fallback_signature")
                return parent(*args, **kwargs)
            st["caller_cache"] = past
            st["invocation_broken"] = False

        target = st["pos"] + 1
        if target > st["capacity"]:
            _sync_caller_cache()
            st["invocation_broken"] = True
            record_route(DECODE_CALL_ID, "fallback_capacity")
            return parent(*args, **kwargs)

        st["input"].copy_(input_ids)
        route = None
        logits = None
        try:
            if st["graphs"] is not None and target in st["graphs"]:
                st["graphs"][target].replay()
                logits = st["graph_outs"][target].logits.clone()
                route = "cuda_graph_replay"
            else:
                if st["graphs"] is None and not st["capture_failed"] and _cuda_ready():
                    try:
                        _capture_all()
                    except Exception as diagnostic_exc:
                        import traceback
                        print("DEVELOPER_CAPTURE_EXCEPTION", type(diagnostic_exc).__name__, str(diagnostic_exc), flush=True)
                        traceback.print_exc()
                        _disable_graphs()
                        st["capture_failed"] = True
                        try:
                            _set_lengths(st["pos"])
                        except Exception:
                            pass
                        record_route(DECODE_CALL_ID, "fallback_capture")
                    if st["graphs"] is not None and target in st["graphs"]:
                        st["graphs"][target].replay()
                        logits = st["graph_outs"][target].logits.clone()
                        route = "cuda_graph_setup_replay"
                if route is None:
                    out = _parent_step()
                    logits = out.logits.clone()
                    route = "eager_window_parent"
        except Exception:
            _disable_graphs()
            st["capture_failed"] = True
            st["eager_failed"] = True
            try:
                _set_lengths(st["pos"])
            except Exception:
                pass
            try:
                _sync_caller_cache()
            except Exception:
                pass
            st["invocation_broken"] = True
            record_route(DECODE_CALL_ID, "fallback_runtime")
            return parent(*args, **kwargs)

        record_route(DECODE_CALL_ID, route)
        if route == "cuda_graph_replay" or route == "cuda_graph_setup_replay":
            st["pos"] = target
            _set_lengths(target)
        else:
            st["pos"] = int(st["window"].layers[0].get_seq_length())
        if st["sync_target"] is not None and st["pos"] >= st["sync_target"]:
            try:
                _sync_caller_cache()
            except Exception:
                pass
        return CausalLMOutputWithPast(
            loss=None,
            logits=logits,
            past_key_values=st["caller_cache"],
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
