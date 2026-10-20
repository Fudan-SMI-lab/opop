"""Candidate: bit-exact Qwen3 decoder-layer forward with lean Python dispatch and
per-layer prefill CUDA Graph replay.

Why bit-exact
-------------
The workload's quality gate compares every element of the captured logits
tensor against a reference produced by the unmodified eager path
(atol=rtol=0.02, exact greedy tokens). Replicating the full 36-layer stack on
CPU with the real weights showed that even a handful of 1-ulp changes inside
the layers' RMSNorm outputs amplify to logits differences that breach that gate
on several percent of vocabulary elements, while an op-for-op identical kernel
sequence stays bit-exact. This candidate therefore keeps the exact eager kernel
sequence of transformers 5.16.1 Qwen3DecoderLayer / Qwen3Attention /
Qwen3RMSNorm / Qwen3MLP (same aten ops, shapes and values) and speeds the
selected region up by removing overhead around those kernels:

1. One pre-bound function per physical layer replaces the HF Python path
   (sub-module __call__ dispatch, hook checks, kwargs plumbing, attention
   interface resolution). The aten kernels issued are unchanged, so outputs
   are bit-identical; only host-side work between kernels shrinks.
2. Each physical layer's prefill signature (batch 1, 256 tokens, contiguous
   BF16, no attention mask, empty DynamicCache) is captured into a CUDA graph
   at its second compatible invocation, which falls inside the framework's
   untimed warmup/reference runs. Replay executes the identical kernel stream,
   so it removes per-kernel launch overhead from the timed region while
   remaining bit-exact. Graph inputs are refilled on every invocation by
   copying the live invocation's hidden states and rotary cos/sin into static
   slots; every kernel re-executes on every invocation and no results or
   fixture data are cached. The first compatible invocation always runs eager,
   decode-length inputs (sequence length 1) never use graphs, and any capture
   or replay problem permanently falls back to the eager path for that layer.

Decode-phase calls (single token, growing cache) stay on the eager path: their
cache shapes change every step and their kernels are issued exactly as the
baseline issues them.

No model parameters are modified, repacked or re-pointed; no aliasing or state
semantics change. The DynamicCache bookkeeping performed once during capture is
restored on replay invocations by rebinding the current cache object's layer to
the graph-owned key/value tensors (Python attribute assignment only, no extra
kernels), so decode steps and the adapter-visible cache are unchanged. All 36
graphs share one capture pool and are captured in live-run order (layer 0..35),
with each layer's graph reading the previous layer's output slot, so pool reuse
between sequential replays is safe and steady-state prefill performs one input
copy, one cos/sin copy and 36 graph replays.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class _LayerReplacement:
    """Pre-bound, kernel-identical Qwen3DecoderLayer forward for one physical layer."""

    def __init__(self, layer, shared):
        attn = layer.self_attn
        mlp = layer.mlp
        self.q_proj = attn.q_proj
        self.k_proj = attn.k_proj
        self.v_proj = attn.v_proj
        self.o_proj = attn.o_proj
        self.q_norm = attn.q_norm
        self.k_norm = attn.k_norm
        self.input_layernorm = layer.input_layernorm
        self.post_attention_layernorm = layer.post_attention_layernorm
        self.gate_proj = mlp.gate_proj
        self.up_proj = mlp.up_proj
        self.down_proj = mlp.down_proj
        self.act_fn = mlp.act_fn
        self.layer_idx = attn.layer_idx
        self.num_kv_groups = attn.num_key_value_groups
        self.head_dim = attn.head_dim
        self.is_causal = bool(getattr(attn, "is_causal", True))
        self.scaling = attn.scaling
        self.in_features = attn.q_proj.in_features

        self._shared = shared          # one dict per root model
        self._graphs_ok = torch.cuda.is_available()
        self._graph = None
        self._sig = None               # captured prefill signature
        self._prefill_calls = 0
        self._slot_in = None           # graph input slot (chained to previous graph output)
        self._slot_out = None          # graph output tensor (capture-pool)
        self._graph_keys = None        # graph-owned cache tensors (capture-pool)
        self._graph_values = None

    # ------------------------------------------------------------------ #
    # bit-exact eager forward: identical aten op sequence to HF 5.16.1   #
    # ------------------------------------------------------------------ #
    def _eager_forward(self, hidden_states, attention_mask, position_embeddings, past_key_values):
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)
        half = self.head_dim // 2
        q_rot = torch.cat((-query_states[..., half:], query_states[..., :half]), dim=-1)
        k_rot = torch.cat((-key_states[..., half:], key_states[..., :half]), dim=-1)
        query_states = (query_states * cos) + (q_rot * sin)
        key_states = (key_states * cos) + (k_rot * sin)

        if past_key_values is not None:
            key_states, value_states = past_key_values.update(
                key_states, value_states, self.layer_idx
            )

        sdpa_kwargs = {}
        if self.num_kv_groups > 1:
            key_head_dim = key_states.shape[-1]
            value_head_dim = value_states.shape[-1]
            if attention_mask is None and key_head_dim == value_head_dim <= 256:
                sdpa_kwargs = {"enable_gqa": True}
            else:
                key_states = _repeat_kv(key_states, self.num_kv_groups)
                value_states = _repeat_kv(value_states, self.num_kv_groups)

        q_length = query_states.shape[2]
        kv_length = key_states.shape[2]
        is_causal = self.is_causal
        is_causal = q_length > 1 and attention_mask is None and is_causal
        if is_causal and attention_mask is None and q_length > 1 and kv_length > q_length:
            key_states = key_states[:, :, :q_length, :]
            value_states = value_states[:, :, :q_length, :]

        attn_output = F.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=attention_mask,
            dropout_p=0.0,
            scale=self.scaling,
            is_causal=is_causal,
            **sdpa_kwargs,
        )
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = self.o_proj(attn_output.reshape(*input_shape, -1).contiguous())

        hidden_states = residual + attn_output

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.down_proj(
            self.act_fn(self.gate_proj(hidden_states)) * self.up_proj(hidden_states)
        )
        hidden_states = residual + hidden_states
        return hidden_states

    # ------------------------------------------------------------------ #
    # guards                                                             #
    # ------------------------------------------------------------------ #
    def _graph_signature(self, hidden_states, attention_mask, position_embeddings, past_key_values):
        if not self._graphs_ok or attention_mask is not None:
            return None
        if past_key_values is None or position_embeddings is None:
            return None
        if type(past_key_values).__name__ != "DynamicCache":
            return None
        layers = getattr(past_key_values, "layers", None)
        if layers is None:
            return None
        if hidden_states.dtype is not torch.bfloat16 or not hidden_states.is_cuda:
            return None
        if hidden_states.dim() != 3 or hidden_states.shape[-1] != self.in_features:
            return None
        if hidden_states.shape[1] <= 1:
            return None  # decode-like single-token step: eager only
        if not hidden_states.is_contiguous():
            return None
        cos, sin = position_embeddings
        if cos.dtype is not torch.bfloat16 or sin.dtype is not torch.bfloat16:
            return None
        if not cos.is_contiguous() or not sin.is_contiguous():
            return None
        # prefill graph assumes an empty cache for this layer (kv_len == q_len)
        if self.layer_idx < len(layers):
            dl = layers[self.layer_idx]
            if getattr(dl, "is_initialized", False):
                keys = getattr(dl, "keys", None)
                if keys is not None and keys.dim() == 4 and keys.shape[-2] != 0:
                    return None
        return (tuple(hidden_states.shape), tuple(cos.shape))

    # ------------------------------------------------------------------ #
    # cache helpers (DynamicCache semantics without extra kernels)       #
    # ------------------------------------------------------------------ #
    def _cache_layer(self, cache):
        layers = getattr(cache, "layers", None)
        if layers is None:
            return None
        replicate = getattr(cache, "layer_class_to_replicate", None)
        if replicate is not None:
            while len(layers) <= self.layer_idx:
                layers.append(replicate())
        if self.layer_idx >= len(layers):
            return None
        return layers[self.layer_idx]

    def _preinit_cache_layer(self, cache, device, dtype):
        dl = self._cache_layer(cache)
        if dl is None:
            return False
        if not getattr(dl, "is_initialized", False):
            dl.dtype = dtype
            dl.device = device
            dl.keys = torch.empty((0,), dtype=dtype, device=device)
            dl.values = torch.empty((0,), dtype=dtype, device=device)
            dl.is_initialized = True
        return True

    def _rebind_cache_layer(self, cache):
        dl = self._cache_layer(cache)
        if dl is None:
            return False
        dl.keys = self._graph_keys
        dl.values = self._graph_values
        dl.is_initialized = True
        return True

    # ------------------------------------------------------------------ #
    # static slot management                                             #
    # ------------------------------------------------------------------ #
    def _ensure_statics(self, hidden_states, position_embeddings):
        shared = self._shared
        cos, sin = position_embeddings
        if shared["cos"] is None or shared["cos"].shape != cos.shape:
            shared["cos"] = torch.empty_like(cos)
            shared["sin"] = torch.empty_like(sin)
            shared["cos_src"] = None
            shared["sin_src"] = None
        if shared["cos_src"] is not cos:
            shared["cos"].copy_(cos)
            shared["sin"].copy_(sin)
            shared["cos_src"] = cos
            shared["sin_src"] = sin
        if self._slot_in is None:
            prev = shared["prev_out"]
            if prev is not None and prev.shape == hidden_states.shape and prev.is_cuda:
                self._slot_in = prev  # chain: read previous layer's output slot directly
            else:
                self._slot_in = torch.empty_like(hidden_states)
        if self._slot_in is not hidden_states:
            self._slot_in.copy_(hidden_states)

    # ------------------------------------------------------------------ #
    # capture                                                            #
    # ------------------------------------------------------------------ #
    def _try_capture(self, hidden_states, position_embeddings, past_key_values):
        try:
            device = hidden_states.device
            self._ensure_statics(hidden_states, position_embeddings)
            if self._shared["pool"] is None:
                self._shared["pool"] = torch.cuda.graph_pool_handle()

            side = self._shared["side"]
            if side is None or side.device != device:
                side = torch.cuda.Stream(device=device)
                self._shared["side"] = side
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                for _ in range(2):
                    warm_cache = type(past_key_values)()
                    if not self._preinit_cache_layer(warm_cache, device, hidden_states.dtype):
                        raise RuntimeError("cache layer preinit failed")
                    self._eager_forward(
                        self._slot_in, None,
                        (self._shared["cos"], self._shared["sin"]), warm_cache,
                    )
            torch.cuda.current_stream().wait_stream(side)
            torch.cuda.synchronize(device)

            graph = torch.cuda.CUDAGraph()
            cap_cache = type(past_key_values)()
            if not self._preinit_cache_layer(cap_cache, device, hidden_states.dtype):
                raise RuntimeError("cache layer preinit failed")
            with torch.cuda.graph(graph, pool=self._shared["pool"]):
                out = self._eager_forward(
                    self._slot_in, None,
                    (self._shared["cos"], self._shared["sin"]), cap_cache,
                )
            dl = cap_cache.layers[self.layer_idx]
            self._graph = graph
            self._slot_out = out
            self._graph_keys = dl.keys
            self._graph_values = dl.values
            self._shared["prev_out"] = out
            graph.replay()
            torch.cuda.synchronize(device)
            return True
        except Exception:
            self._graph = None
            self._sig = None
            self._slot_out = None
            self._graph_keys = None
            self._graph_values = None
            self._graphs_ok = False
            return False

    # ------------------------------------------------------------------ #
    # entry point installed by the framework                             #
    # ------------------------------------------------------------------ #
    def __call__(self, hidden_states, attention_mask=None, position_ids=None,
                 past_key_values=None, use_cache=None, position_embeddings=None, **kwargs):
        if (
            position_embeddings is not None
            and past_key_values is not None
            and self._graphs_ok
        ):
            sig = self._graph_signature(
                hidden_states, attention_mask, position_embeddings, past_key_values
            )
            if sig is not None:
                if self._graph is None:
                    if self._sig is None or self._sig == sig:
                        self._prefill_calls += 1
                    if (self._sig is None or self._sig == sig) and self._prefill_calls >= 2:
                        if self._try_capture(hidden_states, position_embeddings, past_key_values):
                            if not self._rebind_cache_layer(past_key_values):
                                self._graph = None
                                self._graphs_ok = False
                                return self._eager_forward(
                                    hidden_states, attention_mask, position_embeddings, past_key_values
                                )
                            self._sig = sig
                            return self._slot_out
                        return self._eager_forward(
                            hidden_states, attention_mask, position_embeddings, past_key_values
                        )
                    if self._sig is None:
                        self._sig = sig
                    return self._eager_forward(
                        hidden_states, attention_mask, position_embeddings, past_key_values
                    )
                if sig == self._sig:
                    try:
                        self._ensure_statics(hidden_states, position_embeddings)
                        self._graph.replay()
                        if not self._rebind_cache_layer(past_key_values):
                            return self._eager_forward(
                                hidden_states, attention_mask, position_embeddings, past_key_values
                            )
                        return self._slot_out
                    except Exception:
                        self._graph = None
                        self._graphs_ok = False
                        return self._eager_forward(
                            hidden_states, attention_mask, position_embeddings, past_key_values
                        )
        return self._eager_forward(
            hidden_states, attention_mask, position_embeddings, past_key_values
        )


def build(context):
    targets = context.get("targets") or {}
    modules = context.get("modules") or {}
    models = context.get("models") or {}
    root = models.get("model")
    replacements = {}
    for call_id, spec in targets.items():
        if not isinstance(spec, dict) or spec.get("kind") != "module":
            continue
        semantics = spec.get("semantics") or ""
        if not semantics.endswith("Qwen3DecoderLayer"):
            continue
        layer = modules.get(call_id)
        if layer is None:
            continue
        holder = root if root is not None else layer
        state = holder.__dict__
        shared = state.get("_opop_layer_shared")
        if shared is None:
            shared = {
                "pool": None,       # shared graph capture pool
                "prev_out": None,   # previous layer's graph output slot (chaining)
                "cos": None,        # shared static cos/sin slots
                "sin": None,
                "cos_src": None,    # source tensors already copied this invocation
                "sin_src": None,
                "side": None,       # shared warmup side stream
            }
            state["_opop_layer_shared"] = shared
        state = layer.__dict__
        replacement = state.get("_opop_layer_replacement")
        if replacement is None or replacement._shared is not shared:
            replacement = _LayerReplacement(layer, shared)
            state["_opop_layer_replacement"] = replacement
        replacements[call_id] = replacement
    return replacements
