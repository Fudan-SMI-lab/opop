"""Candidate: bit-exact eager composite of the Qwen3-4B prefill region.

Repairs trial-af78dd3283, whose CUDA-graph capture/replay of the prefill call
failed quality (logits max abs error 5.0625, 41.5% mismatched elements, greedy
tokens diverging late in the sequence).  A captured graph replays kernels whose
pointer/workspace context differs from the eager parent, so prefill logits and
KV states were not bit-identical and a near-tie argmax flipped during decode,
breaking the exact greedy-token requirement.  The same fate hit the earlier
Triton attempt (trial-9a65eeeca9), showing this workload punishes any
numerical perturbation of prefill.

This attempt keeps every aten operator, operand order, shape, stride and dtype
of the parent's eager prefill path and removes only host-side Python
dispatch: nn.Module ``_call_impl`` machinery, GradientCheckpointingLayer /
capture_outputs / merge_with_config_defaults decorators, per-call attention
interface lookups, config ``__getattribute__`` layers and per-layer module
traversal.  Measured evidence: the warm prefill spans ~47.5 ms wall with only
~12.1 ms of GPU-active time (inputs/observations/current timeline), so the
region is host-bound and dispatch removal attacks the dominant cost.

Bit-exactness rules used (each is value-identity, not approximation):
- identical ``F.linear``/embedding/RMSNorm/rotary/SDPA op chains with the same
  parameter tensors, operand order and derived strides;
- skipped ops are exact identities: RoPE ``* attention_scaling`` with scaling
  1.0, ``arange + 0`` for the empty-cache position ids, the ``cat`` with an
  empty KV tensor, and RMSNorm/``lm_head`` on positions other than the last
  (row-wise reduction);
- the value projection is materialized contiguous exactly like the parent's
  cache ``cat`` output, and SDPA receives tensors with the parent's strides;
- the returned ``past_key_values`` is a real ``DynamicCache`` (fresh container
  objects per invocation, per-layer ``DynamicLayer`` state identical to what
  ``DynamicCache(config=...)`` plus ``update`` would produce), so the untouched
  parent decode loop extends it bit-identically.

Decode-phase calls are not replaced.  Unsupported signatures and any runtime
failure fall back to the parent callable with distinct route labels.
"""

from __future__ import annotations

import torch

try:
    from transformers.cache_utils import DynamicCache, DynamicLayer
    from transformers.modeling_outputs import CausalLMOutputWithPast
except Exception:  # pragma: no cover - defensive import for foreign envs
    DynamicCache = None
    DynamicLayer = None
    CausalLMOutputWithPast = None

PREFILL_CALL = "900c23e99c808d2c9bda215a"

_LINEAR = torch._C._nn.linear
_SILU = torch._C._nn.silu
_SDPA = torch._C._nn.scaled_dot_product_attention
_EMBED = torch.nn.functional.embedding

_KNOWN_KWARGS = frozenset(
    {
        "input_ids",
        "attention_mask",
        "position_ids",
        "past_key_values",
        "inputs_embeds",
        "labels",
        "use_cache",
        "logits_to_keep",
    }
)
_NONE_ONLY_KWARGS = (
    "attention_mask",
    "position_ids",
    "past_key_values",
    "inputs_embeds",
    "labels",
)

_F32 = torch.float32


class _PrefillComposite:
    """Lean, op-exact driver of the parent prefill forward chain."""

    def __init__(self, model, parent, record_route):
        self.call_id = PREFILL_CALL
        self.parent = parent
        self._route = record_route

        inner = model.model
        config = model.config
        if getattr(config, "_attn_implementation", "sdpa") != "sdpa":
            raise RuntimeError("composite prefill: non-sdpa attention implementation")
        if getattr(config, "use_sliding_window", False) or getattr(config, "sliding_window", None):
            raise RuntimeError("composite prefill: sliding window unsupported")
        layer_types = getattr(config, "layer_types", None) or ["full_attention"] * config.num_hidden_layers
        if any(t != "full_attention" for t in layer_types):
            raise RuntimeError("composite prefill: non-full-attention layers unsupported")
        if config.hidden_act != "silu":
            raise RuntimeError("composite prefill: non-silu activation")

        self.device = model.lm_head.weight.device
        self.heads = int(config.num_attention_heads)
        self.kv_heads = int(config.num_key_value_heads)
        self.head_dim = int(getattr(config, "head_dim", config.hidden_size // config.num_attention_heads))
        self.q_dim = self.heads * self.head_dim
        self.half = self.head_dim // 2
        self.scale = float(self.head_dim) ** -0.5
        self.gqa = self.heads // self.kv_heads
        if self.gqa > 1 and self.head_dim > 256:
            raise RuntimeError("composite prefill: head_dim > 256 disables gqa path")
        if len(inner.layers) != int(config.num_hidden_layers):
            raise RuntimeError("composite prefill: layer count mismatch")

        rotary = inner.rotary_emb
        self.inv_freq = rotary.inv_freq
        self.attention_scaling = float(rotary.attention_scaling)

        self.embed_w = inner.embed_tokens.weight
        self.norm_w = inner.norm.weight
        self.norm_eps = float(inner.norm.variance_epsilon)
        self.lm_w = model.lm_head.weight
        if model.lm_head.bias is not None:
            raise RuntimeError("composite prefill: lm_head bias unsupported")

        layers = []
        for layer in inner.layers:
            attn = layer.self_attn
            mlp = layer.mlp
            if attn.head_dim != self.head_dim or attn.scaling != self.scale:
                raise RuntimeError("composite prefill: layer attention geometry mismatch")
            layers.append(
                (
                    layer.input_layernorm.weight,
                    float(layer.input_layernorm.variance_epsilon),
                    attn.q_proj.weight,
                    attn.q_proj.bias,
                    attn.q_norm.weight,
                    float(attn.q_norm.variance_epsilon),
                    attn.k_proj.weight,
                    attn.k_proj.bias,
                    attn.k_norm.weight,
                    float(attn.k_norm.variance_epsilon),
                    attn.v_proj.weight,
                    attn.v_proj.bias,
                    attn.o_proj.weight,
                    attn.o_proj.bias,
                    layer.post_attention_layernorm.weight,
                    float(layer.post_attention_layernorm.variance_epsilon),
                    mlp.gate_proj.weight,
                    mlp.gate_proj.bias,
                    mlp.up_proj.weight,
                    mlp.up_proj.bias,
                    mlp.down_proj.weight,
                    mlp.down_proj.bias,
                )
            )
        self.layer_specs = tuple(layers)

        proto = DynamicCache(config=config)
        self.cache_cls = type(proto)
        self.cache_base = {k: v for k, v in proto.__dict__.items() if k != "layers"}
        self.layer_classes = tuple(type(l) for l in proto.layers)

    def _guard(self, args, kwargs):
        try:
            if args:
                return None
            for key in kwargs:
                if key not in _KNOWN_KWARGS and kwargs.get(key) is not None:
                    return None
            for key in _NONE_ONLY_KWARGS:
                if kwargs.get(key) is not None:
                    return None
            if kwargs.get("use_cache") is not True:
                return None
            l2k = kwargs.get("logits_to_keep", None)
            if not isinstance(l2k, int) or l2k != 1:
                return None
            ids = kwargs.get("input_ids")
            if not torch.is_tensor(ids):
                return None
            if ids.dim() != 2 or ids.shape[0] != 1 or ids.shape[1] < 2:
                return None
            if ids.dtype != torch.long or ids.device != self.device:
                return None
            return ids
        except Exception:
            return None

    def _make_cache(self):
        cache = object.__new__(self.cache_cls)
        cache.__dict__.update(self.cache_base)
        cache.layers = [cls() for cls in self.layer_classes]
        return cache

    def __call__(self, *args, **kwargs):
        ids = self._guard(args, kwargs)
        if ids is None:
            self._route(self.call_id, "prefill_parent_signature_fallback")
            return self.parent(*args, **kwargs)
        try:
            out = self._forward(ids)
        except Exception:
            self._route(self.call_id, "prefill_runtime_failure_parent_fallback")
            return self.parent(*args, **kwargs)
        self._route(self.call_id, "prefill_bitexact_eager_composite")
        return out

    def _forward(self, ids):
        batch, seq = ids.shape
        cache = self._make_cache()
        cache_layers = cache.layers

        hidden = _EMBED(ids, self.embed_w)

        position_ids = torch.arange(seq, device=ids.device).unsqueeze(0)
        pos_exp = position_ids[:, None, :].float()
        inv_exp = self.inv_freq[None, :, None].float().expand(batch, -1, 1)
        freqs = (inv_exp @ pos_exp).transpose(1, 2)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos()
        sin = emb.sin()
        if self.attention_scaling != 1.0:
            cos = cos * self.attention_scaling
            sin = sin * self.attention_scaling
        cos = cos.to(hidden.dtype).unsqueeze(1)
        sin = sin.to(hidden.dtype).unsqueeze(1)

        heads, kv_heads, head_dim = self.heads, self.kv_heads, self.head_dim
        half, q_dim = self.half, self.q_dim
        gqa_kwargs = {"enable_gqa": True} if self.gqa > 1 else {}
        scale = self.scale

        for idx, spec in enumerate(self.layer_specs):
            (
                ln1_w,
                ln1_eps,
                q_w,
                q_b,
                qn_w,
                qn_eps,
                k_w,
                k_b,
                kn_w,
                kn_eps,
                v_w,
                v_b,
                o_w,
                o_b,
                ln2_w,
                ln2_eps,
                g_w,
                g_b,
                u_w,
                u_b,
                d_w,
                d_b,
            ) = spec

            x32 = hidden.to(_F32)
            var = x32.pow(2).mean(-1, keepdim=True)
            normed = ln1_w * (x32 * torch.rsqrt(var + ln1_eps)).to(hidden.dtype)

            query = _LINEAR(normed, q_w, q_b).view(batch, seq, heads, head_dim)
            q32 = query.to(_F32)
            qvar = q32.pow(2).mean(-1, keepdim=True)
            query = qn_w * (q32 * torch.rsqrt(qvar + qn_eps)).to(query.dtype)
            query = query.transpose(1, 2)

            key = _LINEAR(normed, k_w, k_b).view(batch, seq, kv_heads, head_dim)
            k32 = key.to(_F32)
            kvar = k32.pow(2).mean(-1, keepdim=True)
            key = kn_w * (k32 * torch.rsqrt(kvar + kn_eps)).to(key.dtype)
            key = key.transpose(1, 2)

            value = _LINEAR(normed, v_w, v_b).view(batch, seq, kv_heads, head_dim).transpose(1, 2)

            query = (query * cos) + (
                torch.cat((-query[..., half:], query[..., :half]), dim=-1) * sin
            )
            key = (key * cos) + (torch.cat((-key[..., half:], key[..., :half]), dim=-1) * sin)
            value = value.contiguous()

            cache_layer = cache_layers[idx]
            cache_layer.is_initialized = True
            cache_layer.dtype = key.dtype
            cache_layer.device = key.device
            cache_layer.keys = key
            cache_layer.values = value

            attn = _SDPA(
                query,
                key,
                value,
                attn_mask=None,
                dropout_p=0.0,
                scale=scale,
                is_causal=True,
                **gqa_kwargs,
            )
            attn = _LINEAR(
                attn.transpose(1, 2).contiguous().view(batch, seq, q_dim), o_w, o_b
            )
            hidden = hidden + attn

            x32 = hidden.to(_F32)
            var = x32.pow(2).mean(-1, keepdim=True)
            normed = ln2_w * (x32 * torch.rsqrt(var + ln2_eps)).to(hidden.dtype)
            hidden = hidden + _LINEAR(
                _SILU(_LINEAR(normed, g_w, g_b)) * _LINEAR(normed, u_w, u_b), d_w, d_b
            )

        last = hidden[:, -1:, :].to(_F32)
        lvar = last.pow(2).mean(-1, keepdim=True)
        last = self.norm_w * (last * torch.rsqrt(lvar + self.norm_eps)).to(hidden.dtype)
        logits = _LINEAR(last, self.lm_w, None)

        return CausalLMOutputWithPast(
            loss=None,
            logits=logits,
            past_key_values=cache,
            hidden_states=None,
            attentions=None,
        )


def build(context):
    """Install the bit-exact prefill composite for the region root call."""
    record_route = context.get("record_route") or (lambda *_: None)
    modules = context.get("modules") or {}
    models = context.get("models") or {}
    references = context.get("references") or {}

    model = modules.get(PREFILL_CALL) or models.get("model")
    parent = references.get(PREFILL_CALL)
    if (
        model is None
        or parent is None
        or DynamicCache is None
        or DynamicLayer is None
        or CausalLMOutputWithPast is None
    ):
        return {}

    try:
        engine = _PrefillComposite(model, parent, record_route)
    except Exception:
        def _setup_failed(*args, **kwargs):
            record_route(PREFILL_CALL, "prefill_setup_failure_parent_fallback")
            return parent(*args, **kwargs)

        return {PREFILL_CALL: _setup_failed}

    return {PREFILL_CALL: engine}
