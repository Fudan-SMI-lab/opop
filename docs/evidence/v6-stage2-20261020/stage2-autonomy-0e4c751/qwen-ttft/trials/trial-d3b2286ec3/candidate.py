"""Lean eager prefill forward for the frozen Qwen3-4B 'interactive' workload.

Selected site: the prefill-phase call of the root ``Qwen3ForCausalLM`` module
(call ``900c23e99c808d2c9bda215a``, target ``model``). The replacement
re-executes the parent forward's exact operator sequence -- embed_tokens
lookup, per-layer RMSNorm math (fp32 upcast identical to
``Qwen3RMSNorm.forward``), ``F.linear`` q/k/v/o and gate/up/down projections
on the live module parameters, the imported transformers
``apply_rotary_pos_emb`` / ``sdpa_attention_forward`` callables,
``DynamicCache.update`` per layer, final norm and the ``logits_to_keep=1``
lm_head slice -- from one flat Python loop with weights pre-bound at build
time. Kernel selection and accumulation order are therefore unchanged (same
ops, dtypes, shapes, strides, order, same weight tensors), unlike the
rejected CUDA-graph trial whose captured/re-selected kernels flipped a
marginal argmax. Host-side, the region's nn.Module ``__call__`` machinery,
hook traversal, per-layer config lookups and mask bookkeeping are removed.

The decode phase is untouched: the framework's phase guard routes
decode-phase calls of the same physical module to the parent forward, which
consumes the ``DynamicCache`` produced here exactly as it consumes its own.
Any signature outside the observed prefill signature, or any runtime error,
falls back to the parent callable.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

PREFILL_MODEL_CALL = "900c23e99c808d2c9bda215a"

_EXPECTED_SHAPE = (1, 256)
_EXPECTED_STRIDE = (256, 1)

_ALLOWED_KWARGS = frozenset((
    "input_ids",
    "attention_mask",
    "position_ids",
    "past_key_values",
    "inputs_embeds",
    "labels",
    "use_cache",
    "logits_to_keep",
))

_RMS_ORIGIN = "transformers.models.qwen3.modeling_qwen3"


def _rms_norm(hidden_states, weight, eps):
    input_dtype = hidden_states.dtype
    hidden_states = hidden_states.to(torch.float32)
    variance = hidden_states.pow(2).mean(-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance + eps)
    return weight * hidden_states.to(input_dtype)


def _build_parts(context):
    from transformers.cache_utils import DynamicCache
    from transformers.integrations.sdpa_attention import sdpa_attention_forward
    from transformers.modeling_outputs import CausalLMOutputWithPast
    from transformers.models.qwen3 import modeling_qwen3

    model = context["modules"][PREFILL_MODEL_CALL]
    parent = context["references"][PREFILL_MODEL_CALL]
    inner = model.model
    config = model.config

    if config._attn_implementation != "sdpa":
        raise TypeError("attention implementation is not sdpa")
    if getattr(inner, "has_sliding_layers", False):
        raise TypeError("model has sliding-attention layers")
    for module in (inner.norm, inner.layers[0].input_layernorm):
        if type(module).forward.__module__ != _RMS_ORIGIN:
            raise TypeError("Qwen3RMSNorm.forward is not the plain modeling implementation")

    head_dim = inner.layers[0].self_attn.head_dim
    device = inner.embed_tokens.weight.device
    plan = []
    for layer in inner.layers:
        attn = layer.self_attn
        mlp = layer.mlp
        plan.append((
            layer.input_layernorm.weight,
            layer.input_layernorm.variance_epsilon,
            attn,
            attn.q_proj.weight, attn.q_proj.bias,
            attn.k_proj.weight, attn.k_proj.bias,
            attn.v_proj.weight, attn.v_proj.bias,
            attn.q_norm.weight, attn.q_norm.variance_epsilon,
            attn.k_norm.weight, attn.k_norm.variance_epsilon,
            attn.o_proj.weight, attn.o_proj.bias,
            layer.post_attention_layernorm.weight,
            layer.post_attention_layernorm.variance_epsilon,
            mlp.gate_proj.weight,
            mlp.up_proj.weight,
            mlp.down_proj.weight,
            attn.layer_idx,
            attn.scaling,
        ))
    return {
        "model": model,
        "parent": parent,
        "embed": inner.embed_tokens,
        "rotary": inner.rotary_emb,
        "final_weight": inner.norm.weight,
        "final_eps": inner.norm.variance_epsilon,
        "lm_head_weight": model.lm_head.weight,
        "config": config,
        "head_dim": head_dim,
        "device": device,
        "plan": tuple(plan),
        "DynamicCache": DynamicCache,
        "sdpa_attention_forward": sdpa_attention_forward,
        "apply_rotary_pos_emb": modeling_qwen3.apply_rotary_pos_emb,
        "CausalLMOutputWithPast": CausalLMOutputWithPast,
        "disabled": False,
    }


def _compatible(args, kwargs, device):
    if args or set(kwargs) - _ALLOWED_KWARGS:
        return False
    if (
        kwargs.get("attention_mask") is not None
        or kwargs.get("position_ids") is not None
        or kwargs.get("past_key_values") is not None
        or kwargs.get("inputs_embeds") is not None
        or kwargs.get("labels") is not None
    ):
        return False
    if kwargs.get("use_cache") is not True:
        return False
    logits_to_keep = kwargs.get("logits_to_keep")
    if isinstance(logits_to_keep, bool) or logits_to_keep != 1:
        return False
    input_ids = kwargs.get("input_ids")
    if not isinstance(input_ids, torch.Tensor):
        return False
    if input_ids.device != device or input_ids.dtype != torch.int64:
        return False
    if input_ids.dim() != 2 or tuple(input_ids.shape) != _EXPECTED_SHAPE:
        return False
    if tuple(input_ids.stride()) != _EXPECTED_STRIDE:
        return False
    if input_ids.requires_grad:
        return False
    return True


def build(context):
    record = context.get("record_route", lambda *_: None)
    try:
        parts = _build_parts(context)
    except Exception:
        return {}

    parent = parts["parent"]
    model = parts["model"]

    def prefill_forward(*args, **kwargs):
        if parts["disabled"]:
            record(PREFILL_MODEL_CALL, "fallback_parent_disabled")
            return parent(*args, **kwargs)
        if not _compatible(args, kwargs, parts["device"]):
            record(PREFILL_MODEL_CALL, "fallback_signature")
            return parent(*args, **kwargs)
        if model.training:
            record(PREFILL_MODEL_CALL, "fallback_parent_training")
            return parent(*args, **kwargs)
        input_ids = kwargs["input_ids"]
        try:
            batch, seq = input_ids.shape
            head_dim = parts["head_dim"]
            linear = F.linear
            silu = F.silu
            sdpa_forward = parts["sdpa_attention_forward"]
            apply_rotary = parts["apply_rotary_pos_emb"]
            hidden = parts["embed"].forward(input_ids)
            past = parts["DynamicCache"](config=parts["config"])
            update = past.update
            position_ids = torch.arange(seq, device=input_ids.device).unsqueeze(0)
            cos, sin = parts["rotary"].forward(hidden, position_ids)
            for plan in parts["plan"]:
                (
                    w_ln, e_ln, attn,
                    w_q, b_q, w_k, b_k, w_v, b_v,
                    w_qn, e_qn, w_kn, e_kn,
                    w_o, b_o, w_pln, e_pln,
                    w_gate, w_up, w_down,
                    layer_idx, scaling,
                ) = plan
                residual = hidden
                normed = _rms_norm(residual, w_ln, e_ln)
                query = _rms_norm(
                    linear(normed, w_q, b_q).view(batch, seq, -1, head_dim), w_qn, e_qn
                ).transpose(1, 2)
                key = _rms_norm(
                    linear(normed, w_k, b_k).view(batch, seq, -1, head_dim), w_kn, e_kn
                ).transpose(1, 2)
                value = linear(normed, w_v, b_v).view(batch, seq, -1, head_dim).transpose(1, 2)
                query, key = apply_rotary(query, key, cos, sin)
                key, value = update(key, value, layer_idx)
                attn_out, _ = sdpa_forward(
                    attn, query, key, value, None,
                    dropout=0.0, scaling=scaling, sliding_window=None,
                )
                attn_out = linear(attn_out.reshape(batch, seq, -1).contiguous(), w_o, b_o)
                hidden = residual + attn_out
                residual = hidden
                normed = _rms_norm(residual, w_pln, e_pln)
                hidden = residual + linear(
                    silu(linear(normed, w_gate)) * linear(normed, w_up), w_down
                )
            hidden = _rms_norm(hidden, parts["final_weight"], parts["final_eps"])
            logits = linear(hidden[:, -1:, :], parts["lm_head_weight"])
            record(PREFILL_MODEL_CALL, "lean_prefill_eager")
            return parts["CausalLMOutputWithPast"](logits=logits, past_key_values=past)
        except Exception:
            parts["disabled"] = True
            record(PREFILL_MODEL_CALL, "runtime_failure_fallback_parent")
            return parent(*args, **kwargs)

    return {PREFILL_MODEL_CALL: prefill_forward}
