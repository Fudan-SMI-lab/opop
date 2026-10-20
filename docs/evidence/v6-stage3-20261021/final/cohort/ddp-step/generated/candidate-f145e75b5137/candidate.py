"""Dispatch-minimized flattened forward for the frozen GPT-2 DDP training step.

Replaces the root ``gpt2`` module boundary (GPT2LMHeadModel.forward) with a
flattened same-order eager reimplementation that emits the identical compute
aten op sequence (embedding, addmm, bmm, layer_norm, mul/add, softmax, clone,
mm, log_softmax/nll) with the identical operand shapes/strides/offsets the
parent emits, while removing per-call Python overhead:

* no per-boundary nn.Module.__call__ machinery for the 151 observed modules,
* no HF wrapper glue (mask re-derivation, padding warnings, output recorders),
* the constant causal mask and position ids are hoisted (built once per
  (device, seq_len) with the same values the parent computes every call),
* attention operand folding is emitted directly via as_strided with exactly
  the strides/offsets the parent's matmul folding produces,
* no-op dropout (p=0.0) and no-op .float()/views are skipped.

Every numeric decision keeps values bitwise identical to the parent, so the
loss, every gradient and every updated parameter are bitwise unchanged.
Unsupported signatures/configs fall back to the parent reference.
"""

import math

import torch
import torch.nn.functional as F
from transformers.models.gpt2.modeling_gpt2 import CausalLMOutputWithCrossAttentions

_SQRT_2_OVER_PI = math.sqrt(2.0 / math.pi)
_GELU_COEFF = 0.044715
_IGNORE_INDEX = -100


def _build_plan(module):
    """Extract static per-layer references. Returns None when unsupported."""
    try:
        config = module.config
        if config._attn_implementation != "eager":
            return None
        if config.activation_function != "gelu_new":
            return None
        if getattr(config, "reorder_and_upcast_attn", False):
            return None
        if getattr(config, "scale_attn_by_inverse_layer_idx", False):
            return None
        if not getattr(config, "scale_attn_weights", True):
            return None
        if getattr(config, "add_cross_attention", False):
            return None
        transformer = module.transformer
        blocks = list(transformer.h)
        if len(blocks) != config.n_layer:
            return None
        embed_dim = config.n_embd
        num_heads = config.n_head
        head_dim = embed_dim // num_heads
        if head_dim * num_heads != embed_dim:
            return None
        inner_dim = config.n_inner if config.n_inner is not None else 4 * embed_dim
        eps = float(config.layer_norm_epsilon)
        scaling = head_dim ** -0.5

        wte = transformer.wte
        wpe = transformer.wpe
        lm_head = module.lm_head
        for dropout in (transformer.drop,):
            if float(dropout.p) != 0.0:
                return None
        if lm_head.bias is not None:
            return None
        wte_weight = wte.weight
        wpe_weight = wpe.weight
        head_weight = lm_head.weight
        ln_f = transformer.ln_f
        vocab_size = int(config.vocab_size)

        layers = []
        for block in blocks:
            attn = block.attn
            mlp = block.mlp
            ln_1 = block.ln_1
            ln_2 = block.ln_2
            c_attn = attn.c_attn
            c_proj = attn.c_proj
            c_fc = mlp.c_fc
            m_proj = mlp.c_proj
            for dropout in (attn.attn_dropout, attn.resid_dropout, mlp.dropout):
                if float(dropout.p) != 0.0:
                    return None
            if c_attn.weight.shape != (embed_dim, 3 * embed_dim):
                return None
            if c_proj.weight.shape != (embed_dim, embed_dim):
                return None
            if c_fc.weight.shape != (embed_dim, inner_dim):
                return None
            if m_proj.weight.shape != (inner_dim, embed_dim):
                return None
            if getattr(attn, "is_cross_attention", False):
                return None
            if float(attn.scaling) != float(scaling):
                return None
            layers.append((
                ln_1.weight, ln_1.bias,
                c_attn.weight, c_attn.bias,
                c_proj.weight, c_proj.bias,
                ln_2.weight, ln_2.bias,
                c_fc.weight, c_fc.bias,
                m_proj.weight, m_proj.bias,
            ))
        for tensor in (wte_weight, wpe_weight, head_weight, ln_f.weight, ln_f.bias):
            if tensor.dtype is not torch.float32:
                return None
        for layer in layers:
            for tensor in layer:
                if tensor.dtype is not torch.float32:
                    return None
        device = wte_weight.device
        return {
            "layers": layers,
            "wte_weight": wte_weight,
            "wpe_weight": wpe_weight,
            "head_weight": head_weight,
            "ln_f_weight": ln_f.weight,
            "ln_f_bias": ln_f.bias,
            "eps": eps,
            "scaling": scaling,
            "embed_dim": embed_dim,
            "num_heads": num_heads,
            "head_dim": head_dim,
            "qkv_dim": 3 * embed_dim,
            "vocab_size": vocab_size,
            "device": device,
        }
    except Exception:
        return None


def _position_and_mask(device, seq_len, cache):
    key = (str(device), seq_len)
    cached = cache.get(key)
    if cached is None:
        position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
        min_value = torch.finfo(torch.float32).min
        bool_mask = torch.arange(seq_len, device=device).view(seq_len, 1) >= torch.arange(seq_len, device=device).view(1, seq_len)
        bool_mask = bool_mask.unsqueeze(0)
        mask = torch.where(bool_mask, torch.zeros((), device=device, dtype=torch.float32),
                           torch.tensor(min_value, device=device, dtype=torch.float32))
        cached = (position_ids, mask)
        cache[key] = cached
    return cached


def _flat_forward(plan, cache, input_ids, labels):
    """Flattened GPT2LMHeadModel.forward: identical compute ops, minimal dispatch."""
    scaling = plan["scaling"]
    num_heads = plan["num_heads"]
    head_dim = plan["head_dim"]
    embed_dim = plan["embed_dim"]
    qkv_dim = plan["qkv_dim"]
    eps = plan["eps"]
    as_strided = torch.as_strided
    addmm = torch.addmm
    bmm = torch.bmm
    mul = torch.mul
    add = torch.add
    layer_norm = torch.native_layer_norm
    softmax = F.softmax
    embedding = F.embedding
    mm = torch.mm
    seq_len = input_ids.size(1)
    position_ids, causal_mask = _position_and_mask(plan["device"], seq_len, cache)

    hidden = embedding(input_ids, plan["wte_weight"])
    hidden = add(hidden, embedding(position_ids, plan["wpe_weight"]))

    for (ln1_w, ln1_b, qkv_w, qkv_b, ao_w, ao_b, ln2_w, ln2_b, fc_w, fc_b, mp_w, mp_b) in plan["layers"]:
        residual = hidden
        normed = layer_norm(hidden, [embed_dim], ln1_w, ln1_b, eps)[0]
        qkv = addmm(qkv_b, normed.view(-1, embed_dim), qkv_w).view(1, seq_len, qkv_dim)
        query = as_strided(qkv, (num_heads, seq_len, head_dim), (head_dim, qkv_dim, 1), 0)
        key_t = as_strided(qkv, (num_heads, head_dim, seq_len), (head_dim, 1, qkv_dim), embed_dim)
        value = as_strided(qkv, (num_heads, seq_len, head_dim), (head_dim, qkv_dim, 1), 2 * embed_dim)
        scores = bmm(query, key_t)
        scores = mul(scores, scaling)
        scores = add(scores, causal_mask)
        probs = softmax(scores, dim=-1)
        context = bmm(probs, value)
        attended = addmm(ao_b, context.transpose(0, 1).contiguous().view(seq_len, embed_dim), ao_w).view(1, seq_len, embed_dim)
        hidden = add(attended, residual)

        residual = hidden
        normed = layer_norm(hidden, [embed_dim], ln2_w, ln2_b, eps)[0]
        projected = addmm(fc_b, normed.view(-1, embed_dim), fc_w).view(1, seq_len, -1)
        half = mul(projected, 0.5)
        cubed = torch.pow(projected, 3.0)
        inner = mul(cubed, _GELU_COEFF)
        inner = add(projected, inner)
        inner = mul(inner, _SQRT_2_OVER_PI)
        inner = torch.tanh(inner)
        inner = add(inner, 1.0)
        projected = mul(half, inner)
        projected = addmm(mp_b, projected.view(-1, projected.size(-1)), mp_w).view(1, seq_len, embed_dim)
        hidden = add(residual, projected)

    hidden = layer_norm(hidden, [embed_dim], plan["ln_f_weight"], plan["ln_f_bias"], eps)[0]
    logits = mm(hidden.view(-1, embed_dim), torch.t(plan["head_weight"])).view(1, seq_len, plan["vocab_size"])

    padded = F.pad(labels, (0, 1), value=_IGNORE_INDEX)
    shift_labels = padded[..., 1:]
    loss = F.cross_entropy(logits.view(-1, plan["vocab_size"]), shift_labels.view(-1),
                           ignore_index=_IGNORE_INDEX, reduction="mean")
    return CausalLMOutputWithCrossAttentions(
        loss=loss,
        logits=logits,
        past_key_values=None,
        hidden_states=None,
        attentions=None,
        cross_attentions=None,
    )


def build(context):
    targets = context.get("targets") or {}
    modules = context.get("modules") or {}
    models = context.get("models") or {}
    references = context["references"]
    record_route = context.get("record_route", lambda *_: None)

    call_id = None
    for cid, spec in targets.items():
        if spec.get("kind") == "module" and spec.get("target") == "gpt2":
            call_id = cid
            break
    if call_id is None:
        return {}

    parent = references.get(call_id)
    if parent is None:
        return {}
    module = modules.get(call_id)
    if module is None:
        module = models.get("gpt2")
    plan = _build_plan(module) if module is not None else None
    cache = {}

    def replacement(*args, **kwargs):
        if plan is not None and not args and set(kwargs) == {"input_ids", "labels", "use_cache"}:
            input_ids = kwargs["input_ids"]
            labels = kwargs["labels"]
            if not kwargs["use_cache"]:
                if (isinstance(input_ids, torch.Tensor) and isinstance(labels, torch.Tensor)
                        and input_ids.dim() == 2 and input_ids.size(0) == 1
                        and input_ids.dtype is torch.int64 and labels.dtype is torch.int64
                        and labels.shape == input_ids.shape
                        and input_ids.device == plan["device"]):
                    try:
                        result = _flat_forward(plan, cache, input_ids, labels)
                        record_route(call_id, "flat:eager_same_op_sequence")
                        return result
                    except Exception:
                        record_route(call_id, "parent:runtime_error")
                        return parent(*args, **kwargs)
            record_route(call_id, "parent:signature_guard")
            return parent(*args, **kwargs)
        if plan is None:
            record_route(call_id, "parent:setup_config_unsupported")
        else:
            record_route(call_id, "parent:signature_guard")
        return parent(*args, **kwargs)

    return {call_id: replacement}
