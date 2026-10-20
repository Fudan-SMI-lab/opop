"""Dispatch-lean eager rewrite of the GPT-2 accumulation forward tree.

Replaces the GPT2LMHeadModel module-call boundary with one fused eager
implementation that issues the same aten kernels (identical shapes, dtypes and
per-element arithmetic) while removing the per-module Python dispatch, the
per-forward causal-mask rebuild and the 7-op NewGELU chain (replaced by the
single aten::gelu approximate='tanh' kernel of the same closed form).
Communication, DDP hooks, optimizer and observation semantics are untouched.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

try:
    from transformers.models.gpt2.modeling_gpt2 import CausalLMOutputWithCrossAttentions as _Output
except Exception:  # pragma: no cover - worker always ships transformers
    class _Output:
        def __init__(self, loss=None, logits=None):
            self.loss, self.logits = loss, logits
            self.past_key_values = None
            self.hidden_states = None
            self.attentions = None
            self.cross_attentions = None


def _pack_model(model):
    transformer = getattr(model, "transformer", None)
    block_list = getattr(transformer, "h", None)
    lm_head = getattr(model, "lm_head", None)
    wte = getattr(transformer, "wte", None)
    wpe = getattr(transformer, "wpe", None)
    ln_f = getattr(transformer, "ln_f", None)
    drop = getattr(transformer, "drop", None)
    if block_list is None or lm_head is None or wte is None or wpe is None or ln_f is None or drop is None:
        return None
    if len(block_list) == 0 or lm_head.bias is not None:
        return None
    config = getattr(model, "config", None)
    if config is None:
        return None
    if getattr(config, "_attn_implementation", "eager") != "eager":
        return None
    if getattr(config, "add_cross_attention", False):
        return None
    if float(drop.p) != 0.0:
        return None
    head_weight = lm_head.weight
    wte_weight, wpe_weight = wte.weight, wpe.weight
    blocks = []
    first = block_list[0].attn
    embed_dim = int(first.embed_dim)
    num_heads = int(first.num_heads)
    head_dim = int(first.head_dim)
    if num_heads * head_dim != embed_dim or embed_dim != wte_weight.shape[1]:
        return None
    for block in block_list:
        attn = block.attn
        mlp = block.mlp
        ln_1, ln_2 = block.ln_1, block.ln_2
        if getattr(attn, "is_cross_attention", False) or hasattr(attn, "q_attn"):
            return None
        if getattr(attn, "reorder_and_upcast_attn", False) or getattr(attn, "scale_attn_by_inverse_layer_idx", False):
            return None
        if not getattr(attn, "scale_attn_weights", True):
            return None
        if float(attn.resid_dropout.p) != 0.0 or float(attn.attn_dropout.p) != 0.0 or float(mlp.dropout.p) != 0.0:
            return None
        if attn.c_attn.weight.shape != (embed_dim, 3 * embed_dim) or attn.c_proj.weight.shape != (embed_dim, embed_dim):
            return None
        if ln_1.weight.shape != (embed_dim,) or ln_2.weight.shape != (embed_dim,):
            return None
        for parameter in (ln_1.weight, ln_1.bias, attn.c_attn.weight, attn.c_attn.bias, attn.c_proj.weight,
                          attn.c_proj.bias, ln_2.weight, ln_2.bias, mlp.c_fc.weight, mlp.c_fc.bias,
                          mlp.c_proj.weight, mlp.c_proj.bias, ln_f.weight, ln_f.bias, head_weight,
                          wte_weight, wpe_weight):
            if parameter is None or parameter.dtype is not torch.float32:
                return None
        blocks.append((ln_1.weight, ln_1.bias, float(ln_1.eps), attn.c_attn.weight, attn.c_attn.bias,
                       float(attn.scaling), attn.c_proj.weight, attn.c_proj.bias,
                       ln_2.weight, ln_2.bias, float(ln_2.eps),
                       mlp.c_fc.weight, mlp.c_fc.bias, mlp.c_proj.weight, mlp.c_proj.bias))
    inner_dim = int(blocks[0][11].shape[1])
    return {
        "wte": wte_weight, "wpe": wpe_weight, "blocks": tuple(blocks),
        "lnf": (ln_f.weight, ln_f.bias, float(ln_f.eps)),
        "head": head_weight, "embed_dim": embed_dim, "inner_dim": inner_dim,
        "num_heads": num_heads, "head_dim": head_dim, "vocab": int(head_weight.shape[0]),
        "positions": int(wpe_weight.shape[0]), "masks": {},
    }


def build(context):
    record_route = context.get("record_route") or (lambda *_: None)
    targets = context["targets"]
    modules = context["modules"]
    references = context["references"]

    root_id = next((ident for ident, spec in targets.items()
                    if spec.get("kind") == "module" and spec.get("target") == "model"), None)
    if root_id is None:
        return {}
    parent = references[root_id]
    model = modules[root_id]
    try:
        packed = _pack_model(model)
    except Exception:
        packed = None
    if packed is None:

        def setup_failed(*args, **kwargs):
            record_route(root_id, "fallback:setup-failure")
            return parent(*args, **kwargs)

        return {root_id: setup_failed}

    wte_weight = packed["wte"]
    wpe_weight = packed["wpe"]
    blocks = packed["blocks"]
    lnf_w, lnf_b, lnf_eps = packed["lnf"]
    head_weight = packed["head"]
    embed_dim = packed["embed_dim"]
    inner_dim = packed["inner_dim"]
    num_heads = packed["num_heads"]
    head_dim = packed["head_dim"]
    vocab = packed["vocab"]
    positions = packed["positions"]
    masks = packed["masks"]
    ln_shape = (embed_dim,)
    triple = 3 * embed_dim

    def forward(input_ids=None, attention_mask=None, past_key_values=None, token_type_ids=None,
                position_ids=None, inputs_embeds=None, encoder_hidden_states=None,
                encoder_attention_mask=None, labels=None, use_cache=None, logits_to_keep=0, **kwargs):
        supported = (
            input_ids is not None and labels is not None and inputs_embeds is None
            and attention_mask is None and token_type_ids is None and position_ids is None
            and encoder_hidden_states is None and encoder_attention_mask is None
            and past_key_values is None and not use_cache and not kwargs
            and logits_to_keep in (0, None) and input_ids.dtype is torch.long
            and input_ids.dim() == 2 and labels.shape == input_ids.shape
            and 0 < input_ids.shape[1] <= positions
        )
        if not supported:
            record_route(root_id, "fallback:unsupported-signature")
            return parent(input_ids=input_ids, attention_mask=attention_mask, past_key_values=past_key_values,
                          token_type_ids=token_type_ids, position_ids=position_ids, inputs_embeds=inputs_embeds,
                          encoder_hidden_states=encoder_hidden_states,
                          encoder_attention_mask=encoder_attention_mask, labels=labels,
                          use_cache=use_cache, logits_to_keep=logits_to_keep, **kwargs)
        try:
            batch, seq = input_ids.shape
            key = (seq, input_ids.device)
            mask = masks.get(key)
            if mask is None:
                keep = torch.ones(seq, seq, dtype=torch.bool, device=input_ids.device).tril()
                mask = torch.where(keep, torch.zeros((), dtype=torch.float32, device=input_ids.device),
                                   torch.full((), torch.finfo(torch.float32).min, device=input_ids.device,
                                              dtype=torch.float32)).view(1, 1, seq, seq)
                masks[key] = mask
            hidden = F.embedding(input_ids, wte_weight) + wpe_weight[:seq]
            for (ln1_w, ln1_b, ln1_eps, ca_w, ca_b, scaling, cp_w, cp_b,
                 ln2_w, ln2_b, ln2_eps, cf_w, cf_b, mp_w, mp_b) in blocks:
                residual = hidden
                h = F.layer_norm(hidden, ln_shape, ln1_w, ln1_b, ln1_eps)
                qkv = torch.addmm(ca_b, h.view(-1, embed_dim), ca_w).view(batch, seq, triple)
                query, key_t, value = qkv.split(embed_dim, dim=2)
                key_t = key_t.view(batch, seq, num_heads, head_dim).transpose(1, 2)
                value = value.view(batch, seq, num_heads, head_dim).transpose(1, 2)
                query = query.view(batch, seq, num_heads, head_dim).transpose(1, 2)
                weights = torch.matmul(query, key_t.transpose(-1, -2)) * scaling
                weights = weights + mask
                weights = F.softmax(weights, dim=-1)
                attn_out = torch.matmul(weights, value).transpose(1, 2).reshape(batch, seq, embed_dim)
                attn_out = torch.addmm(cp_b, attn_out.view(-1, embed_dim), cp_w).view(batch, seq, embed_dim)
                hidden = attn_out + residual
                residual = hidden
                h = F.layer_norm(hidden, ln_shape, ln2_w, ln2_b, ln2_eps)
                h = F.gelu(torch.addmm(cf_b, h.view(-1, embed_dim), cf_w), approximate="tanh")
                hidden = residual + torch.addmm(mp_b, h.view(-1, inner_dim), mp_w).view(batch, seq, embed_dim)
            h = F.layer_norm(hidden, ln_shape, lnf_w, lnf_b, lnf_eps)
            logits = F.linear(h, head_weight)
            shift_labels = F.pad(labels, (0, 1), value=-100)[..., 1:].contiguous()
            loss = F.cross_entropy(logits.view(-1, vocab), shift_labels.view(-1),
                                   ignore_index=-100, reduction="mean")
            record_route(root_id, "fused-eager-tree")
            return _Output(loss=loss, logits=logits)
        except Exception as exc:
            record_route(root_id, "fallback:runtime-error:" + type(exc).__name__)
            return parent(input_ids=input_ids, labels=labels, use_cache=use_cache)

    return {root_id: forward}
