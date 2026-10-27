"""Candidate: host-lean fused composite for the GPT2Block module call sites.

Replaces each of the 12 observed ``model.transformer.h.*`` GPT2Block forward
sites (group 3b09e47895ee88ac921d, phase ``gradient_accumulation``) with one
Python callable that issues the exact aten kernel sequence of the installed
parent path (transformers 5.16.1 eager GPT-2):

  native_layer_norm -> addmm(c_attn) -> view/split/view/transpose ->
  bmm(q,k^T) -> mul.Scalar(0.125) -> add.Tensor(mask) -> _softmax ->
  bmm(w,v) -> transpose -> reshape(clone) -> addmm(c_proj) -> add.Tensor ->
  native_layer_norm -> addmm(c_fc) -> NewGELU(tanh formula: pow, mul, add,
  mul, tanh, add, mul) -> addmm(mlp c_proj) -> add.Tensor

The p=0.0 dropout hops and the float32->float32 ``Tensor.type`` call of the
parent are identity operations (verified: ``F.dropout(x, 0.0, True) is x`` and
``x.type(torch.float32) is x``) and are skipped; every non-identity kernel,
its operand order and tensor layout are reproduced verbatim, so forward
outputs, autograd graph and backward kernels are bitwise identical to the
parent. CPU validation with the frozen 12x768 architecture (seed 20261020)
shows bitwise-equal loss, logits, all gradients, accumulated gradients over
two microbatches and post-SGD parameters versus the untouched parent path.

Motivation (selection de4d95e2 hypothesis): removing the per-block Python
dispatch (module _call_impl chains, attention branching/config lookups,
Conv1D/NewGELU/Dropout module hops) reduces host submission time and its
jitter, which the execution_timeline collector attributes the waiting rank's
in-kernel NCCL excess to. Communication behavior itself is untouched.
"""

import math

import torch
import torch.nn.functional as F

_BLOCK_SEMANTICS_MARKER = "GPT2Block"
_IGNORED_KWARGS = frozenset({"position_ids", "output_attentions", "output_hidden_states", "cache_position"})

_route_composite = "composite:gpt2_block:eager_fused"
_route_signature = "fallback:gpt2_block:signature"
_route_setup = "fallback:gpt2_block:setup_guard"


def _make_fused_block(block, call_id, parent, record):
    """Build a fused replacement for one GPT2Block site.

    Returns None when a build-time guard fails (unexpected module structure);
    the caller then installs a labelled parent passthrough so the setup
    failure stays observable instead of silently vanishing into the parent.
    """
    try:
        ln_1 = block.ln_1
        ln_2 = block.ln_2
        attn = block.attn
        mlp = block.mlp
        c_attn = attn.c_attn
        attn_c_proj = attn.c_proj
        c_fc = mlp.c_fc
        mlp_c_proj = mlp.c_proj
        embed_dim = attn.embed_dim
        heads = attn.num_heads
        head_dim = attn.head_dim
        split_size = attn.split_size
        inter = c_fc.nf
        eps = ln_1.eps
        scaling = attn.scaling
        ok = (
            not attn.is_cross_attention
            and not hasattr(block, "crossattention")
            and attn.config._attn_implementation == "eager"
            and not attn.reorder_and_upcast_attn
            and attn.scale_attn_weights
            and not attn.scale_attn_by_inverse_layer_idx
            and float(attn.attn_dropout.p) == 0.0
            and float(attn.resid_dropout.p) == 0.0
            and float(mlp.dropout.p) == 0.0
            and head_dim * heads == embed_dim
            and split_size == embed_dim
            and float(ln_1.eps) == float(ln_2.eps)
            and tuple(c_attn.weight.shape) == (embed_dim, 3 * embed_dim)
            and tuple(attn_c_proj.weight.shape) == (embed_dim, embed_dim)
            and tuple(c_fc.weight.shape) == (embed_dim, inter)
            and tuple(mlp_c_proj.weight.shape) == (inter, embed_dim)
            and ln_1.weight is not None and ln_1.bias is not None
            and ln_2.weight is not None and ln_2.bias is not None
            and isinstance(scaling, float)
        )
    except Exception:
        return None
    if not ok:
        return None

    ln_1_w = ln_1.weight
    ln_1_b = ln_1.bias
    ln_2_w = ln_2.weight
    ln_2_b = ln_2.bias
    c_attn_w = c_attn.weight
    c_attn_b = c_attn.bias
    a_proj_w = attn_c_proj.weight
    a_proj_b = attn_c_proj.bias
    fc_w = c_fc.weight
    fc_b = c_fc.bias
    mp_w = mlp_c_proj.weight
    mp_b = mlp_c_proj.bias
    norm_shape = (embed_dim,)
    gelu_c = math.sqrt(2.0 / math.pi)
    float32 = torch.float32

    layer_norm = F.layer_norm
    addmm = torch.addmm
    matmul = torch.matmul
    softmax = F.softmax
    tanh = torch.tanh
    pow_ = torch.pow
    tensor_t = torch.Tensor

    def fused_block(hidden_states, past_key_values=None, attention_mask=None,
                    encoder_hidden_states=None, encoder_attention_mask=None,
                    use_cache=False, **kwargs):
        if (past_key_values is not None
                or encoder_hidden_states is not None
                or encoder_attention_mask is not None
                or use_cache
                or not isinstance(hidden_states, tensor_t)
                or hidden_states.dtype is not float32
                or hidden_states.dim() != 3
                or hidden_states.size(-1) != embed_dim
                or (attention_mask is not None and (
                    attention_mask.dtype is not float32 or attention_mask.dim() != 4))
                or not _kwargs_ignorable(kwargs)):
            record(call_id, _route_signature)
            return parent(hidden_states, past_key_values, attention_mask,
                          encoder_hidden_states,
                          encoder_attention_mask=encoder_attention_mask,
                          use_cache=use_cache, **kwargs)
        record(call_id, _route_composite)
        b = hidden_states.size(0)
        s = hidden_states.size(1)
        residual = hidden_states
        x = layer_norm(hidden_states, norm_shape, ln_1_w, ln_1_b, eps)
        qkv = addmm(c_attn_b, x.view(-1, embed_dim), c_attn_w).view(b, s, 3 * embed_dim)
        query, key, value = qkv.split(split_size, dim=2)
        query = query.view(b, s, heads, head_dim).transpose(1, 2)
        key = key.view(b, s, heads, head_dim).transpose(1, 2)
        value = value.view(b, s, heads, head_dim).transpose(1, 2)
        weights = matmul(query, key.transpose(-1, -2)) * scaling
        if attention_mask is not None:
            weights = weights + attention_mask
        weights = softmax(weights, dim=-1)
        attn_out = matmul(weights, value)
        attn_out = attn_out.transpose(1, 2).reshape(b, s, embed_dim)
        attn_out = addmm(a_proj_b, attn_out.view(-1, embed_dim), a_proj_w).view(b, s, embed_dim)
        hidden = attn_out + residual
        residual2 = hidden
        x2 = layer_norm(hidden, norm_shape, ln_2_w, ln_2_b, eps)
        h_fc = addmm(fc_b, x2.view(-1, embed_dim), fc_w).view(b, s, inter)
        act = 0.5 * h_fc * (1.0 + tanh(gelu_c * (h_fc + 0.044715 * pow_(h_fc, 3.0))))
        proj = addmm(mp_b, act.view(-1, inter), mp_w).view(b, s, embed_dim)
        return residual2 + proj

    return fused_block


def _kwargs_ignorable(kwargs):
    """Extra kwargs are ignored by the parent eager path too; keep the guard
    conservative so unexpected requests fall back to the parent."""
    for name in kwargs:
        if name not in _IGNORED_KWARGS:
            return False
    return True


def _make_guard_failed(call_id, parent, record):
    def setup_guard_failed(*args, **kwargs):
        record(call_id, _route_setup)
        return parent(*args, **kwargs)

    return setup_guard_failed


def build(context):
    targets = context.get("targets") or {}
    modules = context.get("modules") or {}
    references = context.get("references") or {}
    record = context.get("record_route") or (lambda *_: None)
    replacements = {}
    for call_id, spec in targets.items():
        try:
            kind = spec.get("kind")
            semantics = str(spec.get("semantics", ""))
        except Exception:
            continue
        if kind != "module" or _BLOCK_SEMANTICS_MARKER not in semantics:
            continue
        module = modules.get(call_id)
        parent = references.get(call_id)
        if module is None or not callable(parent):
            continue
        fused = _make_fused_block(module, call_id, parent, record)
        if fused is None:
            replacements[call_id] = _make_guard_failed(call_id, parent, record)
        else:
            replacements[call_id] = fused
    return replacements
