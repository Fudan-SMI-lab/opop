"""Fused eager reimplementation of the GPT-2 transformer-block region.

Replaces every observed GPT2Block call site (12 layers x 2 accumulation
phases) with a numerically equivalent, dispatch-minimized eager path:

* one ``F.layer_norm`` per pre-norm (no per-call ``nn.Module.__call__``);
* the c_attn / attn.c_proj / mlp.c_fc / mlp.c_proj projections stay
  ``addmm`` GEMMs over the original parameter tensors with the original GEMM
  geometry, so DDP still sees the same parameters becoming ready in the same
  reverse order and the 13 gradient allreduces are untouched;
* the attention core collapses QK^T, the 1/sqrt(head_dim) scaling and the
  causal-mask add into a single ``baddbmm`` cuBLAS epilogue (``alpha=scale``,
  ``beta=1`` over a stride-0 expanded mask view), followed by ``softmax`` and
  one ``bmm`` against V - three aten kernels instead of five;
* NewGELU becomes ``F.gelu(approximate='tanh')``, the identical
  ``0.5*x*(1+tanh(sqrt(2/pi)*(x+0.044715*x^3)))`` formula as one fused forward
  and one fused backward kernel instead of an eight-op pow/tanh chain;
* the frozen config has attn_pdrop=resid_pdrop=embd_pdrop=0.0, so every
  in-block dropout in training mode is an algebraic identity and is skipped.

Every originally observed aten/module site inside the block therefore stops
dispatching; the returned tensor is value-identical to the parent block output
(fp32 rounding-level differences from fused epilogues only), autograd produces
the exact gradients of the same mathematics, and no in-place rewrites are used.

Any signature or attribute drift (cross-attention, KV cache, use_cache,
non-FP32 dtypes, batched masks, dropout p>0, reordered/upcast attention,
unexpected shapes) falls back to the parent bound forward.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def _probe_baddbmm(device) -> bool:
    """Build-time setup check that cuBLAS baddbmm accepts a stride-0 expanded
    bias operand with an alpha epilogue on this worker's bound device; the
    candidate falls back to the explicit scale/add kernels otherwise."""
    if device is None or device.type != "cuda":
        return True
    try:
        a = torch.randn(4, 32, 16, device=device, dtype=torch.float32)
        b = torch.randn(4, 32, 16, device=device, dtype=torch.float32)
        bias = torch.zeros(16, 16, device=device, dtype=torch.float32)
        got = torch.baddbmm(
            bias.unsqueeze(0).expand(4, 16, 16), a, b.transpose(1, 2), beta=1.0, alpha=0.125
        )
        want = torch.bmm(a, b.transpose(1, 2)).mul(0.125)
        torch.cuda.synchronize(device)
        return bool(torch.allclose(got, want, atol=1e-4, rtol=1e-4))
    except Exception:
        return False


def _make_block_forward(block, parent_forward, use_baddbmm):
    ln_1 = block.ln_1
    ln_2 = block.ln_2
    attn = block.attn
    mlp = block.mlp

    ln1_w, ln1_b, ln1_eps = ln_1.weight, ln_1.bias, ln_1.eps
    ln2_w, ln2_b, ln2_eps = ln_2.weight, ln_2.bias, ln_2.eps
    c_attn_w, c_attn_b = attn.c_attn.weight, attn.c_attn.bias
    c_proj_w, c_proj_b = attn.c_proj.weight, attn.c_proj.bias
    fc_w, fc_b = mlp.c_fc.weight, mlp.c_fc.bias
    out_w, out_b = mlp.c_proj.weight, mlp.c_proj.bias

    heads = int(attn.num_heads)
    scale = float(attn.scaling)
    embed_dim = int(c_attn_w.shape[0])

    skip_dropout = (
        float(getattr(attn.attn_dropout, "p", 0.0) or 0.0) == 0.0
        and float(getattr(attn.resid_dropout, "p", 0.0) or 0.0) == 0.0
        and float(getattr(mlp.dropout, "p", 0.0) or 0.0) == 0.0
    )
    plain_eager_attn = (
        not bool(getattr(attn, "is_cross_attention", False))
        and bool(getattr(attn, "scale_attn_weights", True))
        and not bool(getattr(attn, "scale_attn_by_inverse_layer_idx", False))
        and not bool(getattr(attn, "reorder_and_upcast_attn", False))
    )
    fp32 = torch.float32

    def _fast(hidden_states, attention_mask):
        seq = hidden_states.shape[1]
        x = hidden_states.reshape(seq, embed_dim)
        h = F.layer_norm(x, (embed_dim,), ln1_w, ln1_b, ln1_eps)
        qkv = torch.addmm(c_attn_b, h, c_attn_w)
        width = qkv.shape[1] // 3
        head_dim = width // heads
        q, k, v = torch.split(qkv, width, dim=1)
        q = q.view(seq, heads, head_dim).transpose(0, 1)
        k = k.view(seq, heads, head_dim).transpose(0, 1)
        v = v.view(seq, heads, head_dim).transpose(0, 1)
        mask = attention_mask[0, 0]
        if use_baddbmm:
            scores = torch.baddbmm(
                mask.unsqueeze(0).expand(heads, seq, seq),
                q,
                k.transpose(1, 2),
                beta=1.0,
                alpha=scale,
            )
        else:
            scores = torch.bmm(q, k.transpose(1, 2)).mul_(scale).add_(mask)
        probs = torch.softmax(scores, dim=-1)
        context = torch.bmm(probs, v)
        context = context.transpose(0, 1).reshape(seq, width)
        attn_out = torch.addmm(c_proj_b, context, c_proj_w)
        hidden = x + attn_out
        mlp_in = F.layer_norm(hidden, (embed_dim,), ln2_w, ln2_b, ln2_eps)
        ff = torch.addmm(fc_b, mlp_in, fc_w)
        ff = F.gelu(ff, approximate="tanh")
        ff = torch.addmm(out_b, ff, out_w)
        return (hidden + ff).view(hidden_states.shape)

    def _block_forward(
        hidden_states,
        past_key_values=None,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        use_cache=False,
        **kwargs,
    ):
        if (
            skip_dropout
            and plain_eager_attn
            and past_key_values is None
            and encoder_hidden_states is None
            and not use_cache
            and attention_mask is not None
            and hidden_states.dim() == 3
            and hidden_states.shape[0] == 1
            and hidden_states.dtype is fp32
            and hidden_states.shape[2] == embed_dim
            and attention_mask.dim() == 4
            and attention_mask.dtype is fp32
            and attention_mask.shape[0] == 1
            and attention_mask.shape[1] == 1
            and attention_mask.shape[2] == hidden_states.shape[1]
            and attention_mask.shape[3] == hidden_states.shape[1]
        ):
            try:
                return _fast(hidden_states, attention_mask)
            except Exception:
                pass
        return parent_forward(
            hidden_states,
            past_key_values,
            attention_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            use_cache=use_cache,
            **kwargs,
        )

    return _block_forward


def _is_block_site(spec):
    if spec.get("kind") != "module":
        return False
    if not str(spec.get("semantics", "")).endswith(".GPT2Block"):
        return False
    parts = str(spec.get("target", "")).split(".")
    return len(parts) == 4 and parts[0] == "model" and parts[1] == "transformer" and parts[2] == "h"


def build(context):
    targets = context.get("targets") or {}
    modules = context.get("modules") or {}
    references = context.get("references") or {}

    state = context.get("state") or {}
    use_baddbmm = _probe_baddbmm(state.get("device"))

    replacements = {}
    for call_id, spec in targets.items():
        if not _is_block_site(spec):
            continue
        module = modules.get(call_id)
        if module is None:
            continue
        parent_forward = references.get(call_id)
        if not callable(parent_forward):
            parent_forward = module.forward
        replacements[call_id] = _make_block_forward(module, parent_forward, use_baddbmm)
    return replacements
