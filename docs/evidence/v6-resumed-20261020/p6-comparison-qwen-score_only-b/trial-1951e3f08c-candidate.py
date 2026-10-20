"""Prefill-specialized composite replacement for the frozen Qwen3-4B interactive workload.

Replaces the single observed prefill call of ``model`` (Qwen3ForCausalLM.forward,
batch=1, prompt 256, use_cache=True, logits_to_keep=1) with a restructured forward:

  * one-time build-time weight repacking: q/k/v projections concatenated into a
    single [6144, 2560] GEMM operand and gate/up into [19456, 2560]; the original
    ``nn.Linear.weight`` parameters are repointed to row-view ``nn.Parameter``s of
    the packed storage (same values, same row-major layout), so total weight memory
    is unchanged and the untouched decode path keeps working unchanged;
  * per-layer fused Triton kernels (with eager-aten fallbacks): fused residual-add +
    RMSNorm (one kernel instead of add + 8-op eager norm chain), fused per-head
    q/k RMSNorm + rotary embedding writing SDPA-ready [B, H, S, D] tensors, and a
    fused SiLU*up kernel;
  * the SDPA call, o/down GEMMs, embedding, rotary cos/sin computation (through the
    original module) and lm_head are preserved; lm_head uses a bandwidth-oriented
    Triton GEMV (cublas gemv measured 827 us for a 778 MB weight read) with
    F.linear fallback;
  * the prefill KV cache is written directly into a genuine ``DynamicCache``
    (config-constructed exactly like the original), skipping only the
    concatenate-with-empty kernels; all 31 decode steps then run the untouched
    original forward.

Numerics mirror the reference op-by-op rounding: every intermediate that the
eager chain materializes as bf16 is rounded to bf16 inside the kernels, and all
norm/variance math accumulates in fp32 like ``Qwen3RMSNorm``.

A strict signature guard falls back to the parent bound forward for anything that
is not the observed prefill signature (e.g. decode-shaped calls), and any failure
of the Triton toolchain degrades to an aten path that keeps the fused GEMMs.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from transformers.cache_utils import DynamicCache
from transformers.modeling_outputs import CausalLMOutputWithPast

try:  # Triton is available on the worker's torch 2.8 + CUDA build; degrade if not.
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except Exception:  # pragma: no cover - exercised only on CPU-only environments
    triton = None
    tl = None
    _HAS_TRITON = False


# --------------------------------------------------------------------------- #
# Triton kernels (no autotuning, deterministic, fixed launch parameters)
# --------------------------------------------------------------------------- #
if _HAS_TRITON:

    @triton.jit
    def _rmsnorm_kernel(x_ptr, w_ptr, o_ptr, N, eps, BLOCK: tl.constexpr):
        row = tl.program_id(0)
        cols = tl.arange(0, BLOCK)
        m = cols < N
        x = tl.load(x_ptr + row * N + cols, mask=m, other=0.0).to(tl.float32)
        var = tl.sum(tl.where(m, x * x, 0.0), axis=0) / N
        inv = tl.rsqrt(var + eps)
        t = (x * inv).to(tl.bfloat16)  # reference casts normed value to bf16 first
        w = tl.load(w_ptr + cols, mask=m, other=0.0).to(tl.float32)
        out = (t.to(tl.float32) * w).to(tl.bfloat16)  # aten bf16 mul: fp32 math, bf16 round
        tl.store(o_ptr + row * N + cols, out, mask=m)

    @triton.jit
    def _add_rmsnorm_kernel(x_ptr, d_ptr, w_ptr, h_ptr, o_ptr, N, eps, BLOCK: tl.constexpr):
        row = tl.program_id(0)
        cols = tl.arange(0, BLOCK)
        m = cols < N
        x = tl.load(x_ptr + row * N + cols, mask=m, other=0.0).to(tl.float32)
        d = tl.load(d_ptr + row * N + cols, mask=m, other=0.0).to(tl.float32)
        hn = (x + d).to(tl.bfloat16)  # residual add is materialized in bf16
        tl.store(h_ptr + row * N + cols, hn, mask=m)
        h = hn.to(tl.float32)  # the norm then reads the rounded bf16 sum
        var = tl.sum(tl.where(m, h * h, 0.0), axis=0) / N
        inv = tl.rsqrt(var + eps)
        t = (h * inv).to(tl.bfloat16)
        w = tl.load(w_ptr + cols, mask=m, other=0.0).to(tl.float32)
        out = (t.to(tl.float32) * w).to(tl.bfloat16)
        tl.store(o_ptr + row * N + cols, out, mask=m)

    @triton.jit
    def _qk_norm_rope_kernel(
        qkv_ptr,
        cos_ptr,
        sin_ptr,
        qw_ptr,
        kw_ptr,
        out_ptr,  # [num_q_heads + num_kv_heads, S, head_dim]
        S,
        QH: tl.constexpr,
        KOFF: tl.constexpr,  # column offset of k block inside qkv (= QH * head_dim)
        HD: tl.constexpr,
        HQ: tl.constexpr,  # heads handled per program (both q and k tiles)
        NQB: tl.constexpr,  # number of q head-blocks (= QH // HQ)
        EPS: tl.constexpr,
        QKVN: tl.constexpr,  # row width of the fused qkv GEMM output
    ):
        s = tl.program_id(0)
        hb = tl.program_id(1)
        HALF: tl.constexpr = HD // 2
        d = tl.arange(0, HALF)
        heads = tl.arange(0, HQ)
        is_q = hb < NQB
        hid = hb * HQ + heads  # q heads are 0..QH-1; k heads continue at QH in out
        src = tl.where(is_q, hid * HD, KOFF + heads * HD)  # column base inside qkv row
        c1 = tl.load(cos_ptr + s * HD + d).to(tl.float32)
        s1 = tl.load(sin_ptr + s * HD + d).to(tl.float32)
        c2 = tl.load(cos_ptr + s * HD + HALF + d).to(tl.float32)
        s2 = tl.load(sin_ptr + s * HD + HALF + d).to(tl.float32)
        qw1 = tl.load(qw_ptr + d).to(tl.float32)
        kw1 = tl.load(kw_ptr + d).to(tl.float32)
        w1 = tl.where(is_q, qw1, kw1)
        qw2 = tl.load(qw_ptr + HALF + d).to(tl.float32)
        kw2 = tl.load(kw_ptr + HALF + d).to(tl.float32)
        w2 = tl.where(is_q, qw2, kw2)
        a = tl.load(qkv_ptr + s * QKVN + src[:, None] + d[None, :]).to(tl.float32)
        b = tl.load(qkv_ptr + s * QKVN + src[:, None] + (HALF + d)[None, :]).to(tl.float32)
        var = (tl.sum(a * a, axis=1) + tl.sum(b * b, axis=1)) / HD
        inv = tl.rsqrt(var + EPS)
        na = ((a * inv[:, None]).to(tl.bfloat16).to(tl.float32) * w1[None, :]).to(tl.bfloat16)
        nb = ((b * inv[:, None]).to(tl.bfloat16).to(tl.float32) * w2[None, :]).to(tl.bfloat16)
        naf = na.to(tl.float32)
        nbf = nb.to(tl.float32)
        # q_embed = (q * cos) + (rotate_half(q) * sin), with bf16 rounding after
        # each aten-visible step exactly like the eager chain.
        t1a = (naf * c1[None, :]).to(tl.bfloat16).to(tl.float32)
        t2a = ((-nbf) * s1[None, :]).to(tl.bfloat16).to(tl.float32)
        outa = (t1a + t2a).to(tl.bfloat16)
        t1b = (nbf * c2[None, :]).to(tl.bfloat16).to(tl.float32)
        t2b = (naf * s2[None, :]).to(tl.bfloat16).to(tl.float32)
        outb = (t1b + t2b).to(tl.bfloat16)
        dst = out_ptr + hid[:, None] * (S * HD) + s * HD
        tl.store(dst + d[None, :], outa)
        tl.store(dst + (HALF + d)[None, :], outb)

    @triton.jit
    def _silu_mul_kernel(x_ptr, o_ptr, I, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        nblk = tl.cdiv(I, BLOCK)
        row = pid // nblk
        blk = pid % nblk
        cols = blk * BLOCK + tl.arange(0, BLOCK)
        m = cols < I
        g = tl.load(x_ptr + row * (2 * I) + cols, mask=m, other=0.0).to(tl.float32)
        u = tl.load(x_ptr + row * (2 * I) + I + cols, mask=m, other=0.0).to(tl.float32)
        sg = (g * tl.sigmoid(g)).to(tl.bfloat16)  # aten silu: fp32 math, bf16 store
        out = (sg.to(tl.float32) * u).to(tl.bfloat16)  # aten mul: fp32 math, bf16 store
        tl.store(o_ptr + row * I + cols, out, mask=m)

    @triton.jit
    def _gemv_kernel(w_ptr, x_ptr, y_ptr, N, K: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
        pid = tl.program_id(0)
        rn = pid * BN + tl.arange(0, BN)
        rm = rn < N
        acc = tl.zeros((BN,), dtype=tl.float32)
        for k0 in range(0, K, BK):
            rk = k0 + tl.arange(0, BK)
            km = rk < K
            w = tl.load(
                w_ptr + rn[:, None] * K + rk[None, :],
                mask=rm[:, None] & km[None, :],
                other=0.0,
            ).to(tl.float32)
            x = tl.load(x_ptr + rk, mask=km, other=0.0).to(tl.float32)
            acc += tl.sum(w * x[None, :], axis=1)
        tl.store(y_ptr + rn, acc.to(tl.bfloat16), mask=rm)


# --------------------------------------------------------------------------- #
# aten fallbacks: exact replicas of the transformers eager op chains
# --------------------------------------------------------------------------- #
def _aten_rmsnorm(x, w, eps):
    dt = x.dtype
    h = x.to(torch.float32)
    var = h.pow(2).mean(-1, keepdim=True)
    h = h * torch.rsqrt(var + eps)
    return w * h.to(dt)


def _aten_add_rmsnorm(x, delta, w, eps):
    hn = x + delta
    return hn, _aten_rmsnorm(hn, w, eps)


def _aten_rope(q, k, cos, sin):
    # q, k: [1, H, S, D]; cos/sin: [1, S, D] (bf16)
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)

    def rot(x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    return (q * cos) + rot(q) * sin, (k * cos) + rot(k) * sin


# --------------------------------------------------------------------------- #
# candidate build
# --------------------------------------------------------------------------- #
def _pack_linears(mods):
    """Concatenate [A, K] weights row-wise; repoint originals to row views."""
    packed = torch.cat([m.weight.data for m in mods], dim=0).contiguous()
    rg = mods[0].weight.requires_grad
    off = 0
    for m in mods:
        n = m.weight.shape[0]
        m.weight = torch.nn.Parameter(packed[off : off + n], requires_grad=rg)
        off += n
    return packed


class _Prefill:  # closure capturing packed per-layer state
    def __init__(self, model, ref, use_triton):
        self.model = model
        self.ref = ref
        self.triton = use_triton
        self.fallbacks = 0
        cfg = model.config
        self.num_layers = cfg.num_hidden_layers
        self.hidden = cfg.hidden_size
        self.eps = cfg.rms_norm_eps
        self.scale = cfg.head_dim ** -0.5
        base = model.model
        self.embed = base.embed_tokens
        self.rotary = base.rotary_emb
        self.final_w = base.norm.weight.data
        self.lm_head = model.lm_head
        self.head_dim = cfg.head_dim if getattr(cfg, "head_dim", None) else cfg.hidden_size // cfg.num_attention_heads
        self.q_heads = cfg.num_attention_heads
        self.kv_heads = cfg.num_key_value_heads
        self.q_out = self.q_heads * self.head_dim
        self.kv_out = self.kv_heads * self.head_dim
        self.inter = cfg.intermediate_size
        # qk norm+rope tiling: one program tile covers HQ = kv_heads heads; the
        # grid is (S, q_heads // kv_heads + 1): q blocks then a single k block.
        self.hq = self.kv_heads
        self.qk_ok = (
            use_triton
            and self.q_heads % self.kv_heads == 0
            and self.kv_heads & (self.kv_heads - 1) == 0
            and self.head_dim % 2 == 0
            and self.head_dim <= 4096
        )
        self.layers = []
        for lyr in base.layers:
            sa = lyr.self_attn
            self.layers.append(
                {
                    "in_w": lyr.input_layernorm.weight.data,
                    "post_w": lyr.post_attention_layernorm.weight.data,
                    "qw": sa.q_norm.weight.data,
                    "kw": sa.k_norm.weight.data,
                    "o_w": sa.o_proj.weight.data,
                    "down_w": lyr.mlp.down_proj.weight.data,
                }
            )
        self.qkv_ws = None  # filled by build() after weight packing
        self.gu_ws = None

    # -- cache -------------------------------------------------------------
    def _new_cache(self):
        return DynamicCache(config=self.model.config)

    # -- triton wrappers ----------------------------------------------------
    def _rmsnorm(self, x, w, n):
        out = torch.empty_like(x)
        _rmsnorm_kernel[(n,)](x, w, out, x.shape[-1], self.eps, BLOCK=4096, num_warps=8)
        return out

    def _add_rmsnorm(self, x, delta, w, n):
        hn = torch.empty_like(x)
        out = torch.empty_like(x)
        _add_rmsnorm_kernel[(n,)](x, delta, w, hn, out, x.shape[-1], self.eps, BLOCK=4096, num_warps=8)
        return hn, out

    def _silu_mul(self, x, S):
        inter = x.shape[-1] // 2
        out = torch.empty((S, inter), dtype=x.dtype, device=x.device)
        grid = (S * triton.cdiv(inter, 2048),)
        _silu_mul_kernel[grid](x, out, inter, BLOCK=2048, num_warps=8)
        return out

    def _lm_gemv(self, w, x, N, K):
        y = torch.empty((N,), dtype=x.dtype, device=x.device)
        _gemv_kernel[(triton.cdiv(N, 128),)](w, x, y, N, K=K, BN=128, BK=256, num_warps=8)
        return y

    # -- one fused decoder layer (attention part) ---------------------------
    def _attention(self, i, normed, cos, sin, cache, S):
        L = self.layers[i]
        qkv = F.linear(normed, self.qkv_ws[i])  # [S, q+2k]
        if self.qk_ok:
            qw, kw = L["qw"], L["kw"]
            dev = qkv.device
            QKVN = self.q_out + 2 * self.kv_out
            qk = torch.empty(
                (self.q_heads + self.kv_heads, S, self.head_dim), dtype=qkv.dtype, device=dev
            )
            nqb = self.q_heads // self.hq
            _qk_norm_rope_kernel[(S, nqb + 1)](
                qkv,
                cos,
                sin,
                qw,
                kw,
                qk,
                S,
                QH=self.q_heads,
                KOFF=self.q_out,
                HD=self.head_dim,
                HQ=self.hq,
                NQB=nqb,
                EPS=self.eps,
                QKVN=QKVN,
                num_warps=4,
            )
            q = qk[: self.q_heads].unsqueeze(0)
            k = qk[self.q_heads :].unsqueeze(0)
            v = torch.as_strided(
                qkv,
                (1, self.kv_heads, S, self.head_dim),
                (S * QKVN, self.head_dim, QKVN, 1),
                self.q_out + self.kv_out,
            )
        else:
            hq = qkv[:, : self.q_out].reshape(S, self.q_heads, self.head_dim)
            hk = qkv[:, self.q_out : self.q_out + self.kv_out].reshape(S, self.kv_heads, self.head_dim)
            hq = _aten_rmsnorm(hq, L["qw"], self.eps).transpose(0, 1).unsqueeze(0)
            hk = _aten_rmsnorm(hk, L["kw"], self.eps).transpose(0, 1).unsqueeze(0)
            q, k = _aten_rope(hq, hk, cos.view(1, S, self.head_dim), sin.view(1, S, self.head_dim))
            v = qkv[
                :, self.q_out + self.kv_out : self.q_out + 2 * self.kv_out
            ].reshape(S, self.kv_heads, self.head_dim).transpose(0, 1).unsqueeze(0)
        # store into cache without concatenate-with-empty kernels
        dl = cache.layers[i]
        if not dl.is_initialized:
            dl.lazy_initialization(k, v)
        dl.keys = k.contiguous() if not k.is_contiguous() else k
        dl.values = v
        attn = F.scaled_dot_product_attention(
            q, k, v, attn_mask=None, dropout_p=0.0, is_causal=True, scale=self.scale, enable_gqa=True
        )
        attn = attn.transpose(1, 2).contiguous().view(S, self.q_out)
        return F.linear(attn, L["o_w"])

    def _mlp(self, i, normed, S):
        L = self.layers[i]
        gu = F.linear(normed, self.gu_ws[i])  # [S, 2*inter]
        if self.triton:
            act = self._silu_mul(gu, S)
        else:
            act = F.silu(gu[:, : self.inter]) * gu[:, self.inter :]
        return F.linear(act, L["down_w"])

    # -- main ---------------------------------------------------------------
    def __call__(self, *args, **kwargs):
        (input_ids, attention_mask, position_ids, past_key_values, inputs_embeds, labels,
         use_cache, logits_to_keep, more) = _split(kwargs)
        if args or not self._is_prefill(input_ids, attention_mask, position_ids,
                                        past_key_values, inputs_embeds, labels,
                                        use_cache, logits_to_keep, more):
            return self.ref(*args, **kwargs)
        try:
            return self.forward(input_ids)
        except Exception:
            # never compromise the workload: recompute through the parent path
            self.fallbacks += 1
            return self.ref(
                input_ids=input_ids, use_cache=use_cache, logits_to_keep=logits_to_keep, **more
            )

    def _is_prefill(self, input_ids, attention_mask, position_ids, past_key_values,
                    inputs_embeds, labels, use_cache, logits_to_keep, more):
        return (
            input_ids is not None
            and attention_mask is None
            and position_ids is None
            and past_key_values is None
            and inputs_embeds is None
            and labels is None
            and more == {}
            and use_cache is True
            and isinstance(logits_to_keep, int)
            and logits_to_keep == 1
            and input_ids.dim() == 2
            and input_ids.shape[0] == 1
            and input_ids.shape[1] > 1
            and input_ids.is_cuda
        )

    def forward(self, input_ids):
        S = input_ids.shape[1]
        dev = input_ids.device
        H = self.hidden
        hs = self.embed(input_ids).view(S, H)  # [S, H] residual stream
        cache = self._new_cache()
        pos = torch.arange(S, device=dev).unsqueeze(0)
        cos, sin = self.rotary(hs.view(1, S, H), pos)  # bf16 [1, S, D], exact module call
        cos2 = cos.view(S, self.head_dim)
        sin2 = sin.view(S, self.head_dim)

        if self.triton:
            normed = self._rmsnorm(hs, self.layers[0]["in_w"], S)
        else:
            normed = _aten_rmsnorm(hs, self.layers[0]["in_w"], self.eps)

        for i in range(self.num_layers):
            attn_out = self._attention(i, normed, cos2, sin2, cache, S)
            if self.triton:
                hs, normed = self._add_rmsnorm(hs, attn_out, self.layers[i]["post_w"], S)
            else:
                hs, normed = _aten_add_rmsnorm(hs, attn_out, self.layers[i]["post_w"], self.eps)
            mlp_out = self._mlp(i, normed, S)
            if i + 1 < self.num_layers:
                if self.triton:
                    hs, normed = self._add_rmsnorm(hs, mlp_out, self.layers[i + 1]["in_w"], S)
                else:
                    hs, normed = _aten_add_rmsnorm(hs, mlp_out, self.layers[i + 1]["in_w"], self.eps)
            else:
                last = hs[-1:] + mlp_out[-1:]  # only the final position feeds logits
                if self.triton:
                    last = self._rmsnorm(last, self.final_w, 1)
                else:
                    last = _aten_rmsnorm(last, self.final_w, self.eps)
                logits = self._lm_logits(last)

        return CausalLMOutputWithPast(
            loss=None, logits=logits, past_key_values=cache, hidden_states=None, attentions=None
        )

    def _lm_logits(self, last):
        w = self.lm_head.weight
        V, K = w.shape
        if self.triton and K == last.shape[-1] and last.shape[-2] == 1:
            return self._lm_gemv(w, last.view(K), V, K).view(1, 1, V)
        return F.linear(last.view(1, 1, K), w)  # [1, 1, V] like the reference


def _split(kwargs):
    more = dict(kwargs)
    get = lambda k: more.pop(k, None)
    input_ids = get("input_ids")
    attention_mask = get("attention_mask")
    position_ids = get("position_ids")
    past_key_values = get("past_key_values")
    inputs_embeds = get("inputs_embeds")
    labels = get("labels")
    use_cache = get("use_cache")
    logits_to_keep = get("logits_to_keep")
    return (
        input_ids, attention_mask, position_ids, past_key_values, inputs_embeds, labels,
        use_cache, logits_to_keep, more,
    )


def _warm_kernels(pf, S):
    """JIT-compile the Triton kernels and warm the new GEMM/SDPA shapes at build
    time on dummy data (real kernel specializations, real weight tensors, random
    activations only - no fixture input is involved)."""
    dev = next(pf.model.parameters()).device
    H = pf.hidden
    dt = torch.bfloat16
    with torch.cuda.device(dev):
        with torch.no_grad():
            x = torch.randn(S, H, device=dev, dtype=dt)
            d = torch.randn(S, H, device=dev, dtype=dt)
            w = torch.randn(H, device=dev, dtype=dt)
            pf._rmsnorm(x, w, S)
            pf._add_rmsnorm(x, d, w, S)
            # fused qkv + norm/rope kernel at real geometry
            qkv = torch.randn(S, pf.q_out + 2 * pf.kv_out, device=dev, dtype=dt)
            cos = torch.randn(S, pf.head_dim, device=dev, dtype=dt)
            sin = torch.randn(S, pf.head_dim, device=dev, dtype=dt)
            qk = torch.empty((pf.q_heads + pf.kv_heads, S, pf.head_dim), device=dev, dtype=dt)
            nqb = pf.q_heads // pf.hq
            _qk_norm_rope_kernel[(S, nqb + 1)](
                qkv, cos, sin, pf.layers[0]["qw"], pf.layers[0]["kw"], qk, S,
                QH=pf.q_heads, KOFF=pf.q_out, HD=pf.head_dim, HQ=pf.hq, NQB=nqb,
                EPS=pf.eps, QKVN=pf.q_out + 2 * pf.kv_out, num_warps=4,
            )
            v = torch.as_strided(
                qkv, (1, pf.kv_heads, S, pf.head_dim),
                (S * (pf.q_out + 2 * pf.kv_out), pf.head_dim, pf.q_out + 2 * pf.kv_out, 1),
                pf.q_out + pf.kv_out,
            )
            attn = F.scaled_dot_product_attention(
                qk[: pf.q_heads].unsqueeze(0), qk[pf.q_heads :].unsqueeze(0), v,
                attn_mask=None, dropout_p=0.0, is_causal=True, scale=pf.scale, enable_gqa=True,
            )
            F.linear(attn.transpose(1, 2).contiguous().view(S, pf.q_out), pf.layers[0]["o_w"])
            # gate_up + silu*up + down at real geometry
            gu = F.linear(x, pf.gu_ws[0])
            pf._silu_mul(gu, S)
            F.linear(gu[:, : pf.inter], pf.layers[0]["down_w"])
            # fused qkv GEMM + lm_head gemv (real weights, dummy activations)
            F.linear(x, pf.qkv_ws[0])
            pf._lm_logits(torch.randn(1, 1, H, device=dev, dtype=dt))
        torch.cuda.synchronize(dev)


def build(context):
    targets = context.get("targets") or {}
    state = context.get("state") or {}
    models = context.get("models") or {}
    references = context.get("references") or {}

    model = models.get("model") or state.get("model")
    if model is None:
        return {}

    call_id = None
    for cid, spec in targets.items():
        if (
            spec.get("kind") == "module"
            and spec.get("target") == "model"
            and spec.get("phase") == "prefill"
        ):
            call_id = cid
            break
    if call_id is None:
        return {}

    ref = references.get(call_id) or model.forward

    use_triton = _HAS_TRITON and torch.cuda.is_available()

    prefill = _Prefill(model, ref, use_triton)

    # Build-time invariant weight packing (once per physical module): fused
    # qkv [q+2k, H] and gate_up [2*inter, H] operands. The original Linear
    # weights are repointed to row views of the packed storage, so the values,
    # row-major layout and total bytes are unchanged and the untouched decode
    # path (and any fallback) sees identical weights.
    with torch.no_grad():
        prefill.qkv_ws = []
        prefill.gu_ws = []
        for lyr in model.model.layers:
            sa = lyr.self_attn
            prefill.qkv_ws.append(_pack_linears([sa.q_proj, sa.k_proj, sa.v_proj]))
            prefill.gu_ws.append(_pack_linears([lyr.mlp.gate_proj, lyr.mlp.up_proj]))

    if use_triton:
        try:
            if next(model.parameters()).device.type != "cuda":
                raise RuntimeError("model is not CUDA-resident")
            _warm_kernels(prefill, 256)
        except Exception:
            use_triton = False
            prefill.triton = False
            prefill.qk_ok = False

    return {call_id: prefill}
