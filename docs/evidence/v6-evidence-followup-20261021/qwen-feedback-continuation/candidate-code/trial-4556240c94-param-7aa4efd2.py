"""Repair candidate for the Qwen3-4B TTFT region: elementwise-only Triton fusion.

Repair of trial-e262267559, whose fused per-head q/k RMSNorm+rotary Triton kernel
mismatched the parent (first mismatch 0:q_rotary max_abs 0.0625, final logits
15511 tolerance violations) and only passed quality through the permanent
parent_fallback_validation route. Developer isolation showed: with FP fusion
disabled the layer-0 norm+rotary was bitwise exact, but at layer 3 the per-head
norm *reduction* (tl.sum) alone still differed from aten mean by one element /
one bf16 ulp, which amplified to 14631 logit violations.

This repair removes every Triton reduction instead of trying to match one:

- Every reduction-bearing op stays on parent calls: each RMSNorm variance is
  `aten::mean` over an fp32 tensor of exact squares (bf16->fp32 squares are
  exact, `pow(2)` is x*x), GEMMs are the same F.linear on the same weight
  tensors and fresh 512B-aligned buffers, SDPA is called with the parent's
  exact arguments, and embed_tokens / rotary_emb / final norm / lm_head run as
  parent module calls.
- Triton is used only for strictly elementwise, reduction-free chains that
  reproduce the parent per-op bf16 rounding chain (one fp32 rounding per torch
  op, explicit bf16 round-trips so no FMA contraction can occur, plus
  enable_fp_fusion=False): residual add + squares, norm apply
  (w * rnd(x * rsqrt(var+eps))), fused per-head norm apply + rotate_half
  rotary writing q into a [b,s,h,hd]-based buffer (parent q_embed strides) and
  k directly into the contiguous [b,h,s,hd] KV cache slice (parent cat output
  layout), v copy into the cache, and SiLU*up (x / (1 + exp(-x)) with
  __nv_expf / __nv_fdiv_rn, the same libdevice symbols aten lowers to).
  rsqrt is __nv_rsqrtf via libdevice, the same symbol aten's rsqrt uses.
- The KV-cache cat, rotate_half cat/neg, norm to/pow/mul chains and their
  ~290 tiny add/rsqrt launches disappear; the decode loop is untouched and
  consumes a parent-compatible DynamicCache.

First-invocation self-validation on the current inputs (quality phase, before
warmup/timed repeats): tier 1 requires bitwise-equal logits and bitwise-equal
per-layer KV cache vs a live parent run; tier 2 (only if tier 1 fails) runs
two full 31-step decode chains and accepts only torch.equal tokens with logits
within half the frozen tolerance. A one-time bitwise probe of the SiLU kernel
against aten silu degrades to aten silu*up for all layers if it mismatches.
Any failure falls back permanently to the parent with distinct route labels
(signature, context, setup failure incl. Triton JIT, validation, runtime), so
a mismatched run shows no gain instead of a quality failure.

The CPU-simulated rounding chains of every kernel are bitwise-equal to the
transformers forward for logits, all KV tensors and decode steps; the full CPU
model-level simulation (parent modules + emulated kernels) matched logits,
cache and a decode step bitwise.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
    from triton.language.extra.cuda import libdevice as _ld

    _HAS_TRITON = True
except Exception:  # pragma: no cover - worker has triton 3.7.1
    _HAS_TRITON = False


if _HAS_TRITON:

    @triton.jit
    def _sq_kernel(x_ptr, sq_ptr, n, BLOCK: tl.constexpr):
        # sq = f32(x)^2, exact in fp32 for bf16-valued inputs (parent to+pow)
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        m = offs < n
        x = tl.load(x_ptr + offs, mask=m, other=0.0).to(tl.float32)
        tl.store(sq_ptr + offs, x * x, mask=m)

    @triton.jit
    def _add_sq_kernel(x_ptr, r_ptr, s_ptr, sq_ptr, n, BLOCK: tl.constexpr):
        # s = rnd_bf16(f32(x)+f32(r))  (parent residual add, one rounding)
        # sq = f32(s)^2 exact (parent to+pow on the add output)
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        m = offs < n
        x = tl.load(x_ptr + offs, mask=m, other=0.0).to(tl.float32)
        r = tl.load(r_ptr + offs, mask=m, other=0.0).to(tl.float32)
        s = (x + r).to(tl.bfloat16)
        tl.store(s_ptr + offs, s, mask=m)
        sf = s.to(tl.float32)
        tl.store(sq_ptr + offs, sf * sf, mask=m)

    @triton.jit
    def _norm_apply_kernel(x_ptr, w_ptr, var_ptr, y_ptr, n_cols, eps, BLOCK: tl.constexpr):
        # inv = rsqrt(var + eps)  (__nv_rsqrtf, same symbol as aten rsqrt)
        # xn = rnd_bf16(x * inv); y = rnd_bf16(w * f32(xn))  (parent mul/to/mul)
        row = tl.program_id(0)
        chunk = tl.program_id(1)
        cols = chunk * BLOCK + tl.arange(0, BLOCK)
        m = cols < n_cols
        var = tl.load(var_ptr + row)
        inv = _ld.rsqrt(var + eps)
        x = tl.load(x_ptr + row * n_cols + cols, mask=m, other=0.0).to(tl.float32)
        xn = (x * inv).to(tl.bfloat16)
        w = tl.load(w_ptr + cols, mask=m, other=0.0).to(tl.float32)
        y = (w * xn.to(tl.float32)).to(tl.bfloat16)
        tl.store(y_ptr + row * n_cols + cols, y, mask=m)

    @triton.jit
    def _norm_rotary_kernel(
        q_ptr, w_ptr, var_ptr, cos_ptr, sin_ptr, out_ptr,
        seq, n_heads, eps,
        HEAD_DIM: tl.constexpr, OUT_BSHD: tl.constexpr,
    ):
        # Per-head RMSNorm apply (variance from aten mean, loaded here) fused
        # with rotate_half rotary, reproducing the parent rounding chain:
        # qn = rnd_bf16(w * rnd_bf16(x*inv)); a = rnd_bf16(qn*cos);
        # b = rnd_bf16(qn2 * (sin*sign)); y = rnd_bf16(f32(a)+f32(b)).
        # Explicit bf16 round-trips between multiplies/adds prevent FMA
        # contraction; the sign multiply is exact ( +-1 ).
        pid = tl.program_id(0)
        h = pid % n_heads
        t = pid // n_heads  # token row = bi*seq + p
        p = t % seq
        bi = t // seq
        d = tl.arange(0, HEAD_DIM)
        half = HEAD_DIM // 2
        d2 = tl.where(d < half, d + half, d - half)
        src = q_ptr + t * (n_heads * HEAD_DIM) + h * HEAD_DIM
        x = tl.load(src + d).to(tl.float32)
        x2 = tl.load(src + d2).to(tl.float32)
        w = tl.load(w_ptr + d).to(tl.float32)
        w2 = tl.load(w_ptr + d2).to(tl.float32)
        var = tl.load(var_ptr + t * n_heads + h)
        inv = _ld.rsqrt(var + eps)
        xn = (x * inv).to(tl.bfloat16)
        xn2 = (x2 * inv).to(tl.bfloat16)
        qn = (w * xn.to(tl.float32)).to(tl.bfloat16)
        qn2 = (w2 * xn2.to(tl.float32)).to(tl.bfloat16)
        sign = tl.where(d < half, -1.0, 1.0)
        c = tl.load(cos_ptr + t * HEAD_DIM + d).to(tl.float32)
        s = tl.load(sin_ptr + t * HEAD_DIM + d).to(tl.float32)
        a = (qn.to(tl.float32) * c).to(tl.bfloat16)
        b = (qn2.to(tl.float32) * (s * sign)).to(tl.bfloat16)
        y = (a.to(tl.float32) + b.to(tl.float32)).to(tl.bfloat16)
        if OUT_BSHD:
            off = t * (n_heads * HEAD_DIM) + h * HEAD_DIM + d
        else:
            off = bi * (n_heads * seq * HEAD_DIM) + h * (seq * HEAD_DIM) + p * HEAD_DIM + d
        tl.store(out_ptr + off, y)

    @triton.jit
    def _silu_mul_kernel(g_ptr, u_ptr, o_ptr, n, BLOCK: tl.constexpr):
        # sv = rnd_bf16(g / (1 + exp(-g)))  (__nv_expf / __nv_fdiv_rn, the
        # same libdevice symbols aten's silu kernel lowers to);
        # y = rnd_bf16(f32(sv) * f32(u))  (parent bf16 mul)
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        m = offs < n
        g = tl.load(g_ptr + offs, mask=m, other=0.0).to(tl.float32)
        u = tl.load(u_ptr + offs, mask=m, other=0.0).to(tl.float32)
        e = _ld.exp(-g)
        sv = _ld.div_rn(g, 1.0 + e).to(tl.bfloat16)
        y = (sv.to(tl.float32) * u).to(tl.bfloat16)
        tl.store(o_ptr + offs, y, mask=m)


_LABEL_FUSED = "fused_elementwise_prefill"
_LABEL_SIGNATURE = "parent_fallback_signature"
_LABEL_CONTEXT = "parent_fallback_context"
_LABEL_SETUP = "parent_fallback_setup_failure"
_LABEL_RUNTIME = "parent_fallback_runtime_failure"
_LABEL_VALIDATION = "parent_fallback_validation"

_VAL_ATOL = 0.01  # half of the frozen quality tolerance (0.02)
_VAL_RTOL = 0.01

_EB = 1024  # elementwise block
_NB = 1024  # norm-apply block


def _find_region_root(targets):
    fallback = None
    for call_id, spec in targets.items():
        if not isinstance(spec, dict):
            continue
        if spec.get("kind") != "module" or spec.get("phase") != "prefill":
            continue
        semantics = str(spec.get("semantics") or "")
        if semantics.endswith("Qwen3ForCausalLM"):
            return call_id
        if fallback is None and "ForCausalLM" in semantics:
            fallback = call_id
    return fallback


def build(context):
    if not _HAS_TRITON:
        return {}
    targets = context.get("targets") or {}
    references = context.get("references") or {}
    modules = context.get("modules") or {}
    record_route = context.get("record_route")
    if not callable(record_route):
        record_route = lambda *_: None

    call_id = _find_region_root(targets)
    if call_id is None or call_id not in references or call_id not in modules:
        return {}
    parent = references[call_id]
    module = modules.get(call_id)

    try:
        from transformers.cache_utils import DynamicCache, DynamicLayer
        from transformers.modeling_outputs import CausalLMOutputWithPast
    except Exception:
        return {}

    # ---- extract invariant references (no copies, no transforms) ----
    try:
        inner = module.model
        embed_tokens = inner.embed_tokens
        rotary = inner.rotary_emb
        final_norm = inner.norm
        lm_head = module.lm_head
        layer_modules = list(inner.layers)
        config = module.config
    except Exception:
        return {}

    layer_data = []
    for lyr in layer_modules:
        sa = lyr.self_attn
        mlp = lyr.mlp
        mods = (sa.q_proj, sa.k_proj, sa.v_proj, sa.o_proj,
                mlp.gate_proj, mlp.up_proj, mlp.down_proj)
        if any(getattr(m, "bias", None) is not None for m in mods):
            return {}
        layer_data.append(
            {
                "iln_w": lyr.input_layernorm.weight,
                "iln_eps": float(lyr.input_layernorm.variance_epsilon),
                "pln_w": lyr.post_attention_layernorm.weight,
                "pln_eps": float(lyr.post_attention_layernorm.variance_epsilon),
                "q_w": sa.q_proj.weight,
                "k_w": sa.k_proj.weight,
                "v_w": sa.v_proj.weight,
                "o_w": sa.o_proj.weight,
                "qn_w": sa.q_norm.weight,
                "qn_eps": float(sa.q_norm.variance_epsilon),
                "kn_w": sa.k_norm.weight,
                "kn_eps": float(sa.k_norm.variance_epsilon),
                "gw": mlp.gate_proj.weight,
                "uw": mlp.up_proj.weight,
                "dw": mlp.down_proj.weight,
                "scale": float(sa.scaling),
            }
        )
    if lm_head.bias is not None:
        return {}

    head_dim = int(layer_modules[0].self_attn.head_dim)
    n_q_heads = int(layer_data[0]["q_w"].shape[0]) // head_dim
    n_kv_heads = int(layer_data[0]["k_w"].shape[0]) // head_dim
    hidden = int(layer_data[0]["q_w"].shape[1])
    inter = int(layer_data[0]["gw"].shape[0])
    n_layers = len(layer_data)
    # elementwise kernels need a power-of-two head dim; guard everything else
    if (
        n_q_heads * head_dim != int(layer_data[0]["q_w"].shape[0])
        or n_kv_heads * head_dim != int(layer_data[0]["k_w"].shape[0])
        or head_dim <= 0 or head_dim & (head_dim - 1)
        or n_q_heads <= 0 or n_kv_heads <= 0 or n_q_heads % n_kv_heads
        or hidden <= 0 or inter <= 0 or n_layers <= 0
    ):
        return {}

    state_ctx = context.get("state") or {}
    try:
        new_tokens = int(state_ctx.get("new_tokens", 0) or 0)
    except Exception:
        new_tokens = 0
    if not (1 <= new_tokens <= 512):
        new_tokens = 32

    st = {
        "failed": False,
        "fail_label": _LABEL_SETUP,
        "reason": None,
        "validated": False,
        "attempts": 0,
        "silu_checked": False,
        "silu_fused": True,
    }

    # ------------------------------------------------------------------
    def _cdiv(a, b):
        return (a + b - 1) // b

    def _new_cache(keys, values):
        cache = None
        layers = []
        try:
            candidate_cache = DynamicCache(config=config)
            if len(candidate_cache.layers) == len(keys):
                cache = candidate_cache
                layers = candidate_cache.layers
        except Exception:
            cache = None
        if cache is None:
            cache = DynamicCache()
            for _ in keys:
                layers.append(DynamicLayer())
            cache.layers.extend(layers)
        for layer, key, value in zip(layers, keys, values):
            layer.keys = key
            layer.values = value
            layer.is_initialized = True
            layer.dtype = key.dtype
            layer.device = key.device
        return cache

    def _clone_cache(cache):
        try:
            layers = list(cache.layers)
            return _new_cache(
                [layer.keys.clone() for layer in layers],
                [layer.values.clone() for layer in layers],
            )
        except Exception:
            return None

    def _close(a, ref):
        if a is None or ref is None or a.shape != ref.shape:
            return False
        af = a.detach().float()
        rf = ref.detach().float()
        diff = (af - rf).abs()
        tol = _VAL_ATOL + _VAL_RTOL * rf.abs()
        return bool(torch.all(diff <= tol).item())

    def _run_decode(first_logits, cache, steps):
        token = first_logits[:, -1].argmax(dim=-1, keepdim=True)
        tokens = [token]
        logits = [first_logits[:, -1]]
        for _ in range(steps - 1):
            out = parent(
                input_ids=token,
                past_key_values=cache,
                use_cache=True,
                logits_to_keep=1,
            )
            cache = out.past_key_values
            token = out.logits[:, -1].argmax(dim=-1, keepdim=True)
            tokens.append(token)
            logits.append(out.logits[:, -1])
        return torch.cat(tokens, dim=1), torch.stack(logits, dim=1)

    # ------------------------------------------------------------------
    def _fused(input_ids):
        device = input_ids.device
        b, s = input_ids.shape
        rows = b * s

        hidden_states = embed_tokens(input_ids)  # parent module, [b,s,hidden]
        if hidden_states.dtype != torch.bfloat16 or not hidden_states.is_contiguous():
            raise RuntimeError("unexpected embedding output")
        dtype = hidden_states.dtype
        position_ids = torch.arange(s, device=device).unsqueeze(0)
        cos, sin = rotary(hidden_states, position_ids)  # parent module [b,s,hd]
        if cos.shape[-1] != head_dim:
            raise RuntimeError("rotary dim differs from attention head dim")
        if not cos.is_contiguous():
            cos = cos.contiguous()
        if not sin.is_contiguous():
            sin = sin.contiguous()

        cur = hidden_states.reshape(rows, hidden)  # layer-0 residual input
        in_a = torch.empty_like(cur)
        in_b = torch.empty_like(cur)
        parity = 0
        mid = torch.empty_like(cur)
        sq_in = torch.empty((rows, hidden), dtype=torch.float32)
        sq_mid = torch.empty((rows, hidden), dtype=torch.float32)
        na = torch.empty((rows, hidden), dtype=dtype)
        sq_q = torch.empty((rows, n_q_heads * head_dim), dtype=torch.float32)
        sq_k = torch.empty((rows, n_kv_heads * head_dim), dtype=torch.float32)
        qrot = torch.empty((b, s, n_q_heads, head_dim), dtype=dtype, device=device)
        kv = torch.empty(
            (2, n_layers, b, n_kv_heads, s, head_dim), dtype=dtype, device=device
        )
        act = torch.empty((rows, inter), dtype=dtype, device=device)

        n_hid = rows * hidden
        n_q = rows * n_q_heads * head_dim
        n_kv = rows * n_kv_heads * head_dim
        n_inter = rows * inter

        _sq_kernel[(_cdiv(n_hid, _EB),)](cur, sq_in, n_hid, BLOCK=_EB, num_warps=4,
                                         enable_fp_fusion=False)
        for i, L in enumerate(layer_data):
            var = sq_in.view(b, s, hidden).mean(-1, keepdim=True)  # aten reduction
            _norm_apply_kernel[(rows, _cdiv(hidden, _NB))](
                cur, L["iln_w"], var, na, hidden, L["iln_eps"],
                BLOCK=_NB, num_warps=4, enable_fp_fusion=False,
            )
            na3 = na.view(b, s, hidden)
            q = F.linear(na3, L["q_w"])
            k = F.linear(na3, L["k_w"])
            v = F.linear(na3, L["v_w"])

            _sq_kernel[(_cdiv(n_q, _EB),)](q, sq_q, n_q, BLOCK=_EB, num_warps=4,
                                           enable_fp_fusion=False)
            _sq_kernel[(_cdiv(n_kv, _EB),)](k, sq_k, n_kv, BLOCK=_EB, num_warps=4,
                                            enable_fp_fusion=False)
            var_q = sq_q.view(b, s, n_q_heads, head_dim).mean(-1, keepdim=True)
            var_k = sq_k.view(b, s, n_kv_heads, head_dim).mean(-1, keepdim=True)

            _norm_rotary_kernel[(rows * n_q_heads,)](
                q, L["qn_w"], var_q, cos, sin, qrot,
                s, n_q_heads, L["qn_eps"],
                HEAD_DIM=head_dim, OUT_BSHD=True, num_warps=1, enable_fp_fusion=False,
            )
            ck = kv[0, i]
            cv = kv[1, i]
            _norm_rotary_kernel[(rows * n_kv_heads,)](
                k, L["kn_w"], var_k, cos, sin, ck,
                s, n_kv_heads, L["kn_eps"],
                HEAD_DIM=head_dim, OUT_BSHD=False, num_warps=1, enable_fp_fusion=False,
            )
            cv.copy_(v.view(b, s, n_kv_heads, head_dim).permute(0, 2, 1, 3))

            q_view = qrot.permute(0, 2, 1, 3)  # parent q_embed strides
            attn = F.scaled_dot_product_attention(
                q_view, ck, cv, attn_mask=None, dropout_p=0.0,
                scale=L["scale"], is_causal=True, enable_gqa=True,
            )
            attn = attn.transpose(1, 2).contiguous()       # sdpa_attention_forward
            attn2 = attn.reshape(b, s, n_q_heads * head_dim).contiguous()  # Qwen3Attention
            o = F.linear(attn2, L["o_w"])

            _add_sq_kernel[(_cdiv(n_hid, _EB),)](
                o.view(rows, hidden), cur, mid, sq_mid, n_hid,
                BLOCK=_EB, num_warps=4, enable_fp_fusion=False,
            )
            var_m = sq_mid.view(b, s, hidden).mean(-1, keepdim=True)
            _norm_apply_kernel[(rows, _cdiv(hidden, _NB))](
                mid, L["pln_w"], var_m, na, hidden, L["pln_eps"],
                BLOCK=_NB, num_warps=4, enable_fp_fusion=False,
            )
            nb3 = na.view(b, s, hidden)
            g = F.linear(nb3, L["gw"])
            u = F.linear(nb3, L["uw"])

            if st["silu_fused"]:
                _silu_mul_kernel[(_cdiv(n_inter, _EB),)](
                    g.view(rows, inter), u.view(rows, inter), act, n_inter,
                    BLOCK=_EB, num_warps=4, enable_fp_fusion=False,
                )
            else:
                act.copy_((F.silu(g) * u).view(rows, inter))
            if not st["silu_checked"]:
                # one-time bitwise probe of the Triton silu against aten silu
                st["silu_checked"] = True
                ref_act = (F.silu(g) * u).view(rows, inter)
                if st["silu_fused"] and not torch.equal(act, ref_act):
                    st["silu_fused"] = False
                    act.copy_(ref_act)

            dout = F.linear(act.view(b, s, inter), L["dw"])
            if i == n_layers - 1:
                nxt = torch.add(mid, dout.view(rows, hidden))  # parent final add
            else:
                nxt = in_a if parity == 0 else in_b
                parity ^= 1
                _add_sq_kernel[(_cdiv(n_hid, _EB),)](
                    dout.view(rows, hidden), mid, nxt, sq_in, n_hid,
                    BLOCK=_EB, num_warps=4, enable_fp_fusion=False,
                )
            cur = nxt

        norm_out = final_norm(cur.view(b, s, hidden))  # parent module
        logits = F.linear(norm_out[:, -1:, :], lm_head.weight)
        cache = _new_cache([kv[0, i] for i in range(n_layers)],
                           [kv[1, i] for i in range(n_layers)])
        return CausalLMOutputWithPast(
            loss=None, logits=logits, past_key_values=cache,
            hidden_states=None, attentions=None,
        )

    # ------------------------------------------------------------------
    def _validate(ref_out, fused_out):
        """True only if the fused path is safe for the full frozen workload."""
        logits_ok = bool(
            fused_out.logits is not None
            and fused_out.logits.shape == ref_out.logits.shape
            and torch.equal(fused_out.logits, ref_out.logits)
        )
        ref_cache = ref_out.past_key_values
        fused_cache = fused_out.past_key_values
        cache_ok = False
        try:
            rl = list(ref_cache.layers)
            fl = list(fused_cache.layers)
            cache_ok = len(rl) == len(fl) and len(rl) > 0 and all(
                torch.equal(a.keys, b.keys) and torch.equal(a.values, b.values)
                for a, b in zip(rl, fl)
            )
        except Exception:
            cache_ok = False
        if logits_ok and cache_ok:
            return True  # tier 1: bitwise logits + bitwise KV cache
        # tier 2: near-tied is acceptable only if the full generation matches;
        # decode validation runs on cloned caches so the returned caches (which
        # the harness decode consumes) stay pristine prefill-only caches.
        if not _close(fused_out.logits, ref_out.logits):
            return False
        if not bool(torch.equal(
            fused_out.logits[:, -1].argmax(dim=-1),
            ref_out.logits[:, -1].argmax(dim=-1),
        )):
            return False
        ref_clone = _clone_cache(ref_cache)
        fused_clone = _clone_cache(fused_cache)
        if ref_clone is None or fused_clone is None:
            return False
        try:
            ref_tokens, ref_logits = _run_decode(ref_out.logits, ref_clone, new_tokens)
            fused_tokens, fused_logits = _run_decode(
                fused_out.logits, fused_clone, new_tokens
            )
        except Exception:
            return False
        return bool(
            torch.equal(fused_tokens, ref_tokens) and _close(fused_logits, ref_logits)
        )

    # ------------------------------------------------------------------
    def prefill_forward(
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        logits_to_keep=0,
        **kwargs,
    ):
        call_kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "past_key_values": past_key_values,
            "inputs_embeds": inputs_embeds,
            "labels": labels,
            "use_cache": use_cache,
            "logits_to_keep": logits_to_keep,
        }
        call_kwargs.update(kwargs)

        tensor_ok = isinstance(input_ids, torch.Tensor) and input_ids.is_cuda
        if (
            not tensor_ok
            or torch.is_grad_enabled()
            or torch.cuda.is_current_stream_capturing()
        ):
            record_route(call_id, _LABEL_CONTEXT)
            return parent(**call_kwargs)
        if not (
            input_ids.ndim == 2
            and input_ids.dtype == torch.int64
            and 0 < input_ids.shape[1] <= 8192
            and input_ids.shape[0] * input_ids.shape[1] * n_q_heads <= 8_388_608
            and attention_mask is None
            and position_ids is None
            and past_key_values is None
            and inputs_embeds is None
            and labels is None
            and use_cache is True
            and isinstance(logits_to_keep, int)
            and logits_to_keep == 1
            and not kwargs
        ):
            record_route(call_id, _LABEL_SIGNATURE)
            return parent(**call_kwargs)
        if st["failed"]:
            record_route(call_id, st["fail_label"])
            return parent(**call_kwargs)

        try:
            if not st["validated"]:
                st["validated"] = True
                ref_out = parent(
                    input_ids=input_ids, use_cache=True, logits_to_keep=1
                )
                fused_out = _fused(input_ids)
                if not _validate(ref_out, fused_out):
                    st["failed"] = True
                    st["reason"] = "first-invocation validation mismatch"
                    st["fail_label"] = _LABEL_VALIDATION
                    record_route(call_id, _LABEL_VALIDATION)
                    return ref_out
                st["attempts"] += 1
                record_route(call_id, _LABEL_FUSED)
                return fused_out

            output = _fused(input_ids)
        except Exception as exc:
            st["failed"] = True
            st["reason"] = repr(exc)[:300]
            st["fail_label"] = _LABEL_SETUP if st["attempts"] == 0 else _LABEL_RUNTIME
            record_route(call_id, st["fail_label"])
            return parent(**call_kwargs)
        st["attempts"] += 1
        record_route(call_id, _LABEL_FUSED)
        return output

    return {call_id: prefill_forward}
