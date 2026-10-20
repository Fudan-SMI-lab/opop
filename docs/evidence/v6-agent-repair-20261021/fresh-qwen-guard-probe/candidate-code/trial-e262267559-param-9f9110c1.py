"""TTFT repair candidate: fused eager prefill for Qwen3-4B BF16 SDPA.

Previous trial (trial-9221b5ebdd) replayed a CUDA Graph of the parent prefill
and passed quality/feasibility, but its paired gains (~0.13 ms) stayed below
the MAD noise bounds: on this Ampere GPU the prefill window is GPU-bound
(~21.6 ms kernel time out of ~24.6 ms wall, ~16.5 ms of which is cuBLAS
bf16 GEMMs), so removing launch gaps alone cannot move ttft_ms. This attempt
reduces actual GPU work instead, replacing the region-root prefill forward
(call id of the observed `qwen3_4b` module) with an eager composite that:

- keeps every GEMM, the SDPA call, the embedding lookup and the rotary
  cos/sin computation bit-identical to the parent (same F.linear/F.sdpa calls
  on the same weight tensors, shapes, dtypes and contiguity);
- replaces the ~4.4 ms of unfused elementwise traffic (fp32 RMSNorm chains,
  residual adds, rotate_half rotary with q/k RMSNorm, SiLU*up, KV cat/copy)
  with single-pass Triton kernels that reproduce the parent's per-op rounding
  chain (fp32 opmath, one bf16 rounding per torch op);
- writes K/V directly into preallocated contiguous [b, heads, seq, head_dim]
  cache buffers and returns a parent-compatible DynamicCache, so the 31
  decode steps keep running the unmodified parent;
- on the first invocation validates itself against the parent on the current
  inputs (prefill logits plus two full 31-step decode chains run on cloned
  caches) and falls back permanently to the parent unless every greedy token
  matches exactly and logits stay within half the quality tolerance.

Fallback categories: unsupported signature, foreign context (grad enabled or
foreign graph capture), setup failure (first-attempt exception, including
Triton JIT compile), validation mismatch and runtime failure.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
    _HAS_TRITON = True
except Exception:  # pragma: no cover - worker has triton 3.7.1
    _HAS_TRITON = False

if _HAS_TRITON:

    @triton.jit
    def _rmsnorm_kernel(x_ptr, w_ptr, out_ptr, n_cols, eps, BLOCK: tl.constexpr):
        row = tl.program_id(0)
        cols = tl.arange(0, BLOCK)
        mask = cols < n_cols
        x = tl.load(x_ptr + row * n_cols + cols, mask=mask, other=0.0).to(tl.float32)
        var = tl.sum(x * x, axis=0) / n_cols
        inv = tl.rsqrt(var + eps)
        # parent: weight * (x * rsqrt(var+eps)).to(bf16), computed in fp32 opmath
        xn = (x * inv).to(tl.bfloat16).to(tl.float32)
        w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        y = (w * xn).to(tl.bfloat16)
        tl.store(out_ptr + row * n_cols + cols, y, mask=mask)

    @triton.jit
    def _add_rmsnorm_kernel(x_ptr, r_ptr, w_ptr, out_ptr, res_ptr, n_cols, eps, BLOCK: tl.constexpr):
        row = tl.program_id(0)
        cols = tl.arange(0, BLOCK)
        mask = cols < n_cols
        x = tl.load(x_ptr + row * n_cols + cols, mask=mask, other=0.0).to(tl.float32)
        r = tl.load(r_ptr + row * n_cols + cols, mask=mask, other=0.0).to(tl.float32)
        s = (x + r).to(tl.bfloat16)  # parent: residual + hidden (bf16 rounded)
        tl.store(res_ptr + row * n_cols + cols, s, mask=mask)
        sf = s.to(tl.float32)
        var = tl.sum(sf * sf, axis=0) / n_cols
        inv = tl.rsqrt(var + eps)
        xn = (sf * inv).to(tl.bfloat16).to(tl.float32)
        w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        y = (w * xn).to(tl.bfloat16)
        tl.store(out_ptr + row * n_cols + cols, y, mask=mask)

    @triton.jit
    def _norm_rotary_kernel(
        src_ptr, w_ptr, cos_ptr, sin_ptr, out_ptr, n_heads, seq, eps,
        HEAD_DIM: tl.constexpr,
    ):
        pid = tl.program_id(0)
        h = pid % n_heads
        t = pid // n_heads  # global token row (batch * seq + pos)
        p = t % seq
        d = tl.arange(0, HEAD_DIM)
        half = HEAD_DIM // 2
        base = src_ptr + t * (n_heads * HEAD_DIM) + h * HEAD_DIM
        x = tl.load(base + d).to(tl.float32)
        var = tl.sum(x * x, axis=0) / HEAD_DIM
        inv = tl.rsqrt(var + eps)
        w = tl.load(w_ptr + d).to(tl.float32)
        q = (w * (x * inv).to(tl.bfloat16).to(tl.float32)).to(tl.bfloat16).to(tl.float32)
        # rotate_half(q)[d] = -q[d+half] if d < half else q[d-half]; reuse the
        # same normalization scalar so the bf16 rounding chain matches torch.
        off2 = tl.where(d < half, d + half, d - half)
        x2 = tl.load(base + off2).to(tl.float32)
        w2 = tl.load(w_ptr + off2).to(tl.float32)
        rot = (w2 * (x2 * inv).to(tl.bfloat16).to(tl.float32)).to(tl.bfloat16).to(tl.float32)
        sign = tl.where(d < half, -1.0, 1.0)
        rotf = rot * sign
        cos = tl.load(cos_ptr + p * HEAD_DIM + d).to(tl.float32)
        sin = tl.load(sin_ptr + p * HEAD_DIM + d).to(tl.float32)
        a = (q * cos).to(tl.bfloat16).to(tl.float32)
        b = (rotf * sin).to(tl.bfloat16).to(tl.float32)
        y = (a + b).to(tl.bfloat16)
        bi = t // seq
        out_off = bi * (n_heads * seq * HEAD_DIM) + h * (seq * HEAD_DIM) + p * HEAD_DIM
        tl.store(out_ptr + out_off + d, y)

    @triton.jit
    def _silu_and_mul_kernel(g_ptr, u_ptr, o_ptr, numel, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        off = pid * BLOCK + tl.arange(0, BLOCK)
        mask = off < numel
        g = tl.load(g_ptr + off, mask=mask, other=0.0).to(tl.float32)
        u = tl.load(u_ptr + off, mask=mask, other=0.0).to(tl.float32)
        s = (g * tl.sigmoid(g)).to(tl.bfloat16).to(tl.float32)
        y = (s * u).to(tl.bfloat16)
        tl.store(o_ptr + off, y, mask=mask)


_LABEL_FUSED = "fused_triton_prefill"
_LABEL_SIGNATURE = "parent_fallback_signature"
_LABEL_CONTEXT = "parent_fallback_context"
_LABEL_SETUP = "parent_fallback_setup_failure"
_LABEL_RUNTIME = "parent_fallback_runtime_failure"
_LABEL_VALIDATION = "parent_fallback_validation"

_VAL_ATOL = 0.01  # half of the frozen quality tolerance (0.02)
_VAL_RTOL = 0.01


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

    # ---- extract invariant weight references (no copies, no transforms) ----
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
        if any(
            getattr(m, "bias", None) is not None
            for m in (sa.q_proj, sa.k_proj, sa.v_proj, sa.o_proj,
                      mlp.gate_proj, mlp.up_proj, mlp.down_proj)
        ):
            return {}
        if lm_head.bias is not None:
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

    head_dim = int(layer_modules[0].self_attn.head_dim)
    n_q_heads = int(layer_data[0]["q_w"].shape[0]) // head_dim
    n_kv_heads = int(layer_data[0]["k_w"].shape[0]) // head_dim
    hidden = int(layer_data[0]["q_w"].shape[1])
    inter = int(layer_data[0]["gw"].shape[0])
    n_layers = len(layer_data)
    if n_q_heads * head_dim != int(layer_data[0]["q_w"].shape[0]):
        return {}
    n_block = triton.next_power_of_2(hidden)
    state_ctx = context.get("state") or {}
    new_tokens = int(state_ctx.get("new_tokens", 0) or 0)

    st = {"failed": False, "reason": None, "fail_label": _LABEL_SETUP, "validated": False, "attempts": 0}

    # ------------------------------------------------------------------
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
        cloned = _new_cache(
            [layer.keys.clone() for layer in cache.layers],
            [layer.values.clone() for layer in cache.layers],
        )
        return cloned

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
        hidden_states = embed_tokens(input_ids)  # [b, s, hidden] bf16
        if hidden_states.dtype != torch.bfloat16 or not hidden_states.is_contiguous():
            raise RuntimeError("unexpected embedding output")
        dtype = hidden_states.dtype
        position_ids = torch.arange(s, device=device).unsqueeze(0)
        cos, sin = rotary(hidden_states, position_ids)  # parent module, [1, s, hd]
        if cos.shape[-1] != head_dim:
            raise RuntimeError("rotary dim differs from attention head dim")
        cos2 = cos.reshape(s, head_dim)
        sin2 = sin.reshape(s, head_dim)

        h2 = hidden_states.reshape(rows, hidden)
        ra = torch.empty_like(h2)  # mid-layer residual buffer
        rb = torch.empty_like(h2)  # layer output residual buffer
        na = torch.empty_like(h2)  # input_layernorm output buffer
        nb = torch.empty_like(h2)  # post_attention_layernorm output buffer
        qrot = torch.empty((b, n_q_heads, s, head_dim), dtype=dtype, device=device)
        cache_block = torch.empty(
            (2, n_layers, b, n_kv_heads, s, head_dim), dtype=dtype, device=device
        )
        act = torch.empty((rows, inter), dtype=dtype, device=device)

        cur = h2  # running residual, layer input
        for i, L in enumerate(layer_data):
            _rmsnorm_kernel[(rows,)](cur, L["iln_w"], na, hidden, L["iln_eps"], BLOCK=n_block)
            na3 = na.view(b, s, hidden)
            q = F.linear(na3, L["q_w"])
            k = F.linear(na3, L["k_w"])
            v = F.linear(na3, L["v_w"])
            _norm_rotary_kernel[(rows * n_q_heads,)](
                q, L["qn_w"], cos2, sin2, qrot, n_q_heads, s, L["qn_eps"],
                HEAD_DIM=head_dim,
            )
            ck = cache_block[0, i]
            cv = cache_block[1, i]
            _norm_rotary_kernel[(rows * n_kv_heads,)](
                k, L["kn_w"], cos2, sin2, ck, n_kv_heads, s, L["kn_eps"],
                HEAD_DIM=head_dim,
            )
            cv.copy_(v.view(b, s, n_kv_heads, head_dim).permute(0, 2, 1, 3))
            attn = F.scaled_dot_product_attention(
                qrot, ck, cv, attn_mask=None, dropout_p=0.0,
                scale=L["scale"], is_causal=True, enable_gqa=True,
            )
            attn2 = attn.transpose(1, 2).contiguous().view(b, s, n_q_heads * head_dim)
            o = F.linear(attn2, L["o_w"])
            _add_rmsnorm_kernel[(rows,)](
                o.view(rows, hidden), cur, L["pln_w"], nb, ra, hidden, L["pln_eps"],
                BLOCK=n_block,
            )
            g = F.linear(nb.view(b, s, hidden), L["gw"])
            u = F.linear(nb.view(b, s, hidden), L["uw"])
            numel = rows * inter
            _silu_and_mul_kernel[(triton.cdiv(numel, 1024),)](g, u, act, numel, BLOCK=1024)
            d = F.linear(act.view(b, s, inter), L["dw"])
            torch.add(ra, d.view(rows, hidden), out=rb)
            cur = rb

        last = cur[rows - 1: rows]
        out_last = torch.empty((1, hidden), dtype=dtype, device=device)
        _rmsnorm_kernel[(1,)](last, final_norm.weight, out_last, hidden,
                              float(final_norm.variance_epsilon), BLOCK=n_block)
        logits = F.linear(out_last.view(1, 1, hidden), lm_head.weight)
        cache = _new_cache(
            [cache_block[0, i] for i in range(n_layers)],
            [cache_block[1, i] for i in range(n_layers)],
        )
        return CausalLMOutputWithPast(
            loss=None, logits=logits, past_key_values=cache,
            hidden_states=None, attentions=None,
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
            and 0 < input_ids.shape[1] <= 32768
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
                ok = _close(fused_out.logits, ref_out.logits) and bool(
                    torch.equal(
                        fused_out.logits[:, -1].argmax(dim=-1),
                        ref_out.logits[:, -1].argmax(dim=-1),
                    )
                )
                if ok and new_tokens > 1:
                    ref_cache = _clone_cache(ref_out.past_key_values)
                    fused_cache = _clone_cache(fused_out.past_key_values)
                    if ref_cache is None or fused_cache is None:
                        ok = False
                    else:
                        ref_tokens, ref_logits = _run_decode(
                            ref_out.logits, ref_cache, new_tokens
                        )
                        fused_tokens, fused_logits = _run_decode(
                            fused_out.logits, fused_cache, new_tokens
                        )
                        ok = bool(torch.equal(fused_tokens, ref_tokens)) and _close(
                            fused_logits, ref_logits
                        )
                if not ok:
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
