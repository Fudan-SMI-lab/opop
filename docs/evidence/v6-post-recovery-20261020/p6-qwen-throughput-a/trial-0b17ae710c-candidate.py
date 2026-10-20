"""Flattened eager decode executor for the Qwen3-4B offline workload.

Repair of the failure recorded in inputs/previous-feedback.txt (trial-6002d1ef9b,
decision no_valid_parameter_candidate): four decode/root CUDA-graph attempts
(trials 7ee056ddc4, eeb66b190e, 82eb37adf6, 6002d1ef9b) all failed quality with
the byte-identical deterministic signature 10,431,056 / 38,895,616 mismatched
logits elements (26.8 percent, max abs 4.0), and the latest additionally failed
exact greedy-token identity on its first (capture) run. Root cause evidenced by
the actual implementation: torch.cuda.graph stream capture RECORDS work without
executing it, so on every capture invocation the eager lm_head read
uninitialized private-pool memory (wrong tokens), while the replay run diverged
deterministically from eager numerics through capture-time kernel/backend
selection and the pool-resident KV cat chain - an output-ownership clone cannot
cure either defect, matching the audit that the hypothesis must change.

This candidate removes stream capture entirely. The replacement of call
323bcbe006b343648b346af9 (decode-phase Qwen3ForCausalLM.forward, kwargs
input_ids [8,1] int64 / past_key_values DynamicCache / use_cache True /
logits_to_keep 1) executes the SAME eager aten op sequence as the pristine
parent (embedding; per layer input RMSNorm, q/k/v linear + view, q/k RMSNorm,
transpose, rotary mul/add/cat chains, KV cache append, SDPA with the exact
parent signature, o_proj, residual adds, post RMSNorm, gate/up/silu/down MLP;
final RMSNorm; lm_head on the last position) with the same parameter tensors,
shapes, dtypes and values, so every kernel is dispatched eagerly exactly as in
the reference run:

1. Per-run preallocated KV workspace: step 0 of each run binds the run's fresh
   eager prefill K/V by device copies into persistent per-layer
   [batch, kv_heads, base_len+32, head_dim] workspaces; each decode step writes
   its one-token k/v in place (narrow-slice copy_) and hands SDPA the prefix
   slice, which holds byte-identical values to what DynamicCache.update
   torch.cat would have produced. This deletes the per-step whole-cache cat
   rewrites (72 per step), their allocation churn and the DynamicCache Python
   bookkeeping without changing any kernel input value.
2. A RoPE cos/sin table for all decode positions is built once from the live
   inv_freq buffer with the parent's exact op sequence (fp32 outer-product
   matmul, cat, cos/sin, scaling mul, dtype cast); per-step slices are
   byte-identical to the parent's per-step tensors because every element is a
   scalar-only product/cast with no reduction-order freedom.
3. The returned past_key_values stays the incoming DynamicCache object with
   layer tensors reassigned to prefix views of the workspace, so a mid-run
   fallback to the pristine parent consumes a fully valid cache (parent cats
   onto the views). Guards fall back per-call and poison the remainder of the
   run; unexpected exceptions latch to the parent permanently.

No CUDA work happens at import or build time; the workspace/table are created
on the first decode invocation (warmup, outside timed measure runs) and reused
with a 72-copy rebind at the start of every subsequent run.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from transformers.cache_utils import DynamicCache
from transformers.modeling_outputs import CausalLMOutputWithPast

CALL_ID = "323bcbe006b343648b346af9"
_EXPECTED_KWARGS = frozenset(("input_ids", "past_key_values", "use_cache", "logits_to_keep"))
_SLOTS_MARGIN = 32
_TABLE_STEPS = 40


class _GuardFailure(Exception):
    pass


def build(context):
    modules = context.get("modules") or {}
    module = modules.get(CALL_ID)
    references = context.get("references") or {}
    parent = references.get(CALL_ID)
    if module is None or parent is None or not callable(parent):
        return {}
    inner = getattr(module, "model", None)
    lm_head = getattr(module, "lm_head", None)
    if inner is None or lm_head is None:
        return {}
    rotary = getattr(inner, "rotary_emb", None)
    embed = getattr(inner, "embed_tokens", None)
    final_norm = getattr(inner, "norm", None)
    layer_modules = getattr(inner, "layers", None)
    if rotary is None or embed is None or final_norm is None or not layer_modules:
        return {}

    record_route = context.get("record_route")
    if not callable(record_route):

        def record_route(*_args, **_kwargs):
            return None

    # Static structure validation; attribute reads only, no CUDA work here.
    if getattr(inner, "has_sliding_layers", True):
        return {}
    if getattr(rotary, "rope_type", None) != "default":
        return {}
    inv_freq = getattr(rotary, "inv_freq", None)
    if not isinstance(inv_freq, torch.Tensor) or inv_freq.dim() != 1 or inv_freq.numel() < 2:
        return {}
    try:
        rope_scaling = float(rotary.attention_scaling)
    except Exception:
        return {}
    try:
        emb_weight = embed.weight
        emb_pad = embed.padding_idx
        emb_max_norm = embed.max_norm
        emb_norm_type = embed.norm_type
        emb_sgf = embed.scale_grad_by_freq
        emb_sparse = embed.sparse
        lm_weight = lm_head.weight
        lm_bias = lm_head.bias
        layers = list(layer_modules)
    except Exception:
        return {}
    n_layers = len(layers)
    if n_layers == 0:
        return {}

    meta = []
    q_heads = kv_heads = head_dim = hidden = None
    for lyr in layers:
        attn = getattr(lyr, "self_attn", None)
        mlp = getattr(lyr, "mlp", None)
        ln1 = getattr(lyr, "input_layernorm", None)
        ln2 = getattr(lyr, "post_attention_layernorm", None)
        if attn is None or mlp is None or ln1 is None or ln2 is None:
            return {}
        if getattr(attn, "sliding_window", None) is not None:
            return {}
        try:
            q_w, q_b = attn.q_proj.weight, attn.q_proj.bias
            k_w, k_b = attn.k_proj.weight, attn.k_proj.bias
            v_w, v_b = attn.v_proj.weight, attn.v_proj.bias
            o_w, o_b = attn.o_proj.weight, attn.o_proj.bias
            qn_w, qn_eps = attn.q_norm.weight, float(attn.q_norm.variance_epsilon)
            kn_w, kn_eps = attn.k_norm.weight, float(attn.k_norm.variance_epsilon)
            g_w, g_b = mlp.gate_proj.weight, mlp.gate_proj.bias
            u_w, u_b = mlp.up_proj.weight, mlp.up_proj.bias
            d_w, d_b = mlp.down_proj.weight, mlp.down_proj.bias
            ln1_w, ln1_eps = ln1.weight, float(ln1.variance_epsilon)
            ln2_w, ln2_eps = ln2.weight, float(ln2.variance_epsilon)
            scaling = float(attn.scaling)
            cur_hd = int(attn.head_dim)
        except Exception:
            return {}
        if q_b is not None or k_b is not None or v_b is not None or o_b is not None:
            return {}  # this asset has attention_bias=false; keep the fast path bias-free
        cur_qh = q_w.shape[0] // cur_hd
        cur_kvh = k_w.shape[0] // cur_hd
        if cur_qh * cur_hd != q_w.shape[0] or cur_kvh * cur_hd != k_w.shape[0]:
            return {}
        if v_w.shape != k_w.shape or o_w.shape[1] != q_w.shape[0]:
            return {}
        if q_w.shape[1] != ln1_w.shape[0] or k_w.shape[1] != ln1_w.shape[0]:
            return {}
        if (
            g_w.shape[1] != ln1_w.shape[0]
            or g_w.shape[0] != u_w.shape[0]
            or d_w.shape[0] != ln1_w.shape[0]
            or d_w.shape[1] != g_w.shape[0]
        ):
            return {}
        if q_heads is None:
            q_heads, kv_heads, head_dim, hidden = cur_qh, cur_kvh, cur_hd, ln1_w.shape[0]
        elif (cur_qh, cur_kvh, cur_hd, ln1_w.shape[0]) != (q_heads, kv_heads, head_dim, hidden):
            return {}
        meta.append(
            (
                ln1_w, ln1_eps,
                ln2_w, ln2_eps,
                q_w, k_w, v_w, o_w,
                qn_w, qn_eps,
                kn_w, kn_eps,
                g_w, g_b, u_w, u_b, d_w, d_b,
                scaling,
            )
        )
    if q_heads % kv_heads != 0 or q_heads // kv_heads < 2:
        return {}  # parent takes the enable_gqa branch only for groups > 1
    emb_dtype = emb_weight.dtype
    if emb_dtype is None or emb_weight.dim() != 2:
        return {}

    st = {
        "broken": False,
        "run_ok": False,
        "last_cache": None,
        "meta": meta,
        "n": n_layers,
        "qh": q_heads,
        "kvh": kv_heads,
        "hd": head_dim,
        "hidden": hidden,
        "groups": q_heads // kv_heads,
        "inv_freq": inv_freq,
        "rope_scaling": rope_scaling,
        "emb_w": emb_weight,
        "emb_pad": emb_pad,
        "emb_max_norm": emb_max_norm,
        "emb_norm_type": emb_norm_type,
        "emb_sgf": emb_sgf,
        "emb_sparse": emb_sparse,
        "emb_dtype": emb_dtype,
        "lm_w": lm_weight,
        "lm_b": lm_bias,
        "fn_w": final_norm.weight,
        "fn_eps": float(final_norm.variance_epsilon),
        "module": module,
        "parent": parent,
        "record_route": record_route,
        "wk": None,
        "wv": None,
        "cos": None,
        "sin": None,
        "base_len": None,
        "slots": None,
        "batch": None,
        "dtype": None,
        "device": None,
    }

    def _build_table(base_len, device):
        # Mirrors Qwen3RotaryEmbedding.forward for positions base_len..base_len+_TABLE_STEPS-1
        # with the parent's exact op sequence; scalar-only products are bit-stable
        # regardless of the matrogram width, so per-step slices are byte-identical
        # to the parent's per-step [1,1,head_dim] tensors.
        with torch.inference_mode(False):
            inv_exp = inv_freq[None, :, None].float().expand(1, -1, 1).to(device)
            pos = torch.arange(base_len, base_len + _TABLE_STEPS, device=device, dtype=torch.float32)
            freqs = (inv_exp.float() @ pos[None, None, :].float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            st["cos"] = (emb.cos() * st["rope_scaling"]).to(st["emb_dtype"])
            st["sin"] = (emb.sin() * st["rope_scaling"]).to(st["emb_dtype"])

    def _bind(cache, input_ids, k0):
        if getattr(st["module"], "training", False):
            raise _GuardFailure("module training")
        shape = tuple(k0.shape)
        if len(shape) != 4:
            raise _GuardFailure("cache ndim")
        batch, kvh, base_len, hd = shape
        if kvh != st["kvh"] or hd != st["hd"]:
            raise _GuardFailure("cache geometry")
        if input_ids.shape[0] != batch:
            raise _GuardFailure("batch mismatch")
        device, dtype = k0.device, k0.dtype
        if dtype != st["emb_dtype"]:
            raise _GuardFailure("cache dtype")
        layers = cache.layers
        if len(layers) != st["n"]:
            raise _GuardFailure("layer count")
        for lyr in layers:
            kk, vv = lyr.keys, lyr.values
            if kk is None or vv is None:
                raise _GuardFailure("uninitialized cache layer")
            if tuple(kk.shape) != shape or kk.dtype != dtype or kk.device != device:
                raise _GuardFailure("nonuniform cache")
        slots = base_len + _SLOTS_MARGIN
        if (
            st["wk"] is None
            or st["base_len"] != base_len
            or st["slots"] != slots
            or st["batch"] != batch
            or st["device"] != device
            or st["dtype"] != dtype
        ):
            with torch.inference_mode(False):
                st["wk"] = [
                    torch.empty((batch, kvh, slots, hd), dtype=dtype, device=device)
                    for _ in range(st["n"])
                ]
                st["wv"] = [
                    torch.empty((batch, kvh, slots, hd), dtype=dtype, device=device)
                    for _ in range(st["n"])
                ]
            _build_table(base_len, device)
            st["base_len"] = base_len
            st["slots"] = slots
            st["batch"] = batch
            st["device"] = device
            st["dtype"] = dtype
        elif st["cos"] is None:
            _build_table(base_len, device)
        wk, wv = st["wk"], st["wv"]
        for lyr, wki, wvi in zip(layers, wk, wv):
            wki[:, :, :base_len, :].copy_(lyr.keys)
            wvi[:, :, :base_len, :].copy_(lyr.values)
        st["run_ok"] = True

    def _fast_step(token, step, cache):
        base_len = st["base_len"]
        pos = base_len + step
        new_len = pos + 1
        batch, qh, kvh, hd = st["batch"], st["qh"], st["kvh"], st["hd"]
        half = hd // 2
        cos = st["cos"][:, step : step + 1, :].unsqueeze(1)
        sin = st["sin"][:, step : step + 1, :].unsqueeze(1)
        h = F.embedding(
            token,
            st["emb_w"],
            padding_idx=st["emb_pad"],
            max_norm=st["emb_max_norm"],
            norm_type=st["emb_norm_type"],
            scale_grad_by_freq=st["emb_sgf"],
            sparse=st["emb_sparse"],
        )
        dt = h.dtype
        f32 = torch.float32
        linear = F.linear
        sdpa = F.scaled_dot_product_attention
        cat = torch.cat
        rsqrt = torch.rsqrt
        silu = F.silu
        wk, wv = st["wk"], st["wv"]
        for m, wki, wvi in zip(st["meta"], wk, wv):
            (
                ln1_w, ln1_eps,
                ln2_w, ln2_eps,
                q_w, k_w, v_w, o_w,
                qn_w, qn_eps,
                kn_w, kn_eps,
                g_w, g_b, u_w, u_b, d_w, d_b,
                scaling,
            ) = m
            residual = h
            x32 = h.to(f32)
            x = ln1_w * (x32 * rsqrt(x32.pow(2).mean(-1, keepdim=True) + ln1_eps)).to(dt)
            q = linear(x, q_w).view(batch, 1, qh, hd)
            t32 = q.to(f32)
            q = (qn_w * (t32 * rsqrt(t32.pow(2).mean(-1, keepdim=True) + qn_eps)).to(dt)).transpose(1, 2)
            k = linear(x, k_w).view(batch, 1, kvh, hd)
            t32 = k.to(f32)
            k = (kn_w * (t32 * rsqrt(t32.pow(2).mean(-1, keepdim=True) + kn_eps)).to(dt)).transpose(1, 2)
            v = linear(x, v_w).view(batch, 1, kvh, hd).transpose(1, 2)
            # apply_rotary_pos_emb(q, k, cos, sin) with cos/sin already unsqueezed once
            q1 = q[..., :half]
            q2 = q[..., half:]
            q = (q * cos) + (cat((-q2, q1), dim=-1) * sin)
            k1 = k[..., :half]
            k2 = k[..., half:]
            k = (k * cos) + (cat((-k2, k1), dim=-1) * sin)
            # DynamicCache.update equivalent: in-place slot write, prefix view for SDPA
            wki[:, :, pos:new_len, :].copy_(k)
            wvi[:, :, pos:new_len, :].copy_(v)
            kk = wki[:, :, :new_len, :]
            vv = wvi[:, :, :new_len, :]
            o = sdpa(
                q,
                kk,
                vv,
                attn_mask=None,
                dropout_p=0.0,
                scale=scaling,
                is_causal=False,
                enable_gqa=True,
            )
            o = o.transpose(1, 2).contiguous().reshape(batch, 1, -1).contiguous()
            o = linear(o, o_w)
            h = residual + o
            residual = h
            x32 = h.to(f32)
            x = ln2_w * (x32 * rsqrt(x32.pow(2).mean(-1, keepdim=True) + ln2_eps)).to(dt)
            h = residual + linear(silu(linear(x, g_w, g_b)) * linear(x, u_w, u_b), d_w, d_b)
        x32 = h.to(f32)
        h = st["fn_w"] * (x32 * rsqrt(x32.pow(2).mean(-1, keepdim=True) + st["fn_eps"])).to(dt)
        logits = linear(h[:, -1:, :], st["lm_w"], st["lm_b"])
        # Cache layers are re-pointed only after the whole step succeeded, so any
        # mid-step exception leaves the incoming cache fully parent-consumable.
        for lyr, wki, wvi in zip(cache.layers, wk, wv):
            lyr.keys = wki[:, :, :new_len, :]
            lyr.values = wvi[:, :, :new_len, :]
        return logits

    def replacement(*args, **kwargs):
        try:
            if st["broken"]:
                raise _GuardFailure("broken")
            if args or frozenset(kwargs) != _EXPECTED_KWARGS:
                raise _GuardFailure("signature")
            input_ids = kwargs["input_ids"]
            cache = kwargs["past_key_values"]
            if kwargs["use_cache"] is not True or kwargs["logits_to_keep"] != 1:
                raise _GuardFailure("flags")
            if (
                not isinstance(input_ids, torch.Tensor)
                or input_ids.dim() != 2
                or input_ids.shape[1] != 1
                or input_ids.dtype != torch.int64
            ):
                raise _GuardFailure("input_ids")
            if type(cache) is not DynamicCache:
                raise _GuardFailure("cache type")
            layers = cache.layers
            if len(layers) != st["n"]:
                raise _GuardFailure("layer count")
            k0 = layers[0].keys
            if k0 is None or k0.dim() != 4:
                raise _GuardFailure("cache geometry")
            length = int(k0.shape[-2])
            if cache is st["last_cache"]:
                if not st["run_ok"]:
                    raise _GuardFailure("poisoned run")
                step = length - st["base_len"]
                if step < 1 or step >= _TABLE_STEPS or length + 1 > st["slots"]:
                    raise _GuardFailure("step range")
                if (
                    k0.shape[0] != st["batch"]
                    or k0.shape[1] != st["kvh"]
                    or k0.shape[3] != st["hd"]
                    or k0.dtype != st["dtype"]
                    or k0.device != st["device"]
                ):
                    raise _GuardFailure("geometry drift")
            else:
                # Any cache object we did not return last call is a fresh run (or a
                # run recovered from fallbacks): the adapter only ever passes the
                # prefill cache or the object we/parent returned, and the identity
                # branch above owns the latter, so binding from it is authoritative.
                _bind(cache, input_ids, k0)
                step = 0
            logits = _fast_step(input_ids, step, cache)
            st["last_cache"] = cache
            record_route(CALL_ID, "eager_flat_decode_ws")
            return CausalLMOutputWithPast(loss=None, logits=logits, past_key_values=cache)
        except _GuardFailure:
            st["run_ok"] = False
            record_route(CALL_ID, "parent_fallback")
            return parent(*args, **kwargs)
        except Exception:
            st["broken"] = True
            st["run_ok"] = False
            record_route(CALL_ID, "parent_fallback")
            return parent(*args, **kwargs)

    return {CALL_ID: replacement}
