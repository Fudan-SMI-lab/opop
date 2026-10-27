"""Joint candidate for the selected Qwen3Model boundary (scope: joint, groups
7f49ae8c64c12e322632 prefill + 92e2f6fb9310a942405a decode).

Prefill call 2441f10600ba127de5cb753c (input_ids [1,256], past_key_values=None,
use_cache=True): the 36-layer eager stack is re-expressed as a fused composite
(fused RMSNorm via aten rms_norm, fused q/k-norm + RoPE, fused SiLU*up, direct
SDPA flash calls with the parent's exact is_causal/enable_gqa configuration and
position-only precomputed rotary cos/sin) and captured once into a CUDA graph at
installation time.  Every invocation copies the current input_ids into the
static buffer and replays the graph, so the workload executes on the current
inputs each time.  A fallback ladder is verified at installation time against
the parent reference on random inputs: composite-graph -> eager composite ->
parent-graph -> parent-eager; any call signature that does not match the
observed prefill specification goes to the parent reference unchanged.

Decode call e7df98b905ed8aa688d5227a (input_ids [1,1], DynamicCache): parent
passthrough, preserving cache growth, continuous execution and reset semantics.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from transformers.cache_utils import DynamicCache
from transformers.modeling_outputs import BaseModelOutputWithPast

PREFILL_CALL_ID = "2441f10600ba127de5cb753c"
DECODE_CALL_ID = "e7df98b905ed8aa688d5227a"

_HAS_TRITON = False
try:  # optional fused kernels; a pure-torch reference path is used when absent
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except Exception:  # pragma: no cover
    triton = None
    tl = None


if _HAS_TRITON:

    @triton.jit
    def _qk_norm_rope_kernel(
        x_ptr,  # [B, S, H, D] contiguous
        w_ptr,  # [D] norm weight
        cos_ptr,  # [S, HALF] rotary cos (halves are identical by construction)
        sin_ptr,  # [S, HALF]
        out_ptr,  # [B, H, S, D] contiguous
        S,
        H,
        eps,
        HALF: tl.constexpr,
        D: tl.constexpr,
    ):
        pid = tl.program_id(0)
        sh = S * H
        b = pid // sh
        rem = pid % sh
        s = rem // H
        h = rem % H
        d = tl.arange(0, HALF)
        base_in = ((b * S + s) * H + h) * D
        x1 = tl.load(x_ptr + base_in + d).to(tl.float32)
        x2 = tl.load(x_ptr + base_in + HALF + d).to(tl.float32)
        var = (tl.sum(x1 * x1, axis=0) + tl.sum(x2 * x2, axis=0)) / D
        rstd = 1.0 / tl.sqrt(var + eps)
        w1 = tl.load(w_ptr + d).to(tl.float32)
        w2 = tl.load(w_ptr + HALF + d).to(tl.float32)
        y1 = x1 * rstd * w1
        y2 = x2 * rstd * w2
        cos = tl.load(cos_ptr + s * HALF + d).to(tl.float32)
        sin = tl.load(sin_ptr + s * HALF + d).to(tl.float32)
        o1 = y1 * cos - y2 * sin
        o2 = y2 * cos + y1 * sin
        base_out = ((b * H + h) * S + s) * D
        tl.store(out_ptr + base_out + d, o1.to(out_ptr.dtype.element_ty))
        tl.store(out_ptr + base_out + HALF + d, o2.to(out_ptr.dtype.element_ty))

    @triton.jit
    def _silu_mul_kernel(g_ptr, u_ptr, o_ptr, n, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        off = pid * BLOCK + tl.arange(0, BLOCK)
        m = off < n
        g = tl.load(g_ptr + off, mask=m, other=0).to(tl.float32)
        u = tl.load(u_ptr + off, mask=m, other=0).to(tl.float32)
        o = (g / (1.0 + tl.exp(-g))) * u
        tl.store(o_ptr + off, o.to(o_ptr.dtype.element_ty), mask=m)


# ---------------------------------------------------------------------------
# Pure-torch reference implementations (also used when Triton is unavailable).
# Both compute the parent math with fp32 accumulation and a single bf16
# rounding; they are deterministic and within bf16 rounding of the parent.
# ---------------------------------------------------------------------------


def _qk_norm_rope_ref(x, weight, cos_h, sin_h, eps):
    """x: [B, S, H, D] -> rotated [B, H, S, D] with per-head RMSNorm applied."""
    xf = x.float()
    var = xf.pow(2).mean(-1, keepdim=True)
    xn = (xf * torch.rsqrt(var + eps)) * weight.float()
    xn = xn.to(x.dtype).transpose(1, 2)
    half = x.shape[-1] // 2
    x1 = xn[..., :half]
    x2 = xn[..., half:]
    rot = torch.cat((-x2, x1), dim=-1)
    cos = torch.cat((cos_h, cos_h), dim=-1).to(x.dtype).view(1, 1, *cos_h.shape[:-1], -1)
    sin = torch.cat((sin_h, sin_h), dim=-1).to(x.dtype).view(1, 1, *sin_h.shape[:-1], -1)
    return (xn * cos) + (rot * sin)


def _silu_mul_ref(gate, up):
    return F.silu(gate) * up


def _make_fused_ops(device, head_dim, dtype):
    """Return (qk_norm_rope, silu_mul) callables, Triton-backed when possible."""
    if not _HAS_TRITON:
        return _qk_norm_rope_ref, _silu_mul_ref
    half = head_dim // 2

    def qk_norm_rope_triton(x, weight, cos_h, sin_h, eps):
        B, S, H, D = x.shape
        out = torch.empty((B, H, S, D), dtype=x.dtype, device=x.device)
        grid = (B * H * S,)
        _qk_norm_rope_kernel[grid](
            x, weight, cos_h, sin_h, out, S, H, eps, HALF=half, D=D, num_warps=2
        )
        return out

    def silu_mul_triton(gate, up):
        out = torch.empty_like(gate)
        n = gate.numel()
        grid = (triton.cdiv(n, 4096),)
        _silu_mul_kernel[grid](gate, up, out, n, BLOCK=4096, num_warps=8)
        return out

    try:  # compile-check both kernels once on tiny dummies at installation time
        probe = torch.zeros((1, 2, 3, head_dim), dtype=dtype, device=device)
        w = torch.ones(head_dim, dtype=dtype, device=device)
        c = torch.zeros((2, half), dtype=dtype, device=device)
        qk_norm_rope_triton(probe, w, c, c, 1e-6)
        a = torch.zeros((4, 4), dtype=dtype, device=device)
        silu_mul_triton(a, a)
        torch.cuda.synchronize(device)
        return qk_norm_rope_triton, silu_mul_triton
    except Exception:
        return _qk_norm_rope_ref, _silu_mul_ref


def _make_composite(module, cos_h, sin_h, qk_norm_rope, silu_mul):
    """Build the fused prefill stack over the parent's own parameters."""
    cfg = module.config
    num_heads = cfg.num_attention_heads
    num_kv = cfg.num_key_value_heads
    head_dim = getattr(cfg, "head_dim", cfg.hidden_size // num_heads)
    eps = cfg.rms_norm_eps
    layers = list(module.layers[: cfg.num_hidden_layers])
    norm_shape = (cfg.hidden_size,)

    def run_prefill(input_ids):
        B, S = input_ids.shape
        hidden = module.embed_tokens(input_ids)
        ks = []
        vs = []
        for layer in layers:
            attn = layer.self_attn
            residual = hidden
            h = F.rms_norm(hidden, norm_shape, layer.input_layernorm.weight, eps)
            q = qk_norm_rope(
                F.linear(h, attn.q_proj.weight).view(B, S, num_heads, head_dim),
                attn.q_norm.weight,
                cos_h,
                sin_h,
                eps,
            )
            k = qk_norm_rope(
                F.linear(h, attn.k_proj.weight).view(B, S, num_kv, head_dim),
                attn.k_norm.weight,
                cos_h,
                sin_h,
                eps,
            )
            v = F.linear(h, attn.v_proj.weight).view(B, S, num_kv, head_dim).transpose(1, 2)
            a = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=None,
                dropout_p=0.0,
                scale=attn.scaling,
                is_causal=True,
                enable_gqa=True,
            )
            a = a.transpose(1, 2).reshape(B, S, num_heads * head_dim)
            hidden = residual + F.linear(a, attn.o_proj.weight)

            residual = hidden
            h2 = F.rms_norm(hidden, norm_shape, layer.post_attention_layernorm.weight, eps)
            m = silu_mul(
                F.linear(h2, layer.mlp.gate_proj.weight),
                F.linear(h2, layer.mlp.up_proj.weight),
            )
            hidden = residual + F.linear(m, layer.mlp.down_proj.weight)
            ks.append(k)
            vs.append(v)
        hidden = F.rms_norm(hidden, norm_shape, module.norm.weight, eps)
        return hidden, ks, vs

    return run_prefill


def _wrap_output(module, hidden, ks, vs):
    """Return the parent's output structure with a fresh DynamicCache view."""
    cache = DynamicCache(config=module.config)
    layers = cache.layers
    for i, layer in enumerate(layers):
        k = ks[i]
        v = vs[i]
        layer.dtype = k.dtype
        layer.device = k.device
        layer.keys = k
        layer.values = v
        layer.is_initialized = True
    return BaseModelOutputWithPast(last_hidden_state=hidden, past_key_values=cache)


def _capture_graph(fn, static_ids, device):
    """Side-stream warmup + single CUDA graph capture of fn(static_ids)."""
    stream = torch.cuda.Stream(device=device)
    stream.wait_stream(torch.cuda.current_stream(device))
    with torch.cuda.stream(stream):
        for _ in range(2):
            fn(static_ids)
    torch.cuda.current_stream(device).wait_stream(stream)
    torch.cuda.synchronize(device)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = fn(static_ids)
    torch.cuda.synchronize(device)
    return graph, out


def build(context):
    record_route = context.get("record_route", lambda *_: None)
    modules = context.get("modules") or {}
    references = context.get("references") or {}
    models = context.get("models") or {}
    state = context.get("state") or {}

    root = models.get("model")
    module = modules.get(PREFILL_CALL_ID)
    if module is None and root is not None:
        module = getattr(root, "model", None)
    prefill_parent = references.get(PREFILL_CALL_ID)
    decode_parent = references.get(DECODE_CALL_ID)

    replacements = {}

    if decode_parent is not None:

        def decode_replacement(*args, **kwargs):
            record_route(DECODE_CALL_ID, "parent-passthrough")
            return decode_parent(*args, **kwargs)

        replacements[DECODE_CALL_ID] = decode_replacement

    if module is None or prefill_parent is None or not torch.cuda.is_available():
        return replacements

    device = next(module.parameters()).device
    cfg = module.config
    head_dim = getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)
    half = head_dim // 2

    batch = None
    seq = None
    target = (context.get("targets") or {}).get(PREFILL_CALL_ID)
    try:
        sig = (target.get("signatures") or [{}])[0]
        shape = (sig.get("kwargs") or {}).get("input_ids", {}).get("shape")
        if shape and len(shape) == 2:
            batch, seq = int(shape[0]), int(shape[1])
    except Exception:
        batch, seq = None, None
    if batch is None:
        workload = state.get("workload") or {}
        batch = int(workload.get("batch", 1))
        seq = int(workload.get("prompt_tokens", 0))

    def _parent_call(ids):
        return prefill_parent(
            input_ids=ids,
            attention_mask=None,
            position_ids=None,
            past_key_values=None,
            inputs_embeds=None,
            use_cache=True,
        )

    mode = "parent-eager"
    graph_runner = None
    composite = None

    try:
        with torch.no_grad(), torch.cuda.device(device):
            # Position-only rotary tables: the parent recomputes these each
            # prefill for positions arange(S); compute them once with the
            # parent's own rotary module (identical deterministic values).
            pos = torch.arange(seq, device=device).unsqueeze(0)
            dummy = torch.zeros(
                (batch, seq, cfg.hidden_size),
                dtype=next(module.parameters()).dtype,
                device=device,
            )
            cos, sin = module.rotary_emb(dummy, pos)
            cos_h = cos[0, :, :half].contiguous()
            sin_h = sin[0, :, :half].contiguous()

            qk_rope, silu_mul = _make_fused_ops(device, head_dim, cos_h.dtype)
            composite = _make_composite(module, cos_h, sin_h, qk_rope, silu_mul)

            verify_ids = [
                torch.randint(0, cfg.vocab_size, (batch, seq), device=device)
                for _ in range(2)
            ]

            def _check(fn):
                # Structural guard only: a wrong norm/rope/position/layout
                # diverges by O(1) or more through 36 layers; correct math with
                # fewer bf16 roundings stays within accumulated bf16 noise.
                for ids in verify_ids:
                    hidden, ks, vs = fn(ids)
                    ref = _parent_call(ids)
                    if not torch.allclose(
                        hidden.float(), ref.last_hidden_state.float(), atol=0.15, rtol=0.15
                    ):
                        return False
                    ref_layers = ref.past_key_values.layers
                    for i, ref_layer in enumerate(ref_layers):
                        if not torch.allclose(
                            ks[i].float(), ref_layer.keys.float(), atol=0.15, rtol=0.15
                        ):
                            return False
                        if not torch.allclose(
                            vs[i].float(), ref_layer.values.float(), atol=0.15, rtol=0.15
                        ):
                            return False
                return True

            composite_ok = _check(composite)

            static_ids = torch.randint(
                0, cfg.vocab_size, (batch, seq), device=device
            )

            def _replay_fn(graph, static, hidden, ks, vs):
                def fn(ids):
                    static.copy_(ids)
                    graph.replay()
                    return hidden, ks, vs

                return fn

            if composite_ok:
                try:
                    graph, out = _capture_graph(composite, static_ids, device)
                    if _check(_replay_fn(graph, static_ids, *out)):
                        mode = "graph-composite"
                        graph_runner = (graph, static_ids, *out)
                    else:
                        record_route(PREFILL_CALL_ID, "setup-failure:graph-verify")
                except Exception:
                    record_route(PREFILL_CALL_ID, "setup-failure:graph-capture-composite")
            else:
                record_route(PREFILL_CALL_ID, "setup-failure:composite-verify")
                try:

                    def parent_fn(ids):
                        out = _parent_call(ids)
                        layers = out.past_key_values.layers
                        return (
                            out.last_hidden_state,
                            [l.keys for l in layers],
                            [l.values for l in layers],
                        )

                    graph, out = _capture_graph(parent_fn, static_ids, device)
                    if _check(_replay_fn(graph, static_ids, *out)):
                        mode = "graph-parent"
                        graph_runner = (graph, static_ids, *out)
                    else:
                        record_route(PREFILL_CALL_ID, "setup-failure:graph-verify-parent")
                except Exception:
                    record_route(PREFILL_CALL_ID, "setup-failure:graph-capture-parent")

            if mode not in ("graph-composite", "graph-parent") and composite_ok:
                mode = "eager-composite"

            try:
                torch.cuda.synchronize(device)
                torch.cuda.empty_cache()
            except Exception:
                pass
    except Exception:
        record_route(PREFILL_CALL_ID, "setup-failure:setup")
        mode = "parent-eager"
        graph_runner = None
        composite = None

    expected_keys = {
        "input_ids",
        "attention_mask",
        "position_ids",
        "past_key_values",
        "inputs_embeds",
        "use_cache",
    }

    def _fast_guard(kwargs):
        if set(kwargs.keys()) != expected_keys:
            return False
        ids = kwargs["input_ids"]
        if not isinstance(ids, torch.Tensor) or ids.device.type != "cuda":
            return False
        if ids.dtype != torch.int64 or tuple(ids.shape) != (batch, seq):
            return False
        if kwargs["attention_mask"] is not None or kwargs["position_ids"] is not None:
            return False
        if kwargs["past_key_values"] is not None or kwargs["inputs_embeds"] is not None:
            return False
        return kwargs["use_cache"] is True

    if graph_runner is not None:
        graph, static_in, s_hidden, s_ks, s_vs = graph_runner

        def graph_path(input_ids):
            static_in.copy_(input_ids, non_blocking=True)
            graph.replay()
            return _wrap_output(module, s_hidden, s_ks, s_vs)

    def prefill_replacement(*args, **kwargs):
        nonlocal mode
        if args or not _fast_guard(kwargs):
            record_route(PREFILL_CALL_ID, "parent-fallback-signature")
            return prefill_parent(*args, **kwargs)
        try:
            if mode == "graph-composite":
                record_route(PREFILL_CALL_ID, "graph-replay:composite")
                return graph_path(kwargs["input_ids"])
            if mode == "graph-parent":
                record_route(PREFILL_CALL_ID, "graph-replay:parent")
                return graph_path(kwargs["input_ids"])
            if mode == "eager-composite":
                record_route(PREFILL_CALL_ID, "eager-composite")
                hidden, ks, vs = composite(kwargs["input_ids"])
                return _wrap_output(module, hidden, ks, vs)
            record_route(PREFILL_CALL_ID, "parent-eager")
            return prefill_parent(**kwargs)
        except Exception:
            mode = "parent-eager"
            record_route(PREFILL_CALL_ID, "runtime-failure:parent-fallback")
            return prefill_parent(**kwargs)

    replacements[PREFILL_CALL_ID] = prefill_replacement
    return replacements
