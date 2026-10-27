"""Joint candidate for the selected Qwen3Model boundary (scope: joint, groups
7f49ae8c64c12e322632 prefill + 92e2f6fb9310a942405a decode).

Repair of trial-1a25041f29: that candidate's build-time verification failed
(``_check`` returned False at candidate.py:338) because its fused q/k norm
computed ``(x_norm_fp32 * weight_fp32)`` rounded once, while the parent
Qwen3RMSNorm returns ``weight * round_bf16(x_norm)`` - the round-to-bf16
happens BEFORE the weight multiply.  Qwen3-4B q/k norm weights reach 44.0, so
the missing intermediate rounding amplified to ~2.0 absolute error on layer-0
keys (reproduced on CPU), drifted the stack by ~1.0 in last_hidden_state and
silently degraded every quality invocation to the parent-eager fallback
(route telemetry: ``parent-eager`` count 2), whose ~2-3 ms gain then failed
the paired noise test.

This replacement keeps the parent's exact rounding sequence everywhere:
per-head q/k RMSNorm is round-then-weight in bf16, RoPE arithmetic runs in
bf16 with the same per-op rounding order as ``apply_rotary_pos_emb``, SiLU
uses the ATen form ``x / (1 + exp(-x))`` in fp32 rounded once, every GEMM and
the SDPA flash call use the parent's own weights/shapes/strides/scaling, and
the KV cache entries are written directly in the parent's contiguous
[B, H, S, D] post-rope layout (replacing DynamicCache's empty-cat copies).
The 36-layer stack is fused into three Triton kernels (add+RMSNorm,
q/k-norm+RoPE+transpose, SiLU*up) plus the unchanged GEMMs/SDPA, captured
once into a CUDA graph at installation; every invocation copies the current
input_ids into the static buffer and replays.  Installation verifies the
eager composite and two graph replays against the live parent reference on
fresh random token ids with tight per-tensor thresholds and falls back
(composite-graph -> eager-composite -> parent-graph -> parent-eager) with
distinct route labels for setup failure, unsupported signature and runtime
failure.  Decode (31 calls per run) passes through the parent unchanged.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from transformers.cache_utils import DynamicCache
from transformers.modeling_outputs import BaseModelOutputWithPast

PREFILL_CALL_ID = "2441f10600ba127de5cb753c"
DECODE_CALL_ID = "e7df98b905ed8aa688d5227a"

_HAS_TRITON = False
try:  # fused kernels; a bitwise-exact pure-torch path is used when absent
    import triton
    import triton.language as tl

    _HAS_TRITON = hasattr(tl, "rsqrt")
except Exception:  # pragma: no cover
    triton = None
    tl = None


if _HAS_TRITON:

    @triton.jit
    def _k_add_rmsnorm(
        x_ptr,  # [R, N] bf16 norm input (previous residual stream)
        d_ptr,  # [R, N] bf16 delta (attn/mlp output) when HAS_DELTA
        w_ptr,  # [N] bf16 norm weight
        sum_ptr,  # [R, N] bf16 rounded residual sum when STORE_SUM
        out_ptr,  # [R, N] bf16 normed output
        inv_n,
        eps,
        N: tl.constexpr,
        BLOCK: tl.constexpr,
        HAS_DELTA: tl.constexpr,
        STORE_SUM: tl.constexpr,
    ):
        # Parent semantics: s = round_bf16(x + d); n = round_bf16(w * round_bf16(s32 * rsqrt(mean(s32^2)+eps)))
        # ATen mean = tree-sum * float(1/N); ATen bf16 elementwise ops compute in fp32 and round once.
        row = tl.program_id(0)
        offs = tl.arange(0, BLOCK)
        m = offs < N
        base = row.to(tl.int64) * N
        xf = tl.load(x_ptr + base + offs, mask=m, other=0.0).to(tl.float32)
        if HAS_DELTA:
            df = tl.load(d_ptr + base + offs, mask=m, other=0.0).to(tl.float32)
            sf = (xf + df).to(tl.bfloat16)
            xf = sf.to(tl.float32)
            if STORE_SUM:
                tl.store(sum_ptr + base + offs, sf, mask=m)
        var = tl.sum(xf * xf, axis=0) * inv_n
        rstd = tl.rsqrt(var + eps)
        xn = (xf * rstd).to(tl.bfloat16).to(tl.float32)
        wf = tl.load(w_ptr + offs, mask=m, other=0.0).to(tl.float32)
        y = (wf * xn).to(tl.bfloat16)
        tl.store(out_ptr + base + offs, y, mask=m)

    @triton.jit
    def _k_qknorm_rope(
        x_ptr,  # [B, S, H, D] bf16 contiguous projection output
        w_ptr,  # [D] bf16 per-head norm weight
        cos_ptr,  # [S, HALF] bf16 rotary cos half
        sin_ptr,  # [S, HALF] bf16 rotary sin half
        out_ptr,  # [B, H, S, D] bf16 contiguous post-rope output
        S,
        H,
        inv_d,
        eps,
        HALF: tl.constexpr,
        D: tl.constexpr,
    ):
        # Parent: y = round_bf16(w * round_bf16(x_norm)); rope o = round_bf16(round_bf16(y*cos) + round_bf16(rot*sin)) in bf16
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
        var = (tl.sum(x1 * x1, axis=0) + tl.sum(x2 * x2, axis=0)) * inv_d
        rstd = tl.rsqrt(var + eps)
        xn1 = (x1 * rstd).to(tl.bfloat16).to(tl.float32)
        xn2 = (x2 * rstd).to(tl.bfloat16).to(tl.float32)
        w1 = tl.load(w_ptr + d).to(tl.float32)
        w2 = tl.load(w_ptr + HALF + d).to(tl.float32)
        y1 = (w1 * xn1).to(tl.bfloat16).to(tl.float32)
        y2 = (w2 * xn2).to(tl.bfloat16).to(tl.float32)
        cos = tl.load(cos_ptr + s * HALF + d).to(tl.float32)
        sin = tl.load(sin_ptr + s * HALF + d).to(tl.float32)
        t1 = (y1 * cos).to(tl.bfloat16).to(tl.float32)
        u1 = ((0.0 - y2) * sin).to(tl.bfloat16).to(tl.float32)
        o1 = (t1 + u1).to(tl.bfloat16)
        t2 = (y2 * cos).to(tl.bfloat16).to(tl.float32)
        u2 = (y1 * sin).to(tl.bfloat16).to(tl.float32)
        o2 = (t2 + u2).to(tl.bfloat16)
        base_out = ((b * H + h) * S + s) * D
        tl.store(out_ptr + base_out + d, o1)
        tl.store(out_ptr + base_out + HALF + d, o2)

    @triton.jit
    def _k_silu_mul(
        g_ptr,  # [N] bf16 gate
        u_ptr,  # [N] bf16 up
        o_ptr,  # [N] bf16 silu(gate) * up
        n,
        BLOCK: tl.constexpr,
    ):
        # ATen: round_bf16(g32 / (1 + exp(-g32))) then round_bf16(s32 * u32)
        pid = tl.program_id(0)
        off = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
        msk = off < n
        g = tl.load(g_ptr + off, mask=msk, other=0.0).to(tl.float32)
        u = tl.load(u_ptr + off, mask=msk, other=0.0).to(tl.float32)
        s = (g / (1.0 + tl.exp(-g))).to(tl.bfloat16).to(tl.float32)
        o = (s * u).to(tl.bfloat16)
        tl.store(o_ptr + off, o, mask=msk)


# ---------------------------------------------------------------------------
# Pure-torch ops: the parent's exact op sequences (bitwise-identical math).
# Used for CPU validation, as the no-Triton fallback, and as the eager
# reference when verifying the Triton kernels / graph replays.
# ---------------------------------------------------------------------------


def _rms_norm_exact(x, weight, eps):
    xf = x.float()
    var = xf.pow(2).mean(-1, keepdim=True)
    xn = (xf * torch.rsqrt(var + eps)).to(x.dtype)
    return weight * xn


class _TorchOps:
    """Bitwise-exact emulation of the parent's per-op math (no fusion)."""

    def __init__(self, cos_h, sin_h):
        self.cos_h = cos_h
        self.sin_h = sin_h

    def norm_only(self, x, w, eps):
        return _rms_norm_exact(x, w, eps)

    def add_norm(self, x, d, w, eps):
        s = x + d
        return s, _rms_norm_exact(s, w, eps)

    def add_norm_final(self, x, d, w, eps):
        return _rms_norm_exact(x + d, w, eps)

    def qk_rope(self, x, w, eps):
        # x: [B, S, H, D] -> [B, H, S, D]; mirrors Qwen3Attention + apply_rotary_pos_emb
        y = _rms_norm_exact(x, w, eps).transpose(1, 2)
        half = x.shape[-1] // 2
        y1, y2 = y[..., :half], y[..., half:]
        rot = torch.cat((-y2, y1), dim=-1)
        cos = torch.cat((self.cos_h, self.cos_h), dim=-1).view(1, 1, *self.cos_h.shape[:-1], -1)
        sin = torch.cat((self.sin_h, self.sin_h), dim=-1).view(1, 1, *self.sin_h.shape[:-1], -1)
        return ((y * cos) + (rot * sin)).contiguous()

    def silu_mul(self, g, u):
        return F.silu(g) * u


def _make_triton_ops(device, hidden, head_dim, cos_h, sin_h, dtype):
    half = head_dim // 2
    inv_n = float(1.0 / hidden)
    inv_d = float(1.0 / head_dim)
    block_n = triton.next_power_of_2(hidden)
    cos_c = cos_h.contiguous()
    sin_c = sin_h.contiguous()

    def norm_only(x, w, eps):
        out = torch.empty_like(x)
        rows = x.shape[0] * x.shape[1]
        _k_add_rmsnorm[(rows,)](
            x, x, w, out, out, inv_n, eps,
            N=hidden, BLOCK=block_n, HAS_DELTA=False, STORE_SUM=False, num_warps=8,
        )
        return out

    def add_norm(x, d, w, eps):
        s = torch.empty_like(x)
        out = torch.empty_like(x)
        rows = x.shape[0] * x.shape[1]
        _k_add_rmsnorm[(rows,)](
            x, d, w, s, out, inv_n, eps,
            N=hidden, BLOCK=block_n, HAS_DELTA=True, STORE_SUM=True, num_warps=8,
        )
        return s, out

    def add_norm_final(x, d, w, eps):
        out = torch.empty_like(x)
        rows = x.shape[0] * x.shape[1]
        _k_add_rmsnorm[(rows,)](
            x, d, w, out, out, inv_n, eps,
            N=hidden, BLOCK=block_n, HAS_DELTA=True, STORE_SUM=False, num_warps=8,
        )
        return out

    def qk_rope(x, w, eps):
        B, S, H, D = x.shape
        out = torch.empty((B, H, S, D), dtype=x.dtype, device=x.device)
        _k_qknorm_rope[(B * H * S,)](
            x, w, cos_c, sin_c, out, S, H, inv_d, eps,
            HALF=half, D=D, num_warps=2,
        )
        return out

    def silu_mul(g, u):
        out = torch.empty_like(g)
        n = g.numel()
        _k_silu_mul[(triton.cdiv(n, 8192),)](g, u, out, n, BLOCK=8192, num_warps=8)
        return out

    try:  # compile-check all three kernels on tiny dummies at installation time
        probe = torch.zeros((1, 4, hidden), dtype=dtype, device=device)
        w = torch.ones(hidden, dtype=dtype, device=device)
        norm_only(probe, w, 1e-6)
        s, o = add_norm(probe, probe, w, 1e-6)
        add_norm_final(probe, probe, w, 1e-6)
        xq = torch.zeros((1, 2, 3, head_dim), dtype=dtype, device=device)
        wq = torch.ones(head_dim, dtype=dtype, device=device)
        qk_rope(xq, wq, 1e-6)
        a = torch.zeros((64, 64), dtype=dtype, device=device)
        silu_mul(a, a)
        torch.cuda.synchronize(device)
        return _Ops(norm_only, add_norm, add_norm_final, qk_rope, silu_mul)
    except Exception:
        return None


class _Ops:
    def __init__(self, norm_only, add_norm, add_norm_final, qk_rope, silu_mul):
        self.norm_only = norm_only
        self.add_norm = add_norm
        self.add_norm_final = add_norm_final
        self.qk_rope = qk_rope
        self.silu_mul = silu_mul


def _make_composite(module, ops):
    """Fused prefill stack over the parent's own parameters."""
    cfg = module.config
    nh = cfg.num_attention_heads
    nkv = cfg.num_key_value_heads
    hd = getattr(cfg, "head_dim", None) or cfg.hidden_size // nh
    eps = cfg.rms_norm_eps
    layers = list(module.layers[: cfg.num_hidden_layers])
    last = len(layers) - 1
    final_w = module.norm.weight

    def run_prefill(input_ids):
        B, S = input_ids.shape
        hidden = module.embed_tokens(input_ids)
        ks = []
        vs = []
        mlp_delta = None
        for i, layer in enumerate(layers):
            attn = layer.self_attn
            if i == 0:
                h = ops.norm_only(hidden, layer.input_layernorm.weight, eps)
            else:
                hidden, h = ops.add_norm(hidden, mlp_delta, layer.input_layernorm.weight, eps)
            q = ops.qk_rope(F.linear(h, attn.q_proj.weight).view(B, S, nh, hd), attn.q_norm.weight, eps)
            k = ops.qk_rope(F.linear(h, attn.k_proj.weight).view(B, S, nkv, hd), attn.k_norm.weight, eps)
            v = F.linear(h, attn.v_proj.weight).view(B, S, nkv, hd).transpose(1, 2).contiguous()
            a = F.scaled_dot_product_attention(
                q, k, v, attn_mask=None, dropout_p=0.0, scale=attn.scaling, is_causal=True, enable_gqa=True
            )
            delta = F.linear(a.transpose(1, 2).reshape(B, S, nh * hd), attn.o_proj.weight)
            hidden, h2 = ops.add_norm(hidden, delta, layer.post_attention_layernorm.weight, eps)
            g = F.linear(h2, layer.mlp.gate_proj.weight)
            u = F.linear(h2, layer.mlp.up_proj.weight)
            mlp_delta = F.linear(ops.silu_mul(g, u), layer.mlp.down_proj.weight)
            if i == last:
                hidden = ops.add_norm_final(hidden, mlp_delta, final_w, eps)
            ks.append(k)
            vs.append(v)
        return hidden, ks, vs

    return run_prefill


def _wrap_output(module, hidden, ks, vs):
    """Parent output structure: fresh DynamicCache view over the produced K/V."""
    cache = DynamicCache(config=module.config)
    layers = cache.layers
    for i, layer in enumerate(layers):
        k = ks[i]
        v = vs[i]
        layer.keys = k
        layer.values = v
        layer.is_initialized = True
        layer.dtype = k.dtype
        layer.device = k.device
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


# Verification thresholds vs the eager parent (max over two random inputs):
#   hidden <= 0.10 (previous failing version measured 1.0; expected ulp noise < 0.05)
#   k      <= 1.20 with mean <= 2e-3 (previous failing version measured 2.0 max)
#   v      <= 0.10 with mean <= 1e-3 (weight-amplified bf16-ulp flips stay rare)
_HIDDEN_MAX = 0.10
_KV_MAX = 1.20
_K_MEAN = 2e-3
_V_MEAN = 1e-3


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
        if module is not None and prefill_parent is not None:
            record_route(PREFILL_CALL_ID, "setup-failure:no-cuda")
        return replacements

    device = next(module.parameters()).device
    cfg = module.config
    head_dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
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
            # Position-only rotary tables for positions 0..S-1, computed once
            # with the parent's own rotary module (input-value independent).
            pos = torch.arange(seq, device=device).unsqueeze(0)
            dummy = torch.zeros((batch, seq, cfg.hidden_size), dtype=torch.bfloat16, device=device)
            cos, sin = module.rotary_emb(dummy, pos)
            cos_h = cos[0, :, :half].contiguous()
            sin_h = sin[0, :, :half].contiguous()

            torch_ops = _TorchOps(cos_h, sin_h)
            triton_ops = None
            if _HAS_TRITON:
                triton_ops = _make_triton_ops(device, cfg.hidden_size, head_dim, cos_h, sin_h, torch.bfloat16)
            if triton_ops is None:
                record_route(PREFILL_CALL_ID, "setup-failure:triton-unavailable")
            ops = triton_ops if triton_ops is not None else torch_ops
            composite = _make_composite(module, ops)

            verify_ids = [
                torch.randint(0, cfg.vocab_size, (batch, seq), device=device)
                for _ in range(2)
            ]

            def _stats(fn):
                worst = {"hidden": 0.0, "k": 0.0, "k_mean": 0.0, "v": 0.0, "v_mean": 0.0}
                for ids in verify_ids:
                    hidden, ks, vs = fn(ids)
                    ref = _parent_call(ids)
                    ref_layers = ref.past_key_values.layers
                    dh = (hidden.float() - ref.last_hidden_state.float()).abs().max().item()
                    worst["hidden"] = max(worst["hidden"], dh)
                    kmax = 0.0
                    kmean = 0.0
                    vmax = 0.0
                    vmean = 0.0
                    for i, ref_layer in enumerate(ref_layers):
                        dk = (ks[i].float() - ref_layer.keys.float()).abs()
                        dv = (vs[i].float() - ref_layer.values.float()).abs()
                        kmax = max(kmax, dk.max().item())
                        kmean = max(kmean, dk.mean().item())
                        vmax = max(vmax, dv.max().item())
                        vmean = max(vmean, dv.mean().item())
                    worst["k"] = max(worst["k"], kmax)
                    worst["k_mean"] = max(worst["k_mean"], kmean)
                    worst["v"] = max(worst["v"], vmax)
                    worst["v_mean"] = max(worst["v_mean"], vmean)
                return worst

            def _ok(s):
                return (
                    s["hidden"] <= _HIDDEN_MAX
                    and s["k"] <= _KV_MAX
                    and s["k_mean"] <= _K_MEAN
                    and s["v"] <= _KV_MAX
                    and s["v_mean"] <= _V_MEAN
                )

            stats = _stats(composite)
            composite_ok = _ok(stats)
            if not composite_ok:
                record_route(
                    PREFILL_CALL_ID,
                    "setup-failure:composite-verify:h={hidden:.3f},k={k:.3f}/{k_mean:.2e},v={v:.3f}/{v_mean:.2e}".format(
                        **stats
                    ),
                )

            if composite_ok:
                try:
                    static_ids = torch.randint(0, cfg.vocab_size, (batch, seq), device=device)
                    graph, out = _capture_graph(composite, static_ids, device)
                    s_hidden, s_ks, s_vs = out

                    def _replay(ids):
                        static_ids.copy_(ids)
                        graph.replay()
                        return s_hidden, s_ks, s_vs

                    gstats = _stats(_replay)
                    if _ok(gstats):
                        mode = "graph-composite"
                        graph_runner = (graph, static_ids, s_hidden, s_ks, s_vs)
                    else:
                        record_route(
                            PREFILL_CALL_ID,
                            "setup-failure:graph-verify:h={hidden:.3f},k={k:.3f},v={v:.3f}".format(**gstats),
                        )
                        mode = "eager-composite"
                except Exception:
                    record_route(PREFILL_CALL_ID, "setup-failure:graph-capture-composite")
                    mode = "eager-composite"
            else:
                # Verified fallback: capture the unmodified parent module (its
                # own nn.Module forward, not the binding wrapper) so replay
                # still removes exposed dispatch while keeping parent math.
                try:

                    def _module_out(ids):
                        out = module(
                            input_ids=ids,
                            attention_mask=None,
                            position_ids=None,
                            past_key_values=None,
                            inputs_embeds=None,
                            use_cache=True,
                        )
                        layers = out.past_key_values.layers
                        return (
                            out.last_hidden_state,
                            [l.keys for l in layers],
                            [l.values for l in layers],
                        )

                    static_ids = torch.randint(0, cfg.vocab_size, (batch, seq), device=device)
                    graph, out = _capture_graph(_module_out, static_ids, device)
                    s_hidden, s_ks, s_vs = out

                    def _replay(ids):
                        static_ids.copy_(ids)
                        graph.replay()
                        return s_hidden, s_ks, s_vs

                    pstats = _stats(_replay)
                    if pstats["hidden"] <= 0.02 and pstats["k"] <= 0.02 and pstats["v"] <= 0.02:
                        mode = "graph-parent"
                        graph_runner = (graph, static_ids, s_hidden, s_ks, s_vs)
                    else:
                        record_route(
                            PREFILL_CALL_ID,
                            "setup-failure:graph-verify-parent:h={hidden:.3f},k={k:.3f},v={v:.3f}".format(**pstats),
                        )
                except Exception:
                    record_route(PREFILL_CALL_ID, "setup-failure:graph-capture-parent")

            try:
                torch.cuda.synchronize(device)
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

    graph = None
    static_in = None
    s_hidden = s_ks = s_vs = None
    if graph_runner is not None:
        graph, static_in, s_hidden, s_ks, s_vs = graph_runner

    def prefill_replacement(*args, **kwargs):
        nonlocal mode
        if args or not _fast_guard(kwargs):
            record_route(PREFILL_CALL_ID, "parent-fallback-signature")
            return prefill_parent(*args, **kwargs)
        try:
            if graph is not None:
                record_route(
                    PREFILL_CALL_ID,
                    "graph-replay:composite" if mode == "graph-composite" else "graph-replay:parent",
                )
                static_in.copy_(kwargs["input_ids"])
                graph.replay()
                return _wrap_output(module, s_hidden, s_ks, s_vs)
            if mode == "eager-composite" and composite is not None:
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
