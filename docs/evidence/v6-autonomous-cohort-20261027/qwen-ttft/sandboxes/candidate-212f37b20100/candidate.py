"""Joint candidate for the selected Qwen3Model boundary (prefill call
2441f10600ba127de5cb753c of group 7f49ae8c64c12e322632 + decode call
e7df98b905ed8aa688d5227a of group 92e2f6fb9310a942405a).

Repair of trial-cf72c8a39d-param-65bdefc5 (and trial-1a25041f29 before it):
both priors degraded silently to the 'parent-eager' route because installation
was gated on self-imposed absolute hidden/k/v error budgets (candidate.py:472
`_ok` returned False), budgets that no refusion can meet bit-for-bit since
Triton fp32 reductions/exp/rsqrt differ from ATen by ulps that q/k-norm weights
up to 44.0 amplify over 36 layers.  The measured candidate was therefore the
parent plus wrapper overhead and both paired gains were parent-vs-parent noise.

This attempt gates installation on the project's actual quality bar instead:
for every engaged path it requires greedy first-token equality with the live
parent reference and logits closeness at the source tolerances through the
boundary's own lm_head projection on fresh random [1,256] inputs, per-input
determinism across repeated executions, and exact multi-step greedy-token
agreement (candidate prefill + parent decode vs full parent, 8 steps, 2
inputs).  The ladder is graph-composite -> graph-parent -> eager-composite ->
parent-eager and records the failing stat vector or exception text at every
transition, so quality telemetry can never show a silent 'parent-eager'
installation again.  The fused composite keeps the parent's per-op rounding
sequence (round-then-weight RMSNorm, bf16 RoPE with rotate_half, ATen silu
order x*(1/(1+exp(-x)))), writes K/V directly in the parent's contiguous
[B,H,S,D] post-rope layout (replacing all 145 prefill cache cats), passes v
into SDPA exactly as the parent does, and replays as a CUDA graph with a
static input copy per invocation.  Decode passes through the live parent
reference unchanged (31 calls per run).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from transformers.cache_utils import DynamicCache
from transformers.modeling_outputs import BaseModelOutputWithPast

PREFILL_CALL_ID = "2441f10600ba127de5cb753c"
DECODE_CALL_ID = "e7df98b905ed8aa688d5227a"

_LOGITS_ATOL = 0.02
_LOGITS_RTOL = 0.02
_N_CHECK = 3
_N_STEP = 8

_HAS_TRITON = False
try:
    import triton
    import triton.language as tl

    _HAS_TRITON = hasattr(tl, "rsqrt")
except Exception:  # pragma: no cover
    triton = None
    tl = None


if _HAS_TRITON:

    @triton.jit
    def _k_add_rmsnorm(
        x_ptr,
        d_ptr,
        w_ptr,
        sum_ptr,
        out_ptr,
        inv_n,
        eps,
        N: tl.constexpr,
        BLOCK: tl.constexpr,
        HAS_DELTA: tl.constexpr,
        STORE_SUM: tl.constexpr,
    ):
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
        x_ptr,
        w_ptr,
        cos_ptr,
        sin_ptr,
        out_ptr,
        S,
        H,
        inv_d,
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
    def _k_silu_mul(g_ptr, u_ptr, o_ptr, n, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        off = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
        msk = off < n
        g = tl.load(g_ptr + off, mask=msk, other=0.0).to(tl.float32)
        u = tl.load(u_ptr + off, mask=msk, other=0.0).to(tl.float32)
        s = (g * (1.0 / (1.0 + tl.exp(-g)))).to(tl.bfloat16).to(tl.float32)
        o = (s * u).to(tl.bfloat16)
        tl.store(o_ptr + off, o, mask=msk)


def _rms_norm_exact(x, weight, eps):
    xf = x.float()
    var = xf.pow(2).mean(-1, keepdim=True)
    xn = (xf * torch.rsqrt(var + eps)).to(x.dtype)
    return weight * xn


class _TorchOps:
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
        y = _rms_norm_exact(x, w, eps).transpose(1, 2)
        half = x.shape[-1] // 2
        seq = x.shape[1]
        y1, y2 = y[..., :half], y[..., half:]
        rot = torch.cat((-y2, y1), dim=-1)
        cos = torch.cat((self.cos_h, self.cos_h), dim=-1).view(1, 1, seq, half * 2)
        sin = torch.cat((self.sin_h, self.sin_h), dim=-1).view(1, 1, seq, half * 2)
        return (y * cos) + (rot * sin)

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
        b, s, h, d = x.shape
        out = torch.empty((b, h, s, d), dtype=x.dtype, device=x.device)
        _k_qknorm_rope[(b * h * s,)](
            x, w, cos_c, sin_c, out, s, h, inv_d, eps,
            HALF=half, D=d, num_warps=2,
        )
        return out

    def silu_mul(g, u):
        out = torch.empty_like(g)
        n = g.numel()
        _k_silu_mul[(triton.cdiv(n, 8192),)](g, u, out, n, BLOCK=8192, num_warps=8)
        return out

    try:
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
    cfg = module.config
    nh = cfg.num_attention_heads
    nkv = cfg.num_key_value_heads
    hd = getattr(cfg, "head_dim", None) or cfg.hidden_size // nh
    eps = cfg.rms_norm_eps
    layers = list(module.layers[: cfg.num_hidden_layers])
    last = len(layers) - 1
    final_w = module.norm.weight

    def run_prefill(input_ids):
        b, s = input_ids.shape
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
            q = ops.qk_rope(F.linear(h, attn.q_proj.weight).view(b, s, nh, hd), attn.q_norm.weight, eps)
            k = ops.qk_rope(F.linear(h, attn.k_proj.weight).view(b, s, nkv, hd), attn.k_norm.weight, eps)
            v = F.linear(h, attn.v_proj.weight).view(b, s, nkv, hd).transpose(1, 2)
            a = F.scaled_dot_product_attention(
                q, k, v, attn_mask=None, dropout_p=0.0, scale=attn.scaling, is_causal=True, enable_gqa=True
            )
            delta = F.linear(a.transpose(1, 2).reshape(b, s, nh * hd), attn.o_proj.weight)
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


def _wrap_output(cfg, hidden, ks, vs):
    cache = DynamicCache(config=cfg)
    for lyr, k, v in zip(cache.layers, ks, vs):
        lyr.keys = k
        lyr.values = v
        lyr.is_initialized = True
        lyr.dtype = k.dtype
        lyr.device = k.device
    return BaseModelOutputWithPast(last_hidden_state=hidden, past_key_values=cache)


def _capture_graph(fn, static_ids, device):
    stream = torch.cuda.Stream(device=device)
    stream.wait_stream(torch.cuda.current_stream(device))
    with torch.cuda.stream(stream):
        for _ in range(3):
            fn(static_ids)
    torch.cuda.current_stream(device).wait_stream(stream)
    torch.cuda.synchronize(device)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = fn(static_ids)
    torch.cuda.synchronize(device)
    return graph, out


def _exc(exc):
    text = "{}:{}".format(type(exc).__name__, exc)
    return text[:200].replace("\n", " ")


_EXPECTED_KEYS = {
    "input_ids",
    "attention_mask",
    "position_ids",
    "past_key_values",
    "inputs_embeds",
    "use_cache",
}


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

    if module is None or prefill_parent is None:
        record_route(PREFILL_CALL_ID, "setup-failure:missing-boundary")
        return replacements
    if not torch.cuda.is_available():
        record_route(PREFILL_CALL_ID, "setup-failure:no-cuda")
        return replacements

    quality = state.get("quality") or {}
    atol = float(quality.get("logits_atol", quality.get("atol", _LOGITS_ATOL)))
    rtol = float(quality.get("logits_rtol", quality.get("rtol", _LOGITS_RTOL)))

    device = next(module.parameters()).device
    cfg = module.config
    head_dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
    half = head_dim // 2

    lm_head = None
    if root is not None and getattr(root, "lm_head", None) is not None:
        lm_head = root.lm_head

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

    run_state = {"mode": "parent-eager", "graph": None, "ids": None, "h": None, "ks": None, "vs": None}
    composite = None

    try:
        with torch.no_grad(), torch.cuda.device(device):
            pos = torch.arange(seq, device=device).unsqueeze(0)
            dummy = torch.zeros((batch, seq, cfg.hidden_size), dtype=torch.bfloat16, device=device)
            cos, sin = module.rotary_emb(dummy, pos)
            cos_h = cos[0, :, :half].contiguous()
            sin_h = sin[0, :, :half].contiguous()
            del dummy

            torch_ops = _TorchOps(cos_h, sin_h)
            triton_ops = None
            if _HAS_TRITON:
                triton_ops = _make_triton_ops(device, cfg.hidden_size, head_dim, cos_h, sin_h, torch.bfloat16)
            if triton_ops is None:
                record_route(PREFILL_CALL_ID, "setup-failure:triton-unavailable")

            check_ids = [
                torch.randint(0, cfg.vocab_size, (batch, seq), device=device, dtype=torch.int64)
                for _ in range(_N_CHECK)
            ]

            def parent_call(ids):
                return prefill_parent(
                    input_ids=ids,
                    attention_mask=None,
                    position_ids=None,
                    past_key_values=None,
                    inputs_embeds=None,
                    use_cache=True,
                )

            def first_logits(hidden):
                h = hidden[:, -1:, :]
                if lm_head is not None:
                    return lm_head(h)
                return F.linear(h, module.embed_tokens.weight)

            def gen(boundary, ids, steps):
                out = boundary(ids)
                lg = first_logits(out.last_hidden_state)
                tok = lg[:, -1].argmax(dim=-1, keepdim=True)
                toks = [tok]
                logits = [lg]
                cache = out.past_key_values
                for _ in range(steps - 1):
                    out = decode_parent(
                        input_ids=tok,
                        attention_mask=None,
                        position_ids=None,
                        past_key_values=cache,
                        inputs_embeds=None,
                        use_cache=True,
                    )
                    cache = out.past_key_values
                    lg = first_logits(out.last_hidden_state)
                    tok = lg[:, -1].argmax(dim=-1, keepdim=True)
                    toks.append(tok)
                    logits.append(lg)
                return torch.cat(toks, dim=1), torch.cat(logits, dim=1)

            def verify(boundary):
                tok_ok = True
                close_ok = True
                err = 0.0
                for ids in check_ids:
                    lg = first_logits(boundary(ids).last_hidden_state).float()
                    rg = first_logits(parent_call(ids).last_hidden_state).float()
                    tok_ok &= bool(torch.equal(lg.argmax(dim=-1), rg.argmax(dim=-1)))
                    close_ok &= bool(torch.allclose(lg, rg, atol=atol, rtol=rtol))
                    err = max(err, (lg - rg).abs().max().item())
                h1 = boundary(check_ids[0]).last_hidden_state.clone()
                h2 = boundary(check_ids[0]).last_hidden_state
                det_ok = bool(torch.equal(h1, h2))
                steps_ok = True
                step_err = 0.0
                for ids in check_ids[:2]:
                    tc, lc = gen(boundary, ids, _N_STEP)
                    tr, lr = gen(parent_call, ids, _N_STEP)
                    steps_ok &= bool(torch.equal(tc, tr))
                    step_err = max(step_err, (lc.float() - lr.float()).abs().max().item())
                ok = bool(tok_ok and close_ok and det_ok and steps_ok)
                return {
                    "ok": ok,
                    "tok": bool(tok_ok),
                    "close": bool(close_ok),
                    "det": det_ok,
                    "steps": bool(steps_ok),
                    "err": err,
                    "step_err": step_err,
                }

            def fmt(st):
                return "tok={},close={},det={},steps={},err={:.4f},step_err={:.4f}".format(
                    st["tok"], st["close"], st["det"], st["steps"], st["err"], st["step_err"]
                )

            ladder = []
            if triton_ops is not None:
                ladder.append(("triton", triton_ops))
            ladder.append(("torch", torch_ops))
            for name, ops in ladder:
                comp = _make_composite(module, ops)

                def eager_boundary(ids, _c=comp):
                    return _wrap_output(cfg, *_c(ids))

                st = verify(eager_boundary)
                if st["ok"]:
                    composite = comp
                    record_route(PREFILL_CALL_ID, "verify-eager-{}:ok:{}".format(name, fmt(st)))
                    break
                record_route(PREFILL_CALL_ID, "setup-failure:verify-eager-{}:{}".format(name, fmt(st)))

            if composite is not None:
                run_state["mode"] = "eager-composite"
                try:
                    static_ids = torch.randint(0, cfg.vocab_size, (batch, seq), device=device, dtype=torch.int64)
                    graph, (s_hidden, s_ks, s_vs) = _capture_graph(composite, static_ids, device)

                    def replay_boundary(ids):
                        static_ids.copy_(ids)
                        graph.replay()
                        return _wrap_output(cfg, s_hidden, s_ks, s_vs)

                    st = verify(replay_boundary)
                    if st["ok"]:
                        run_state.update(
                            {"mode": "graph-composite", "graph": graph, "ids": static_ids,
                             "h": s_hidden, "ks": s_ks, "vs": s_vs}
                        )
                        record_route(PREFILL_CALL_ID, "verify-graph-composite:ok:{}".format(fmt(st)))
                    else:
                        record_route(PREFILL_CALL_ID, "setup-failure:verify-graph-composite:{}".format(fmt(st)))
                except Exception as exc:
                    record_route(PREFILL_CALL_ID, "setup-failure:capture-composite:{}".format(_exc(exc)))

            if run_state["mode"] != "graph-composite":
                try:

                    def parent_module_fn(ids):
                        out = module(
                            input_ids=ids,
                            attention_mask=None,
                            position_ids=None,
                            past_key_values=None,
                            inputs_embeds=None,
                            use_cache=True,
                        )
                        layers = out.past_key_values.layers
                        return out.last_hidden_state, [l.keys for l in layers], [l.values for l in layers]

                    static_ids = torch.randint(0, cfg.vocab_size, (batch, seq), device=device, dtype=torch.int64)
                    graph, (p_hidden, p_ks, p_vs) = _capture_graph(parent_module_fn, static_ids, device)

                    def replay_parent(ids):
                        static_ids.copy_(ids)
                        graph.replay()
                        return _wrap_output(cfg, p_hidden, p_ks, p_vs)

                    st = verify(replay_parent)
                    if st["ok"]:
                        run_state.update(
                            {"mode": "graph-parent", "graph": graph, "ids": static_ids,
                             "h": p_hidden, "ks": p_ks, "vs": p_vs}
                        )
                        record_route(PREFILL_CALL_ID, "verify-graph-parent:ok:{}".format(fmt(st)))
                    else:
                        record_route(PREFILL_CALL_ID, "setup-failure:verify-graph-parent:{}".format(fmt(st)))
                except Exception as exc:
                    record_route(PREFILL_CALL_ID, "setup-failure:capture-parent:{}".format(_exc(exc)))

            try:
                torch.cuda.synchronize(device)
            except Exception:
                pass
    except Exception as exc:
        record_route(PREFILL_CALL_ID, "setup-failure:setup:{}".format(_exc(exc)))
        run_state.update({"mode": "parent-eager", "graph": None, "ids": None, "h": None, "ks": None, "vs": None})
        composite = None

    record_route(PREFILL_CALL_ID, "install:{}".format(run_state["mode"]))

    def _fast_guard(kwargs):
        if set(kwargs.keys()) != _EXPECTED_KEYS:
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

    def prefill_replacement(*args, **kwargs):
        if args or not _fast_guard(kwargs):
            record_route(PREFILL_CALL_ID, "parent-fallback-signature")
            return prefill_parent(*args, **kwargs)
        st = run_state
        try:
            if st["graph"] is not None:
                record_route(PREFILL_CALL_ID, "graph-replay:{}".format(st["mode"]))
                st["ids"].copy_(kwargs["input_ids"])
                st["graph"].replay()
                return _wrap_output(cfg, st["h"], st["ks"], st["vs"])
            if st["mode"] == "eager-composite" and composite is not None:
                record_route(PREFILL_CALL_ID, "eager-composite")
                hidden, ks, vs = composite(kwargs["input_ids"])
                return _wrap_output(cfg, hidden, ks, vs)
            record_route(PREFILL_CALL_ID, "parent-eager")
            return prefill_parent(**kwargs)
        except Exception as exc:
            if st["mode"] in ("graph-composite", "graph-parent") and composite is not None:
                st["mode"] = "eager-composite"
                st["graph"] = None
                record_route(PREFILL_CALL_ID, "runtime-failure-graph:{}".format(_exc(exc)))
                try:
                    hidden, ks, vs = composite(kwargs["input_ids"])
                    return _wrap_output(cfg, hidden, ks, vs)
                except Exception as exc2:
                    st["mode"] = "parent-eager"
                    record_route(PREFILL_CALL_ID, "runtime-failure-composite:{}".format(_exc(exc2)))
                    return prefill_parent(**kwargs)
            st["mode"] = "parent-eager"
            st["graph"] = None
            record_route(PREFILL_CALL_ID, "runtime-failure:{}".format(_exc(exc)))
            return prefill_parent(**kwargs)

    replacements[PREFILL_CALL_ID] = prefill_replacement
    return replacements
