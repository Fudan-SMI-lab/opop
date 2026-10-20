"""Candidate: dispatch-collapsed fused prefill region for the Qwen3-4B TTFT objective.

Replaces the phase-guarded prefill call of ``Qwen3ForCausalLM`` (call id
``900c23e99c808d2c9bda215a``, signature ``input_ids=[1,256] int64,
use_cache=True, logits_to_keep=1``) with an equivalent implementation that

* packs q/k/v and gate/up projections into single cuBLAS GEMMs (the packed
  storages are shared with the rebound ``nn.Linear`` module weights, so the
  untouched decode path reads identical values with no weight duplication),
* fuses residual-add+RMSNorm, q/k RMSNorm+rotary+head split into the KV cache
  layout, and SiLU*gate into single Triton kernels that reproduce the parent's
  bf16 materialization points,
* keeps the parent's math library calls (cuBLAS GEMMs, flash SDPA with
  ``enable_gqa=True`` and ``is_causal=True``, embedding gather, lm_head GEMM),
* lazily captures the whole region into a CUDA graph inside the first real
  invocation (after side-stream warmup) and replays it on every invocation on
  the current inputs, with an eager composite and the parent forward as
  fallback paths.

The replacement returns ``CausalLMOutputWithPast`` with last-position logits
``[1,1,vocab]`` bf16 and a standard ``DynamicCache`` (36 layers of
``[1,8,256,128]`` keys/values) that the untouched parent decode loop consumes.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

try:  # worker always has transformers; kept defensive for import-time safety
    from transformers.cache_utils import DynamicCache
    from transformers.modeling_outputs import CausalLMOutputWithPast
except Exception:  # pragma: no cover
    DynamicCache = None
    CausalLMOutputWithPast = None

try:
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except Exception:  # pragma: no cover
    _HAS_TRITON = False

PREFILL_CALL = "900c23e99c808d2c9bda215a"

_KNOWN_KWARGS = frozenset(
    {
        "input_ids",
        "attention_mask",
        "position_ids",
        "past_key_values",
        "inputs_embeds",
        "labels",
        "use_cache",
        "logits_to_keep",
    }
)


if _HAS_TRITON:

    @triton.jit
    def _rmsnorm_kernel(x_ptr, out_ptr, w_ptr, stride_x, stride_o, N, eps, BLOCK: tl.constexpr):
        row = tl.program_id(0).to(tl.int64)
        xp = x_ptr + row * stride_x
        op = out_ptr + row * stride_o
        acc = tl.zeros((BLOCK,), dtype=tl.float32)
        for off in range(0, N, BLOCK):
            idx = off + tl.arange(0, BLOCK)
            m = idx < N
            x = tl.load(xp + idx, mask=m, other=0.0).to(tl.float32)
            acc += x * x
        var = tl.sum(acc, axis=0) / N
        rstd = tl.rsqrt(var + eps)
        for off in range(0, N, BLOCK):
            idx = off + tl.arange(0, BLOCK)
            m = idx < N
            x = tl.load(xp + idx, mask=m, other=0.0).to(tl.float32)
            w = tl.load(w_ptr + idx, mask=m, other=0.0).to(tl.float32)
            xhat = (x * rstd).to(tl.bfloat16).to(tl.float32)
            tl.store(op + idx, (w * xhat).to(tl.bfloat16), mask=m)

    @triton.jit
    def _add_rmsnorm_kernel(
        a_ptr, b_ptr, h_ptr, out_ptr, w_ptr, stride_a, stride_b, stride_h, stride_o, N, eps, BLOCK: tl.constexpr
    ):
        # h = round_bf16(a + b) written to h_ptr; out = RMSNorm(h) * w to out_ptr.
        # h is recomputed (not reloaded) in the second pass to keep memory ops ordered.
        row = tl.program_id(0).to(tl.int64)
        ap = a_ptr + row * stride_a
        bp = b_ptr + row * stride_b
        hp = h_ptr + row * stride_h
        op = out_ptr + row * stride_o
        acc = tl.zeros((BLOCK,), dtype=tl.float32)
        for off in range(0, N, BLOCK):
            idx = off + tl.arange(0, BLOCK)
            m = idx < N
            a = tl.load(ap + idx, mask=m, other=0.0).to(tl.float32)
            b = tl.load(bp + idx, mask=m, other=0.0).to(tl.float32)
            t = (a + b).to(tl.bfloat16)
            tl.store(hp + idx, t, mask=m)
            tf = t.to(tl.float32)
            acc += tf * tf
        var = tl.sum(acc, axis=0) / N
        rstd = tl.rsqrt(var + eps)
        for off in range(0, N, BLOCK):
            idx = off + tl.arange(0, BLOCK)
            m = idx < N
            a = tl.load(ap + idx, mask=m, other=0.0).to(tl.float32)
            b = tl.load(bp + idx, mask=m, other=0.0).to(tl.float32)
            t = (a + b).to(tl.bfloat16)
            tf = t.to(tl.float32)
            w = tl.load(w_ptr + idx, mask=m, other=0.0).to(tl.float32)
            xhat = (tf * rstd).to(tl.bfloat16).to(tl.float32)
            tl.store(op + idx, (w * xhat).to(tl.bfloat16), mask=m)

    @triton.jit
    def _qkv_norm_rope_kernel(
        qkv_ptr,
        q_ptr,
        k_ptr,
        v_ptr,
        cos_ptr,
        sin_ptr,
        wqn_ptr,
        wkn_ptr,
        S,
        eps,
        QH: tl.constexpr,
        KH: tl.constexpr,
        D: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        # BLOCK == D // 2. qkv row layout: [S, QH + 2*KH, D] (GEMM output of the
        # packed [q; k; v] weight). Outputs are head-major [., heads, S, D]:
        # q [1, QH, S, D], k/v [1, KH, S, D] written straight into the cache
        # buffers, replicating q_norm -> k_norm -> rotary (NeoX rotate_half)
        # with the parent's bf16 rounding chain.
        pid = tl.program_id(0)
        nheads = QH + 2 * KH
        head = pid % nheads
        s = pid // nheads
        half: tl.constexpr = D // 2
        j = tl.arange(0, BLOCK)
        base = qkv_ptr + (s * nheads + head) * D
        x1 = tl.load(base + j).to(tl.float32)
        x2 = tl.load(base + half + j).to(tl.float32)
        c1 = tl.load(cos_ptr + s * D + j).to(tl.float32)
        c2 = tl.load(cos_ptr + s * D + half + j).to(tl.float32)
        sn1 = tl.load(sin_ptr + s * D + j).to(tl.float32)
        sn2 = tl.load(sin_ptr + s * D + half + j).to(tl.float32)
        if head < QH:
            var = (tl.sum(x1 * x1, axis=0) + tl.sum(x2 * x2, axis=0)) / D
            rstd = tl.rsqrt(var + eps)
            w1 = tl.load(wqn_ptr + j).to(tl.float32)
            w2 = tl.load(wqn_ptr + half + j).to(tl.float32)
            x1n = (w1 * (x1 * rstd).to(tl.bfloat16).to(tl.float32)).to(tl.bfloat16).to(tl.float32)
            x2n = (w2 * (x2 * rstd).to(tl.bfloat16).to(tl.float32)).to(tl.bfloat16).to(tl.float32)
            o1 = (
                (x1n * c1).to(tl.bfloat16).to(tl.float32) + ((-x2n) * sn1).to(tl.bfloat16).to(tl.float32)
            ).to(tl.bfloat16)
            o2 = (
                (x2n * c2).to(tl.bfloat16).to(tl.float32) + (x1n * sn2).to(tl.bfloat16).to(tl.float32)
            ).to(tl.bfloat16)
            op = q_ptr + (head.to(tl.int64) * S + s) * D
            tl.store(op + j, o1)
            tl.store(op + half + j, o2)
        elif head < QH + KH:
            var = (tl.sum(x1 * x1, axis=0) + tl.sum(x2 * x2, axis=0)) / D
            rstd = tl.rsqrt(var + eps)
            w1 = tl.load(wkn_ptr + j).to(tl.float32)
            w2 = tl.load(wkn_ptr + half + j).to(tl.float32)
            x1n = (w1 * (x1 * rstd).to(tl.bfloat16).to(tl.float32)).to(tl.bfloat16).to(tl.float32)
            x2n = (w2 * (x2 * rstd).to(tl.bfloat16).to(tl.float32)).to(tl.bfloat16).to(tl.float32)
            o1 = (
                (x1n * c1).to(tl.bfloat16).to(tl.float32) + ((-x2n) * sn1).to(tl.bfloat16).to(tl.float32)
            ).to(tl.bfloat16)
            o2 = (
                (x2n * c2).to(tl.bfloat16).to(tl.float32) + (x1n * sn2).to(tl.bfloat16).to(tl.float32)
            ).to(tl.bfloat16)
            op = k_ptr + ((head - QH).to(tl.int64) * S + s) * D
            tl.store(op + j, o1)
            tl.store(op + half + j, o2)
        else:
            op = v_ptr + ((head - QH - KH).to(tl.int64) * S + s) * D
            tl.store(op + j, x1.to(tl.bfloat16))
            tl.store(op + half + j, x2.to(tl.bfloat16))

    @triton.jit
    def _silu_mul_kernel(g_ptr, u_ptr, o_ptr, stride_g, stride_u, stride_o, N, BLOCK: tl.constexpr):
        row = tl.program_id(0).to(tl.int64)
        gp = g_ptr + row * stride_g
        up = u_ptr + row * stride_u
        op = o_ptr + row * stride_o
        for off in range(0, N, BLOCK):
            idx = off + tl.arange(0, BLOCK)
            m = idx < N
            g = tl.load(gp + idx, mask=m, other=0.0).to(tl.float32)
            u = tl.load(up + idx, mask=m, other=0.0).to(tl.float32)
            s = (g * tl.sigmoid(g)).to(tl.bfloat16).to(tl.float32)
            tl.store(op + idx, (s * u).to(tl.bfloat16), mask=m)


def _rms_torch(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    """Bit-compatible re-implementation of Qwen3RMSNorm.forward."""
    input_dtype = x.dtype
    xf = x.to(torch.float32)
    variance = xf.pow(2).mean(-1, keepdim=True)
    xf = xf * torch.rsqrt(variance + eps)
    return w * xf.to(input_dtype)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _rope_torch(q, k, cos, sin):
    cu = cos.unsqueeze(1)
    su = sin.unsqueeze(1)
    q_embed = (q * cu) + (_rotate_half(q) * su)
    k_embed = (k * cu) + (_rotate_half(k) * su)
    return q_embed, k_embed


class _SimpleCache:
    """Duck-typed DynamicCache fallback if transformers cache_utils is unusable."""

    def __init__(self, ks, vs):
        self._k = list(ks)
        self._v = list(vs)

    def get_seq_length(self, layer_idx: int = 0) -> int:
        t = self._k[layer_idx] if layer_idx < len(self._k) else None
        if t is None or t.numel() == 0:
            return 0
        return t.shape[-2]

    def update(self, key_states, value_states, layer_idx: int, *args, **kwargs):
        if layer_idx < len(self._k):
            self._k[layer_idx] = torch.cat([self._k[layer_idx], key_states], dim=-2)
            self._v[layer_idx] = torch.cat([self._v[layer_idx], value_states], dim=-2)
        else:
            self._k.append(key_states)
            self._v.append(value_states)
        return self._k[layer_idx], self._v[layer_idx]


class _FusedPrefill:
    def __init__(self, model, parent, route, call_id, state):
        self.parent = parent
        self._route = route
        self.call_id = call_id
        cfg = model.config
        self.cfg = cfg
        self.L = cfg.num_hidden_layers
        self.H = cfg.hidden_size
        self.QH = cfg.num_attention_heads
        self.KH = cfg.num_key_value_heads
        self.D = getattr(cfg, "head_dim", None) or self.H // self.QH
        self.I = cfg.intermediate_size
        self.eps = float(cfg.rms_norm_eps)
        self.vocab = cfg.vocab_size
        self.scale = float(self.D) ** -0.5
        self.seq = int(state.get("prompt_tokens", 256)) if isinstance(state, dict) else 256
        self.dev = next(model.parameters()).device
        self.dtype = next(model.parameters()).dtype
        self.mode = "init"
        self.graph = None
        self.use_tr = False
        with torch.inference_mode(False):
            self._pack_weights(model)
            self._rope_tables(model)
        self.st_in = None
        self.st_k = None
        self.st_v = None
        self.st_logits = None

    # ------------------------------------------------------------------ setup
    def _rebind(self, module, name, view):
        old = getattr(module, name)
        newp = torch.nn.Parameter(view.detach(), requires_grad=old.requires_grad)
        module._parameters[name] = newp

    def _pack_weights(self, model):
        m = model.model
        self.embed_w = m.embed_tokens.weight
        self.lm_w = model.lm_head.weight
        self.final_w = m.norm.weight
        self.w_qkv_t = []
        self.w_gu_t = []
        self.w_o_t = []
        self.w_d_t = []
        self.w_in = []
        self.w_post = []
        self.w_qn = []
        self.w_kn = []
        self._packed_keepalive = []
        q_rows = self.QH * self.D
        kv_rows = self.KH * self.D
        for lyr in m.layers:
            at = lyr.self_attn
            mlp = lyr.mlp
            qkv = torch.cat(
                [at.q_proj.weight.detach(), at.k_proj.weight.detach(), at.v_proj.weight.detach()], dim=0
            ).contiguous()
            gu = torch.cat(
                [mlp.gate_proj.weight.detach(), mlp.up_proj.weight.detach()], dim=0
            ).contiguous()
            # Rebind the parent modules to storage-sharing views so the untouched
            # decode path (and any parent-path fallback) reads identical weights.
            self._rebind(at.q_proj, "weight", qkv[:q_rows])
            self._rebind(at.k_proj, "weight", qkv[q_rows : q_rows + kv_rows])
            self._rebind(at.v_proj, "weight", qkv[q_rows + kv_rows :])
            self._rebind(mlp.gate_proj, "weight", gu[: self.I])
            self._rebind(mlp.up_proj, "weight", gu[self.I :])
            self._packed_keepalive.append((qkv, gu))
            self.w_qkv_t.append(qkv.t())
            self.w_gu_t.append(gu.t())
            self.w_o_t.append(at.o_proj.weight.t())
            self.w_d_t.append(mlp.down_proj.weight.t())
            self.w_in.append(lyr.input_layernorm.weight)
            self.w_post.append(lyr.post_attention_layernorm.weight)
            self.w_qn.append(at.q_norm.weight)
            self.w_kn.append(at.k_norm.weight)
        self.lm_w_t = self.lm_w.t()

    def _rope_tables(self, model):
        # Derive cos/sin from the module's own inv_freq buffer, matching the
        # parent's fp32 outer-product math and bf16 cast bit-for-bit.
        rot = model.model.rotary_emb
        inv_freq = rot.inv_freq.detach().to(torch.float32).reshape(-1)
        scaling = float(getattr(rot, "attention_scaling", 1.0) or 1.0)
        pos = torch.arange(self.seq, dtype=torch.float32, device=self.dev)
        freqs = torch.outer(pos, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.cos = (emb.cos() * scaling).to(self.dtype)[None].contiguous()
        self.sin = (emb.sin() * scaling).to(self.dtype)[None].contiguous()

    # ------------------------------------------------------------- computation
    def _compute(self, ids, ks, vs, logits_out, use_tr):
        S, H, D, QH, KH, I, L = self.seq, self.H, self.D, self.QH, self.KH, self.I, self.L
        nheads = QH + 2 * KH
        h = F.embedding(ids, self.embed_w).view(S, H)
        x = torch.empty_like(h)
        if use_tr:
            _rmsnorm_kernel[(S,)](h, x, self.w_in[0], H, H, H, self.eps, BLOCK=512)
        else:
            x = _rms_torch(h, self.w_in[0], self.eps)
        for i in range(L):
            qkv = torch.mm(x, self.w_qkv_t[i])
            if use_tr:
                q = torch.empty((1, QH, S, D), dtype=torch.bfloat16, device=self.dev)
                _qkv_norm_rope_kernel[(S * nheads,)](
                    qkv,
                    q,
                    ks[i],
                    vs[i],
                    self.cos,
                    self.sin,
                    self.w_qn[i],
                    self.w_kn[i],
                    S,
                    self.eps,
                    QH=QH,
                    KH=KH,
                    D=D,
                    BLOCK=D // 2,
                )
            else:
                qkv4 = qkv.view(1, S, nheads, D)
                q_, k_, v_ = qkv4.split([QH, KH, KH], dim=2)
                q_ = _rms_torch(q_, self.w_qn[i], self.eps).transpose(1, 2)
                k_ = _rms_torch(k_, self.w_kn[i], self.eps).transpose(1, 2)
                v_ = v_.transpose(1, 2)
                q_, k_ = _rope_torch(q_, k_, self.cos, self.sin)
                ks[i].copy_(k_)
                vs[i].copy_(v_)
                q = q_
            attn = F.scaled_dot_product_attention(
                q,
                ks[i],
                vs[i],
                attn_mask=None,
                dropout_p=0.0,
                scale=self.scale,
                is_causal=True,
                enable_gqa=True,
            )
            a2 = attn.transpose(1, 2).reshape(S, QH * D)
            o = torch.mm(a2, self.w_o_t[i])
            if use_tr:
                h1 = torch.empty_like(h)
                x2 = torch.empty_like(h)
                _add_rmsnorm_kernel[(S,)](h, o, h1, x2, self.w_post[i], H, H, H, H, H, self.eps, BLOCK=512)
            else:
                h1 = h + o
                x2 = _rms_torch(h1, self.w_post[i], self.eps)
            gu = torch.mm(x2, self.w_gu_t[i])
            g = gu[:, :I]
            u = gu[:, I:]
            if use_tr:
                act = torch.empty((S, I), dtype=self.dtype, device=self.dev)
                _silu_mul_kernel[(S,)](g, u, act, 2 * I, 2 * I, I, I, BLOCK=1024)
            else:
                act = F.silu(g) * u
            d = torch.mm(act, self.w_d_t[i])
            if i + 1 < L:
                if use_tr:
                    hn = torch.empty_like(h1)
                    xn = torch.empty_like(h1)
                    _add_rmsnorm_kernel[(S,)](h1, d, hn, xn, self.w_in[i + 1], H, H, H, H, H, self.eps, BLOCK=512)
                else:
                    hn = h1 + d
                    xn = _rms_torch(hn, self.w_in[i + 1], self.eps)
                h, x = hn, xn
            else:
                h = h1 + d
        last = h[S - 1 : S, :]
        if use_tr:
            ln = torch.empty((1, H), dtype=self.dtype, device=self.dev)
            _rmsnorm_kernel[(1,)](last, ln, self.final_w, H, H, H, self.eps, BLOCK=512)
        else:
            ln = _rms_torch(last, self.final_w, self.eps)
        torch.mm(ln, self.lm_w_t, out=logits_out.view(1, self.vocab))

    # ------------------------------------------------------------------ cache
    def _build_cache(self, ks, vs):
        if DynamicCache is not None:
            try:
                cache = DynamicCache(config=self.cfg)
                layers = cache.layers
                if len(layers) != self.L:
                    raise RuntimeError("unexpected cache layer count")
                for i, lyr in enumerate(layers):
                    lyr.keys = ks[i]
                    lyr.values = vs[i]
                    lyr.is_initialized = True
                    lyr.dtype = ks[i].dtype
                    lyr.device = ks[i].device
                return cache
            except Exception:
                pass
            try:
                cache = DynamicCache()
                for i in range(self.L):
                    cache.update(ks[i], vs[i], i)
                return cache
            except Exception:
                pass
        return _SimpleCache(ks, vs)

    def _outputs(self, ks, vs, logits):
        return CausalLMOutputWithPast(
            loss=None,
            logits=logits.clone(),
            past_key_values=self._build_cache(ks, vs),
        )

    # -------------------------------------------------------------- invocation
    def _invoke_eager(self, ids):
        S, KH, D = self.seq, self.KH, self.D
        ks = [torch.empty((1, KH, S, D), dtype=self.dtype, device=self.dev) for _ in range(self.L)]
        vs = [torch.empty((1, KH, S, D), dtype=self.dtype, device=self.dev) for _ in range(self.L)]
        logits = torch.empty((1, 1, self.vocab), dtype=self.dtype, device=self.dev)
        self._compute(ids, ks, vs, logits, self.use_tr)
        self._route(self.call_id, "fused_prefill_eager_composite")
        return self._outputs(ks, vs, logits)

    def _initialize(self, ids):
        S, KH, D = self.seq, self.KH, self.D
        torch.cuda.synchronize()
        self.st_in = torch.empty((1, S), dtype=torch.long, device=self.dev)
        self.st_in.copy_(ids)
        self.st_k = [torch.empty((1, KH, S, D), dtype=self.dtype, device=self.dev) for _ in range(self.L)]
        self.st_v = [torch.empty((1, KH, S, D), dtype=self.dtype, device=self.dev) for _ in range(self.L)]
        self.st_logits = torch.empty((1, 1, self.vocab), dtype=self.dtype, device=self.dev)
        use_tr = _HAS_TRITON and self.dtype == torch.bfloat16
        if use_tr:
            try:
                self._compute(self.st_in, self.st_k, self.st_v, self.st_logits, True)
                torch.cuda.synchronize()
            except Exception:
                use_tr = False
        self.use_tr = use_tr
        if not use_tr:
            self._compute(self.st_in, self.st_k, self.st_v, self.st_logits, False)
            torch.cuda.synchronize()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                self._compute(self.st_in, self.st_k, self.st_v, self.st_logits, self.use_tr)
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        captured = False
        try:
            with torch.cuda.graph(graph):
                self._compute(self.st_in, self.st_k, self.st_v, self.st_logits, self.use_tr)
            captured = True
        except Exception:
            captured = False
        if captured:
            self.graph = graph
            self.mode = "graph"
            self.graph.replay()
            self._route(self.call_id, "fused_prefill_graph_replay")
            return self._outputs(self.st_k, self.st_v, self.st_logits)
        self.mode = "eager"
        return self._invoke_eager(ids)

    def _guard(self, args, kwargs):
        if any(k not in _KNOWN_KWARGS for k in kwargs):
            return None
        n = len(args)
        ids = kwargs.get("input_ids", args[0] if n > 0 else None)
        am = kwargs.get("attention_mask", args[1] if n > 1 else None)
        pos = kwargs.get("position_ids", args[2] if n > 2 else None)
        past = kwargs.get("past_key_values", args[3] if n > 3 else None)
        emb = kwargs.get("inputs_embeds", args[4] if n > 4 else None)
        labels = kwargs.get("labels", args[5] if n > 5 else None)
        uc = kwargs.get("use_cache", args[6] if n > 6 else None)
        l2k = kwargs.get("logits_to_keep", args[7] if n > 7 else None)
        if ids is None or am is not None or pos is not None or past is not None:
            return None
        if emb is not None or labels is not None:
            return None
        if not (uc is True or uc is None):
            return None
        if l2k != 1:
            return None
        if not (
            torch.is_tensor(ids)
            and ids.dim() == 2
            and tuple(ids.shape) == (1, self.seq)
            and ids.dtype == torch.long
            and ids.is_cuda
            and ids.device == self.dev
        ):
            return None
        return ids

    def __call__(self, *args, **kwargs):
        try:
            ids = self._guard(args, kwargs)
        except Exception:
            ids = None
        if ids is None:
            self._route(self.call_id, "parent_signature_fallback")
            return self.parent(*args, **kwargs)
        mode = self.mode
        try:
            if mode == "graph":
                self.st_in.copy_(ids, non_blocking=True)
                self.graph.replay()
                self._route(self.call_id, "fused_prefill_graph_replay")
                return self._outputs(self.st_k, self.st_v, self.st_logits)
            if mode == "eager":
                return self._invoke_eager(ids)
            return self._initialize(ids)
        except Exception:
            pass
        # Degrade one level and retry once within this invocation.
        try:
            if mode in ("init", "graph"):
                self.graph = None
                self.mode = "eager"
                out = self._invoke_eager(ids)
                self._route(self.call_id, "fused_prefill_capture_failed_eager")
                return out
        except Exception:
            pass
        self.mode = "parent"
        self._route(self.call_id, "parent_runtime_fallback")
        return self.parent(*args, **kwargs)


def build(context):
    record_route = context.get("record_route") or (lambda *_: None)
    modules = context.get("modules") or {}
    models = context.get("models") or {}
    references = context.get("references") or {}
    state = context.get("state") or {}
    call_id = PREFILL_CALL
    model = modules.get(call_id) or models.get("model")
    parent = references.get(call_id)
    if model is None or parent is None:
        def _unavailable(*args, **kwargs):
            record_route(call_id, "setup_failure_missing_model_or_parent")
            if parent is None:
                raise RuntimeError("fused prefill: no parent reference available")
            return parent(*args, **kwargs)

        return {call_id: _unavailable}
    try:
        engine = _FusedPrefill(model, parent, record_route, call_id, state)
    except Exception:
        def _setup_failed(*args, **kwargs):
            record_route(call_id, "setup_failure_parent_fallback")
            return parent(*args, **kwargs)

        return {call_id: _setup_failed}
    return {call_id: engine}
