from __future__ import annotations

import torch
import torch.nn.functional as F
from transformers.modeling_outputs import CausalLMOutputWithPast

try:
    import triton
    import triton.language as tl
    _TRITON_OK = True
except Exception:
    _TRITON_OK = False

try:
    from transformers.models.qwen3.modeling_qwen3 import (
        apply_rotary_pos_emb as _apply_rope,
    )
except Exception:
    _apply_rope = None

DECODE_MODEL_CALL = "323bcbe006b343648b346af9"
PREFILL_MODEL_CALL = "900c23e99c808d2c9bda215a"

_EXPECTED_DECODE_KWARGS = frozenset(
    {"input_ids", "past_key_values", "use_cache", "logits_to_keep"}
)

_VERIFY_ATOL = 0.012
_VERIFY_RTOL = 0.012
_KV_ATOL = 0.02
_KV_RTOL = 0.02


class _GuardFailure(Exception):
    pass


class _SetupFailure(Exception):
    pass


def _layer_pairs(cache):
    layers = getattr(cache, "layers", None)
    if layers is None:
        return None
    pairs = []
    for layer in layers:
        key = getattr(layer, "keys", None)
        value = getattr(layer, "values", None)
        if not isinstance(key, torch.Tensor) or not isinstance(value, torch.Tensor):
            return None
        pairs.append((key, value))
    return pairs or None


def _view_cache(template, pairs):
    cache = object.__new__(type(template))
    cache.__dict__.update(template.__dict__)
    cloned_layers = []
    for layer, (key, value) in zip(template.layers, pairs):
        clone = object.__new__(type(layer))
        clone.__dict__.update(layer.__dict__)
        clone.keys = key
        clone.values = value
        clone.is_initialized = True
        cloned_layers.append(clone)
    cache.layers = cloned_layers
    return cache


if _TRITON_OK:

    @triton.jit
    def _rmsnorm_kernel(x_ptr, w_ptr, y_ptr, eps, N: tl.constexpr,
                        BLOCK: tl.constexpr):
        sumsq = tl.zeros((BLOCK,), dtype=tl.float32)
        for n0 in range(0, N, BLOCK):
            offs = n0 + tl.arange(0, BLOCK)
            x = tl.load(x_ptr + offs).to(tl.float32)
            sumsq += x * x
        var = tl.sum(sumsq, axis=0) / N
        rstd = tl.math.rsqrt(var + eps)
        for n0 in range(0, N, BLOCK):
            offs = n0 + tl.arange(0, BLOCK)
            x = tl.load(x_ptr + offs).to(tl.float32)
            w = tl.load(w_ptr + offs).to(tl.float32)
            h = (x * rstd).to(tl.bfloat16).to(tl.float32)
            tl.store(y_ptr + offs, (w * h).to(tl.bfloat16))

    @triton.jit
    def _silumul_kernel(g_ptr, u_ptr, out_ptr, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        g = tl.load(g_ptr + offs).to(tl.float32)
        u = tl.load(u_ptr + offs).to(tl.float32)
        s = (g / (1.0 + tl.math.exp(-g))).to(tl.bfloat16).to(tl.float32)
        tl.store(out_ptr + offs, (s * u).to(tl.bfloat16))

    @triton.jit
    def _qkrope_kernel(q_src, k_src, v_src, qnw_ptr, knw_ptr, cos_ptr, sin_ptr,
                       q_dst, km_ptr, vm_ptr, pos,
                       NQ: tl.constexpr, NKV: tl.constexpr,
                       MAXLEN: tl.constexpr, HD: tl.constexpr,
                       EPS: tl.constexpr):
        t = tl.program_id(0)
        d = tl.arange(0, HD)
        half = HD // 2
        if t < NQ:
            src = q_src + t * HD
            w_ptr = qnw_ptr
            dst = q_dst + t * HD
            x = tl.load(src + d).to(tl.float32)
            var = tl.sum(x * x, axis=0) / HD
            rstd = tl.math.rsqrt(var + EPS)
            y = (x * rstd).to(tl.bfloat16).to(tl.float32)
            wn = tl.load(w_ptr + d).to(tl.float32)
            xn = (wn * y).to(tl.bfloat16).to(tl.float32)
            i1 = tl.where(d < half, d + half, d - half)
            xp = tl.load(src + i1).to(tl.float32)
            yp = (xp * rstd).to(tl.bfloat16).to(tl.float32)
            wp = tl.load(w_ptr + i1).to(tl.float32)
            xnp = (wp * yp).to(tl.bfloat16).to(tl.float32)
            rot = tl.where(d < half, -xnp, xnp)
            cos = tl.load(cos_ptr + pos * HD + d).to(tl.float32)
            sin = tl.load(sin_ptr + pos * HD + d).to(tl.float32)
            t1 = (xn * cos).to(tl.bfloat16)
            t2 = (rot * sin).to(tl.bfloat16)
            out = (t1.to(tl.float32) + t2.to(tl.float32)).to(tl.bfloat16)
            tl.store(dst + d, out)
        elif t < NQ + NKV:
            h = t - NQ
            src = k_src + h * HD
            w_ptr = knw_ptr
            dst = km_ptr + h * (MAXLEN * HD) + pos * HD
            x = tl.load(src + d).to(tl.float32)
            var = tl.sum(x * x, axis=0) / HD
            rstd = tl.math.rsqrt(var + EPS)
            y = (x * rstd).to(tl.bfloat16).to(tl.float32)
            wn = tl.load(w_ptr + d).to(tl.float32)
            xn = (wn * y).to(tl.bfloat16).to(tl.float32)
            i1 = tl.where(d < half, d + half, d - half)
            xp = tl.load(src + i1).to(tl.float32)
            yp = (xp * rstd).to(tl.bfloat16).to(tl.float32)
            wp = tl.load(w_ptr + i1).to(tl.float32)
            xnp = (wp * yp).to(tl.bfloat16).to(tl.float32)
            rot = tl.where(d < half, -xnp, xnp)
            cos = tl.load(cos_ptr + pos * HD + d).to(tl.float32)
            sin = tl.load(sin_ptr + pos * HD + d).to(tl.float32)
            t1 = (xn * cos).to(tl.bfloat16)
            t2 = (rot * sin).to(tl.bfloat16)
            out = (t1.to(tl.float32) + t2.to(tl.float32)).to(tl.bfloat16)
            tl.store(dst + d, out)
        else:
            h = t - NQ - NKV
            val = tl.load(v_src + h * HD + d)
            tl.store(vm_ptr + h * (MAXLEN * HD) + pos * HD + d, val)


class _FastDecodeRunner:
    def __init__(self, model, home, prompt_len, steps, build_inputs,
                 decode_parent, prefill_parent):
        self.model = model
        self.home = home
        self.prompt_len = prompt_len
        self.steps = steps
        self.build_inputs = build_inputs
        self.decode_parent = decode_parent
        self.prefill_parent = prefill_parent
        self.fast_enabled = False
        self.disable_reason = "not_built"
        self.mode = None
        self.decode_graphs = []
        self.n_layers = self.n_heads = self.kv_heads = self.head_dim = 0
        self.hidden = self.inter = self.vocab = 0
        self.eps = 0.0
        self.scale = 1.0
        self.groups = 1
        self.batch = 1
        self.max_len = prompt_len + steps
        self.layer_w = []
        self.masters_k = []
        self.masters_v = []
        self.cos_t = None
        self.sin_t = None
        self.template = None
        self.mirror_template = None
        self.static_input = None
        self.logits_static = [None]
        self.logits_flat = [None]
        self.scratch_logits = None

    def _extract(self):
        model = self.model
        cfg = getattr(model, "config", None)
        if cfg is None:
            raise _SetupFailure("config")
        try:
            self.hidden = int(cfg.hidden_size)
            self.n_layers = int(cfg.num_hidden_layers)
            self.n_heads = int(cfg.num_attention_heads)
            self.kv_heads = int(cfg.num_key_value_heads)
            self.head_dim = int(getattr(cfg, "head_dim", 0)
                                or self.hidden // self.n_heads)
            self.inter = int(cfg.intermediate_size)
            self.vocab = int(cfg.vocab_size)
            self.eps = float(cfg.rms_norm_eps)
        except Exception:
            raise _SetupFailure("config_fields")
        if self.n_heads % self.kv_heads != 0:
            raise _SetupFailure("gqa_groups")
        self.groups = self.n_heads // self.kv_heads
        if self.head_dim < 2 or (self.head_dim & (self.head_dim - 1)) != 0:
            raise _SetupFailure("head_dim")
        self.scale = self.head_dim ** -0.5
        base = getattr(model, "model", None)
        if base is None or not getattr(base, "layers", None):
            raise _SetupFailure("model_tree")
        self.base = base
        embed = getattr(base, "embed_tokens", None)
        final_norm = getattr(base, "norm", None)
        lm_head = getattr(model, "lm_head", None)
        if embed is None or final_norm is None or lm_head is None:
            raise _SetupFailure("model_modules")
        self.embed_w = embed.weight
        self.final_w = final_norm.weight
        self.lm_w = lm_head.weight
        self._check_weight(self.embed_w, "embed")
        self._check_weight(self.final_w, "final_norm")
        self._check_weight(self.lm_w, "lm_head")
        if tuple(self.lm_w.shape) != (self.vocab, self.hidden):
            raise _SetupFailure("lm_head_shape")
        if tuple(self.embed_w.shape) != (self.vocab, self.hidden):
            raise _SetupFailure("embed_shape")
        if self.hidden % 512 != 0:
            raise _SetupFailure("hidden_layout")
        if self.inter % 512 != 0:
            raise _SetupFailure("inter_layout")
        for layer in base.layers:
            attn = getattr(layer, "self_attn", None)
            mlp = getattr(layer, "mlp", None)
            if attn is None or mlp is None:
                raise _SetupFailure("layer_modules")
            projs = (getattr(attn, "q_proj", None), getattr(attn, "k_proj", None),
                     getattr(attn, "v_proj", None), getattr(attn, "o_proj", None),
                     getattr(mlp, "gate_proj", None), getattr(mlp, "up_proj", None),
                     getattr(mlp, "down_proj", None))
            if any(proj is None for proj in projs):
                raise _SetupFailure("layer_projections")
            for proj in projs:
                if proj.bias is not None:
                    raise _SetupFailure("linear_bias_unsupported")
            weights = {
                "q": attn.q_proj.weight, "k": attn.k_proj.weight,
                "v": attn.v_proj.weight, "o": attn.o_proj.weight,
                "g": mlp.gate_proj.weight, "u": mlp.up_proj.weight,
                "d": mlp.down_proj.weight,
                "qn": attn.q_norm.weight, "kn": attn.k_norm.weight,
                "ln1": layer.input_layernorm.weight,
                "ln2": layer.post_attention_layernorm.weight,
            }
            for name, weight in weights.items():
                self._check_weight(weight, name)
            if tuple(weights["q"].shape) != (self.n_heads * self.head_dim, self.hidden):
                raise _SetupFailure("q_shape")
            if tuple(weights["k"].shape) != (self.kv_heads * self.head_dim, self.hidden):
                raise _SetupFailure("k_shape")
            if tuple(weights["v"].shape) != (self.kv_heads * self.head_dim, self.hidden):
                raise _SetupFailure("v_shape")
            if tuple(weights["o"].shape) != (self.hidden, self.n_heads * self.head_dim):
                raise _SetupFailure("o_shape")
            if tuple(weights["g"].shape) != (self.inter, self.hidden):
                raise _SetupFailure("g_shape")
            if tuple(weights["u"].shape) != (self.inter, self.hidden):
                raise _SetupFailure("u_shape")
            if tuple(weights["d"].shape) != (self.hidden, self.inter):
                raise _SetupFailure("d_shape")
            if tuple(weights["qn"].shape) != (self.head_dim,):
                raise _SetupFailure("qn_shape")
            if tuple(weights["kn"].shape) != (self.head_dim,):
                raise _SetupFailure("kn_shape")
            self.layer_w.append(weights)
        rotary = getattr(base, "rotary_emb", None)
        if rotary is None:
            rotary = getattr(base.layers[0].self_attn, "rotary_emb", None)
        if rotary is None:
            raise _SetupFailure("rotary_emb_unavailable")
        try:
            dummy = torch.zeros((1, 1, self.hidden), dtype=torch.bfloat16,
                                device=self.home)
            pos_ids = torch.arange(self.max_len, device=self.home).unsqueeze(0)
            cos, sin = rotary(dummy, pos_ids)
        except Exception:
            raise _SetupFailure("rotary_emb_call")
        try:
            self.cos_t = cos.reshape(self.max_len, self.head_dim).contiguous()
            self.sin_t = sin.reshape(self.max_len, self.head_dim).contiguous()
        except Exception:
            raise _SetupFailure("rope_layout")
        if self.cos_t.dtype != torch.bfloat16 or self.sin_t.dtype != torch.bfloat16:
            raise _SetupFailure("rope_dtype")
        self.q_rows = self.n_heads * self.head_dim
        self.kv_rows = self.kv_heads * self.head_dim

    def _check_weight(self, weight, name):
        if not isinstance(weight, torch.Tensor):
            raise _SetupFailure("weight_%s" % name)
        if weight.dtype != torch.bfloat16 or weight.device != self.home:
            raise _SetupFailure("weight_%s_layout" % name)
        if not weight.is_contiguous():
            raise _SetupFailure("weight_%s_contig" % name)

    def _fused_step(self, step, logits_out):
        home = self.home
        h = F.embedding(self.static_input, self.embed_w).view(1, self.hidden)
        pos = self.prompt_len + step - 1
        seq = pos + 1
        for idx in range(self.n_layers):
            w = self.layer_w[idx]
            y1 = torch.empty((1, self.hidden), dtype=torch.bfloat16, device=home)
            _rmsnorm_kernel[(1,)](h.view(-1), w["ln1"], y1.view(-1), self.eps,
                                  N=self.hidden, BLOCK=512, num_warps=4)
            q = F.linear(y1, w["q"])
            k = F.linear(y1, w["k"])
            v = F.linear(y1, w["v"])
            qrope = torch.empty((1, self.q_rows), dtype=torch.bfloat16, device=home)
            _qkrope_kernel[(self.n_heads + 2 * self.kv_heads,)](
                q, k, v, w["qn"], w["kn"], self.cos_t, self.sin_t, qrope,
                self.masters_k[idx], self.masters_v[idx], pos,
                NQ=self.n_heads, NKV=self.kv_heads,
                MAXLEN=self.max_len, HD=self.head_dim, EPS=self.eps, num_warps=2)
            att = F.scaled_dot_product_attention(
                qrope.view(1, self.n_heads, 1, self.head_dim),
                self.masters_k[idx][:, :, :seq, :],
                self.masters_v[idx][:, :, :seq, :],
                attn_mask=None, dropout_p=0.0, is_causal=False,
                scale=self.scale, enable_gqa=True)
            attc = att.transpose(1, 2).contiguous().view(1, self.q_rows)
            x1 = F.linear(attc, w["o"])
            x1 = x1 + h
            y2 = torch.empty((1, self.hidden), dtype=torch.bfloat16, device=home)
            _rmsnorm_kernel[(1,)](x1.view(-1), w["ln2"], y2.view(-1), self.eps,
                                  N=self.hidden, BLOCK=512, num_warps=4)
            g = F.linear(y2, w["g"])
            u = F.linear(y2, w["u"])
            act = torch.empty((1, self.inter), dtype=torch.bfloat16, device=home)
            _silumul_kernel[(self.inter // 512,)](g, u, act, BLOCK=512,
                                                  num_warps=4)
            d = F.linear(act, w["d"])
            h = d + x1
        yf = torch.empty((1, self.hidden), dtype=torch.bfloat16, device=home)
        _rmsnorm_kernel[(1,)](h.view(-1), self.final_w, yf.view(-1), self.eps,
                              N=self.hidden, BLOCK=512, num_warps=4)
        logits = F.linear(yf, self.lm_w)
        logits_out.copy_(logits.view(-1))
        return logits_out

    def _exact_step(self, step, logits_out):
        if _apply_rope is None:
            raise _SetupFailure("rope_import")
        base = self.base
        h = base.embed_tokens(self.static_input)
        pos = self.prompt_len + step - 1
        seq = pos + 1
        cos = self.cos_t[pos:pos + 1].unsqueeze(0)
        sin = self.sin_t[pos:pos + 1].unsqueeze(0)
        shape = (1, 1, -1, self.head_dim)
        for idx, layer in enumerate(base.layers):
            y1 = layer.input_layernorm(h)
            attn = layer.self_attn
            q = attn.q_norm(attn.q_proj(y1).view(shape)).transpose(1, 2)
            k = attn.k_norm(attn.k_proj(y1).view(shape)).transpose(1, 2)
            v = attn.v_proj(y1).view(shape).transpose(1, 2)
            q, k = _apply_rope(q, k, cos, sin)
            self.masters_k[idx][:, :, pos:pos + 1, :].copy_(k)
            self.masters_v[idx][:, :, pos:pos + 1, :].copy_(v)
            att = F.scaled_dot_product_attention(
                q, self.masters_k[idx][:, :, :seq, :],
                self.masters_v[idx][:, :, :seq, :],
                attn_mask=None, dropout_p=0.0, is_causal=False,
                scale=self.scale, enable_gqa=True)
            att = att.transpose(1, 2).contiguous().view(1, 1, -1)
            h = h + attn.o_proj(att)
            y2 = layer.post_attention_layernorm(h)
            mlp = layer.mlp
            act = mlp.act_fn(mlp.gate_proj(y2)) * mlp.up_proj(y2)
            h = h + mlp.down_proj(act)
        h = base.norm(h)
        logits = self.model.lm_head(h)
        logits_out.copy_(logits.view(-1))
        return logits_out

    def _views(self, length):
        return [
            (key[:, :, :length, :], value[:, :, :length, :])
            for key, value in zip(self.masters_k, self.masters_v)
        ]

    def _refresh(self, pairs, seq):
        expected = (self.batch, self.kv_heads, seq, self.head_dim)
        for master_k, master_v, (key, value) in zip(self.masters_k, self.masters_v,
                                                    pairs):
            if tuple(key.shape) != expected or key.dtype != torch.bfloat16:
                raise _GuardFailure("cache_shape")
            if key.device != self.home or value.device != self.home:
                raise _GuardFailure("cache_device")
            master_k[:, :, :seq, :].copy_(key, non_blocking=True)
            master_v[:, :, :seq, :].copy_(value, non_blocking=True)

    def _is_master_view(self, pairs, seq):
        expected = (self.batch, self.kv_heads, seq, self.head_dim)
        for idx in range(self.n_layers):
            key, value = pairs[idx]
            master_k = self.masters_k[idx]
            master_v = self.masters_v[idx]
            if tuple(key.shape) != expected or tuple(value.shape) != expected:
                return False
            if key.data_ptr() != master_k.data_ptr():
                return False
            if value.data_ptr() != master_v.data_ptr():
                return False
        return True

    def _resolve_step(self, past):
        pairs = _layer_pairs(past)
        if pairs is None or len(pairs) != self.n_layers:
            raise _GuardFailure("cache_layers")
        try:
            seq = int(past.get_seq_length())
        except Exception as error:
            raise _GuardFailure("seq_length") from error
        if int(pairs[0][0].shape[-2]) != seq:
            raise _GuardFailure("cache_length_incoherent")
        if seq < self.prompt_len or seq > self.prompt_len + self.steps - 1:
            raise _GuardFailure("cache_length")
        step = seq - self.prompt_len + 1
        if self._is_master_view(pairs, seq):
            return step, False
        self._refresh(pairs, seq)
        return step, True

    def _fresh_mirror(self, length):
        return _view_cache(self.mirror_template, self._views(length))

    def _parent_prefill(self, inputs):
        return self.prefill_parent(input_ids=inputs, use_cache=True,
                                   logits_to_keep=1)

    def _parent_decode(self, token, cache):
        return self.decode_parent(input_ids=token, past_key_values=cache,
                                  use_cache=True, logits_to_keep=1)

    def _repl_greedy_chain(self, prefill_repl, decode_repl, inputs):
        out = prefill_repl(input_ids=inputs, use_cache=True, logits_to_keep=1)
        logits_steps = [out.logits[:, -1]]
        token = out.logits[:, -1].argmax(dim=-1, keepdim=True)
        tokens = [token]
        cache = out.past_key_values
        for _ in range(self.steps):
            out = decode_repl(input_ids=token, past_key_values=cache,
                              use_cache=True, logits_to_keep=1)
            cache = out.past_key_values
            token = out.logits[:, -1].argmax(dim=-1, keepdim=True)
            tokens.append(token)
            logits_steps.append(out.logits[:, -1])
        return torch.cat(tokens, dim=1), torch.stack(logits_steps, dim=1)

    def _check_logits(self, mine, reference, what):
        torch.testing.assert_close(
            mine.float(), reference.float(),
            atol=_VERIFY_ATOL, rtol=_VERIFY_RTOL,
            msg="decode logits mismatch (%s)" % what)

    def setup(self, mode, prefill_repl, decode_repl):
        self.mode = mode
        self._extract()
        interior = self._fused_step if mode == "fused" else self._exact_step
        if mode == "fused" and not _TRITON_OK:
            raise _SetupFailure("triton_unavailable")
        current = torch.cuda.current_stream()
        stream = torch.cuda.Stream(device=self.home)

        prefill_out = self._parent_prefill(self.build_inputs)
        template = getattr(prefill_out, "past_key_values", None)
        pairs = _layer_pairs(template) if template is not None else None
        if pairs is None or len(pairs) != self.n_layers:
            raise _SetupFailure("prefill_cache_layers")
        if not isinstance(getattr(prefill_out, "logits", None), torch.Tensor):
            raise _SetupFailure("prefill_logits")
        self.batch = int(self.build_inputs.shape[0])
        expected_kv = (self.batch, self.kv_heads, self.prompt_len, self.head_dim)
        for key, value in pairs:
            if tuple(key.shape) != expected_kv or key.dtype != torch.bfloat16:
                raise _SetupFailure("prefill_kv_shape")
            if key.device != self.home:
                raise _SetupFailure("prefill_kv_device")
        self.template = template
        kv_shape = (self.batch, self.kv_heads, self.max_len, self.head_dim)
        self.masters_k = [torch.empty(kv_shape, dtype=torch.bfloat16,
                                      device=self.home)
                          for _ in range(self.n_layers)]
        self.masters_v = [torch.empty(kv_shape, dtype=torch.bfloat16,
                                      device=self.home)
                          for _ in range(self.n_layers)]
        self._refresh(pairs, self.prompt_len)
        self.mirror_template = _view_cache(self.template,
                                           self._views(self.prompt_len))
        self.static_input = torch.zeros((self.batch, 1), dtype=torch.int64,
                                        device=self.home)
        self.scratch_logits = torch.empty((self.batch, 1, self.vocab),
                                          dtype=torch.bfloat16, device=self.home)
        for _ in range(self.steps):
            static = torch.empty((self.batch, 1, self.vocab),
                                 dtype=torch.bfloat16, device=self.home)
            self.logits_static.append(static)
            self.logits_flat.append(static.view(-1))

        oracle_cache = template
        token = prefill_out.logits[:, -1].argmax(dim=-1, keepdim=True)
        oracle_tokens = [token]
        oracle_logits = [prefill_out.logits[:, -1]]
        scratch_flat = self.scratch_logits.view(-1)
        pool = None
        graphs = []
        for step in range(1, self.steps + 1):
            oracle_out = self._parent_decode(token, oracle_cache)
            oracle_cache = oracle_out.past_key_values
            token_next = oracle_out.logits[:, -1].argmax(dim=-1, keepdim=True)
            self.static_input.copy_(token)
            stream.wait_stream(current)
            with torch.cuda.stream(stream):
                interior(step, scratch_flat)
            current.wait_stream(stream)
            torch.cuda.synchronize()
            my_token = self.scratch_logits[:, -1].argmax(dim=-1, keepdim=True)
            if not bool(torch.equal(my_token, token_next)):
                raise _SetupFailure("verify_token_step_%d" % step)
            self._check_logits(self.scratch_logits[:, -1],
                               oracle_out.logits[:, -1],
                               "%s_eager_vs_parent_step_%d" % (mode, step))
            graph = torch.cuda.CUDAGraph()
            try:
                with torch.cuda.graph(graph, pool=pool, stream=stream):
                    interior(step, self.logits_flat[step])
            except Exception:
                raise _SetupFailure("%s_capture_step_%d" % (mode, step))
            if pool is None:
                pool = graph.pool()
            graphs.append(graph)
            graph.replay()
            torch.cuda.synchronize()
            if not bool(torch.equal(self.logits_static[step], self.scratch_logits)):
                graph.replay()
                torch.cuda.synchronize()
                if not bool(torch.equal(self.logits_static[step],
                                        self.scratch_logits)):
                    raise _SetupFailure("%s_graph_bitwise_step_%d" % (mode, step))
            token = token_next
            oracle_tokens.append(token)
            oracle_logits.append(oracle_out.logits[:, -1])
        self.decode_graphs = graphs

        final_pairs = _layer_pairs(oracle_cache)
        if final_pairs is None:
            raise _SetupFailure("oracle_cache_layers")
        final_len = self.prompt_len + self.steps
        for idx, (key, value) in enumerate(final_pairs):
            torch.testing.assert_close(
                self.masters_k[idx][:, :, :final_len, :].float(), key.float(),
                atol=_KV_ATOL, rtol=_KV_RTOL, msg="final K layer %d" % idx)
            torch.testing.assert_close(
                self.masters_v[idx][:, :, :final_len, :].float(), value.float(),
                atol=_KV_ATOL, rtol=_KV_RTOL, msg="final V layer %d" % idx)

        tokens_ref = torch.cat(oracle_tokens, dim=1)
        logits_ref = torch.stack(oracle_logits, dim=1)

        tokens_a, logits_a = self._repl_greedy_chain(prefill_repl, decode_repl,
                                                     self.build_inputs)
        if not bool(torch.equal(tokens_a, tokens_ref)):
            raise _SetupFailure("relational_tokens_A")
        for step in range(1, self.steps + 1):
            self._check_logits(logits_a[:, step, :], logits_ref[:, step, :],
                               "%s_chain_A_step_%d" % (mode, step))

        teacher = tokens_ref.roll(1, dims=1).contiguous()
        out = prefill_repl(input_ids=self.build_inputs, use_cache=True,
                           logits_to_keep=1)
        logits_b = [out.logits[:, -1]]
        cache = out.past_key_values
        for index in range(self.steps):
            out = decode_repl(input_ids=teacher[:, index:index + 1].contiguous(),
                              past_key_values=cache, use_cache=True,
                              logits_to_keep=1)
            cache = out.past_key_values
            logits_b.append(out.logits[:, -1])
        logits_b = torch.stack(logits_b, dim=1)
        out = self._parent_prefill(self.build_inputs)
        cache = out.past_key_values
        logits_c = [out.logits[:, -1]]
        for index in range(self.steps):
            out = decode_repl(input_ids=teacher[:, index:index + 1],
                              past_key_values=cache, use_cache=True,
                              logits_to_keep=1)
            cache = out.past_key_values
            logits_c.append(out.logits[:, -1])
        logits_c = torch.stack(logits_c, dim=1)
        for step in range(1, self.steps + 1):
            self._check_logits(logits_b[:, step, :], logits_c[:, step, :],
                               "%s_chain_B_step_%d" % (mode, step))

        tokens_d, logits_d = self._repl_greedy_chain(prefill_repl, decode_repl,
                                                     self.build_inputs)
        if not bool(torch.equal(tokens_a, tokens_d)):
            raise _SetupFailure("relational_tokens_D")
        for step in range(1, self.steps + 1):
            self._check_logits(logits_d[:, step, :], logits_a[:, step, :],
                               "%s_chain_D_step_%d" % (mode, step))

        if self.steps >= 3:
            out = self._parent_prefill(self.build_inputs)
            cache = out.past_key_values
            for index in range(2):
                out = self._parent_decode(tokens_ref[:, index:index + 1], cache)
                cache = out.past_key_values
            ref_out = self._parent_decode(tokens_ref[:, 2:3], cache)
            my_out = decode_repl(input_ids=tokens_ref[:, 2:3].contiguous(),
                                 past_key_values=cache, use_cache=True,
                                 logits_to_keep=1)
            self._check_logits(my_out.logits[:, -1], ref_out.logits[:, -1],
                               "%s_chain_E_absorb" % mode)

        torch.cuda.synchronize()
        self.fast_enabled = True
        self.disable_reason = ""


class _PrefillPassthrough:
    def __init__(self, call_id, parent, record_route):
        self._call_id = call_id
        self._parent = parent
        self._record_route = record_route

    def __call__(self, *args, **kwargs):
        self._record_route(self._call_id, "prefill_parent_passthrough")
        try:
            return self._parent(*args, **kwargs)
        except Exception:
            self._record_route(self._call_id, "prefill_parent_error")
            raise


class _DecodeReplacement:
    def __init__(self, runner_box, call_id, parent, record_route):
        self._runner_box = runner_box
        self._call_id = call_id
        self._parent = parent
        self._record_route = record_route

    def set_runner(self, runner):
        self._runner_box[0] = runner

    def __call__(self, *args, **kwargs):
        runner = self._runner_box[0]
        if runner is None or not runner.fast_enabled:
            self._record_route(self._call_id, "parent_fallback_setup_failure")
            return self._parent(*args, **kwargs)
        try:
            if args or frozenset(kwargs) != _EXPECTED_DECODE_KWARGS:
                raise _GuardFailure("kwargs")
            if kwargs["use_cache"] is not True or kwargs["logits_to_keep"] != 1:
                raise _GuardFailure("flags")
            input_ids = kwargs["input_ids"]
            if (not isinstance(input_ids, torch.Tensor)
                    or tuple(input_ids.shape) != (runner.batch, 1)
                    or input_ids.dtype != torch.int64
                    or input_ids.device != runner.home
                    or tuple(input_ids.stride()) != (1, 1)):
                raise _GuardFailure("input_ids")
            step, refreshed = runner._resolve_step(kwargs["past_key_values"])
            if step > runner.steps:
                raise _GuardFailure("range")
        except _GuardFailure:
            self._record_route(self._call_id, "parent_fallback_signature")
            return self._parent(*args, **kwargs)
        except Exception as error:
            runner.fast_enabled = False
            runner.disable_reason = "guard_error:%s" % type(error).__name__
            self._record_route(self._call_id, "parent_fallback_signature")
            return self._parent(*args, **kwargs)
        try:
            runner.static_input.copy_(input_ids)
            runner.decode_graphs[step - 1].replay()
            logits = runner.logits_static[step].clone()
        except Exception as error:
            runner.fast_enabled = False
            runner.disable_reason = "replay_failure:%s" % type(error).__name__
            self._record_route(self._call_id, "parent_fallback_replay_failure")
            return self._parent(*args, **kwargs)
        self._record_route(
            self._call_id,
            "fast_graph_replay_refreshed" if refreshed else "fast_graph_replay")
        return CausalLMOutputWithPast(
            logits=logits,
            past_key_values=runner._fresh_mirror(runner.prompt_len + step))


def build(context):
    record_route = context.get("record_route", lambda *_: None)
    targets = context.get("targets", {})
    references = context.get("references", {})
    prefill_spec = targets.get(PREFILL_MODEL_CALL)
    decode_spec = targets.get(DECODE_MODEL_CALL)
    prefill_parent = references.get(PREFILL_MODEL_CALL)
    decode_parent = references.get(DECODE_MODEL_CALL)
    if prefill_spec is None or decode_spec is None:
        return {}
    if prefill_spec.get("kind") != "module" or decode_spec.get("kind") != "module":
        return {}
    if prefill_parent is None or decode_parent is None:
        return {}
    state = context.get("state", {})
    device = state.get("device")
    if device is None or device.type != "cuda" or not torch.cuda.is_available():
        return {}
    index = device.index if device.index is not None else torch.cuda.current_device()
    home = torch.device("cuda", index)
    model = context.get("models", {}).get("model") or state.get("model")
    if model is None:
        return {}
    new_tokens = int(state.get("new_tokens", 0))
    prompt_tokens = int(state.get("prompt_tokens", 0))
    inputs = state.get("inputs")
    batch = int(state.get("batch", 1))
    if new_tokens < 2 or prompt_tokens < 1 or not isinstance(inputs, torch.Tensor):
        return {}
    if tuple(inputs.shape) != (batch, prompt_tokens) or inputs.device != home:
        return {}
    if inputs.dtype != torch.int64:
        return {}

    runner_box = [None]
    prefill_repl = _PrefillPassthrough(PREFILL_MODEL_CALL, prefill_parent,
                                       record_route)
    decode_repl = _DecodeReplacement(runner_box, DECODE_MODEL_CALL,
                                     decode_parent, record_route)
    failure = "not_attempted"
    for mode in ("fused", "exact"):
        if mode == "fused" and not _TRITON_OK:
            failure = "fused:triton_unavailable"
            continue
        runner = _FastDecodeRunner(model, home, prompt_tokens, new_tokens - 1,
                                   inputs, decode_parent, prefill_parent)
        decode_repl.set_runner(runner)
        try:
            with torch.inference_mode(False):
                runner.setup(mode, prefill_repl, decode_repl)
            failure = ""
            break
        except _SetupFailure as error:
            failure = "%s:%s" % (mode, error.args[0] if error.args else error)
            runner.fast_enabled = False
            runner.disable_reason = failure
            runner.decode_graphs = []
            torch.cuda.synchronize()
        except Exception as error:
            failure = "%s:unexpected_%s" % (mode, type(error).__name__)
            runner.fast_enabled = False
            runner.disable_reason = failure
            runner.decode_graphs = []
            torch.cuda.synchronize()
    if runner_box[0] is None or not runner_box[0].fast_enabled:
        disabled = runner_box[0]
        if disabled is None:
            disabled = _FastDecodeRunner(model, home, prompt_tokens,
                                         new_tokens - 1, inputs, decode_parent,
                                         prefill_parent)
            decode_repl.set_runner(disabled)
        disabled.fast_enabled = False
        if disabled.disable_reason in ("", "not_built"):
            disabled.disable_reason = failure or "setup_failure"
    return {
        PREFILL_MODEL_CALL: prefill_repl,
        DECODE_MODEL_CALL: decode_repl,
    }
