"""Decode-phase whole-bundle replacement for the frozen qwen_inference workload.

Repair of trial-9c4355e642: the previous attempt returned a DynamicCache *subclass*
wrapper from each decode step, so from step 2 onward the installed dispatch's
signature guard (``tensor_description`` -> ``{"type": "DynamicCache"}``) no longer
matched the observed signature and every remaining step silently fell back to the
parent (coverage: 2 replacement invocations vs 60 fallbacks; python profile showed
exactly one composite step per run).

This version keeps the op-faithful unpacked decode composite (bitwise mirror of the
transformers 5.16.1 Qwen3 sdpa decode op sequence, validated against the live parent
at setup) and repairs cache continuation:

* each decode step returns an *exact* ``DynamicCache`` instance whose per-layer
  keys/values are views into the engine's persistent static K/V buffers, so the
  observed kwarg signature keeps matching and the replacement stays on the hot path
  for all 31 decode steps of every run;
* the next step detects the continuation by storage identity (data_ptr of the views)
  and proceeds without any re-copy or materialization;
* per-position CUDA graphs (captured lazily at each position's first occurrence,
  i.e. inside the untimed quality/warmup invocations) are replayed per step;
* the parent path remains a fully compatible consumer of the returned cache
  (DynamicLayer.update cats the views into fresh contiguous tensors), so every
  fallback route preserves true cache state.

Prefill keeps the untouched eager parent path.
"""

import torch
import torch.nn.functional as F
from transformers.cache_utils import DynamicCache, DynamicLayer
from transformers.modeling_outputs import CausalLMOutputWithPast

DECODE_CALL = "323bcbe006b343648b346af9"


def _rms(x, weight, eps):
    x32 = x.to(torch.float32)
    variance = x32.pow(2).mean(-1, keepdim=True)
    x32 = x32 * torch.rsqrt(variance + eps)
    return weight * x32.to(x.dtype)


def _rot_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


class _DecodeEngine:
    def __init__(self, model, parent, record, steps_hint):
        self.parent = parent
        self.record = record
        self.steps_hint = max(int(steps_hint), 8)
        inner = model.model
        self.inner = inner
        self.rotary = inner.rotary_emb
        self.layers = list(inner.layers)
        self.num_layers = len(self.layers)
        self.W_emb = inner.embed_tokens.weight
        self.W_lm = model.lm_head.weight
        self.W_norm = inner.norm.weight
        self.eps_norm = inner.norm.variance_epsilon
        self.W_ln1 = []
        self.W_ln2 = []
        self.eps1 = []
        self.eps2 = []
        self.W_q = []
        self.W_k = []
        self.W_v = []
        self.W_o = []
        self.W_qn = []
        self.W_kn = []
        self.W_g = []
        self.W_u = []
        self.W_d = []
        self.acts = []
        self.eps_qn = []
        self.eps_kn = []
        for lyr in self.layers:
            attn = lyr.self_attn
            mlp = lyr.mlp
            self.W_ln1.append(lyr.input_layernorm.weight)
            self.eps1.append(lyr.input_layernorm.variance_epsilon)
            self.W_ln2.append(lyr.post_attention_layernorm.weight)
            self.eps2.append(lyr.post_attention_layernorm.variance_epsilon)
            self.W_q.append(attn.q_proj.weight)
            self.W_k.append(attn.k_proj.weight)
            self.W_v.append(attn.v_proj.weight)
            self.W_o.append(attn.o_proj.weight)
            self.W_qn.append(attn.q_norm.weight)
            self.W_kn.append(attn.k_norm.weight)
            self.eps_qn.append(attn.q_norm.variance_epsilon)
            self.eps_kn.append(attn.k_norm.variance_epsilon)
            self.W_g.append(mlp.gate_proj.weight)
            self.W_u.append(mlp.up_proj.weight)
            self.W_d.append(mlp.down_proj.weight)
            self.acts.append(mlp.act_fn)
        cfg = model.config
        self.scale = float(self.layers[0].self_attn.scaling)
        self.n_heads = int(cfg.num_attention_heads)
        self.n_kv = int(cfg.num_key_value_heads)
        self.head_dim = int(cfg.head_dim)
        self.hidden = int(cfg.hidden_size)
        self.inter = int(cfg.intermediate_size)
        self.on_cuda = self.W_emb.device.type == "cuda"
        self.ready = False
        self.fatal = False
        self.composite_ok = True
        self.capture_failed = False
        self.validated = False
        self.graphs = {}
        self.graph_logits = {}
        self.pool = None
        self.prompt_len = -1
        self.max_len = -1
        self.K = None
        self.V = None
        self.tok_in = None
        self.cos_tab = None
        self.sin_tab = None
        # Persistent, exact-type DynamicCache reused as the returned continuation
        # object; per step its layer tensors are refreshed as views of the static
        # K/V buffers at the new length.
        self.cache_obj = DynamicCache()
        self.cache_layers = []
        for _ in range(self.num_layers):
            layer = DynamicLayer()
            layer.is_initialized = True
            layer.dtype = self.W_emb.dtype
            layer.device = self.W_emb.device
            self.cache_layers.append(layer)
        self.cache_obj.layers = self.cache_layers
        self.cache_obj.layer_class_to_replicate = None

    # ------------------------------------------------------------------ setup
    def _alloc(self, prompt_len):
        dev = self.W_emb.device
        dt = self.W_emb.dtype
        self.max_len = prompt_len + self.steps_hint + 8
        K_all = torch.zeros((self.num_layers, 1, self.n_kv, self.max_len, self.head_dim), dtype=dt, device=dev)
        V_all = torch.zeros((self.num_layers, 1, self.n_kv, self.max_len, self.head_dim), dtype=dt, device=dev)
        self.K = [K_all[layer] for layer in range(self.num_layers)]
        self.V = [V_all[layer] for layer in range(self.num_layers)]
        self.tok_in = torch.zeros((1, 1), dtype=torch.long, device=dev)
        inv = self.rotary.inv_freq
        pos_ids = torch.arange(self.max_len, device=dev)
        inv_e = inv[None, :, None].float().expand(1, -1, 1)
        pos_e = pos_ids[None, None, :].float()
        freqs = (inv_e @ pos_e).transpose(1, 2)
        emb = torch.cat((freqs, freqs), dim=-1)
        scaling = self.rotary.attention_scaling
        self.cos_tab = (emb.cos() * scaling).to(dt)[0]
        self.sin_tab = (emb.sin() * scaling).to(dt)[0]
        self.ready = True

    # ------------------------------------------------------- cache inspection
    def _aliased_length(self, cache):
        """Length of the static-cache continuation carried by `cache`, else -1.

        A cache returned by a previous step of this engine carries per-layer views
        of the static K/V buffers: view.data_ptr() equals the base buffer's
        data_ptr() and the view length is the written prefix length.
        """
        layers = getattr(cache, "layers", None)
        if layers is None or len(layers) != self.num_layers or self.K is None:
            return -1
        first = layers[0]
        keys = first.keys
        values = first.values
        if keys is None or values is None:
            return -1
        shape = tuple(keys.shape)
        if len(shape) != 4:
            return -1
        if shape[0] != 1 or shape[1] != self.n_kv or shape[3] != self.head_dim:
            return -1
        length = shape[2]
        if tuple(values.shape) != shape:
            return -1
        if keys.data_ptr() != self.K[0].data_ptr() or values.data_ptr() != self.V[0].data_ptr():
            return -1
        for i in range(1, self.num_layers):
            entry = layers[i]
            k = entry.keys
            v = entry.values
            if k is None or v is None or tuple(k.shape) != shape or tuple(v.shape) != shape:
                return -1
            if k.data_ptr() != self.K[i].data_ptr() or v.data_ptr() != self.V[i].data_ptr():
                return -1
        return length

    def _cache_layers_ok(self, cache, length):
        if type(cache) is not DynamicCache:
            return False
        layers = getattr(cache, "layers", None)
        if layers is None or len(layers) != self.num_layers:
            return False
        for entry in layers:
            keys = getattr(entry, "keys", None)
            vals = getattr(entry, "values", None)
            if keys is None or vals is None:
                return False
            if tuple(keys.shape) != (1, self.n_kv, length, self.head_dim):
                return False
            if keys.dtype != self.W_emb.dtype or keys.device != self.W_emb.device:
                return False
        return True

    def init_run(self, cache, length):
        if not self.ready:
            self._alloc(length)
            self.prompt_len = length
        for layer in range(self.num_layers):
            self.K[layer][:, :, :length].copy_(cache.layers[layer].keys)
            self.V[layer][:, :, :length].copy_(cache.layers[layer].values)

    def sync_from_cache(self, cache, length):
        for layer in range(self.num_layers):
            self.K[layer][:, :, :length].copy_(cache.layers[layer].keys)
            self.V[layer][:, :, :length].copy_(cache.layers[layer].values)

    # -------------------------------------------------------- returned output
    def _view_cache(self, new_len):
        for i in range(self.num_layers):
            self.cache_layers[i].keys = self.K[i][:, :, :new_len]
            self.cache_layers[i].values = self.V[i][:, :, :new_len]
        return self.cache_obj

    def _out(self, logits, new_len):
        return CausalLMOutputWithPast(
            loss=None, logits=logits, past_key_values=self._view_cache(new_len)
        )

    # ------------------------------------------------------ decode composite
    def composite(self, pos, tok):
        cos = self.cos_tab[pos].view(1, 1, self.head_dim).unsqueeze(1)
        sin = self.sin_tab[pos].view(1, 1, self.head_dim).unsqueeze(1)
        h = F.embedding(tok, self.W_emb)
        for i in range(self.num_layers):
            residual = h
            h = _rms(h, self.W_ln1[i], self.eps1[i])
            q = F.linear(h, self.W_q[i]).view(1, 1, self.n_heads, self.head_dim)
            k = F.linear(h, self.W_k[i]).view(1, 1, self.n_kv, self.head_dim)
            v = F.linear(h, self.W_v[i]).view(1, 1, self.n_kv, self.head_dim)
            q = _rms(q, self.W_qn[i], self.eps_qn[i]).transpose(1, 2)
            k = _rms(k, self.W_kn[i], self.eps_kn[i]).transpose(1, 2)
            v = v.transpose(1, 2)
            q = (q * cos) + (_rot_half(q) * sin)
            k = (k * cos) + (_rot_half(k) * sin)
            self.K[i][:, :, pos : pos + 1, :].copy_(k)
            self.V[i][:, :, pos : pos + 1, :].copy_(v)
            k_full = torch.cat([self.K[i][:, :, :pos, :], k], dim=-2)
            v_full = torch.cat([self.V[i][:, :, :pos, :], v], dim=-2)
            attn = F.scaled_dot_product_attention(q, k_full, v_full, scale=self.scale, enable_gqa=True)
            attn = attn.transpose(1, 2).contiguous().reshape(1, 1, -1).contiguous()
            h = F.linear(attn, self.W_o[i])
            h = residual + h
            residual = h
            h = _rms(h, self.W_ln2[i], self.eps2[i])
            gate = F.linear(h, self.W_g[i])
            up = F.linear(h, self.W_u[i])
            h = F.linear(self.acts[i](gate) * up, self.W_d[i])
            h = residual + h
        h = _rms(h, self.W_norm, self.eps_norm)
        return F.linear(h[:, -1:, :], self.W_lm)

    # ------------------------------------------------------------- validation
    def validate(self, pos, tok, comp_logits):
        self.validated = True
        scratch = DynamicCache()
        for layer in range(self.num_layers):
            scratch.update(self.K[layer][:, :, :pos], self.V[layer][:, :, :pos], layer)
        out_p = self.parent(input_ids=tok, past_key_values=scratch, use_cache=True, logits_to_keep=1)
        ref = out_p.logits
        if not torch.equal(ref, comp_logits):
            self.composite_ok = False
            self.record("parent_fallback_validation")

    def capture(self, pos, input_ids):
        try:
            tok = self.tok_in
            tok.copy_(input_ids)
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            logits_ref = None
            with torch.cuda.stream(side):
                for _ in range(3):
                    logits_ref = self.composite(pos, tok)
            torch.cuda.current_stream().wait_stream(side)
            if not self.validated:
                self.validate(pos, tok, logits_ref)
                if not self.composite_ok:
                    return
            if self.pool is None:
                self.pool = torch.cuda.graph_pool_handle()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, pool=self.pool):
                cap_logits = self.composite(pos, tok)
            graph.replay()
            if not torch.equal(cap_logits, logits_ref):
                del graph
                self.capture_failed = True
                self.record("setup_capture_mismatch")
                return
            self.graphs[pos] = graph
            self.graph_logits[pos] = cap_logits
        except Exception:
            self.capture_failed = True
            self.record("setup_capture_failure")

    # ------------------------------------------------------------ parent path
    def parent_step(self, input_ids, past_key_values):
        return self.parent(
            input_ids=input_ids,
            past_key_values=past_key_values,
            use_cache=True,
            logits_to_keep=1,
        )

    # ------------------------------------------------------------ decode step
    def step(self, input_ids, past_key_values):
        if self.fatal or not self.composite_ok:
            self.record("parent_fallback")
            return self.parent_step(input_ids, past_key_values)
        length = self._aliased_length(past_key_values)
        if length > 0:
            pos = length
        else:
            length2 = (
                past_key_values.get_seq_length()
                if isinstance(past_key_values, DynamicCache)
                else -1
            )
            if length2 <= 0 or not self._cache_layers_ok(past_key_values, length2):
                self.record("parent_fallback_unsupported_cache")
                return self.parent_step(input_ids, past_key_values)
            if not self.ready:
                self.init_run(past_key_values, length2)
                pos = length2
            elif length2 == self.prompt_len:
                self.init_run(past_key_values, length2)
                pos = length2
            elif self.prompt_len < length2 < self.max_len:
                self.sync_from_cache(past_key_values, length2)
                pos = length2
            else:
                self.record("parent_fallback_overflow")
                return self.parent_step(input_ids, past_key_values)
        if pos >= self.max_len:
            self.record("parent_fallback_overflow")
            return self.parent_step(input_ids, past_key_values)
        if self.on_cuda and not self.capture_failed and pos not in self.graphs:
            self.capture(pos, input_ids)
            if not self.composite_ok or self.fatal:
                self.record("parent_fallback")
                return self.parent_step(input_ids, past_key_values)
        if self.on_cuda and pos in self.graphs:
            self.tok_in.copy_(input_ids)
            self.graphs[pos].replay()
            logits = self.graph_logits[pos].clone()
            self.record("graph_replay")
            return self._out(logits, pos + 1)
        logits = self.composite(pos, input_ids)
        self.record("eager_composite")
        return self._out(logits, pos + 1)


def build(context):
    modules = context.get("modules") or {}
    model = modules.get(DECODE_CALL)
    if model is None:
        return {}
    references = context.get("references") or {}
    parent = references.get(DECODE_CALL)
    if parent is None:
        return {}
    record = context.get("record_route", lambda *_: None)

    def route(label):
        record(DECODE_CALL, label)

    steps_hint = 64
    try:
        spec = (context.get("targets") or {}).get(DECODE_CALL) or {}
        steps_hint = int(spec.get("count", 64))
    except Exception:
        steps_hint = 64
    engine = _DecodeEngine(model, parent, route, steps_hint)

    def decode_forward(
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
        supported = (
            engine.composite_ok
            and not engine.fatal
            and input_ids is not None
            and past_key_values is not None
            and attention_mask is None
            and position_ids is None
            and inputs_embeds is None
            and labels is None
            and not kwargs
            and use_cache in (True, None)
            and logits_to_keep == 1
            and input_ids.dim() == 2
            and tuple(input_ids.shape) == (1, 1)
            and input_ids.dtype == torch.long
            and input_ids.device == engine.W_emb.device
        )
        if not supported:
            route("parent_fallback_unsupported_signature")
            return parent(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                labels=labels,
                use_cache=use_cache,
                logits_to_keep=logits_to_keep,
                **kwargs,
            )
        try:
            return engine.step(input_ids, past_key_values)
        except Exception:
            engine.fatal = True
            route("runtime_failure_parent")
            return engine.parent_step(input_ids, past_key_values)

    return {DECODE_CALL: decode_forward}
