"""Prefill-region CUDA Graph candidate for the Qwen3-4B interactive workload.

Objective: minimize ttft_ms. The first-token boundary is the 256-token prefill
forward of ``model`` (call 900c23e99c808d2c9bda215a) plus the last-position
argmax and the explicit synchronize in the frozen adapter. The observed parent
prefill spends 21458 us of CUDA time in 2145 kernel launches while the measured
TTFT wall is ~24.1 ms, i.e. ~2.6 ms of launch/gap overhead between kernels.

Strategy: capture the *unmodified* parent prefill forward into a CUDA graph once
during build (setup, outside timing) and replay it on every invocation. Replay
re-executes the identical kernel sequence of the parent forward, so logits and
KV cache are bitwise equal to the eager parent (verified at build time before
the replacement is enabled); the saving is the per-launch dispatch overhead of
~2.1k launches inside the TTFT window, not a change in arithmetic or semantics.

Correctness properties:
  * The replacement is installed only for the observed prefill call signature
    (kwargs ``input_ids`` [1, 256] int64 on CUDA, ``use_cache=True``,
    ``logits_to_keep=1``, every other optional kwarg absent/None). Any other
    call shape - in particular every decode step - falls back to the parent
    bound forward, so decode and any auxiliary invocation are untouched.
  * Each replay copies the caller's ``input_ids`` into the graph's static input
    buffer first, so the graph is input-agnostic; nothing about the fixture's
    answer is baked in and the workload executes on every invocation.
  * Each call returns a freshly constructed ``DynamicCache`` whose layers
    reference the graph's static K/V tensors (rewritten by each replay),
    preserving the fresh-cache-per-run state semantics of the workload; decode
    appends by concatenation onto new tensors and never mutates the buffers.
  * If capture or the bitwise build-time verification fails for any reason, the
    callable degrades to a pure parent passthrough (no improvement, no risk).
"""

from __future__ import annotations

import torch

try:  # optional at import time; no CUDA initialization happens here
    from transformers.cache_utils import DynamicCache, DynamicLayer
    from transformers.modeling_outputs import CausalLMOutputWithPast
except Exception:  # pragma: no cover - very old transformers
    DynamicCache = None
    DynamicLayer = None
    CausalLMOutputWithPast = None

MODEL_CALL = "900c23e99c808d2c9bda215a"
_WARMUP_ITERS = 3
_PROMPT_TOKENS = 256
_ALLOWED_KWARGS = frozenset(
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
_NONE_OK_KWARGS = ("attention_mask", "position_ids", "past_key_values", "inputs_embeds", "labels")
_PACK_ATTR = "_opop_prefill_graph_pack"


def _fill_layer(layer, keys, values):
    layer.keys = keys
    layer.values = values
    layer.is_initialized = True
    try:
        layer.dtype = keys.dtype
        layer.device = keys.device
    except Exception:
        pass


def _new_cache(config, depth):
    """A fresh DynamicCache with ``depth`` layers, without any GPU work."""
    try:
        cache = DynamicCache(config=config)
        if len(cache.layers) == depth:
            return cache
    except Exception:
        pass
    cache = DynamicCache()
    if DynamicLayer is not None:
        while len(cache.layers) < depth:
            cache.layers.append(DynamicLayer())
    return cache


def _fresh_cache(config, static_kv):
    """Build a fresh DynamicCache whose layers point at the static K/V tensors."""
    depth = len(static_kv)
    cache = _new_cache(config, depth)
    layers = list(getattr(cache, "layers", []))
    if len(layers) == depth:
        for layer, (keys, values) in zip(layers, static_kv):
            _fill_layer(layer, keys, values)
        if cache.get_seq_length() == int(static_kv[0][0].shape[-2]):
            return cache
    # Documented construction fallback (allocates copies; semantics identical).
    return DynamicCache(ddp_cache_data=list(static_kv))


def _capture_pack(model, state, device):
    """Warm up, capture and verify the prefill graph. Returns None on failure."""
    inputs = state.get("inputs")
    if not isinstance(inputs, torch.Tensor) or tuple(inputs.shape) != (1, _PROMPT_TOKENS) or inputs.dtype != torch.long:
        inputs = torch.zeros((1, _PROMPT_TOKENS), dtype=torch.long, device=device)
    static_ids = torch.empty_like(inputs)
    static_ids.copy_(inputs)

    def prefill(past=None):
        if past is None:
            return model(input_ids=static_ids, use_cache=True, logits_to_keep=1)
        return model(input_ids=static_ids, past_key_values=past, use_cache=True, logits_to_keep=1)

    # 1) Warm up on a side stream so cuBLAS/SDPA workspaces and plans exist
    #    before capture; matches the runtime inference-mode context.
    stream = torch.cuda.Stream(device=device)
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream), torch.inference_mode():
        for _ in range(_WARMUP_ITERS):
            prefill()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize(device)

    # 2) Eager reference outputs for bitwise verification of the capture.
    with torch.inference_mode():
        ref_out = prefill()
    ref_logits = ref_out.logits
    ref_kv = [(layer.keys, layer.values) for layer in ref_out.past_key_values.layers]
    depth = len(ref_kv)
    if depth == 0:
        return None

    # 3) Pre-initialize a template cache (zero-length keys) *outside* capture so
    #    no cache construction runs inside graph capture. Prefill with a fresh
    #    empty cache is semantically identical to prefill with past=None:
    #    position_ids and masks depend only on get_seq_length() == 0.
    template = _new_cache(model.config, depth)
    template_layers = list(getattr(template, "layers", []))
    if len(template_layers) != depth:
        return None
    empty_k = ref_kv[0][0][:, :, :0, :]
    empty_v = ref_kv[0][1][:, :, :0, :]
    for layer in template_layers:
        layer.lazy_initialization(empty_k, empty_v)

    # 4) Capture the unmodified parent forward reading static_ids.
    graph = torch.cuda.CUDAGraph()
    with torch.inference_mode(), torch.cuda.graph(graph):
        cap_out = prefill(template)
    cap_logits = cap_out.logits
    static_kv = [(layer.keys, layer.values) for layer in template_layers]
    if len(static_kv) != depth:
        return None

    # 5) Verify replay is bitwise identical to the eager parent path.
    graph.replay()
    torch.cuda.synchronize(device)
    if not torch.equal(cap_logits, ref_logits):
        return None
    for (gk, gv), (rk, rv) in zip(static_kv, ref_kv):
        if gk.shape != rk.shape or gv.shape != rv.shape:
            return None
        if not (torch.equal(gk, rk) and torch.equal(gv, rv)):
            return None
    token = cap_logits[:, -1].argmax(dim=-1, keepdim=True)
    if not torch.equal(token, ref_logits[:, -1].argmax(dim=-1, keepdim=True)):
        return None

    # 6) Verify the fresh-cache handoff drives one parent decode step bit-exactly.
    with torch.inference_mode():
        graph_step = model(
            input_ids=token,
            past_key_values=_fresh_cache(model.config, static_kv),
            use_cache=True,
            logits_to_keep=1,
        )
        eager_step = model(
            input_ids=token,
            past_key_values=_fresh_cache(model.config, ref_kv),
            use_cache=True,
            logits_to_keep=1,
        )
    if not torch.equal(graph_step.logits, eager_step.logits):
        return None

    output_cls = type(cap_out) if isinstance(cap_out, CausalLMOutputWithPast) else CausalLMOutputWithPast
    if not (isinstance(output_cls, type) and issubclass(output_cls, CausalLMOutputWithPast)):
        output_cls = CausalLMOutputWithPast
    return {
        "graph": graph,
        "static_ids": static_ids,
        "cap_logits": cap_logits,
        "static_kv": static_kv,
        "config": model.config,
        "shape": tuple(static_ids.shape),
        "output_cls": output_cls,
        "device": device,
    }


def build(context):
    specs = context.get("targets") or {}
    references = context.get("references") or {}
    modules = context.get("modules") or {}
    state = context.get("state") or {}
    parameters = context.get("parameters") or {}

    if MODEL_CALL not in specs:
        # Nothing declared locally for this rank: legal empty mapping.
        return {}
    parent = references.get(MODEL_CALL)
    model = modules.get(MODEL_CALL)

    def repl(*args, **kwargs):
        return parent(*args, **kwargs)

    if parent is None or model is None or DynamicCache is None or CausalLMOutputWithPast is None:
        return {MODEL_CALL: repl}
    if parameters:
        return {MODEL_CALL: repl}

    device = None
    try:
        device = next(parameter.device for parameter in model.parameters())
    except (StopIteration, AttributeError, RuntimeError, TypeError):
        device = None
    if device is None or device.type != "cuda" or not torch.cuda.is_available():
        return {MODEL_CALL: repl}

    pack = getattr(model, _PACK_ATTR, None)
    if not (isinstance(pack, dict) and pack.get("device") == device):
        try:
            with torch.cuda.device(device):
                pack = _capture_pack(model, state, device)
        except Exception:
            pack = None
        if pack is None:
            return {MODEL_CALL: repl}
        try:
            setattr(model, _PACK_ATTR, pack)
        except Exception:
            pass

    graph = pack["graph"]
    static_ids = pack["static_ids"]
    cap_logits = pack["cap_logits"]
    static_kv = pack["static_kv"]
    config = pack["config"]
    shape = pack["shape"]
    output_cls = pack["output_cls"]

    def graph_repl(*args, **kwargs):
        # Route only the observed prefill signature to the graph; everything
        # else (decode steps included) uses the parent bound forward.
        if args or not kwargs or set(kwargs) - _ALLOWED_KWARGS:
            return parent(*args, **kwargs)
        if any(kwargs.get(key) is not None for key in _NONE_OK_KWARGS):
            return parent(*args, **kwargs)
        input_ids = kwargs.get("input_ids")
        if not isinstance(input_ids, torch.Tensor):
            return parent(*args, **kwargs)
        logits_to_keep = kwargs.get("logits_to_keep")
        if (
            kwargs.get("use_cache") is not True
            or isinstance(logits_to_keep, bool)
            or logits_to_keep != 1
            or tuple(input_ids.shape) != shape
            or input_ids.dtype != torch.long
            or input_ids.device.type != "cuda"
            or input_ids.device != static_ids.device
        ):
            return parent(*args, **kwargs)
        try:
            static_ids.copy_(input_ids)
            graph.replay()
            cache = _fresh_cache(config, static_kv)
        except Exception:
            return parent(*args, **kwargs)
        return output_cls(
            loss=None,
            logits=cap_logits,
            past_key_values=cache,
            hidden_states=None,
            attentions=None,
        )

    return {MODEL_CALL: graph_repl}
