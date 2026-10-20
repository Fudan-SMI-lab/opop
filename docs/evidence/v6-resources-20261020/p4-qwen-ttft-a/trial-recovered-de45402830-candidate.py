"""Execution-plan replacement: replay the frozen prefill region as one CUDA graph.

Repair of trial-resample-3896551a (see inputs/previous-feedback.txt). That trial
installed 1154 per-site replacements, which forced a TorchDispatchMode over every
aten call of the run; quality passed bit-exactly but ttft regressed from 25.66 ms
to 66.3-76.6 ms because per-call routing overhead dominates when each kernel is
tiny. This candidate therefore replaces exactly ONE site - the observed prefill
Qwen3ForCausalLM module call 900c23e99c808d2c9bda215a - with a CUDA-graph replay
of the unmodified eager forward captured once at build() time:

  * The framework installs a module-kind replacement as one dispatch on the root
    model's forward (phase + signature guarded). No functional sites are
    returned, so the framework never creates a TorchDispatchMode and every decode
    step keeps calling the untouched original forward.
  * Replay launches the byte-identical kernel sequence recorded from the parent
    forward with identical arguments, so no operator, GEMM algorithm, dtype,
    reduction order or rounding is intentionally changed.
  * The request-local KV cache written by the replay lives in the graph's private
    pool; each invocation hands eager decode a fresh transformers 5.x DynamicCache
    whose DynamicLayer entries reference those pool tensors, so decode consumes
    prefill exactly as in the parent (torch.cat allocations outside the pool).
  * Build-time verification: after capture, one eager parent forward and one full
    replacement call (replay + fresh cache handoff) are compared bitwise on the
    logits and on every layer's key/value tensors. Any mismatch, or any capture
    exception, falls back to calling the parent forward unchanged, so the worst
    case reproduces parent behavior instead of different numerics.

The [:, -1] select and greedy argmax stay eager: replacing them would require
functional sites and reintroduce the per-call dispatch overhead that caused the
previous regression.
"""
from __future__ import annotations

import torch

PREFILL_CALL = "900c23e99c808d2c9bda215a"
_WARMUP_ITERATIONS = 3
_ALLOWED_KEYS = frozenset({"input_ids", "use_cache", "logits_to_keep"})


def _eager_passthrough(reference):
    def replacement(*args, **kwargs):
        return reference(*args, **kwargs)

    return replacement


def _capture_engine(reference, inputs):
    """Capture the parent prefill forward over static buffers.

    Returns the replacement callable, or None when capture or the bitwise
    self-check fails (caller then falls back to the parent forward).
    """
    from transformers.cache_utils import DynamicCache, DynamicLayer
    from transformers.modeling_outputs import CausalLMOutputWithPast

    device = inputs.device
    expected_shape = list(inputs.shape)
    expected_stride = list(inputs.stride())

    static_ids = torch.empty_like(inputs)
    static_ids.copy_(inputs)
    static_ptr = static_ids.data_ptr()

    # Warm the exact op sequence (allocator, cuBLAS heuristics, SDPA kernels) on
    # a side stream, then capture on the graph's own side stream.
    side = torch.cuda.Stream(device=device)
    side.wait_stream(torch.cuda.current_stream(device))
    with torch.cuda.stream(side), torch.inference_mode():
        for _ in range(_WARMUP_ITERATIONS):
            reference(input_ids=static_ids, use_cache=True, logits_to_keep=1)
    torch.cuda.current_stream(device).wait_stream(side)
    torch.cuda.synchronize(device)

    graph = torch.cuda.CUDAGraph()
    with torch.inference_mode(), torch.cuda.graph(graph):
        captured = reference(input_ids=static_ids, use_cache=True, logits_to_keep=1)
    static_logits = captured.logits
    static_kv = [(layer.keys, layer.values) for layer in captured.past_key_values.layers]
    num_layers = len(static_kv)
    captured = None

    def fresh_cache():
        layers = []
        for keys, values in static_kv:
            layer = DynamicLayer()
            layer.keys = keys
            layer.values = values
            layer.is_initialized = True
            layer.dtype = keys.dtype
            layer.device = keys.device
            layers.append(layer)
        cache = DynamicCache()
        cache.layers = layers
        return cache

    def compatible(input_ids, kwargs):
        if not isinstance(input_ids, torch.Tensor):
            return False
        if input_ids.device != device or input_ids.dtype != torch.long:
            return False
        if list(input_ids.shape) != expected_shape or list(input_ids.stride()) != expected_stride:
            return False
        if set(kwargs) - _ALLOWED_KEYS:
            return False
        if kwargs.get("use_cache") is not True or kwargs.get("logits_to_keep") != 1:
            return False
        return True

    def prefill(*args, **kwargs):
        # The observed call form is kwargs-only (input_ids, use_cache,
        # logits_to_keep); anything else falls back to the parent.
        if args:
            return reference(*args, **kwargs)
        input_ids = kwargs.get("input_ids")
        if not compatible(input_ids, kwargs):
            return reference(*args, **kwargs)
        if input_ids.data_ptr() != static_ptr:
            static_ids.copy_(input_ids)
        graph.replay()
        return CausalLMOutputWithPast(
            loss=None,
            logits=static_logits,
            past_key_values=fresh_cache(),
            hidden_states=None,
            attentions=None,
        )

    # Bitwise verification: one eager parent forward vs one full replacement call
    # (replay plus fresh-cache handoff). Replay executes the captured kernels, so
    # equality here also validates the cache tensors decode will read.
    with torch.inference_mode():
        eager = reference(input_ids=static_ids, use_cache=True, logits_to_keep=1)
    probe = prefill(input_ids=static_ids, use_cache=True, logits_to_keep=1)
    torch.cuda.synchronize(device)

    if probe.logits.shape != eager.logits.shape or not torch.equal(probe.logits, eager.logits):
        return None
    eager_kv = [(layer.keys, layer.values) for layer in eager.past_key_values.layers]
    if len(eager_kv) != num_layers or len(probe.past_key_values.layers) != num_layers:
        return None
    if probe.past_key_values.get_seq_length() != expected_shape[-1]:
        return None
    for (k_probe, v_probe), (k_eager, v_eager), (k_static, v_static) in zip(
        probe.past_key_values.layers, eager_kv, static_kv
    ):
        if k_probe.data_ptr() != k_static.data_ptr() or v_probe.data_ptr() != v_static.data_ptr():
            return None
        if not (torch.equal(k_static, k_eager) and torch.equal(v_static, v_eager)):
            return None
    del eager, eager_kv, probe
    return prefill


def build(context):
    targets = context.get("targets") or {}
    if PREFILL_CALL not in targets:
        return {}

    reference = (context.get("references") or {}).get(PREFILL_CALL)
    state = context.get("state") or {}
    inputs = state.get("inputs")
    if reference is None or not callable(reference):
        return {}
    if not isinstance(inputs, torch.Tensor) or inputs.device.type != "cuda":
        return {PREFILL_CALL: _eager_passthrough(reference)}

    try:
        engine = _capture_engine(reference, inputs)
    except Exception:
        engine = None
    if engine is None:
        return {PREFILL_CALL: _eager_passthrough(reference)}
    return {PREFILL_CALL: engine}
