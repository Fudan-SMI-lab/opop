"""Candidate: lazily-captured whole-prefill CUDA Graph replay for the Qwen3-4B TTFT region.

Selection (inputs/selection.json): region scope, group e0861b46dbeeb562f67f. The single
replaced call is 900c23e99c808d2c9bda215a (target "model", phase "prefill", semantics
Qwen3ForCausalLM; observed signature input_ids=[1,256] int64 cuda, use_cache=True,
logits_to_keep=1). Decode-phase calls of the same module fall back to the parent.

Repair of the previous failure (inputs/previous-feedback.txt, trial-023e79293c):
  * The prior candidate captured its graph at BUILD time and gated activation on a
    BITWISE torch.equal(replay, eager) comparison; the gate returned False (candidate
    lines 226-229, observed_guard), so every invocation routed to the parent
    (prefill_guard_parent) and effective coexistence was rejected even though capture,
    replay and synchronization had succeeded.
  * Verified framework facts (kernel_optimizer/project/{evaluation,binding,discovery}.py,
    torch 2.13.0+cu129 worker): bundles install before operation(); inherited functional
    replacements engage only inside the BindingSession ReplacementMode that operation()
    enters around adapter.run, and inherited phase-guarded module replacements engage only
    under the matching discovery phase. A build-time capture therefore records the raw
    eager stream, not the inherited implementation - the prior composition assumption was
    unproven and is false in general.

This attempt changes the initialization point and the activation gate:
  * Initialization is LAZY, at the first legitimate invocation: it runs inside
    adapter.run's prefill phase with the inherited dispatch modes entered, so the warmup
    forwards and the capture execute the true current implementation (the inherited
    bundle's fused pointwise kernels and module replacements included) and the captured
    kernel stream preserves that implementation under every later replay.
  * Activation is gated on the frozen request quality contract (logits atol=rtol=0.02,
    exact greedy argmax token, per-layer KV within tolerance, and an end-to-end exact
    greedy token check over the full new_tokens decode), never on bitwise equality. An
    eager-vs-eager bitwise probe is recorded to distinguish eager nondeterminism from
    graph-pool kernel-selection effects; it never disables the path by itself.

Every invocation executes the complete workload: the caller's current input_ids are
copied into the static 2 KiB buffer, one graph replay re-runs the captured prefill kernel
stream, the [1,1,151936] bf16 logits are cloned, and a fresh KV cache is materialized
(torch.cat out of the graph private pool; DynamicCache.update is pure cat on
transformers 5.16.1, verified on the worker Python) for the untouched eager decode loop.
No outputs are cached; nothing about the frozen fixture is precomputed. All run-time GPU
work stays on the calling stream inside the declared ttft boundary with no added
synchronization. Route labels distinguish the active replay path
(prefill_graph_replay) from setup failure (prefill_init_setup_parent,
prefill_init_capture_parent), guard failure (prefill_init_verify_parent), unsupported
signature (prefill_signature_parent), runtime failure (prefill_replay_error_parent) and
cache materialization failure (prefill_clone_failed_parent).
"""
from __future__ import annotations

import torch

try:  # optional phase annotation; the worker environment provides it
    from kernel_optimizer.project.discovery import phase as _discovery_phase
except Exception:  # pragma: no cover - absent in plain environments
    _discovery_phase = None


class _NullPhase:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _decode_phase():
    """Reproduce the adapter's decode phase so verification decodes dispatch like real decode."""
    if _discovery_phase is not None:
        try:
            cm = _discovery_phase("decode")
            if hasattr(cm, "__enter__") and hasattr(cm, "__exit__"):
                return cm
        except Exception:
            pass
    return _NullPhase()


# Frozen quality contract from inputs/request.json.
LOGITS_ATOL = 0.02
LOGITS_RTOL = 0.02

_ALLOWED_KWARGS = {
    "input_ids",
    "use_cache",
    "logits_to_keep",
    "attention_mask",
    "past_key_values",
    "inputs_embeds",
    "position_ids",
    "labels",
    "cache_position",
    "output_attentions",
    "output_hidden_states",
    "return_dict",
}
_NONE_ONLY = (
    "attention_mask",
    "past_key_values",
    "inputs_embeds",
    "position_ids",
    "labels",
    "cache_position",
    "output_attentions",
    "output_hidden_states",
)

_UNINITIALIZED = "uninitialized"
_READY = "ready"
_SETUP_FAILED = "setup_failed"
_CAPTURE_FAILED = "capture_failed"
_VERIFY_FAILED = "verify_failed"
_RUNTIME_FAILED = "runtime_failed"
_CLONE_FAILED = "clone_failed"

_HOLDER = {
    "state": _UNINITIALIZED,
    "graph": None,
    "static_ids": None,
    "static_logits": None,
    "static_pairs": None,
    "out_cls": None,
    "shape": None,
    "device_index": None,
    "model": None,
    "new_tokens": None,
    "eager_bitwise": None,
}


def _cache_pairs(cache):
    """Per-layer (keys, values) extraction across cache API generations."""
    try:
        layers = getattr(cache, "layers", None)
        if isinstance(layers, (list, tuple)) and layers:
            pairs = []
            for layer in layers:
                k = getattr(layer, "keys", None)
                v = getattr(layer, "values", None)
                if isinstance(k, torch.Tensor) and isinstance(v, torch.Tensor):
                    pairs.append((k, v))
                else:
                    return None
            return pairs
        kc = getattr(cache, "key_cache", None)
        vc = getattr(cache, "value_cache", None)
        if (
            isinstance(kc, (list, tuple))
            and isinstance(vc, (list, tuple))
            and len(kc) == len(vc)
        ):
            pairs = []
            for k, v in zip(kc, vc):
                if isinstance(k, torch.Tensor) and isinstance(v, torch.Tensor):
                    pairs.append((k, v))
                elif k is None and v is None:
                    continue
                else:
                    return None
            return pairs if pairs else None
    except Exception:
        return None
    return None


def _pairs_bitwise(pa, pb):
    if pa is None or pb is None or len(pa) != len(pb):
        return False
    for (ka, va), (kb, vb) in zip(pa, pb):
        if ka.shape != kb.shape or va.shape != vb.shape:
            return False
        if not (torch.equal(ka, kb) and torch.equal(va, vb)):
            return False
    return True


def _pairs_close(pa, pb, atol, rtol):
    if pa is None or pb is None or len(pa) != len(pb):
        return False
    for (ka, va), (kb, vb) in zip(pa, pb):
        if ka.shape != kb.shape or va.shape != vb.shape:
            return False
        if not torch.allclose(ka.float(), kb.float(), atol=atol, rtol=rtol):
            return False
        if not torch.allclose(va.float(), vb.float(), atol=atol, rtol=rtol):
            return False
    return True


def _materialize_cache(pairs):
    """Fresh DynamicCache copied out of the graph pool via torch.cat (never aliases it)."""
    if not pairs:
        return None
    try:
        from transformers.cache_utils import DynamicCache

        fresh = DynamicCache()
        for idx, (k, v) in enumerate(pairs):
            try:
                fresh.update(k, v, idx)  # transformers 5.x argument order
            except TypeError:
                fresh.update(idx, k, v)  # transformers 4.x fallback
        fp = _cache_pairs(fresh)
        if fp is None or len(fp) != len(pairs):
            return None
        for (ka, va), (kb, vb) in zip(fp, pairs):
            if ka.shape != kb.shape or va.shape != vb.shape:
                return None
            if ka.data_ptr() == kb.data_ptr() or va.data_ptr() == vb.data_ptr():
                return None
        return fresh
    except Exception:
        return None


def _greedy_decode(ref, first_token, cache, new_tokens):
    """The adapter's decode loop, run under the decode phase so inherited decode sites engage."""
    tokens = [first_token]
    token = first_token
    with _decode_phase():
        for _ in range(new_tokens - 1):
            output = ref(
                input_ids=token,
                past_key_values=cache,
                use_cache=True,
                logits_to_keep=1,
            )
            cache = output.past_key_values
            token = output.logits[:, -1].argmax(dim=-1, keepdim=True)
            tokens.append(token)
    return torch.cat(tokens, dim=1)


def _capture(ref, static_ids, error_mode=None):
    graph = torch.cuda.CUDAGraph()
    if error_mode is None:
        ctx = torch.cuda.graph(graph)
    else:
        ctx = torch.cuda.graph(graph, capture_error_mode=error_mode)
    with ctx:
        static_out = ref(input_ids=static_ids, use_cache=True, logits_to_keep=1)
    return graph, static_out


def _disable(state, graph=None):
    _HOLDER["state"] = state
    _HOLDER["graph"] = None
    _HOLDER["static_logits"] = None
    _HOLDER["static_pairs"] = None
    if graph is not None:
        try:
            graph.reset()
        except Exception:
            pass


def _lazy_init(ref, ids, model, new_tokens):
    """Warmup + capture + verification at the first legitimate invocation.

    Runs inside adapter.run's prefill phase with the inherited dispatch modes entered,
    so the captured stream contains the current parent implementation. Returns True when
    the replay path is enabled; any failure disables it permanently.
    """
    graph = None
    stage = _SETUP_FAILED
    try:
        if not torch.cuda.is_available():
            return _disable(_SETUP_FAILED) or False
        if not (hasattr(torch.cuda, "CUDAGraph") and hasattr(torch.cuda, "graph")):
            return _disable(_SETUP_FAILED) or False
        if not callable(ref) or model is None:
            return _disable(_SETUP_FAILED) or False
        if (
            not isinstance(ids, torch.Tensor)
            or not ids.is_cuda
            or ids.dim() != 2
            or ids.dtype is not torch.int64
        ):
            return _disable(_SETUP_FAILED) or False
        if not isinstance(new_tokens, int) or isinstance(new_tokens, bool) or new_tokens < 1:
            return _disable(_SETUP_FAILED) or False

        device = ids.device
        static_ids = torch.empty_like(ids)
        static_ids.copy_(ids)
        cur = torch.cuda.current_stream()
        side = torch.cuda.Stream(device=device)
        side.wait_stream(cur)
        warm_prev = None
        warm = None
        with torch.cuda.stream(side):
            for _ in range(3):
                warm_prev = warm
                warm = ref(input_ids=static_ids, use_cache=True, logits_to_keep=1)
        cur.wait_stream(side)
        torch.cuda.synchronize(device)
        if warm is None or warm_prev is None:
            return _disable(_SETUP_FAILED) or False
        if getattr(warm, "logits", None) is None:
            return _disable(_SETUP_FAILED) or False

        # Nondeterminism probe (classification only; never disables by itself).
        _HOLDER["eager_bitwise"] = bool(
            torch.equal(warm_prev.logits.detach(), warm.logits.detach())
            and _pairs_bitwise(
                _cache_pairs(warm_prev.past_key_values),
                _cache_pairs(warm.past_key_values),
            )
        )

        stage = _CAPTURE_FAILED
        try:
            graph, static_out = _capture(ref, static_ids)
        except Exception:
            try:
                graph = None
            except Exception:
                pass
            # Single justified fallback: an inconsistent capture is still caught by the
            # full downstream verification against a fresh eager run.
            try:
                graph, static_out = _capture(ref, static_ids, "thread_local")
            except Exception:
                return _disable(_CAPTURE_FAILED) or False

        stage = _VERIFY_FAILED
        graph.replay()
        torch.cuda.synchronize(device)

        static_logits = static_out.logits.detach()
        warm_logits = warm.logits.detach()
        logits_ok = torch.allclose(
            static_logits.float(), warm_logits.float(),
            atol=LOGITS_ATOL, rtol=LOGITS_RTOL,
        )
        token_ok = torch.equal(
            static_logits[:, -1].argmax(dim=-1),
            warm_logits[:, -1].argmax(dim=-1),
        )
        static_pairs = _cache_pairs(static_out.past_key_values)
        warm_pairs = _cache_pairs(warm.past_key_values)
        kv_ok = _pairs_close(static_pairs, warm_pairs, LOGITS_ATOL, LOGITS_RTOL)
        if not (logits_ok and token_ok and kv_ok):
            return _disable(_VERIFY_FAILED, graph) or False
        if static_logits.dim() != 3:
            return _disable(_VERIFY_FAILED, graph) or False

        # Cache materialization and output-construction probes.
        fresh_replay = _materialize_cache(static_pairs)
        fresh_eager = _materialize_cache(warm_pairs)
        if fresh_replay is None or fresh_eager is None:
            return _disable(_VERIFY_FAILED, graph) or False
        out_cls = type(warm)
        probe = out_cls(
            logits=static_logits.clone(),
            past_key_values=_materialize_cache(static_pairs),
        )
        if probe.logits.shape != static_logits.shape:
            del probe
            return _disable(_VERIFY_FAILED, graph) or False
        del probe

        # End-to-end exact greedy token gate over the full decode sequence.
        tokens_eager = _greedy_decode(
            ref, warm_logits[:, -1].argmax(dim=-1, keepdim=True), fresh_eager, new_tokens
        )
        tokens_replay = _greedy_decode(
            ref, static_logits[:, -1].argmax(dim=-1, keepdim=True), fresh_replay, new_tokens
        )
        if not torch.equal(tokens_eager, tokens_replay):
            return _disable(_VERIFY_FAILED, graph) or False

        _HOLDER.update(
            state=_READY,
            graph=graph,
            static_ids=static_ids,
            static_logits=static_logits,
            static_pairs=static_pairs,
            out_cls=out_cls,
            shape=tuple(static_ids.shape),
            device_index=static_ids.get_device(),
            model=model,
            new_tokens=int(new_tokens),
        )
        return True
    except Exception:
        return _disable(stage, graph) or False


def _label_for(state):
    return {
        _SETUP_FAILED: "prefill_init_setup_parent",
        _CAPTURE_FAILED: "prefill_init_capture_parent",
        _VERIFY_FAILED: "prefill_init_verify_parent",
        _RUNTIME_FAILED: "prefill_replay_error_parent",
        _CLONE_FAILED: "prefill_clone_failed_parent",
    }.get(state, "prefill_guard_parent")


def _matching_input(ids):
    return (
        isinstance(ids, torch.Tensor)
        and ids.dim() == 2
        and tuple(ids.shape) == _HOLDER["shape"]
        and ids.dtype is torch.int64
        and ids.is_cuda
        and ids.get_device() == _HOLDER["device_index"]
    )


def _make_repl(call_id, ref, model, new_tokens, record):
    def prefill_repl(*args, **kwargs):
        try:
            if (
                args
                or not set(kwargs) <= _ALLOWED_KWARGS
                or kwargs.get("use_cache") is not True
                or kwargs.get("logits_to_keep") != 1
                or not all(kwargs.get(k) is None for k in _NONE_ONLY)
            ):
                record(call_id, "prefill_signature_parent")
                return ref(*args, **kwargs)
            state = _HOLDER["state"]
            if state == _READY:
                if _matching_input(kwargs.get("input_ids")):
                    _HOLDER["static_ids"].copy_(kwargs["input_ids"])
                    _HOLDER["graph"].replay()
                    logits = _HOLDER["static_logits"].clone()
                    fresh = _materialize_cache(_HOLDER["static_pairs"])
                    if fresh is not None:
                        record(call_id, "prefill_graph_replay")
                        return _HOLDER["out_cls"](logits=logits, past_key_values=fresh)
                    _HOLDER["state"] = _CLONE_FAILED
                    record(call_id, "prefill_clone_failed_parent")
                    return ref(*args, **kwargs)
                record(call_id, "prefill_signature_parent")
                return ref(*args, **kwargs)
            if state == _UNINITIALIZED:
                # Lazy setup inside a legitimate invocation; this invocation still
                # executes the current input workload (warmup + capture + replay all
                # ran with this call's input_ids).
                if _lazy_init(ref, kwargs.get("input_ids"), model, new_tokens):
                    logits = _HOLDER["static_logits"].clone()
                    fresh = _materialize_cache(_HOLDER["static_pairs"])
                    if fresh is not None:
                        record(call_id, "prefill_graph_replay")
                        return _HOLDER["out_cls"](logits=logits, past_key_values=fresh)
                    _HOLDER["state"] = _CLONE_FAILED
                    record(call_id, "prefill_clone_failed_parent")
                    return ref(*args, **kwargs)
                record(call_id, _label_for(_HOLDER["state"]))
                return ref(*args, **kwargs)
            record(call_id, _label_for(state))
            return ref(*args, **kwargs)
        except Exception:
            if _HOLDER["state"] == _READY:
                _HOLDER["state"] = _RUNTIME_FAILED
                _HOLDER["graph"] = None
                _HOLDER["static_logits"] = None
                _HOLDER["static_pairs"] = None
            record(call_id, _label_for(_HOLDER["state"]))
            return ref(*args, **kwargs)

    return prefill_repl


def build(context):
    """Return {call_id: callable} replacements for this rank's local targets.

    No GPU work happens here: capture must occur while the inherited dispatch modes are
    entered and the prefill phase is active, which is only true inside adapter.run.
    """
    state = context.get("state") or {}
    record = context.get("record_route") or (lambda *_: None)
    models = context.get("models") or {}
    model = models.get("model") or state.get("model")
    references = context.get("references") or {}

    targets = context.get("targets") or {}
    if isinstance(targets, dict):
        items = targets.items()
    else:
        items = ((spec["id"], spec) for spec in targets)

    new_tokens = state.get("new_tokens")
    if isinstance(new_tokens, int) and not isinstance(new_tokens, bool):
        new_tokens = int(new_tokens)

    out = {}
    for call_id, spec in items:
        try:
            if spec.get("kind") != "module" or spec.get("target") != "model":
                continue
            if spec.get("phase", "prefill") != "prefill":
                continue
            if "Qwen3ForCausalLM" not in str(spec.get("semantics") or ""):
                continue
            sigs = spec.get("signatures") or []
            if not sigs:
                continue
            ids_meta = (sigs[0].get("kwargs") or {}).get("input_ids") or {}
            if tuple(ids_meta.get("shape") or ()) != (1, 256):
                continue
            if not ids_meta.get("dtype", "torch.int64").endswith("int64"):
                continue
            if model is None:
                continue
            ref = references.get(call_id)
            if not callable(ref):
                ref = model.forward
            if not callable(ref):
                continue
            if _HOLDER["state"] == _READY and _HOLDER.get("model") is not model:
                # Different model object: the previous graph no longer applies.
                _disable(_RUNTIME_FAILED, _HOLDER.get("graph"))
            out[call_id] = _make_repl(call_id, ref, model, new_tokens, record)
        except Exception:
            continue
    return out
