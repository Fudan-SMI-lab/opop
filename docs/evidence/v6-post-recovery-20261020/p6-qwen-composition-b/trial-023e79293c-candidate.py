"""Candidate: whole-prefill CUDA Graph execution policy for the Qwen3-4B TTFT region.

Selection (inputs/selection.json): region scope, group e0861b46dbeeb562f67f - the
Qwen3ForCausalLM module call in the prefill phase (call 900c23e99c808d2c9bda215a,
signature input_ids=[1,256] int64, use_cache=True, logits_to_keep=1).

Diagnosis this attempt repairs (inputs/previous-feedback.txt):
  * The previous trial's custom GEMM replacements never executed (new_routes were
    "kv_parent" and "lm_parent" only: both build-time gates rejected the custom
    paths on the worker device), while its 73 wrapper callables and per-call guard
    machinery stayed installed as pure overhead, and ttft regressed to 89.83 ms vs
    the 70.90 ms reference (paired gains -19.14/-18.71 ms, below the 14.61/4.97 ms
    noise bounds).
  * The measured objective is host-dispatch bound, not GPU bound: the selected
    model prefill call runs 1420 kernels for 20915.875 us diagnostic CUDA
    (target-resources.json), while the parent ttft is ~70.9 ms; decode shows the
    same signature (~2566 ms wall vs ~342 ms diagnostic CUDA).

Mechanism: replace exactly one call - the whole model prefill forward - with a
CUDA Graph replay policy:
  * Build time (untimed): 3 warmup forwards of the current parent reference on a
    side stream, one graph capture, one replay, then BITWISE verification of the
    replayed outputs against the eager parent outputs (logits and every per-layer
    KV tensor via torch.equal). Any failure permanently disables the custom path
    and the callable routes to the parent reference.
  * Every invocation: copy input_ids into the 2 KiB static buffer, replay the
    graph (the identical kernel stream, including the installed bundle's fused
    pointwise kernels captured through the current parent implementation), clone
    the 0.3 MB logits and deep-clone the ~75 MB KV cache so decode receives a
    fresh cache object with regular-allocator tensors, and return a
    CausalLMOutputWithPast built from those fresh values.

The workload executes on each invocation (all 1420 prefill kernels re-run from
the input buffer); no outputs are cached. Decode is untouched: the replacement is
guarded to the observed prefill signature (no past_key_values, [1,256] int64
input, use_cache=True, logits_to_keep=1) and every other signature falls back to
the parent reference. All GPU work stays on the calling stream inside the
declared workload boundary; no extra synchronization is introduced at run time.

Worker-environment note (inputs/worker-environment.json): verified against
torch 2.13.0+cu129 / transformers 5.16.1 - DynamicCache exposes .layers with
.keys/.values, update(key_states, value_states, layer_idx), deepcopy clones
tensors, and CausalLMOutputWithPast(logits=..., past_key_values=...) constructs
normally; the clone helpers below also keep a transformers-4.x fallback
(.key_cache/.value_cache and reversed update argument order).
"""
from __future__ import annotations

import copy as _copy

import torch

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

_HOLDER = {
    "ok": False,
    "graph": None,
    "static_ids": None,
    "static_out": None,
    "shape": None,
    "device": None,
    "out_cls": None,
    "model": None,
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


def _cache_equal(a, b):
    pa = _cache_pairs(a)
    pb = _cache_pairs(b)
    if pa is None or pb is None or len(pa) != len(pb):
        return False
    for (ka, va), (kb, vb) in zip(pa, pb):
        if ka.shape != kb.shape or va.shape != vb.shape:
            return False
        if not (torch.equal(ka, kb) and torch.equal(va, vb)):
            return False
    return True


def _cache_disjoint(a, b):
    pa = _cache_pairs(a)
    pb = _cache_pairs(b)
    if pa is None or pb is None or len(pa) != len(pb) or not pa:
        return False
    return pa[0][0].data_ptr() != pb[0][0].data_ptr()


def _clone_cache(cache):
    """Fresh cache with cloned tensors; never aliases graph-pool storage."""
    try:
        fresh = _copy.deepcopy(cache)
        if fresh is not cache and _cache_equal(fresh, cache) and _cache_disjoint(fresh, cache):
            return fresh
    except Exception:
        pass
    pairs = _cache_pairs(cache)
    if pairs:
        try:
            from transformers.cache_utils import DynamicCache

            fresh = DynamicCache()
            for idx, (k, v) in enumerate(pairs):
                kc = k.detach().clone()
                vc = v.detach().clone()
                try:
                    fresh.update(kc, vc, idx)
                except TypeError:
                    fresh.update(idx, kc, vc)
            if _cache_equal(fresh, cache) and _cache_disjoint(fresh, cache):
                return fresh
        except Exception:
            pass
    return None


def _reset_holder():
    _HOLDER.update(
        ok=False, graph=None, static_ids=None, static_out=None,
        shape=None, device=None, out_cls=None, model=None,
    )


def _capture(ref, static_ids, error_mode=None):
    """Capture the parent prefill forward; returns (graph, static_out)."""
    graph = torch.cuda.CUDAGraph()
    if error_mode is None:
        ctx = torch.cuda.graph(graph)
    else:
        ctx = torch.cuda.graph(graph, capture_error_mode=error_mode)
    with ctx:
        static_out = ref(
            input_ids=static_ids, use_cache=True, logits_to_keep=1
        )
    return graph, static_out


def _try_build_graph(ref, sample_ids, model):
    """One-time capture + bitwise verification. Returns True when enabled."""
    if not torch.cuda.is_available():
        return False
    if not hasattr(torch.cuda, "CUDAGraph") or not hasattr(torch.cuda, "graph"):
        return False
    graph = None
    try:
        with torch.inference_mode():
            device = sample_ids.device
            static_ids = torch.empty_like(sample_ids)
            static_ids.copy_(sample_ids)
            cur = torch.cuda.current_stream()
            side = torch.cuda.Stream(device=device)
            side.wait_stream(cur)
            warm = None
            with torch.cuda.stream(side):
                for _ in range(3):
                    warm = ref(
                        input_ids=static_ids, use_cache=True, logits_to_keep=1
                    )
            cur.wait_stream(side)
            torch.cuda.synchronize(device)
            try:
                graph, static_out = _capture(ref, static_ids)
            except Exception:
                graph, static_out = _capture(ref, static_ids, "thread_local")
            graph.replay()
            torch.cuda.synchronize(device)
            if warm is None or static_out is None:
                return False
            if not torch.equal(
                static_out.logits.detach(), warm.logits.detach()
            ):
                return False
            if not _cache_equal(
                static_out.past_key_values, warm.past_key_values
            ):
                return False
            fresh = _clone_cache(static_out.past_key_values)
            if fresh is None:
                return False
            probe = type(static_out)(
                logits=static_out.logits.detach().clone(),
                past_key_values=fresh,
            )
            if probe.logits.shape != static_out.logits.shape:
                return False
            _HOLDER.update(
                ok=True,
                graph=graph,
                static_ids=static_ids,
                static_out=static_out,
                shape=tuple(static_ids.shape),
                device=device,
                out_cls=type(static_out),
                model=model,
            )
            return True
    except Exception:
        try:
            if graph is not None:
                graph.reset()
        except Exception:
            pass
        return False


def _make_repl(call_id, ref, record):
    def prefill_repl(*args, **kwargs):
        try:
            if (
                _HOLDER["ok"]
                and not args
                and set(kwargs) <= _ALLOWED_KWARGS
                and kwargs.get("use_cache") is True
                and kwargs.get("logits_to_keep") == 1
                and all(kwargs.get(k) is None for k in _NONE_ONLY)
            ):
                ids = kwargs.get("input_ids")
                if (
                    isinstance(ids, torch.Tensor)
                    and ids.dim() == 2
                    and tuple(ids.shape) == _HOLDER["shape"]
                    and ids.dtype is torch.int64
                    and ids.device == _HOLDER["device"]
                ):
                    _HOLDER["static_ids"].copy_(ids)
                    _HOLDER["graph"].replay()
                    logits = _HOLDER["static_out"].logits.detach().clone()
                    fresh = _clone_cache(_HOLDER["static_out"].past_key_values)
                    if fresh is not None:
                        record(call_id, "prefill_graph_replay")
                        return _HOLDER["out_cls"](
                            logits=logits, past_key_values=fresh
                        )
                    record(call_id, "prefill_cache_clone_failed_parent")
                else:
                    record(call_id, "prefill_signature_parent")
            else:
                record(call_id, "prefill_guard_parent")
        except Exception:
            record(call_id, "prefill_error_parent")
        return ref(*args, **kwargs)

    return prefill_repl


def build(context):
    """Return {call_id: callable} replacements for this rank's local targets."""
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
            ref = references.get(call_id)
            if not callable(ref):
                if model is None:
                    continue
                ref = model.forward
            sample = state.get("inputs")
            if not (
                isinstance(sample, torch.Tensor)
                and tuple(sample.shape) == (1, 256)
                and sample.dtype is torch.int64
            ):
                device = state.get("device")
                device = torch.device(device) if device else torch.device("cuda:0")
                if device.type != "cuda" or not torch.cuda.is_available():
                    continue
                sample = torch.zeros((1, 256), dtype=torch.int64, device=device)
            if _HOLDER["ok"] and _HOLDER["model"] is model:
                pass  # reuse the already-verified graph for this state
            else:
                _reset_holder()
                _try_build_graph(ref, sample, model)
            out[call_id] = _make_repl(call_id, ref, record)
        except Exception:
            continue
    return out
