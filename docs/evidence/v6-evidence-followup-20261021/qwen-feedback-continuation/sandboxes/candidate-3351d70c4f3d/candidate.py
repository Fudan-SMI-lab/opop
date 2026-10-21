"""Repair candidate for the Qwen3-4B TTFT region: CUDA Graph capture+replay of the
parent's own prefill at the region root (Qwen3ForCausalLM, prefill phase).

Repair of trial-4556240c94 (its ancestor trial-e262267559 already repaired the
numerics). That bundle passed quality but was rejected for
"paired_gain_not_above_noise"/"objective_not_improved": candidate TTFT medians
were 30.63/25.59 ms vs parent 24.94/25.13 ms. It reimplemented the layer body
with Triton elementwise kernels while keeping aten reductions/GEMMs/SDPA, so the
GPU work was unchanged but ~300 extra Python-side Triton launches per prefill
added host dispatch cost. Measured facts from the collector records:

- target-resources.json root call c9d14173dd4887d4a4133553 (qwen3_4b, prefill):
  2145 kernels, cuda_us 21624.6 (GEMM 15.51 ms, elementwise 3.30 ms, gemv/lm_head
  0.83 ms, reduce 0.71 ms, cat 0.54 ms, flash 0.74 ms) -> GPU busy ~21.6 ms.
- Warm parent wall TTFT is 24.94-25.13 ms -> ~3.3-3.5 ms of per-invocation
  host-launch gap between kernels (python_runtime shows transformers forward
  self-time dominating the host profile; the decode loop shows the same effect
  amplified: ~11.8 ms GPU busy vs ~21.8 ms wall per step).

So the profitable, semantics-preserving lever is launch-gap removal, not kernel
replacement. This candidate captures the CURRENT parent callable
(context.references[call_id]) — not a reimplementation — into a CUDA graph
lazily, inside the first real graph-eligible invocation (the adapter's phase
context and any installed dispatch modes are live there, per the binding
contract), then on every subsequent matching prefill call copies the current
input_ids into the static buffer and replays the graph. The same captured
kernels, in the same order, on the same addresses, execute on every invocation:

- outputs are bitwise identical to an eager parent run (identical kernels and
  order; deterministic stack: benchmark=False, TF32 off, fixed seed), so exact
  greedy tokens and the 0.02 logits tolerance hold by construction, and a
  bitwise validation (logits + every per-layer K/V tensor) runs once at capture
  time before the replay path is ever used;
- no fixture capture: only kernel work is captured; input tokens are copied in
  and all math re-executes per invocation on current inputs;
- the KV cache returned per replay is a fresh DynamicCache wrapper (config-built
  like the one Qwen3Model.forward creates) referencing the static prefill K/V
  tensors, so the untouched eager decode loop sees exactly parent cache state;
- decode-step calls and any other signature/shape fall back to the parent with
  distinct route labels (context / signature / setup failure / validation /
  runtime failure); every failure category permanently reverts to the parent so
  a broken graph can never corrupt quality — it only forgoes the gain.
"""

from __future__ import annotations

import torch

try:
    from transformers.cache_utils import DynamicCache, DynamicLayer
    from transformers.modeling_outputs import CausalLMOutputWithPast

    _IMPORT_OK = True
except Exception:  # pragma: no cover - worker ships transformers 5.16.1
    _IMPORT_OK = False

_LABEL_GRAPH = "graph_replay_prefill"
_LABEL_FIRST = "graph_first_call_eager_capture_validated"
_LABEL_CONTEXT = "parent_fallback_context"
_LABEL_SIGNATURE = "parent_fallback_signature"
_LABEL_SETUP = "parent_fallback_setup_failure"
_LABEL_VALIDATION = "parent_fallback_validation"
_LABEL_RUNTIME = "parent_fallback_runtime_failure"

_WARMUP_ITERS = 2


def _find_region_root(targets):
    for call_id, spec in targets.items():
        if not isinstance(spec, dict):
            continue
        if spec.get("kind") != "module" or spec.get("phase") != "prefill":
            continue
        if str(spec.get("semantics") or "").endswith("Qwen3ForCausalLM"):
            return call_id
    return None


def build(context):
    if not _IMPORT_OK or not hasattr(torch.cuda, "CUDAGraph") or not hasattr(torch.cuda, "graph"):
        return {}
    targets = context.get("targets") or {}
    references = context.get("references") or {}
    modules = context.get("modules") or {}
    record_route = context.get("record_route")
    if not callable(record_route):
        record_route = lambda *_: None

    call_id = _find_region_root(targets)
    if call_id is None or call_id not in references:
        return {}
    parent = references[call_id]

    module = modules.get(call_id)
    config = getattr(module, "config", None) if module is not None else None

    # "idle" -> "ready" | "failed"; single-threaded adapter execution
    st = {"phase": "idle", "label": _LABEL_SETUP, "reason": None}
    gs = {
        "graph": None,
        "static_in": None,
        "logits": None,
        "keys": None,
        "values": None,
        "shape": None,
        "device": None,
        "n_layers": 0,
        "config": config,
    }

    def _build_cache():
        """Fresh DynamicCache wrapper around the static prefill K/V tensors.

        Structurally identical to the cache Qwen3Model.forward builds
        (DynamicCache(config=...)); a fresh object per invocation keeps the
        harness's decode loop from aliasing one mutable cache across runs.
        """
        keys = gs["keys"]
        values = gs["values"]
        cache = None
        layers = None
        cfg = gs["config"]
        if cfg is not None:
            try:
                candidate_cache = DynamicCache(config=cfg)
                if len(candidate_cache.layers) == len(keys):
                    cache = candidate_cache
                    layers = candidate_cache.layers
            except Exception:
                cache = None
        if cache is None:
            cache = DynamicCache()
            layers = [DynamicLayer() for _ in keys]
            cache.layers.extend(layers)
        for layer, key, value in zip(layers, keys, values):
            layer.keys = key
            layer.values = value
            layer.is_initialized = True
            layer.dtype = key.dtype
            layer.device = key.device
        return cache

    def _capture(input_ids):
        """Capture the current parent prefill into a CUDA graph (lazy setup)."""
        device = input_ids.device
        static_in = torch.empty_like(input_ids)
        static_in.copy_(input_ids)

        side = torch.cuda.Stream(device=device)
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(_WARMUP_ITERS):
                parent(input_ids=static_in, use_cache=True, logits_to_keep=1)
        torch.cuda.current_stream().wait_stream(side)

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=side):
            captured = parent(input_ids=static_in, use_cache=True, logits_to_keep=1)

        layers = list(captured.past_key_values.layers)
        if not layers:
            raise RuntimeError("parent returned an empty KV cache during capture")
        gs["graph"] = graph
        gs["static_in"] = static_in
        gs["logits"] = captured.logits
        gs["keys"] = [layer.keys for layer in layers]
        gs["values"] = [layer.values for layer in layers]
        gs["shape"] = tuple(input_ids.shape)
        gs["device"] = device
        gs["n_layers"] = len(layers)

    def _validate_against(ref_out):
        """Replay once and require bitwise equality with the eager parent run."""
        gs["graph"].replay()
        logits = gs["logits"]
        ref_logits = getattr(ref_out, "logits", None)
        if logits is None or ref_logits is None or logits.shape != ref_logits.shape:
            return False
        if not torch.equal(logits, ref_logits):
            return False
        ref_cache = getattr(ref_out, "past_key_values", None)
        try:
            ref_layers = list(ref_cache.layers)
        except Exception:
            return False
        if len(ref_layers) != gs["n_layers"]:
            return False
        for ref_layer, key, value in zip(ref_layers, gs["keys"], gs["values"]):
            if not (
                torch.equal(ref_layer.keys, key) and torch.equal(ref_layer.values, value)
            ):
                return False
        return True

    def prefill_forward(
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
        call_kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "past_key_values": past_key_values,
            "inputs_embeds": inputs_embeds,
            "labels": labels,
            "use_cache": use_cache,
            "logits_to_keep": logits_to_keep,
        }
        call_kwargs.update(kwargs)

        tensor_ok = isinstance(input_ids, torch.Tensor) and input_ids.is_cuda
        if (
            not tensor_ok
            or torch.is_grad_enabled()
            or torch.cuda.is_current_stream_capturing()
        ):
            record_route(call_id, _LABEL_CONTEXT)
            return parent(**call_kwargs)
        if not (
            input_ids.ndim == 2
            and input_ids.dtype == torch.int64
            and 0 < input_ids.shape[0] * input_ids.shape[1] <= 65536
            and attention_mask is None
            and position_ids is None
            and past_key_values is None
            and inputs_embeds is None
            and labels is None
            and use_cache is True
            and isinstance(logits_to_keep, int)
            and not isinstance(logits_to_keep, bool)
            and logits_to_keep == 1
            and not kwargs
        ):
            record_route(call_id, _LABEL_SIGNATURE)
            return parent(**call_kwargs)

        if st["phase"] == "failed":
            record_route(call_id, st["label"])
            return parent(**call_kwargs)

        if st["phase"] == "ready":
            if tuple(input_ids.shape) != gs["shape"] or input_ids.device != gs["device"]:
                record_route(call_id, _LABEL_SIGNATURE)
                return parent(**call_kwargs)
            try:
                gs["static_in"].copy_(input_ids)
                gs["graph"].replay()
                output = CausalLMOutputWithPast(
                    loss=None,
                    logits=gs["logits"],
                    past_key_values=_build_cache(),
                    hidden_states=None,
                    attentions=None,
                )
            except Exception as exc:
                st["phase"] = "failed"
                st["label"] = _LABEL_RUNTIME
                st["reason"] = repr(exc)[:300]
                record_route(call_id, _LABEL_RUNTIME)
                return parent(**call_kwargs)
            record_route(call_id, _LABEL_GRAPH)
            return output

        # First graph-eligible invocation: eager parent run (its result is this
        # invocation's output), then lazy capture + one bitwise validation
        # replay. All of this runs inside the live adapter invocation context,
        # capturing exactly the accepted parent's work.
        ref_out = parent(input_ids=input_ids, use_cache=True, logits_to_keep=1)
        try:
            _capture(input_ids)
        except Exception as exc:
            st["phase"] = "failed"
            st["label"] = _LABEL_SETUP
            st["reason"] = repr(exc)[:300]
            record_route(call_id, _LABEL_SETUP)
            return ref_out
        try:
            if not _validate_against(ref_out):
                st["phase"] = "failed"
                st["label"] = _LABEL_VALIDATION
                st["reason"] = "replay not bitwise-equal to eager parent run"
                record_route(call_id, _LABEL_VALIDATION)
                return ref_out
        except Exception as exc:
            st["phase"] = "failed"
            st["label"] = _LABEL_RUNTIME
            st["reason"] = repr(exc)[:300]
            record_route(call_id, _LABEL_RUNTIME)
            return ref_out
        st["phase"] = "ready"
        record_route(call_id, _LABEL_FIRST)
        return ref_out

    return {call_id: prefill_forward}
