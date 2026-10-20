"""Candidate: CUDA Graph replay of the frozen-shape Qwen3 prefill region.

Replaces the prefill-phase call of the root ``model`` target
(transformers Qwen3ForCausalLM) with a CUDA Graph wrapper:

* first eligible invocation: eager parent warmups on a side stream, one
  capture of the untouched parent forward, one replay, then return the
  captured output object (which then holds real values);
* every later eligible invocation: copy ``input_ids`` into the static input
  buffer, replay the graph (all captured kernels re-execute on the GPU),
  restore the DynamicCache layer tensors to the captured prefill buffers,
  and return the same output object.

The captured kernels are exactly the parent's kernels: no math, dtype, mask,
sampling or data semantics change, so outputs are bit-identical to the parent
run while the eager per-op CPU dispatch/launch overhead inside the ttft
window is removed. Any capture or replay error permanently falls back to the
parent forward.
"""

from __future__ import annotations

import torch

_WARMUP_ITERS = 3
_PREFILL_BATCH = 1
_PREFILL_SEQ = 256


def _find_model_call_id(targets):
    for call_id, spec in targets.items():
        if not isinstance(spec, dict):
            continue
        if spec.get("target") != "model" or spec.get("kind") != "module":
            continue
        if spec.get("phase") not in (None, "prefill"):
            continue
        for sig in spec.get("signatures") or []:
            if not isinstance(sig, dict):
                continue
            kw = sig.get("kwargs") or {}
            ids = kw.get("input_ids")
            if not isinstance(ids, dict):
                continue
            if (
                list(ids.get("shape") or ()) == [_PREFILL_BATCH, _PREFILL_SEQ]
                and kw.get("use_cache") is True
                and kw.get("logits_to_keep") == 1
                and kw.get("past_key_values") is None
            ):
                return call_id
    return None


def build(context):
    targets = context.get("targets") or {}
    references = context.get("references") or {}
    call_id = _find_model_call_id(targets)
    if call_id is None or call_id not in references:
        return {}
    parent = references[call_id]

    state = {
        "graph": None,
        "output": None,
        "static_input": None,
        "cache_reset": (),
        "failed": False,
    }

    def _eligible(input_ids, attention_mask, position_ids, past_key_values,
                  inputs_embeds, labels, use_cache, logits_to_keep, extra):
        if state["failed"] or extra:
            return False
        if input_ids is None or not input_ids.is_cuda:
            return False
        if tuple(input_ids.shape) != (_PREFILL_BATCH, _PREFILL_SEQ):
            return False
        if input_ids.dtype not in (torch.long, torch.int):
            return False
        return (
            attention_mask is None
            and position_ids is None
            and past_key_values is None
            and inputs_embeds is None
            and labels is None
            and use_cache is True
            and isinstance(logits_to_keep, int)
            and logits_to_keep == 1
        )

    def _capture(input_ids):
        torch.cuda.synchronize()
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(_WARMUP_ITERS):
                parent(input_ids=input_ids, use_cache=True, logits_to_keep=1)
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()

        static_input = input_ids.detach().clone()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = parent(input_ids=static_input, use_cache=True, logits_to_keep=1)
        graph.replay()
        torch.cuda.synchronize()

        cache = getattr(out, "past_key_values", None)
        layers = getattr(cache, "layers", None)
        reset = []
        if layers is not None:
            for layer in layers:
                reset.append((layer, layer.keys, layer.values))
        elif cache is not None:
            raise RuntimeError("unsupported cache structure for graph replay")

        state["graph"] = graph
        state["output"] = out
        state["static_input"] = static_input
        state["cache_reset"] = tuple(reset)

    def _prefill(input_ids):
        if state["graph"] is None:
            _capture(input_ids)
        state["static_input"].copy_(input_ids)
        state["graph"].replay()
        for layer, keys, values in state["cache_reset"]:
            layer.keys = keys
            layer.values = values
        return state["output"]

    def replacement(input_ids=None, attention_mask=None, position_ids=None,
                    past_key_values=None, inputs_embeds=None, labels=None,
                    use_cache=None, logits_to_keep=0, **extra):
        if _eligible(input_ids, attention_mask, position_ids,
                     past_key_values, inputs_embeds, labels, use_cache,
                     logits_to_keep, extra):
            try:
                return _prefill(input_ids)
            except Exception:
                state["failed"] = True
        return parent(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            logits_to_keep=logits_to_keep,
            **extra,
        )

    return {call_id: replacement}
