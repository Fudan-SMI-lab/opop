"""Micro-batched ResNet-18 stem re-binding (model.resnet.embedder, call
d16297b8a11b79a2657853d0) at half chunk size for the offline batch-32
peak_allocated_bytes objective.

The accepted stack (stem rows=8 bundle trial-e9279012c6-param-9ac5fd08 plus
stage-0 rows=8 bundle trial-fd9bdc62a2-param-b6738601) measures
177,409,024 B = live FP32 params 46,758,048 + GPU-resident input 19,267,584 +
retained workspace 8,622,944 + 4 x 25,690,112, where 25,690,112 B is one
32x64x56x56 (equivalently 8x64x112x112) FP32 unit. Three of those units
coexist only in the stem phase (full [32,64,56,56] assembly, chunk conv_out
8x64x112x112, chunk bn_out 8x64x112x112); the fourth is not statically
explained (per-operator allocation attribution is an unknown; plausibly a
row-scaled conv workspace). Halving the stem chunk to 4 rows turns conv_out
and bn_out into 12,845,056 B each, so this replacement explicitly replaces the
installed stem bundle (same call site) and runs the unchanged
Conv2d/BatchNorm2d/ReLU parents and the aten.max_pool2d operator - always
through context.references so inherited calls, hooks and their coverage stay
real - on contiguous row chunks, assembling results into one [32,64,56,56]
tensor consumed unchanged by the installed stage-0 bundle and later stages.

Every stem operator (valid 7x7 stride-2 conv, eval BatchNorm, out-of-place
ReLU, max_pool2d) acts per sample, and the evaluator already accepted this
exact pattern twice (stem rows=8, stage-0 rows=8) with max_abs_error
7.6e-06 < atol=rtol=1e-4 and exact top-1.

No parameter transformation, no output caching and no CUDA-graph replay: every
invocation recomputes from the current input through the parent callables.
"""
from __future__ import annotations

import torch

STEM_CALL = "d16297b8a11b79a2657853d0"
CONV_CALL = "11c99c108749a5d6fc1a0e93"
NORM_CALL = "6b96219304ac79385e4ef06c"
ACT_CALL = "c98061ff56600d8872eade43"
POOL_CALL = "0c8cafb1fc30de539c787552"
LEGACY_EMBEDDER_CALL = "1dcfd07639d32233c49eafdf"
LEGACY_POOLER_CALL = "babacb997a72996bf303d028"
DEFAULT_CHUNK_ROWS = 4


def build(context):
    references = context.get("references") or {}
    parent = references.get(STEM_CALL)
    if parent is None:
        return {}
    conv_ref = references.get(CONV_CALL)
    norm_ref = references.get(NORM_CALL)
    act_ref = references.get(ACT_CALL)
    embedder_ref = references.get(LEGACY_EMBEDDER_CALL)
    if conv_ref is not None and norm_ref is not None and act_ref is not None:
        def stem_chain(x):
            return act_ref(norm_ref(conv_ref(x)))
    elif embedder_ref is not None:
        stem_chain = embedder_ref
    else:
        return {}
    pool_ref = references.get(POOL_CALL)
    pooler_ref = references.get(LEGACY_POOLER_CALL)
    if pool_ref is not None:
        def stem_pool(x):
            return pool_ref(x, [3, 3], [2, 2], [1, 1])
    elif pooler_ref is not None:
        stem_pool = pooler_ref
    else:
        return {}
    modules = context.get("modules") or {}
    guard_modules = (
        modules.get(STEM_CALL),
        modules.get(CONV_CALL),
        modules.get(NORM_CALL),
        modules.get(ACT_CALL),
    )
    record_route = context.get("record_route", lambda *_: None)
    parameters = context.get("parameters") or {}
    try:
        chunk_rows = int(parameters.get("stem_chunk_rows", DEFAULT_CHUNK_ROWS))
    except (TypeError, ValueError):
        chunk_rows = DEFAULT_CHUNK_ROWS
    if chunk_rows < 2:
        chunk_rows = 2

    def _training_active():
        for module in guard_modules:
            if module is not None and module.training:
                return True
        return False

    def _compatible(x):
        return (
            isinstance(x, torch.Tensor)
            and x.dim() == 4
            and x.is_cuda
            and x.dtype == torch.float32
            and x.shape[1] == 3
            and x.is_contiguous()
            and not x.requires_grad
        )

    def stem(pixel_values=None, *args, **kwargs):
        if (
            args
            or kwargs
            or not _compatible(pixel_values)
            or pixel_values.shape[0] <= chunk_rows
            or _training_active()
        ):
            record_route(STEM_CALL, "stem_parent_signature_fallback")
            return parent(pixel_values, *args, **kwargs)
        try:
            total = pixel_values.shape[0]
            first = stem_pool(stem_chain(pixel_values[:chunk_rows]))
            head = first.shape[0]
            if head >= total:
                record_route(STEM_CALL, "stem_single_chunk")
                return first
            out = torch.empty(
                (total, *first.shape[1:]), dtype=first.dtype, device=first.device
            )
            out[:head].copy_(first)
            del first
            for start in range(head, total, chunk_rows):
                stop = min(start + chunk_rows, total)
                out[start:stop].copy_(
                    stem_pool(stem_chain(pixel_values[start:stop]))
                )
        except Exception as exc:
            record_route(
                STEM_CALL, "stem_parent_runtime_fallback:" + type(exc).__name__
            )
            return parent(pixel_values)
        record_route(STEM_CALL, "stem_microbatch_rows%d" % chunk_rows)
        return out

    return {STEM_CALL: stem}
