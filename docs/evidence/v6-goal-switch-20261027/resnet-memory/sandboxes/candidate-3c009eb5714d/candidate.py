"""Micro-batched ResNet-18 stem (model.resnet.embedder) for the offline batch-32
peak_allocated_bytes objective.

The baseline peak (280,169,472 B) is created inside the stem: the out-of-place
conv1 output (32x64x112x112x4 = 102,760,448 B) coexists with the equally sized
out-of-place batch_norm1 output next to the live FP32 parameters, the
GPU-resident input and conv workspace. This replacement keeps the frozen
batch-32 request (all 32 rows execute) but runs the unchanged stem sub-chain
(conv -> bn(eval) -> relu -> maxpool, always invoked through
context.references so inherited calls and their coverage remain real) on
contiguous row chunks, assembling the chunk results into the same
[32, 64, 56, 56] hidden state consumed by the untouched later stages.

No parameter transformation, no output caching and no CUDA-graph replay: every
invocation recomputes from the current input through the parent callables.
"""
from __future__ import annotations

import torch

STEM_CALL = "d16297b8a11b79a2657853d0"
CONV_LAYER_CALL = "1dcfd07639d32233c49eafdf"
POOLER_CALL = "babacb997a72996bf303d028"
DEFAULT_CHUNK_ROWS = 8


def build(context):
    references = context.get("references") or {}
    parent = references.get(STEM_CALL)
    conv_layer = references.get(CONV_LAYER_CALL)
    pooler = references.get(POOLER_CALL)
    if parent is None or conv_layer is None or pooler is None:
        return {}
    modules = context.get("modules") or {}
    stem_module = modules.get(STEM_CALL)
    conv_module = modules.get(CONV_LAYER_CALL)
    record_route = context.get("record_route", lambda *_: None)
    parameters = context.get("parameters") or {}
    try:
        chunk_rows = int(parameters.get("stem_chunk_rows", DEFAULT_CHUNK_ROWS))
    except (TypeError, ValueError):
        chunk_rows = DEFAULT_CHUNK_ROWS
    if chunk_rows < 2:
        chunk_rows = 2

    def _training_active():
        for module in (stem_module, conv_module):
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
            first = pooler(conv_layer(pixel_values[:chunk_rows]))
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
                    pooler(conv_layer(pixel_values[start:stop]))
                )
        except Exception as exc:
            record_route(
                STEM_CALL, "stem_parent_runtime_fallback:" + type(exc).__name__
            )
            return parent(pixel_values)
        record_route(STEM_CALL, "stem_microbatch_rows%d" % chunk_rows)
        return out

    return {STEM_CALL: stem}
