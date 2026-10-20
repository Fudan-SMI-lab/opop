from __future__ import annotations

import torch

EMBEDDER_CALL = "fa7ef6548e3a2d54174caeba"
CONV_LAYER_CALL = "fad32b28bd6e0def6ce7593e"
POOLER_CALL = "12da68be9b9e14502fe4702a"
DEFAULT_MICRO_BATCH = 8
LEGAL_MICRO_BATCH = (4, 8, 16)


def build(context):
    references = context.get("references") or {}
    modules = context.get("modules") or {}
    record_route = context.get("record_route", lambda *_: None)

    try:
        parent_embedder = references[EMBEDDER_CALL]
        conv_layer_forward = references[CONV_LAYER_CALL]
        pooler_forward = references[POOLER_CALL]
        embedder_module = modules[EMBEDDER_CALL]
        num_channels = int(embedder_module.num_channels)
    except Exception:
        return {}

    parameters = context.get("parameters") or {}
    raw_micro = parameters.get("stem_micro_batch", DEFAULT_MICRO_BATCH)
    try:
        micro = int(raw_micro)
    except (TypeError, ValueError):
        micro = DEFAULT_MICRO_BATCH
    if micro not in LEGAL_MICRO_BATCH:
        micro = DEFAULT_MICRO_BATCH

    def stem_forward(*args, **kwargs):
        if kwargs or len(args) != 1:
            record_route(EMBEDDER_CALL, "fallback_parent_signature")
            return parent_embedder(*args, **kwargs)
        pixel_values = args[0]
        supported = (
            isinstance(pixel_values, torch.Tensor)
            and pixel_values.dim() == 4
            and pixel_values.dtype == torch.float32
            and pixel_values.is_cuda
            and pixel_values.is_contiguous()
            and not pixel_values.requires_grad
            and pixel_values.shape[1] == num_channels
            and pixel_values.shape[0] > micro
        )
        if not supported or torch.is_grad_enabled() or embedder_module.training:
            record_route(EMBEDDER_CALL, "fallback_parent_signature")
            return parent_embedder(*args, **kwargs)
        try:
            outputs = []
            total = pixel_values.shape[0]
            for start in range(0, total, micro):
                chunk = pixel_values[start : start + micro]
                hidden = conv_layer_forward(chunk)
                outputs.append(pooler_forward(hidden))
        except (RuntimeError, ValueError):
            record_route(EMBEDDER_CALL, "fallback_parent_runtime")
            return parent_embedder(*args, **kwargs)
        record_route(EMBEDDER_CALL, "stem_microbatch")
        return torch.cat(outputs, dim=0)

    return {EMBEDDER_CALL: stem_forward}
