from __future__ import annotations

import torch

RESNET_MODEL_CALL = "fb7277dba44d770ed1cbb22f"
EMBEDDER_CALL = "fa7ef6548e3a2d54174caeba"
ENCODER_CALL = "5784db785cf33b978d46d02b"
POOLER_CALL = "764b049d6a49f5ecc2002175"
DEFAULT_MICRO_BATCH = 8
LEGAL_MICRO_BATCH = (4, 8, 16)


def build(context):
    try:
        from transformers.modeling_outputs import (
            BaseModelOutputWithPoolingAndNoAttention,
        )
    except Exception:
        return {}

    references = context.get("references") or {}
    modules = context.get("modules") or {}
    record_route = context.get("record_route", lambda *_: None)

    try:
        parent_model = references[RESNET_MODEL_CALL]
        embedder_forward = references[EMBEDDER_CALL]
        encoder_forward = references[ENCODER_CALL]
        pooler_forward = references[POOLER_CALL]
        embedder_module = modules[EMBEDDER_CALL]
        model_module = modules[RESNET_MODEL_CALL]
        num_channels = int(embedder_module.num_channels)
        config = model_module.config
        default_ohs = bool(getattr(config, "output_hidden_states", False))
        default_rd = bool(getattr(config, "return_dict", True))
    except Exception:
        return {}

    parameters = context.get("parameters") or {}
    raw_micro = parameters.get("encoder_micro_batch", DEFAULT_MICRO_BATCH)
    try:
        micro = int(raw_micro)
    except (TypeError, ValueError):
        micro = DEFAULT_MICRO_BATCH
    if micro not in LEGAL_MICRO_BATCH:
        micro = DEFAULT_MICRO_BATCH

    def resnet_model_forward(*args, **kwargs):
        extra = set(kwargs) - {"pixel_values", "output_hidden_states", "return_dict"}
        if extra:
            record_route(RESNET_MODEL_CALL, "fallback_parent_signature")
            return parent_model(*args, **kwargs)
        if len(args) == 1 and "pixel_values" not in kwargs:
            pixel_values = args[0]
        elif len(args) == 0 and "pixel_values" in kwargs:
            pixel_values = kwargs["pixel_values"]
        else:
            record_route(RESNET_MODEL_CALL, "fallback_parent_signature")
            return parent_model(*args, **kwargs)

        output_hidden_states = kwargs.get("output_hidden_states")
        if output_hidden_states is None:
            output_hidden_states = default_ohs
        return_dict = kwargs.get("return_dict")
        if return_dict is None:
            return_dict = default_rd

        supported = (
            isinstance(pixel_values, torch.Tensor)
            and pixel_values.dim() == 4
            and pixel_values.dtype == torch.float32
            and pixel_values.is_cuda
            and pixel_values.is_contiguous()
            and not pixel_values.requires_grad
            and pixel_values.shape[1] == num_channels
            and pixel_values.shape[0] > micro
            and not output_hidden_states
            and return_dict
            and not torch.is_grad_enabled()
            and not model_module.training
            and not embedder_module.training
        )
        if not supported:
            record_route(RESNET_MODEL_CALL, "fallback_parent_signature")
            return parent_model(*args, **kwargs)

        try:
            hidden_parts = []
            pooled_parts = []
            total = pixel_values.shape[0]
            for start in range(0, total, micro):
                chunk = pixel_values[start : start + micro]
                embedding = embedder_forward(chunk)
                encoder_outputs = encoder_forward(
                    embedding, output_hidden_states=False, return_dict=True
                )
                last_hidden = encoder_outputs[0]
                hidden_parts.append(last_hidden)
                pooled_parts.append(pooler_forward(last_hidden))
            last_hidden_state = torch.cat(hidden_parts, dim=0)
            pooler_output = torch.cat(pooled_parts, dim=0)
        except (RuntimeError, ValueError):
            record_route(RESNET_MODEL_CALL, "fallback_parent_runtime")
            return parent_model(*args, **kwargs)

        record_route(RESNET_MODEL_CALL, "model_microbatch")
        return BaseModelOutputWithPoolingAndNoAttention(
            last_hidden_state=last_hidden_state,
            pooler_output=pooler_output,
            hidden_states=None,
        )

    return {RESNET_MODEL_CALL: resnet_model_forward}
