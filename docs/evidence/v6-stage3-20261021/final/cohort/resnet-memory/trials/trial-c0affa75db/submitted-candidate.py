"""Micro-batched ResNetModel execution for the frozen ResNet-18 offline batch (region root: model.resnet).

Replaces the contiguous execution owned by transformers ResNetModel.forward: the
GPU-resident (32,3,224,224) FP32 input is sliced into batch chunks and, per chunk,
the current parent callables are invoked through context.references (embedder ->
encoder with output_hidden_states=False/return_dict=True -> adaptive-avg pooler).
Each chunk's last_hidden_state/pooler_output is narrow-copied into preallocated
full-batch outputs and a BaseModelOutputWithPoolingAndNoAttention with unchanged
shapes and fields is returned. Every op in the region is per-sample (eval-mode BN
with frozen running stats, conv windows, out-of-place ReLU, in-place residual add
inside a block, per-sample pooling), so chunked execution is the identical
computation per image. The earlier whole chunked-stem bundle is explicitly
replaced; model-level chunking preserves its behavior (chunk-sized stem
transients) and extends it to the encoder and pooler phases.
"""
import torch

RESNET_CALL = "8fbf233236c985c7650ea8e4"
EMBEDDER_CALL = "683c822136ed7958f3e9ba44"
ENCODER_CALL = "29488c4be587b49a7875c533"
POOLER_CALL = "61beb430723e6c6d4991e78c"

try:
    from transformers.modeling_outputs import (
        BaseModelOutputWithPoolingAndNoAttention as _Output,
    )
except Exception:
    _Output = None


def build(context):
    record_route = context.get("record_route", lambda *_: None)
    modules = context.get("modules") or {}
    references = context.get("references") or {}
    parameters = context.get("parameters") or {}

    chunk = parameters.get("model_chunk", 8)
    try:
        chunk = int(chunk)
    except (TypeError, ValueError):
        chunk = 8
    chunk = max(1, chunk)

    resnet_module = modules.get(RESNET_CALL)
    parent_resnet = references.get(RESNET_CALL)
    if not callable(parent_resnet) and resnet_module is not None:
        parent_resnet = resnet_module.forward

    def _resolve(call_id, attr):
        fn = references.get(call_id)
        if callable(fn):
            return fn
        module = modules.get(call_id)
        fn = getattr(module, "forward", None) if module is not None else None
        if callable(fn):
            return fn
        inner = getattr(resnet_module, attr, None) if resnet_module is not None else None
        return getattr(inner, "forward", None) if inner is not None else None

    embedder = _resolve(EMBEDDER_CALL, "embedder")
    encoder = _resolve(ENCODER_CALL, "encoder")
    pooler = _resolve(POOLER_CALL, "pooler")

    embedder_module = modules.get(EMBEDDER_CALL)
    num_channels = (
        getattr(embedder_module, "num_channels", None)
        if embedder_module is not None
        else None
    )
    if num_channels is None and resnet_module is not None:
        inner = getattr(resnet_module, "embedder", None)
        num_channels = getattr(inner, "num_channels", None) if inner is not None else None
    config = getattr(resnet_module, "config", None)

    setup_ok = (
        callable(parent_resnet)
        and callable(embedder)
        and callable(encoder)
        and callable(pooler)
        and _Output is not None
    )

    def resnet_forward(pixel_values, output_hidden_states=None, return_dict=None, **kwargs):
        if not setup_ok:
            record_route(RESNET_CALL, "micro_batched_model:fallback_setup")
            return parent_resnet(
                pixel_values,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                **kwargs,
            )
        if kwargs or torch.is_grad_enabled():
            record_route(RESNET_CALL, "micro_batched_model:fallback_signature")
            return parent_resnet(
                pixel_values,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                **kwargs,
            )
        resolved_ohs = output_hidden_states
        if resolved_ohs is None:
            resolved_ohs = bool(getattr(config, "output_hidden_states", False))
        resolved_rd = return_dict
        if resolved_rd is None:
            resolved_rd = bool(getattr(config, "return_dict", True))
        if resolved_ohs or not resolved_rd:
            record_route(RESNET_CALL, "micro_batched_model:fallback_signature")
            return parent_resnet(
                pixel_values,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        t = pixel_values
        if (
            not torch.is_tensor(t)
            or t.dim() != 4
            or not t.is_cuda
            or t.dtype != torch.float32
            or not t.is_contiguous()
            or t.shape[0] < 1
            or t.shape[2] < 1
            or t.shape[3] < 1
            or t.requires_grad
            or (num_channels is not None and t.shape[1] != num_channels)
        ):
            record_route(RESNET_CALL, "micro_batched_model:fallback_signature")
            return parent_resnet(
                t,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        n = t.shape[0]
        c = min(chunk, n)
        try:
            first_hidden = encoder(
                embedder(t[0:c]), output_hidden_states=False, return_dict=True
            )[0]
            first_pooled = pooler(first_hidden)
            if n <= c:
                record_route(RESNET_CALL, "micro_batched_model:ok")
                return _Output(
                    last_hidden_state=first_hidden,
                    pooler_output=first_pooled,
                    hidden_states=None,
                )
            last_hidden = torch.empty(
                (n,) + tuple(first_hidden.shape[1:]),
                device=first_hidden.device,
                dtype=first_hidden.dtype,
            )
            pooled = torch.empty(
                (n,) + tuple(first_pooled.shape[1:]),
                device=first_pooled.device,
                dtype=first_pooled.dtype,
            )
            last_hidden[0:c].copy_(first_hidden)
            pooled[0:c].copy_(first_pooled)
            first_hidden = None
            first_pooled = None
            s = c
            while s < n:
                e = min(s + c, n)
                hidden = encoder(
                    embedder(t[s:e]), output_hidden_states=False, return_dict=True
                )[0]
                last_hidden[s:e].copy_(hidden)
                pooled[s:e].copy_(pooler(hidden))
                hidden = None
                s = e
        except Exception:
            record_route(RESNET_CALL, "micro_batched_model:fallback_runtime")
            return parent_resnet(
                t,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        record_route(RESNET_CALL, "micro_batched_model:ok")
        return _Output(
            last_hidden_state=last_hidden,
            pooler_output=pooled,
            hidden_states=None,
        )

    return {RESNET_CALL: resnet_forward}
