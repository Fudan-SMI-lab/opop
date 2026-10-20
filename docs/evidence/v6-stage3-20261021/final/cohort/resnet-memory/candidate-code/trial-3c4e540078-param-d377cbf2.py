"""Chunked stem execution for the frozen ResNet-18 offline batch (region root: ResNetEmbeddings)."""
import torch

STEM_CALL = "683c822136ed7958f3e9ba44"
CONV_LAYER_CALL = "679ccfe137094282d25350c1"
POOLER_CALL = "aa9765a1830bed7ee2727282"


def build(context):
    record_route = context.get("record_route", lambda *_: None)
    modules = context.get("modules") or {}
    references = context.get("references") or {}
    parameters = context.get("parameters") or {}

    chunk = parameters.get("stem_chunk", 8)
    try:
        chunk = int(chunk)
    except (TypeError, ValueError):
        chunk = 8
    chunk = max(1, chunk)

    stem_module = modules.get(STEM_CALL)
    parent_stem = references.get(STEM_CALL)
    if not callable(parent_stem) and stem_module is not None:
        parent_stem = stem_module.forward

    embedder = references.get(CONV_LAYER_CALL)
    if not callable(embedder):
        embedder_module = modules.get(CONV_LAYER_CALL)
        embedder = getattr(embedder_module, "forward", None) if embedder_module is not None else None
    if not callable(embedder) and stem_module is not None:
        embedder = stem_module.embedder.forward

    pooler = references.get(POOLER_CALL)
    if not callable(pooler):
        pooler_module = modules.get(POOLER_CALL)
        pooler = getattr(pooler_module, "forward", None) if pooler_module is not None else None
    if not callable(pooler) and stem_module is not None:
        pooler = stem_module.pooler.forward

    num_channels = getattr(stem_module, "num_channels", None) if stem_module is not None else None
    setup_ok = callable(parent_stem) and callable(embedder) and callable(pooler)

    def stem(pixel_values, *args, **kwargs):
        if not setup_ok:
            record_route(STEM_CALL, "chunked_stem:fallback_setup")
            return parent_stem(pixel_values, *args, **kwargs)
        if args or kwargs or torch.is_grad_enabled():
            record_route(STEM_CALL, "chunked_stem:fallback_signature")
            return parent_stem(pixel_values, *args, **kwargs)
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
            or (num_channels is not None and t.shape[1] != num_channels)
        ):
            record_route(STEM_CALL, "chunked_stem:fallback_signature")
            return parent_stem(t)
        n = t.shape[0]
        c = min(chunk, n)
        try:
            first = pooler(embedder(t[0:c]))
            if n <= c:
                record_route(STEM_CALL, "chunked_stem:ok")
                return first
            out = torch.empty((n,) + tuple(first.shape[1:]), device=first.device, dtype=first.dtype)
            out[0:c].copy_(first)
            first = None
            s = c
            while s < n:
                e = min(s + c, n)
                out[s:e].copy_(pooler(embedder(t[s:e])))
                s = e
        except Exception:
            record_route(STEM_CALL, "chunked_stem:fallback_runtime")
            return parent_stem(t)
        record_route(STEM_CALL, "chunked_stem:ok")
        return out

    return {STEM_CALL: stem}
