"""Micro-batched ResNet-18 encoder stage 0 (model.resnet.encoder.stages.0) for
the offline batch-32 peak_allocated_bytes objective.

On top of the accepted stem micro-batch (bundle trial-e9279012c6-param-9ac5fd08,
route stem_microbatch_rows8), the measured peak 203,099,136 B decomposes exactly
as live FP32 params 46,758,048 + GPU-resident input 19,267,584 + retained cudnn
workspace 8,622,944 + 5 x 25,690,112, where 25,690,112 B is one 32x64x56x56 FP32
activation. Five such tensors coexist only inside encoder stage 0: at the second
BasicLayer's trailing batch_norm the stage input (stem output), the first
BasicLayer's output (residual), the second BasicLayer's relu0 output, its conv1
output and its bn1 output are all live. Every operator in stage 0 (valid 3x3
conv, eval BatchNorm, ReLU, the residual add) acts per sample, so this boundary
can be micro-batched along the frozen batch dimension exactly like the accepted
stem bundle.

This replacement keeps the frozen batch-32 request (all 32 rows execute) but runs
the unchanged two ResNetBasicLayer parents - always invoked through
context.references so inherited calls, hooks and their coverage remain real - on
contiguous row chunks, assembling chunk results into one [32, 64, 56, 56] tensor
consumed unchanged by stages 1-3. The residual add inside each BasicLayer mutates
only tensors freshly allocated inside that parent, never the chunk input views,
so the stage input is never written through this path.

No parameter transformation, no output caching and no CUDA-graph replay: every
invocation recomputes from the current input through the parent callables.
"""
from __future__ import annotations

import torch

STAGE0_CALL = "595c8331c60823c132ca3a4a"
LAYER0_CALL = "f3ec7e68beb5e3d3f846d2fa"
LAYER1_CALL = "dfb96bb329e6c3859d690b9a"
DEFAULT_CHUNK_ROWS = 8


def _run_stage0_chunked(input, layer0, layer1, chunk_rows):
    """Run layer0 -> layer1 on contiguous row chunks of ``input`` and assemble
    the chunk results into one tensor of the parent's output shape."""
    total = input.shape[0]
    first = layer1(layer0(input[:chunk_rows]))
    head = first.shape[0]
    if head >= total:
        return first
    out = torch.empty(
        (total, *first.shape[1:]), dtype=first.dtype, device=first.device
    )
    out[:head].copy_(first)
    del first
    for start in range(head, total, chunk_rows):
        stop = min(start + chunk_rows, total)
        out[start:stop].copy_(layer1(layer0(input[start:stop])))
    return out


def _training_active(modules):
    for module in modules:
        if module is not None and module.training:
            return True
    return False


def _compatible(x):
    return (
        isinstance(x, torch.Tensor)
        and x.dim() == 4
        and x.is_cuda
        and x.dtype == torch.float32
        and x.shape[1] == 64
        and x.is_contiguous()
        and not x.requires_grad
    )


def build(context):
    references = context.get("references") or {}
    parent = references.get(STAGE0_CALL)
    layer0 = references.get(LAYER0_CALL)
    layer1 = references.get(LAYER1_CALL)
    if parent is None or layer0 is None or layer1 is None:
        return {}
    modules = context.get("modules") or {}
    stage_module = modules.get(STAGE0_CALL)
    layer0_module = modules.get(LAYER0_CALL)
    layer1_module = modules.get(LAYER1_CALL)
    record_route = context.get("record_route", lambda *_: None)
    parameters = context.get("parameters") or {}
    try:
        chunk_rows = int(parameters.get("stage0_chunk_rows", DEFAULT_CHUNK_ROWS))
    except (TypeError, ValueError):
        chunk_rows = DEFAULT_CHUNK_ROWS
    if chunk_rows < 2:
        chunk_rows = 2

    def stage0(input=None, *args, **kwargs):
        if (
            args
            or kwargs
            or not _compatible(input)
            or input.shape[0] <= chunk_rows
            or _training_active((stage_module, layer0_module, layer1_module))
        ):
            record_route(STAGE0_CALL, "stage0_parent_signature_fallback")
            return parent(input, *args, **kwargs)
        try:
            out = _run_stage0_chunked(input, layer0, layer1, chunk_rows)
        except Exception as exc:
            record_route(
                STAGE0_CALL, "stage0_parent_runtime_fallback:" + type(exc).__name__
            )
            return parent(input)
        record_route(STAGE0_CALL, "stage0_microbatch_rows%d" % chunk_rows)
        return out

    return {STAGE0_CALL: stage0}
