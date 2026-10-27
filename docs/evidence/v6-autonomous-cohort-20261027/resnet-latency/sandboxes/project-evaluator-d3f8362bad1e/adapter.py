"""Baseline adapter: ResNet-18 FP32 offline batch classification inference.

Reproduces the measured unit of inputs/source/resnet_inference.py for the one
frozen workload in request['workload']: a single
``model(pixel_values=pixels).logits`` forward under ``torch.inference_mode()``
with the source's FP32 numerical policies, pinned pretrained local weights and
a GPU-resident real-image batch. Model loading, input staging/host-to-device
transfer, reference construction, correctness checks, warmup and repeats are
benchmark infrastructure outside run(), exactly as in the source.
"""
from __future__ import annotations

from contextlib import nullcontext

import numpy as np
import torch
from transformers import ResNetForImageClassification

try:
    from kernel_optimizer.project.discovery import phase as _discovery_phase
except Exception:  # annotation only; never required for execution
    _discovery_phase = None

SEED = 20261020


def _configure(device: str) -> None:
    """Source numerical/threading policy (common.py configure)."""
    torch.set_num_threads(4)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass  # host process already started interop workers
    torch.manual_seed(SEED)
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision("highest")


def prepare(context, request):
    device = context["device"]
    _configure(device)
    assets = request["assets"]
    model = (
        ResNetForImageClassification.from_pretrained(
            assets["assets_dir"], local_files_only=True, dtype=torch.float32
        )
        .to(device)
        .eval()
    )
    images = torch.from_numpy(np.load(assets["pixels"])).to(device)
    batch = int(request["workload"]["batch"])
    pixels = images[torch.arange(batch, device=device) % images.shape[0]].contiguous()
    quality = request["quality"]
    return {
        "model": model,
        "images": images,
        "pixels": pixels,
        "batch": batch,
        "device": device,
        "atol": float(quality["logits_atol"]),
        "rtol": float(quality["logits_rtol"]),
        "parameter_count": sum(p.numel() for p in model.parameters()),
    }


def reset(state):
    # Stateless eval inference: the forward records no mutable model state
    # (eval mode, inference_mode, no optimizer/caches). Restore the source's
    # RNG starting point and re-assert eval(); the source clears nothing inside
    # its timed call, so nothing is pre-cleared here.
    torch.manual_seed(SEED)
    state["model"].eval()


@torch.inference_mode()
def _infer(model, pixels):
    context = _discovery_phase("forward") if _discovery_phase is not None else nullcontext()
    with context:
        return model(pixel_values=pixels).logits


def run(state):
    return _infer(state["model"], state["pixels"])


def _corrupted_output_rejected(logits, *, atol, rtol) -> bool:
    # Source negative control (common.py negative_check): the closeness test
    # must reject a +1 corruption of the logits.
    try:
        torch.testing.assert_close(logits + 1, logits, atol=atol, rtol=rtol)
    except AssertionError:
        return True
    raise AssertionError("Correctness control did not reject +1 corruption")


def observe(state, result):
    logits = result.detach()
    repeat = _infer(state["model"], state["pixels"]).detach()
    # Source relational checks (resnet_inference.py:31-33, common.py:74-84),
    # executed outside timed run(): a repeated full forward must be close
    # within the quality tolerance and have identical top-1 predictions.
    torch.testing.assert_close(repeat, logits, atol=state["atol"], rtol=state["rtol"])
    if not bool(torch.equal(repeat.argmax(-1), logits.argmax(-1))):
        raise AssertionError("Repeated run top-1 predictions differ")
    return {
        "logits": logits.to("cpu", copy=True),
        "predicted_top1": logits.argmax(-1).to("cpu", copy=True),
        "repeat_run_max_abs_error": float((repeat.float() - logits.float()).abs().max().item()),
        "repeat_run_close_passed": True,
        "exact_top1": True,
        "corrupted_output_rejected": _corrupted_output_rejected(
            logits, atol=state["atol"], rtol=state["rtol"]
        ),
    }


def metrics(state, result, elapsed_ms):
    elapsed_ms = float(elapsed_ms)
    batch = float(state["batch"])
    return {
        "latency_ms": elapsed_ms,
        "images_per_second": batch * 1000.0 / elapsed_ms,
        "peak_allocated_bytes": float(torch.cuda.max_memory_allocated()),
        "batch": batch,
        "parameter_count": float(state["parameter_count"]),
    }


def models(state):
    return {"model": state["model"]}
