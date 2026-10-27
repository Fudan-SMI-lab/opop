"""Baseline adapter for the resnet_inference offline (batch-32) workload.

Reproduces resnet_inference.py: one FP32 ResNet-18 forward on the fixed
GPU-resident batch of real preprocessed images under @torch.inference_mode(),
returning the full 1000-class logits. Model loading, device setup, input
construction and all correctness checks live outside the timed run().
"""
from __future__ import annotations

import os
from contextlib import contextmanager

import numpy as np
import torch
from transformers import ResNetForImageClassification

try:
    from kernel_optimizer.project.discovery import phase as _phase
except Exception:  # discovery helpers are optional for execution
    @contextmanager
    def _phase(name):  # pragma: no cover - fallback keeps semantics identical
        yield


SEED = 20261020


def _configure(device):
    """Mirror common.py configure() numerical policies (common.py:34-42)."""
    torch.set_num_threads(4)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass  # already initialized in this process
    torch.manual_seed(SEED)
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision("highest")


@torch.inference_mode()
def _infer(model, pixels):
    """resnet_inference.py:14-16 infer()."""
    return model(pixel_values=pixels).logits


def prepare(context, request):
    device = context["device"]
    _configure(device)
    assets = request["assets"]
    asset_dir = os.path.dirname(assets["model_config"])
    model = (
        ResNetForImageClassification.from_pretrained(
            asset_dir, local_files_only=True, dtype=torch.float32
        )
        .to(device)
        .eval()
    )
    images = torch.from_numpy(np.load(assets["pixels"])).to(device)
    batch = int(request["workload"]["batch"])
    pixels = images[torch.arange(batch, device=device) % len(images)].contiguous()
    state = {
        "model": model,
        "pixels": pixels,
        "batch": batch,
        "device": device,
        "quality": request["quality"],
        "id2label": model.config.id2label,
        "parameter_count": int(sum(p.numel() for p in model.parameters())),
        "mode": "measure",
    }
    state["images"] = images
    return state


def reset(state):
    """Stateless inference: frozen eval weights and a fixed GPU-resident input
    tensor leave no mutable state between invocations; the source clears no
    gradients or caches inside its timed loop (resnet_inference.py:38-49)."""
    state["model"].eval()


def run(state):
    with _phase("resnet_forward"):
        return _infer(state["model"], state["pixels"])


def observe(state, result):
    """Source-defined reference-fidelity checks (resnet_inference.py:30-35),
    executed outside the timed region, plus the full required outputs."""
    model, pixels = state["model"], state["pixels"]
    quality = state["quality"]
    atol = float(quality["logits_atol"])
    rtol = float(quality["logits_rtol"])
    with torch.inference_mode():
        reference = _infer(model, pixels)
        actual = result.detach()
        torch.testing.assert_close(actual.cpu(), reference.cpu(), atol=atol, rtol=rtol)
        max_abs_error = (actual.cpu().float() - reference.cpu().float()).abs().max().item()
        rejected = False
        try:
            torch.testing.assert_close(
                reference.cpu() + 1, reference.cpu(), atol=atol, rtol=rtol
            )
        except AssertionError:
            rejected = True
        if not rejected:
            raise AssertionError("Correctness control did not reject +1 corruption")
        pred_classes = actual.argmax(-1).cpu()
        exact_top1 = bool(torch.equal(pred_classes, reference.argmax(-1).cpu()))
        if quality.get("require_exact_top1", True) and not exact_top1:
            raise AssertionError("top-1 predictions differ between repeated invocations")
        predicted_labels = [state["id2label"][int(i)] for i in pred_classes[:3]]
    return {
        "logits": actual.cpu(),
        "pred_classes": pred_classes,
        "predicted_labels_first3": predicted_labels,
        "checks": {
            "logits_close_passed": True,
            "max_abs_error": float(max_abs_error),
            "atol": atol,
            "rtol": rtol,
            "exact_top1": exact_top1,
            "corrupted_output_rejected": True,
        },
    }


def metrics(state, result, elapsed_ms):
    """Objective and project metrics in native units; statistics are queried
    outside the timer (the framework resets CUDA peak stats before it)."""
    batch = state["batch"]
    return {
        "images_per_second": batch * 1000.0 / float(elapsed_ms),
        "latency_ms": float(elapsed_ms),
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "batch": batch,
        "parameter_count": state["parameter_count"],
    }


def models(state):
    return {"model": state["model"]}


def set_mode(state, mode):
    """The source always returns full logits; no separate correctness-capture
    path exists, so modes only differ in intent, never in executed work."""
    state["mode"] = mode
