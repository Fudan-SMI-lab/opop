"""Adapter for the ResNet-18 offline batch inference latency project.

Mirrors inputs/source/resnet_inference.py: one measured invocation is a single
complete ``model(pixel_values=pixels).logits`` forward under
``torch.inference_mode()`` on the FP32 CUDA model, with the batch=32 input
built from the fixture pixels.npy by cycling its images. Model loading,
preprocessing, host->device transfer and reference construction happen in
prepare() outside timed execution.
"""
from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from transformers import ResNetForImageClassification

SEED = 20261020


def _configure(device: torch.device) -> None:
    """Source numerical policies from common.configure(), on the bound device."""
    torch.set_num_threads(4)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    torch.manual_seed(SEED)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision("highest")


def _phase(name: str):
    """Annotate a source task phase if kernel_optimizer discovery is available."""
    try:
        from kernel_optimizer.project.discovery import phase as _discovery_phase

        return _discovery_phase(name)
    except Exception:
        return nullcontext()


def prepare(context, request):
    device = torch.device(context["device"])
    _configure(device)
    assets = request["assets"]
    asset_dir = Path(assets["config.json"]).resolve().parent
    model = ResNetForImageClassification.from_pretrained(
        str(asset_dir), local_files_only=True, dtype=torch.float32
    )
    model.to(device)
    model.eval()
    images = torch.from_numpy(np.load(assets["pixels.npy"])).to(device)
    batch = int(request["workload"]["batch"])
    index = torch.arange(batch, device=device) % images.shape[0]
    pixels = images[index].contiguous()
    return {
        "model": model,
        "images": images,
        "pixels": pixels,
        "batch": batch,
        "device": device,
        "quality": request["quality"],
        "objective": request["objective"],
        "mode": "measure",
    }


def reset(state):
    """Stateless inference: the forward creates no persistent mutable state."""
    state["model"].eval()


def run(state):
    return _infer(state["model"], state["pixels"])


@torch.inference_mode()
def _infer(model, pixels):
    with _phase("forward"):
        return model(pixel_values=pixels).logits


def observe(state, result):
    logits = result.detach()
    return {
        "logits": logits,
        "predicted_class_ids": logits.argmax(dim=-1),
    }


def metrics(state, result, elapsed_ms):
    batch = state["batch"]
    values = {
        "latency_ms": float(elapsed_ms),
        "images_per_second": float(batch) * 1000.0 / float(elapsed_ms),
    }
    if state["device"].type == "cuda":
        values["peak_allocated_bytes"] = float(torch.cuda.max_memory_allocated())
    return values


def models(state):
    return {"model": state["model"]}


def set_mode(state, mode):
    """Source has one execution path; modes differ only in bookkeeping."""
    state["mode"] = mode
