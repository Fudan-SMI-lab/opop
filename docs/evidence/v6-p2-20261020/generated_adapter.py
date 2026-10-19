"""Adapter for the frozen ResNet-18 FP32 single-image online inference workload.

Mirrors inputs/source/resnet_inference.py semantics for workload 'online'
(batch=1): same model loading, same real preprocessed image fixture selection,
same numerical policies (FP32, TF32 disabled, cudnn benchmark off), same seed,
same inference_mode forward pass producing logits.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from transformers import ResNetForImageClassification

SEED = 20261020


def _configure(device: torch.device) -> None:
    """Numerical policies copied from common.configure()."""
    torch.set_num_threads(4)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass  # worker may already have an interop thread pool
    torch.manual_seed(SEED)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision("highest")


def prepare(context, request):
    device = torch.device(context["device"])
    _configure(device)
    assets = Path(request["assets"]["assets"])
    model = (
        ResNetForImageClassification.from_pretrained(
            str(assets), local_files_only=True, dtype=torch.float32
        )
        .to(device)
        .eval()
    )
    images = torch.from_numpy(np.load(assets / "pixels.npy")).to(device)
    batch = int(request["workload"]["batch"])
    pixels = images[torch.arange(batch, device=device) % len(images)].contiguous()
    state = {
        "device": device,
        "model": model,
        "images": images,
        "pixels": pixels,
        "batch": batch,
        "parameter_count": int(sum(p.numel() for p in model.parameters())),
    }
    return state


def reset(state):
    """Stateless inference: restore eval mode and per-repeat memory counters."""
    state["model"].eval()
    device = state["device"]
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


@torch.inference_mode()
def _infer(model, pixels):
    return model(pixel_values=pixels).logits


def run(state):
    logits = _infer(state["model"], state["pixels"])
    return {"logits": logits}


def observe(state, result):
    logits = result["logits"].detach().float().cpu()
    return {
        "logits": logits,
        "predicted_class_ids": logits.argmax(-1).cpu(),
    }


def metrics(state, result, elapsed_ms):
    values = {
        "latency_ms": float(elapsed_ms),
        "batch": float(state["batch"]),
        "images_per_second": float(state["batch"] * 1000.0 / elapsed_ms),
        "parameter_count": float(state["parameter_count"]),
    }
    device = state["device"]
    if device.type == "cuda":
        values["peak_allocated_bytes"] = float(torch.cuda.max_memory_allocated(device))
    return values


def models(state):
    return {"model": state["model"]}
