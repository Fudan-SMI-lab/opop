"""Adapter for the frozen resnet_inference offline batch=32 classification request.

Mirrors inputs/source/resnet_inference.py: FP32 ResNet-18 image classification,
one complete forward over the GPU-resident preprocessed image batch, bounded by
CUDA synchronization (provided by the framework timer). Model loading, numerics
configuration and batch construction happen in prepare(), outside timing.
"""
from __future__ import annotations

import numpy as np
import torch
from transformers import ResNetForImageClassification

try:
    from kernel_optimizer.project.discovery import phase as _phase
except Exception:  # pragma: no cover - annotation only, never required
    from contextlib import contextmanager

    @contextmanager
    def _phase(name):
        yield


SEED = 20261020


def _configure(device):
    """Same numerical policies as common.configure(), on the bound device."""
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.manual_seed(SEED)
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision("highest")


@torch.inference_mode()
def _infer(model, pixels):
    return model(pixel_values=pixels).logits


def prepare(context, request):
    device = torch.device(context["device"])
    _configure(context["device"])
    assets = request["assets"]
    model = (
        ResNetForImageClassification.from_pretrained(
            assets["assets_dir"], local_files_only=True, dtype=torch.float32
        )
        .to(device)
        .eval()
    )
    images = torch.from_numpy(np.load(assets["pixels_npy"])).to(device)
    batch = int(request["workload"]["batch"])
    pixels = images[torch.arange(batch, device=device) % len(images)].contiguous()
    return {
        "device": device,
        "model": model,
        "images": images,
        "pixels": pixels,
        "batch": batch,
        "parameter_count": sum(p.numel() for p in model.parameters()),
    }


def reset(state):
    """Stateless inference: run() builds its own request-local output.

    Nothing is cleared or cached inside the source's timed invocation, so there
    is no source-defined pre-clearing to reproduce here.
    """
    state["model"].eval()


def run(state):
    with _phase("forward"):
        return _infer(state["model"], state["pixels"])


def observe(state, result):
    logits = result.detach()
    return {
        "logits": logits,
        "predicted_label_ids": [int(i) for i in logits.argmax(-1).cpu().tolist()],
    }


def metrics(state, result, elapsed_ms):
    batch = state["batch"]
    elapsed_ms = float(elapsed_ms)
    return {
        "latency_ms": elapsed_ms,
        "images_per_second": float(batch * 1000.0 / elapsed_ms),
        "peak_allocated_bytes": float(torch.cuda.max_memory_allocated()),
        "batch": float(batch),
        "parameter_count": float(state["parameter_count"]),
    }


def models(state):
    return {"model": state["model"]}
