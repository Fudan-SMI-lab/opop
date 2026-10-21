"""Adapter for the ResNet-18 offline batch inference project (source: resnet_inference.py)."""
from __future__ import annotations

import numpy as np
import torch
from transformers import ResNetForImageClassification

try:
    from kernel_optimizer.project.discovery import phase as _phase
except Exception:  # pragma: no cover - annotation only, never load-bearing
    from contextlib import contextmanager

    @contextmanager
    def _phase(name):
        yield


def _configure(device):
    """Mirror common.configure(): seed, thread and FP32/TF32 numerical policies."""
    torch.set_num_threads(4)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    torch.manual_seed(20261020)
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision("highest")


def prepare(context, request):
    device = torch.device(context["device"])
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
    pixels = images[torch.arange(batch, device=device) % len(images)].contiguous()
    return {
        "device": device,
        "model": model,
        "images": images,
        "pixels": pixels,
        "batch": batch,
    }


def reset(state):
    # Stateless single-forward inference: the model stays in eval() and the
    # GPU-resident input is immutable, so there is no mutable state between
    # repeats (the source loop likewise only drops the previous output).
    state["model"].eval()


def run(state):
    model = state["model"]
    pixels = state["pixels"]
    with torch.inference_mode(), _phase("forward"):
        logits = model(pixel_values=pixels).logits
    return {"logits": logits}


def observe(state, result):
    logits = result["logits"].detach()
    return {
        "logits": logits,
        "predicted_class_ids": logits.argmax(-1).cpu(),
    }


def metrics(state, result, elapsed_ms):
    batch = state["batch"]
    return {
        "latency_ms": float(elapsed_ms),
        "images_per_second": float(batch * 1000.0 / elapsed_ms),
        "peak_allocated_bytes": float(torch.cuda.max_memory_allocated()),
    }


def models(state):
    return {"model": state["model"]}
