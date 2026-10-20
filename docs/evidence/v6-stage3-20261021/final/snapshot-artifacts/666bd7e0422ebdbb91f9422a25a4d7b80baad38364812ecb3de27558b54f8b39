"""Adapter for the frozen resnet_inference offline workload (batch 32).

Source semantics (inputs/source/resnet_inference.py + common.py):
  - setup outside timing: configure() numerical policies, from_pretrained
    FP32 model on CUDA in eval mode, pixels.npy loaded to GPU, offline batch
    built as images[arange(batch) % len(images)].contiguous()
  - measured unit: exactly one infer(model, pixels) = model(pixel_values=pixels).logits
    under torch.inference_mode()
  - the call is pure and stateless across repeats, so reset() is a no-op
"""
from __future__ import annotations

import numpy as np
import torch
from transformers import ResNetForImageClassification


def _configure(device):
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
    device = context["device"]
    _configure(device)
    assets = request["assets"]
    assets_dir = assets["model_config"]
    if assets_dir is not None:
        import os

        assets_dir = os.path.dirname(os.path.abspath(assets_dir))
    model = (
        ResNetForImageClassification.from_pretrained(
            assets_dir, local_files_only=True, dtype=torch.float32
        )
        .to(device)
        .eval()
    )
    images = torch.from_numpy(np.load(assets["pixels"] if "pixels" in assets else assets_dir + "/pixels.npy")).to(device)
    workload = request["workload"]
    batch = workload["batch"]
    pixels = images[torch.arange(batch, device=device) % images.shape[0]].contiguous()
    return {
        "device": device,
        "model": model,
        "images": images,
        "pixels": pixels,
        "batch": batch,
    }


def reset(state):
    return None


@torch.inference_mode()
def _infer(model, pixels):
    return model(pixel_values=pixels).logits


def run(state):
    return _infer(state["model"], state["pixels"])


def observe(state, result):
    logits = result.detach()
    return {
        "logits": logits,
        "predicted_classes": logits.argmax(dim=-1),
    }


def metrics(state, result, elapsed_ms):
    batch = state["batch"]
    peak_bytes = int(torch.cuda.max_memory_allocated(state["device"]))
    return {
        "peak_allocated_bytes": float(peak_bytes),
        "latency_ms": float(elapsed_ms),
        "images_per_second": float(batch) * 1000.0 / float(elapsed_ms),
        "batch": float(batch),
        "parameter_count": float(sum(p.numel() for p in state["model"].parameters())),
    }


def models(state):
    return {"model": state["model"]}
