"""Adapter for the frozen offline batch=32 ResNet-18 FP32 inference workload.

Mirrors inputs/source/resnet_inference.py for the single frozen workload
request['workload'] == {"id": "offline", "batch": 32, ...}:

- configure() numerical policy (common.py:34-42)
- ResNetForImageClassification.from_pretrained(assets, local_files_only=True,
  dtype=torch.float32).cuda().eval() (resnet_inference.py:23)
- images = torch.from_numpy(np.load(pixels.npy)).cuda() (resnet_inference.py:24)
- pixels = images[torch.arange(batch, device='cuda') % len(images)].contiguous()
  (resnet_inference.py:29)
- infer(model, pixels) = model(pixel_values=pixels).logits under
  torch.inference_mode() (resnet_inference.py:14-16)

The framework supplies CUDA synchronization at both timer boundaries, which is
exactly the source's completion boundary (torch.cuda.synchronize() after the
forward, resnet_inference.py:42-47), so run() performs only the forward.
"""
from __future__ import annotations

import numpy as np
import torch
from transformers import ResNetForImageClassification

SEED = 20261020


def _configure(device):
    """Source numerical/environment policy (common.py configure)."""
    torch.set_num_threads(4)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass  # already initialized earlier in this process
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
    batch = request["workload"]["batch"]
    pixels = images[torch.arange(batch, device=device) % len(images)].contiguous()
    state = {
        "device": device,
        "model": model,
        "images": images,  # stays live, matching the source's reference at timer time
        "pixels": pixels,
        "batch": batch,
        "parameter_count": sum(p.numel() for p in model.parameters()),
    }
    return state


def reset(state):
    """No-op: stateless inference.

    The model is in eval mode with immutable pretrained weights, the input
    batch is a fixed GPU-resident tensor, and run() builds every output inside
    a fresh request-local torch.inference_mode() context. The source clears or
    resets nothing between timed calls (resnet_inference.py:38-49), so there is
    no mutable state to restore.
    """


def run(state):
    with torch.inference_mode():
        return state["model"](pixel_values=state["pixels"]).logits


def observe(state, result):
    logits = result.detach().to("cpu")
    return {
        "logits": logits,
        "predicted_class_ids": logits.argmax(dim=-1),
    }


def metrics(state, result, elapsed_ms):
    batch = state["batch"]
    return {
        "latency_ms": float(elapsed_ms),
        "images_per_second": float(batch * 1000.0 / elapsed_ms),
        "peak_allocated_bytes": float(torch.cuda.max_memory_allocated()),
        "parameter_count": float(state["parameter_count"]),
        "batch": float(batch),
    }


def models(state):
    return {"model": state["model"]}
