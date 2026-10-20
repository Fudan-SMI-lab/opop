"""Adapter for the frozen resnet_inference 'offline' workload (batch=32, single GPU, FP32).

Reproduces the measured invocation of resnet_inference.py: one complete
ResNetForImageClassification forward over the GPU-resident batch=32 pixel
tensor under torch.inference_mode(), with the source's numerical policies
and seed. Model loading, pixel fixture load and H2D copy happen in
prepare(); the framework times only run().
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from transformers import ResNetForImageClassification

SEED = 20261020


def _configure(device):
    torch.set_num_threads(4)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
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
    assets_dir = Path(assets["model_config"]).resolve().parent
    model = (
        ResNetForImageClassification.from_pretrained(
            assets_dir, local_files_only=True, dtype=torch.float32
        )
        .to(device)
        .eval()
    )
    images = torch.from_numpy(np.load(assets["pixels"])).to(device)
    batch = request["workload"]["batch"]
    pixels = images[torch.arange(batch, device=device) % len(images)].contiguous()
    return {
        "model": model,
        "images": images,
        "pixels": pixels,
        "device": device,
        "batch": batch,
        "num_images": int(len(images)),
    }


def reset(state):
    torch.manual_seed(SEED)
    state["model"].eval()


def run(state):
    return infer(state["model"], state["pixels"])


def observe(state, result):
    logits = result.detach()
    return {
        "logits": logits.cpu(),
        "predicted_class_ids": logits.argmax(-1).cpu(),
    }


def metrics(state, result, elapsed_ms):
    batch = state["batch"]
    return {
        "latency_ms": float(elapsed_ms),
        "images_per_second": float(batch * 1e3 / elapsed_ms),
        "peak_allocated_bytes": float(
            torch.cuda.max_memory_allocated(state["device"])
        ),
    }


def models(state):
    return {"model": state["model"]}


@torch.inference_mode()
def infer(model, pixels):
    return model(pixel_values=pixels).logits
