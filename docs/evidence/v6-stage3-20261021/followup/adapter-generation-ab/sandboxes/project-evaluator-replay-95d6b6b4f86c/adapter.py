"""Adapter for the resnet_inference 'offline' workload: one complete FP32
ResNet-18 image-classification batch inference (batch=32) on GPU-resident
preprocessed pixels, mirroring inputs/source/resnet_inference.py."""
from __future__ import annotations

import os

import numpy as np
import torch
from transformers import ResNetForImageClassification

try:  # optional phase annotation; harmless when unavailable
    from kernel_optimizer.project import discovery as _discovery
except Exception:  # pragma: no cover
    _discovery = None

SEED = 20261020  # common.py configure()


def _configure(device: torch.device) -> None:
    """Source numerical policy, verbatim from common.py configure()."""
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.manual_seed(SEED)
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision("highest")


def _assets_dir(request: dict) -> str:
    # Source loads from the assets directory; the request lists config.json,
    # model.safetensors and preprocessor_config.json inside that one directory.
    return os.path.dirname(os.path.abspath(request["assets"]["model_config"]))


@torch.inference_mode()
def _infer(model, pixels):
    return model(pixel_values=pixels).logits


def prepare(context, request):
    device = torch.device(context["device"])
    _configure(device)
    model = (
        ResNetForImageClassification.from_pretrained(
            _assets_dir(request), local_files_only=True, dtype=torch.float32
        )
        .to(device)
        .eval()
    )
    images = torch.from_numpy(np.load(request["assets"]["pixels"])).to(device)
    batch = int(request["workload"]["batch"])
    # Source fixture construction, resnet_inference.py:29. Kept GPU-resident in
    # state for the workload's lifetime (contributes to the memory objective).
    pixels = images[torch.arange(batch, device=device) % len(images)].contiguous()
    return {
        "model": model,
        "pixels": pixels,
        "device": device,
        "batch": batch,
        "seed": SEED,
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "mode": "measure",
    }


def reset(state):
    # Stateless inference: the only mutable state is the RNG the source seeds
    # once in configure(); restore it so every repeat starts identically.
    torch.manual_seed(state["seed"])


def set_mode(state, mode):
    state["mode"] = mode


def run(state):
    model, pixels = state["model"], state["pixels"]
    if _discovery is not None:
        with _discovery.phase("infer"):
            logits = _infer(model, pixels)
    else:
        logits = _infer(model, pixels)
    return {"logits": logits}


def observe(state, result):
    logits = result["logits"].detach()
    top1 = logits.detach().argmax(-1)
    return {
        "logits": logits,
        "predicted_class_ids": top1,
        "predicted_class_id_list": [int(i) for i in top1.tolist()],
    }


def metrics(state, result, elapsed_ms):
    batch = state["batch"]
    elapsed_ms = float(elapsed_ms)
    # The framework resets CUDA peak stats immediately before its timer; this
    # query outside the timer reproduces the source's per-repeat
    # reset-before / max_memory_allocated-after measurement exactly.
    return {
        "peak_allocated_bytes": float(torch.cuda.max_memory_allocated()),
        "images_per_second": float(batch * 1000.0 / elapsed_ms),
        "latency_ms": elapsed_ms,
        "batch": float(batch),
        "parameter_count": float(state["parameter_count"]),
    }


def models(state):
    return {"resnet18": state["model"]}
