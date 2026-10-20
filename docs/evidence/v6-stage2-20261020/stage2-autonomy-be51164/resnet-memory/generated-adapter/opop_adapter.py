"""Adapter for the frozen resnet_inference 'offline' workload (batch 32).

Mirrors inputs/source/resnet_inference.py and inputs/source/common.py:
FP32 ResNet-18 classification of GPU-resident preprocessed real images with a
single torch.inference_mode() forward pass, minimizing peak CUDA allocated
bytes under an images_per_second constraint.
"""
from __future__ import annotations

import contextlib
from pathlib import Path

import numpy as np
import torch
from transformers import ResNetForImageClassification

try:  # optional discovery annotation; never required for correctness
    from kernel_optimizer.project.discovery import phase as _discovery_phase
except Exception:  # pragma: no cover - annotation is best effort
    _discovery_phase = None

SEED = 20261020


def _phase(name):
    if _discovery_phase is not None:
        return _discovery_phase(name)
    return contextlib.nullcontext()


def _configure(device):
    # Same numerical policies and seed as common.configure(), but on the
    # worker-bound device instead of hardcoded device 0.
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
    device = torch.device(context["device"])
    _configure(device)
    assets = Path(request["assets"]["assets"])
    model = (
        ResNetForImageClassification.from_pretrained(
            assets, local_files_only=True, dtype=torch.float32
        )
        .to(device)
        .eval()
    )
    images = torch.from_numpy(np.load(assets / "pixels.npy")).to(device)
    workload = request["workload"]
    batch = int(workload["batch"])
    pixels = images[torch.arange(batch, device=device) % len(images)].contiguous()
    return {
        "device": device,
        "model": model,
        "images": images,
        "pixels": pixels,
        "batch": batch,
        "workload_id": workload.get("id"),
        "parameter_count": int(sum(p.numel() for p in model.parameters())),
        "last_peak_allocated_bytes": 0.0,
        "last_peak_reserved_bytes": 0.0,
        "mode": "measure",
    }


def set_mode(state, mode):
    state["mode"] = str(mode)


def reset(state):
    # Stateless inference: the forward pass mutates nothing, so restoring the
    # RNG/eval starting point and clearing CUDA peak counters is sufficient.
    torch.manual_seed(SEED)
    state["model"].eval()
    device = state["device"]
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    state["last_peak_allocated_bytes"] = 0.0
    state["last_peak_reserved_bytes"] = 0.0


def run(state):
    # One complete frozen workload iteration with the source timing-loop
    # semantics: synchronize, reset peak stats, forward, synchronize, read peak.
    device = state["device"]
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    with torch.inference_mode(), _phase("forward"):
        logits = state["model"](pixel_values=state["pixels"]).logits
    torch.cuda.synchronize(device)
    state["last_peak_allocated_bytes"] = float(torch.cuda.max_memory_allocated(device))
    state["last_peak_reserved_bytes"] = float(torch.cuda.max_memory_reserved(device))
    return logits


def observe(state, result):
    with torch.no_grad():
        logits = result.detach().to("cpu").clone()
        top1 = logits.argmax(dim=-1)
    return {"logits": logits, "top1": top1}


def metrics(state, result, elapsed_ms):
    elapsed_ms = float(elapsed_ms)
    batch = int(state["batch"])
    values = {
        "peak_allocated_bytes": float(state["last_peak_allocated_bytes"]),
        "peak_reserved_bytes": float(state["last_peak_reserved_bytes"]),
        "latency_ms": elapsed_ms,
        "batch": float(batch),
        "parameter_count": float(state["parameter_count"]),
    }
    values["images_per_second"] = (
        float(batch) * 1000.0 / elapsed_ms if elapsed_ms > 0.0 else 0.0
    )
    return values


def models(state):
    return {"model": state["model"]}
