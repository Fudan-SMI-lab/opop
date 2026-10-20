"""Adapter for the frozen resnet_inference "offline" workload (batch 32, FP32).

Mirrors inputs/source/resnet_inference.py: load the pretrained ResNet-18 from the
request assets, keep the preprocessed pixel fixtures on the bound device, and run
one full-batch forward under inference_mode with the source's numerical policies,
seed, timing and peak-memory measurement boundaries.
"""
from __future__ import annotations

import contextlib
from pathlib import Path

import numpy as np
import torch
from transformers import ResNetForImageClassification

try:  # optional phase annotation; diagnostics-only, never required for execution
    from kernel_optimizer.project.discovery import phase as _discovery_phase
except Exception:  # pragma: no cover - fall back to a no-op context manager

    def _discovery_phase(name):  # noqa: ANN001
        return contextlib.nullcontext()


SEED = 20261020


def _configure(device: torch.device) -> None:
    """Reproduce common.configure() from the source, pinned to the bound device."""
    torch.set_num_threads(4)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass  # interop parallelism already started earlier in this process
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
    batch = int(request["workload"]["batch"])
    model = (
        ResNetForImageClassification.from_pretrained(
            assets, local_files_only=True, dtype=torch.float32
        )
        .to(device)
        .eval()
    )
    images = torch.from_numpy(np.load(assets / "pixels.npy")).to(device)
    pixels = images[torch.arange(batch, device=device) % len(images)].contiguous()
    state = {
        "device": device,
        "model": model,
        "images": images,
        "pixels": pixels,
        "batch": batch,
        "mode": "measure",
        "peak_allocated_bytes": None,
    }
    return state


def reset(state):
    # Stateless inference: the forward pass creates no persistent mutable state
    # (model.eval() + inference_mode leave parameters and buffers untouched).
    # Restore the deterministic RNG and adapter counters anyway.
    torch.manual_seed(SEED)
    state["peak_allocated_bytes"] = None


def set_mode(state, mode):
    state["mode"] = mode


@torch.inference_mode()
def _infer(model, pixels):
    return model(pixel_values=pixels).logits


def run(state):
    model = state["model"]
    pixels = state["pixels"]
    # Source timing protocol: synchronize, reset peak stats, run the forward,
    # synchronize at the completion boundary, then read peak allocated bytes.
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    with _discovery_phase("infer"):
        logits = _infer(model, pixels)
    torch.cuda.synchronize()
    state["peak_allocated_bytes"] = int(torch.cuda.max_memory_allocated())
    return logits


def observe(state, result):
    logits = result.detach().cpu().clone()
    return {
        "logits": logits,
        "top1": logits.argmax(dim=-1),
    }


def metrics(state, result, elapsed_ms):
    batch = state["batch"]
    peak = state["peak_allocated_bytes"]
    if peak is None:
        peak = int(torch.cuda.max_memory_allocated())
    return {
        "images_per_second": float(batch) * 1000.0 / float(elapsed_ms),
        "latency_ms": float(elapsed_ms),
        "peak_allocated_bytes": float(peak),
        "batch_images": float(batch),
    }


def models(state):
    return {"model": state["model"]}
