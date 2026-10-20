"""Adapter for the resnet-memory project: one complete offline batch inference.

Source semantics (inputs/source/resnet_inference.py, common.py):
  - configure(): num_threads=4, num_interop_threads=1, manual_seed(20261020),
    TF32 off for matmul and cudnn, cudnn.benchmark off, float32_matmul_precision
    'highest', single CUDA device.
  - model: ResNetForImageClassification.from_pretrained(assets, local_files_only=True,
    dtype=torch.float32).cuda().eval()
  - pixels: pixels.npy -> cuda tensor; workload batch tensor built once as
    images[torch.arange(batch, device='cuda') % len(images)].contiguous() (outside timing).
  - one measured invocation: infer(model, pixels) = model(pixel_values=pixels).logits
    under torch.inference_mode().
  - objective peak_allocated_bytes: torch.cuda.reset_peak_memory_stats() before the
    timed forward, torch.cuda.max_memory_allocated() after (per-process allocated peak,
    not reserved/driver memory).
"""

from __future__ import annotations

import os

import numpy as np
import torch
from transformers import ResNetForImageClassification

try:  # optional phase annotation; never required for correctness
    from kernel_optimizer.project.discovery import phase as _discovery_phase
except Exception:  # pragma: no cover - annotation is best effort
    from contextlib import contextmanager

    @contextmanager
    def _discovery_phase(name):  # type: ignore[misc]
        yield


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
    assets = request["assets"]
    assets_dir = os.path.dirname(os.path.abspath(assets["model_config"]))
    device = torch.device(context["device"])

    _configure(device)

    model = ResNetForImageClassification.from_pretrained(
        assets_dir, local_files_only=True, dtype=torch.float32
    )
    model.to(device)
    model.eval()

    images = torch.from_numpy(np.load(assets["pixels"])).to(device)
    batch = int(request["workload"]["batch"])
    pixels = images[torch.arange(batch, device=device) % len(images)].contiguous()

    state = {
        "device": device,
        "model": model,
        "pixels": pixels,
        "batch": batch,
        "last_peak_allocated_bytes": 0,
    }
    return state


def reset(state):
    # Stateless inference: the forward creates its own request-local state.
    # Re-affirm eval mode and the RNG so repeats start from the source's setup.
    state["model"].eval()
    torch.manual_seed(SEED)
    state["last_peak_allocated_bytes"] = 0


def run(state):
    model = state["model"]
    pixels = state["pixels"]
    # Source per-repeat semantics: peak stats reset just before the measured
    # forward (after the framework's boundary CUDA sync), read after it.
    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        with _discovery_phase("forward"):
            logits = model(pixel_values=pixels).logits
    state["last_peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
    return {"logits": logits}


def observe(state, result):
    logits = result["logits"].detach().cpu()
    return {
        "logits": logits,
        "predicted_class_ids": logits.argmax(-1).cpu(),
    }


def metrics(state, result, elapsed_ms):
    batch = state["batch"]
    return {
        "peak_allocated_bytes": float(state["last_peak_allocated_bytes"]),
        "images_per_second": float(batch * 1000.0 / elapsed_ms),
        "latency_ms": float(elapsed_ms),
        "batch_size": float(batch),
        "logits_shape_0": float(result["logits"].shape[0]),
        "logits_shape_1": float(result["logits"].shape[1]),
    }


def models(state):
    return {"model": state["model"]}
