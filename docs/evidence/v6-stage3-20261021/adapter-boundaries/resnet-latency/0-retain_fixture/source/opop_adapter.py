from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
from transformers import ResNetForImageClassification

try:
    from kernel_optimizer.project.discovery import phase
except Exception:  # pragma: no cover - worker always provides kernel_optimizer
    @contextmanager
    def phase(name):
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
    device = context["device"]
    _configure(device)
    assets = Path(request["assets"]["assets"])
    model = (
        ResNetForImageClassification.from_pretrained(assets, local_files_only=True, dtype=torch.float32)
        .to(device)
        .eval()
    )
    images = torch.from_numpy(np.load(assets / "pixels.npy")).to(device)
    batch = int(request["workload"]["batch"])
    pixels = images[torch.arange(batch, device=device) % len(images)].contiguous()
    return {
        "device": device,
        "batch": batch,
        "model": model,
        "pixels": pixels,
        "images": images,
        "parameter_count": int(sum(p.numel() for p in model.parameters())),
        "peak_allocated_bytes": 0,
    }


def reset(state):
    torch.manual_seed(SEED)
    state["peak_allocated_bytes"] = 0


@torch.inference_mode()
def run(state):
    device = state["device"]
    torch.cuda.reset_peak_memory_stats(device)
    with phase("forward"):
        logits = state["model"](pixel_values=state["pixels"]).logits
    torch.cuda.synchronize(device)
    state["peak_allocated_bytes"] = int(torch.cuda.max_memory_allocated(device))
    return logits


def observe(state, result):
    logits = result.detach()
    return {"logits": logits, "top1": logits.argmax(dim=-1)}


def metrics(state, result, elapsed_ms):
    elapsed = float(elapsed_ms)
    return {
        "latency_ms": elapsed,
        "images_per_second": float(state["batch"]) * 1000.0 / elapsed,
        "peak_allocated_bytes": float(state["peak_allocated_bytes"]),
        "parameter_count": float(state["parameter_count"]),
    }


def models(state):
    return {"model": state["model"]}


def set_mode(state, mode):
    state["mode"] = mode
