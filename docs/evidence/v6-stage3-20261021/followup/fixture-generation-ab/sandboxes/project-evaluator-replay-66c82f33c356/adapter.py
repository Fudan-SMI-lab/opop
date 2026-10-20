"""Adapter for the frozen resnet_inference 'offline' workload (batch=32).

Reproduces the measured region of inputs/source/resnet_inference.py: one
complete offline image-batch forward pass under torch.inference_mode(),
with the source's numerical policy, seed and GPU-resident input fixtures.
"""
from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
from transformers import ResNetForImageClassification

try:  # task-phase annotation; inert unless the framework activates discovery
    from kernel_optimizer.project.discovery import phase as _phase
except Exception:  # pragma: no cover - fallback keeps the module importable

    @contextmanager
    def _phase(name):  # noqa: D401 - simple stand-in
        yield


SEED = 20261020


def _configure(device):
    """Source numerical policy (common.py configure)."""
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
    assets = request["assets"]
    weights = Path(assets["model_weights"])
    assets_dir = weights.parent
    try:
        model = ResNetForImageClassification.from_pretrained(
            str(assets_dir), local_files_only=True, dtype=torch.float32
        )
    except Exception:
        from safetensors.torch import load_file
        from transformers import ResNetConfig

        config = ResNetConfig.from_json_file(assets["model_config"])
        model = ResNetForImageClassification(config)
        model.load_state_dict(load_file(str(weights)))
    model = model.to(device).eval()
    images = torch.from_numpy(np.load(assets["pixels"])).to(device)
    batch = int(request["workload"]["batch"])
    pixels = images[torch.arange(batch, device=device) % images.shape[0]].contiguous()
    state = {
        "device": device,
        "model": model,
        "images": images,
        "pixels": pixels,
        "batch": batch,
        "parameter_count": int(sum(p.numel() for p in model.parameters())),
    }
    return state


def reset(state):
    # Stateless inference: the eval-mode forward consumes no RNG and mutates no
    # model/optimizer state. GPU-resident fixtures (images/pixels) stay alive in
    # state because their lifetime contributes to peak_allocated_bytes.
    state["model"].eval()


def run(state):
    with torch.inference_mode(), _phase("forward"):
        return state["model"](pixel_values=state["pixels"]).logits


def observe(state, result):
    logits = result.detach().to("cpu", copy=True)
    return {
        "logits": logits,
        "predicted_class_indices": logits.argmax(-1),
    }


def metrics(state, result, elapsed_ms):
    batch = state["batch"]
    return {
        "peak_allocated_bytes": float(torch.cuda.max_memory_allocated()),
        "images_per_second": float(batch * 1e3 / elapsed_ms),
        "latency_ms": float(elapsed_ms),
        "batch": float(batch),
        "parameter_count": float(state["parameter_count"]),
    }


def models(state):
    return {"model": state["model"]}
