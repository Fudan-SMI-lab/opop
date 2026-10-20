"""Adapter for the frozen resnet_inference 'offline' workload (batch 32).

Source: resnet_inference.py + common.py (see inputs/source). The source loads a
pretrained FP32 ResNet-18 from local assets, runs an inference_mode forward over
a GPU-resident batch built by modulo-indexing the fixture images, and times one
forward between CUDA synchronizations.
"""

from __future__ import annotations

import numpy as np
import torch
from transformers import ResNetForImageClassification

try:  # optional discovery annotation; harmless when the package is absent
    from kernel_optimizer.project.discovery import phase as _phase
except Exception:  # pragma: no cover
    from contextlib import contextmanager as _ctxmanager

    @_ctxmanager
    def _phase(name):  # type: ignore[misc]
        yield


SEED = 20261020


def _configure(device):
    """Mirror common.configure() with the worker-bound device."""
    torch.set_num_threads(4)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass  # already initialized in this process
    torch.manual_seed(SEED)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision("highest")
    torch.cuda.set_device(torch.device(device))


def prepare(context, request):
    device = torch.device(context["device"])
    _configure(device)
    assets = request["assets"]["assets"]
    from pathlib import Path

    assets = Path(assets)
    model = (
        ResNetForImageClassification.from_pretrained(
            assets, local_files_only=True, dtype=torch.float32
        )
        .to(device)
        .eval()
    )
    images = torch.from_numpy(np.load(assets / "pixels.npy")).to(device)
    batch = int(request["workload"]["batch"])
    pixels = images[torch.arange(batch, device=images.device) % len(images)].contiguous()
    return {
        "model": model,
        "images": images,
        "pixels": pixels,
        "batch": batch,
        "device": device,
        "seed": SEED,
    }


def reset(state):
    # Stateless inference workload: restore seed policy and reset the CUDA
    # memory-watermark counters so each repeat starts from the source's
    # starting point. The process group (if any) is left untouched.
    torch.manual_seed(state["seed"])
    torch.cuda.reset_peak_memory_stats(state["device"])


def before_run(state):
    # Source resets peak memory stats after its pre-sample synchronize and
    # before starting the timer; this hook is the equivalent out-of-timer spot.
    torch.cuda.reset_peak_memory_stats(state["device"])


@torch.inference_mode()
def run(state):
    with _phase("resnet_forward"):
        logits = state["model"](pixel_values=state["pixels"]).logits
    return logits


def observe(state, result):
    logits = result.detach().to("cpu", copy=True)
    return {
        "logits": logits,
        "top1": logits.argmax(dim=-1),
    }


def metrics(state, result, elapsed_ms):
    batch = state["batch"]
    elapsed_ms = float(elapsed_ms)
    return {
        "latency_ms": elapsed_ms,
        "images_per_second": batch * 1000.0 / elapsed_ms,
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(state["device"])),
        "batch": batch,
    }


def models(state):
    return {"model": state["model"]}


def set_mode(state, mode):
    # No separate correctness-capture path: run() always produces the full
    # logits tensor, so quality/measure/profile share identical semantics.
    state["mode"] = mode
