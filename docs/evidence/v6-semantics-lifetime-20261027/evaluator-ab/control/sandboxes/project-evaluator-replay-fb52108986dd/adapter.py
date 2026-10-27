"""Baseline adapter for the resnet_inference offline workload.

Reproduces the source semantics of resnet_inference.py / common.py for the one
frozen workload row supplied by the framework: a single FP32
ResNetForImageClassification forward over the fixed GPU-resident tensor of
`batch` preprocessed real images, under torch.inference_mode(), ending at the
source's completion boundary (torch.cuda.synchronize()) with the full
1000-class logits returned. Model loading, input construction and correctness
checks stay outside the timed invocation, exactly as in the source.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from transformers import ResNetForImageClassification

SEED = 20261020


def _configure(device: torch.device) -> None:
    """Mirror common.configure() numerical policy (common.py:34-42)."""
    torch.set_num_threads(4)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    torch.manual_seed(SEED)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision("highest")


@torch.inference_mode()
def _infer(model, pixels):
    """Source infer() (resnet_inference.py:14-16)."""
    return model(pixel_values=pixels).logits


def prepare(context, request):
    device = torch.device(context["device"])
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
    batch = int(request["workload"]["batch"])
    index = torch.arange(batch, device=device) % images.shape[0]
    pixels = images[index].contiguous()
    return {
        "device": device,
        "model": model,
        "images": images,
        "pixels": pixels,
        "batch": batch,
        "quality": request["quality"],
        "seed": SEED,
        "parameter_count": int(sum(p.numel() for p in model.parameters())),
    }


def reset(state):
    state["model"].eval()
    torch.manual_seed(state["seed"])


def run(state):
    logits = _infer(state["model"], state["pixels"])
    if state["device"].type == "cuda":
        torch.cuda.synchronize()
    return logits


def observe(state, result):
    logits = result["logits"] if isinstance(result, dict) else result
    logits = logits.detach()
    quality = state["quality"]
    atol = float(quality["logits_atol"])
    rtol = float(quality["logits_rtol"])
    replay = _infer(state["model"], state["pixels"]).detach()
    torch.testing.assert_close(replay, logits, atol=atol, rtol=rtol)
    predicted = logits.argmax(dim=-1)
    if not bool(torch.equal(replay.argmax(dim=-1), predicted)):
        raise AssertionError("exact top-1 mismatch between repeated inferences")
    try:
        torch.testing.assert_close(logits + 1, logits, atol=atol, rtol=rtol)
    except AssertionError:
        corrupted_output_rejected = True
    else:
        raise AssertionError("correctness control did not reject +1 corruption")
    return {
        "logits": logits.cpu(),
        "predicted_classes": predicted.cpu(),
        "exact_top1": True,
        "replay_close": True,
        "max_abs_error": float((replay.float() - logits.float()).abs().max().item()),
        "corrupted_output_rejected": corrupted_output_rejected,
        "atol": atol,
        "rtol": rtol,
        "batch": int(logits.shape[0]),
        "num_classes": int(logits.shape[1]),
        "parameter_count": state["parameter_count"],
    }


def metrics(state, result, elapsed_ms):
    elapsed_ms = float(elapsed_ms)
    values = {
        "images_per_second": state["batch"] * 1000.0 / elapsed_ms,
        "latency_ms": elapsed_ms,
        "batch": state["batch"],
        "parameter_count": state["parameter_count"],
    }
    if state["device"].type == "cuda":
        values["peak_allocated_bytes"] = int(
            torch.cuda.max_memory_allocated(state["device"])
        )
    return values


def models(state):
    return {"model": state["model"]}


def set_mode(state, mode):
    state["mode"] = mode
