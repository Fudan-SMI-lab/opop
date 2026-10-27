from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from transformers import ResNetForImageClassification

SOURCE_SEED = 20261020


def _configure(device):
    torch.set_num_threads(4)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    torch.manual_seed(SOURCE_SEED)
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision("highest")


@torch.inference_mode()
def _infer(model, pixels):
    return model(pixel_values=pixels).logits


def prepare(context, request):
    device = context["device"]
    _configure(device)
    assets = request["assets"]
    asset_dir = Path(assets["model_weights"]).resolve().parent
    model = ResNetForImageClassification.from_pretrained(
        str(asset_dir), local_files_only=True, dtype=torch.float32
    ).to(device)
    model.eval()
    images = torch.from_numpy(np.load(str(assets["pixels"]))).to(device)
    batch = int(request["workload"]["batch"])
    pixels = images[torch.arange(batch, device=device) % len(images)].contiguous()
    return {
        "model": model,
        "images": images,
        "pixels": pixels,
        "batch": batch,
        "device": device,
        "quality": dict(request["quality"]),
        "parameter_count": int(sum(p.numel() for p in model.parameters())),
    }


def reset(state):
    state["model"].eval()


def run(state):
    return _infer(state["model"], state["pixels"])


def observe(state, result):
    quality = state["quality"]
    atol = float(quality["logits_atol"])
    rtol = float(quality["logits_rtol"])
    with torch.inference_mode():
        logits = result.detach()
        replay = _infer(state["model"], state["pixels"]).detach()
        torch.testing.assert_close(replay, logits, atol=atol, rtol=rtol)
        logits_close = True
        max_abs_error = float((replay.float() - logits.float()).abs().max().item())
        exact_top1 = bool(torch.equal(replay.argmax(-1), logits.argmax(-1)))
        try:
            torch.testing.assert_close(replay + 1, logits, atol=atol, rtol=rtol)
            corrupted_output_rejected = False
        except AssertionError:
            corrupted_output_rejected = True
        if not (logits_close and exact_top1 and corrupted_output_rejected):
            raise AssertionError(
                "source-defined relational checks failed: "
                f"logits_close={logits_close} exact_top1={exact_top1} "
                f"corrupted_output_rejected={corrupted_output_rejected}"
            )
        predicted_classes = logits.argmax(-1)
        id2label = state["model"].config.id2label
        predicted_labels_first3 = [
            id2label[int(i)] for i in predicted_classes[:3].cpu().tolist()
        ]
    return {
        "logits": logits,
        "predicted_classes": predicted_classes,
        "predicted_labels_first3": predicted_labels_first3,
        "logits_close": logits_close,
        "exact_top1": exact_top1,
        "corrupted_output_rejected": corrupted_output_rejected,
        "max_abs_error": max_abs_error,
    }


def metrics(state, result, elapsed_ms):
    elapsed_ms = float(elapsed_ms)
    batch = state["batch"]
    peak = int(torch.cuda.max_memory_allocated(torch.device(state["device"])))
    return {
        "latency_ms": elapsed_ms,
        "images_per_second": batch * 1000.0 / elapsed_ms,
        "peak_allocated_bytes": float(peak),
        "batch": float(batch),
        "parameter_count": float(state["parameter_count"]),
    }


def models(state):
    return {"resnet18": state["model"]}
