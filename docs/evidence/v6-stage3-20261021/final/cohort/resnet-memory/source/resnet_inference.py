"""Project input: pretrained CNN image classification with actual image fixtures."""
from __future__ import annotations

import json
import time

import numpy as np
import torch
from transformers import ResNetForImageClassification

from common import arguments, close_check, configure, digest, metadata, negative_check, profile_once, save, summarize


@torch.inference_mode()
def infer(model, pixels):
    return model(pixel_values=pixels).logits


def main():
    args = arguments(__doc__)
    configure()
    config = json.loads(args.workloads.read_text())["resnet_inference"]
    model = ResNetForImageClassification.from_pretrained(args.assets, local_files_only=True, dtype=torch.float32).cuda().eval()
    images = torch.from_numpy(np.load(args.assets / "pixels.npy")).cuda()
    record = metadata(args, "resnet_inference")
    record.update({"asset_manifest_sha256": digest(args.assets / "asset-manifest.json"), "parameter_count": sum(p.numel() for p in model.parameters()), "workloads": []})
    quality = config["correctness"]
    for workload in config["workloads"]:
        pixels = images[torch.arange(workload["batch"], device="cuda") % len(images)].contiguous()
        reference, actual = infer(model, pixels), infer(model, pixels)
        checks = close_check(actual, reference, atol=quality["logits_atol"], rtol=quality["logits_rtol"])
        checks.update(negative_check(reference, atol=quality["logits_atol"], rtol=quality["logits_rtol"]))
        assert torch.equal(actual.argmax(-1), reference.argmax(-1))
        checks["exact_top1"] = True
        checks["predicted_labels"] = [model.config.id2label[int(i)] for i in reference.argmax(-1)[:3].cpu()]
        torch.save({"pixels": pixels.cpu(), "logits": reference.cpu()}, args.output / f"{workload['id']}-reference.pt")
        del reference, actual
        for _ in range(args.warmup):
            infer(model, pixels)
        samples = []
        for _ in range(args.repeats):
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter_ns()
            output = infer(model, pixels)
            torch.cuda.synchronize()
            elapsed = time.perf_counter_ns() - start
            samples.append({"latency_ms": elapsed / 1e6, "images_per_second": workload["batch"] * 1e9 / elapsed, "peak_allocated_bytes": torch.cuda.max_memory_allocated()})
            del output
        row = {"workload": workload, "correctness": checks, "samples": samples, "summary": summarize(samples)}
        if args.profile:
            row["diagnostic"] = profile_once(lambda: infer(model, pixels), args.output / f"{workload['id']}-trace.json")
        record["workloads"].append(row)
        save(args.output / "result.json", record)
        print(workload["id"], json.dumps(row["summary"]), flush=True)


if __name__ == "__main__":
    main()
