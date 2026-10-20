"""Full-graph inference on a seeded directed multigraph and random FP32 weights."""
import argparse
import json
from pathlib import Path
import statistics
import time

import torch
from model import GraphEncoder


def prepare(config, workload, device="cuda"):
    torch.set_num_threads(4)
    torch.manual_seed(config["seed"])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    model = GraphEncoder(**config["model"]).to(device).eval()
    # CPU RNG gives an independently reconstructible input fixture.
    generator = torch.Generator().manual_seed(config["seed"] + 1)
    features = torch.randn(workload["nodes"], config["model"]["input_features"], generator=generator)
    sources = torch.randint(workload["nodes"], (workload["edges"],), generator=generator)
    destinations = torch.randint(workload["nodes"], (workload["edges"],), generator=generator)
    degree = torch.bincount(destinations, minlength=workload["nodes"]).float()
    return model, tuple(t.to(device) for t in (features, sources, destinations, degree))


@torch.inference_mode()
def infer(model, inputs):
    return model(*inputs)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("workloads.json"))
    parser.add_argument("--workload", default="standard")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=15)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    workload = next(w for w in config["workloads"] + config["held_out_workloads"] if w["id"] == args.workload)
    model, inputs = prepare(config, workload)
    reference, actual = infer(model, inputs), infer(model, inputs)
    torch.testing.assert_close(actual, reference, **config["quality"])
    args.output.mkdir(parents=True, exist_ok=False)
    torch.save(reference.cpu(), args.output / "reference.pt")
    del reference, actual
    for _ in range(args.warmup):
        infer(model, inputs)
    samples = []
    for _ in range(args.repeats):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter_ns()
        output = infer(model, inputs)
        torch.cuda.synchronize()
        duration_ms = (time.perf_counter_ns() - start) / 1e6
        samples.append({"latency_ms": duration_ms, "graphs_per_second": 1000 / duration_ms,
                        "peak_allocated_bytes": torch.cuda.max_memory_allocated()})
        del output
    result = {"workload": workload, "config": config, "samples": samples,
              "summary": {k: statistics.median(s[k] for s in samples) for k in samples[0]},
              "quality": "Full output tensor; random fixture fidelity, not downstream graph accuracy"}
    (args.output / "result.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result["summary"]))


if __name__ == "__main__":
    main()
