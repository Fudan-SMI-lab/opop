"""Framework adapter for the graph inference validation project.

Source: inputs/source/{infer.py,model.py,workloads.json}. The frozen workload is
one full forward of the two-layer mean GraphSAGE encoder over the standard
seeded directed multigraph (4096 nodes / 65536 edges), FP32, TF32 disabled.
"""
import json
import sys
from pathlib import Path

import torch

_SOURCE_ROOT = Path(__file__).resolve().parent / "inputs" / "source"
if str(_SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(_SOURCE_ROOT))

from model import GraphEncoder  # noqa: E402  (source-local import)


def _source_config():
    return json.loads((_SOURCE_ROOT / "workloads.json").read_text())


def prepare(context, request):
    cfg = _source_config()
    device = context["device"]
    # Numerical policies and seed order mirror infer.py:12-25 exactly.
    torch.set_num_threads(4)
    torch.manual_seed(cfg["seed"])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    model = GraphEncoder(**cfg["model"]).to(device).eval()
    workload = request["workload"]
    # CPU RNG gives an independently reconstructible input fixture.
    generator = torch.Generator().manual_seed(cfg["seed"] + 1)
    features = torch.randn(workload["nodes"], cfg["model"]["input_features"], generator=generator)
    sources = torch.randint(workload["nodes"], (workload["edges"],), generator=generator)
    destinations = torch.randint(workload["nodes"], (workload["edges"],), generator=generator)
    degree = torch.bincount(destinations, minlength=workload["nodes"]).float()
    inputs = tuple(t.to(device) for t in (features, sources, destinations, degree))
    return {"model": model, "inputs": inputs, "device": device}


def reset(state):
    # Inference has no cross-invocation state (README.md:19): the forward
    # allocates its own buffers and consumes no RNG, so nothing to restore.
    pass


def run(state):
    # One complete graph encoder forward on GPU-resident inputs, matching
    # infer.py:28-30. The framework synchronizes CUDA at both boundaries.
    with torch.inference_mode():
        return state["model"](*state["inputs"])


def observe(state, result):
    # Correctness observation: the complete [nodes, output_features] FP32
    # output tensor, compared with atol=1e-05 / rtol=1e-4 (infer.py:44-45).
    return {"output": result.detach().cpu()}


def metrics(state, result, elapsed_ms):
    # elapsed_ms is the framework's synchronized wall time of one full forward,
    # identical in semantics to the source's latency_ms sample field.
    metrics = {
        "latency_ms": float(elapsed_ms),
        "graphs_per_second": 1000.0 / float(elapsed_ms),
    }
    # Peak statistics query stays outside the timer (source: infer.py:60).
    try:
        metrics["peak_allocated_bytes"] = float(torch.cuda.max_memory_allocated(state["device"]))
    except Exception:
        pass
    return metrics


def models(state):
    return {"model": state["model"]}
