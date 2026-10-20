"""Framework adapter for the graph inference validation project.

Source: inputs/source/{infer.py, model.py, workloads.json}. The frozen workload
is one full forward of the two-layer mean GraphSAGE encoder over the standard
seeded directed multigraph, FP32, TF32 disabled, under torch.inference_mode().

Self-contained by design: the worker deploys this module to a directory where
the original source tree is not guaranteed to be present, so the encoder
definition and the workload-independent source configuration (model
dimensions, seed, numerical policies) are embedded verbatim from the source.
Workload-dependent dimensions (nodes, edges) are read from
request['workload'] so a differently sized variant uses the same adapter.
"""
import torch
from torch import nn


_SOURCE_MODEL = {"input_features": 128, "hidden_features": 256, "output_features": 64}
_SOURCE_SEED = 20261021


class MeanSAGE(nn.Module):
    def __init__(self, input_features, output_features):
        super().__init__()
        self.project = nn.Linear(2 * input_features, output_features)

    def forward(self, features, sources, destinations, degree):
        messages = features.index_select(0, sources)
        neighbors = torch.zeros_like(features).index_add_(0, destinations, messages)
        neighbors = neighbors / degree.clamp_min(1).unsqueeze(1)
        return self.project(torch.cat((features, neighbors), dim=1))


class GraphEncoder(nn.Module):
    def __init__(self, input_features=128, hidden_features=256, output_features=64):
        super().__init__()
        self.first = MeanSAGE(input_features, hidden_features)
        self.second = MeanSAGE(hidden_features, output_features)

    def forward(self, features, sources, destinations, degree):
        hidden = torch.relu(self.first(features, sources, destinations, degree))
        return self.second(hidden, sources, destinations, degree)


def prepare(context, request):
    workload = request["workload"]
    nodes = workload["nodes"]
    edges = workload["edges"]
    device = context["device"]
    # Numerical policies and construction order mirror infer.py:12-25.
    torch.set_num_threads(4)
    torch.manual_seed(_SOURCE_SEED)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    model = GraphEncoder(**_SOURCE_MODEL).to(device).eval()
    # CPU RNG gives an independently reconstructible input fixture.
    generator = torch.Generator().manual_seed(_SOURCE_SEED + 1)
    features = torch.randn(nodes, _SOURCE_MODEL["input_features"], generator=generator)
    sources = torch.randint(nodes, (edges,), generator=generator)
    destinations = torch.randint(nodes, (edges,), generator=generator)
    degree = torch.bincount(destinations, minlength=nodes).float()
    inputs = tuple(t.to(device) for t in (features, sources, destinations, degree))
    return {"model": model, "inputs": inputs, "device": device, "nodes": nodes, "edges": edges}


def reset(state):
    # Inference has no cross-invocation state (README.md:19): each forward
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
    return {"output": result.detach().to("cpu")}


def metrics(state, result, elapsed_ms):
    # elapsed_ms is the framework's synchronized wall time of one full forward,
    # identical in semantics to the source's latency_ms sample field
    # (infer.py:53-60). The peak statistics query stays outside the timer.
    latency_ms = float(elapsed_ms)
    out = {
        "latency_ms": latency_ms,
        "graphs_per_second": 1000.0 / latency_ms,
    }
    if torch.cuda.is_available():
        out["peak_allocated_bytes"] = float(torch.cuda.max_memory_allocated(state["device"]))
    return out


def models(state):
    return {"model": state["model"]}
