"""Framework adapter for the graph inference validation project.

Frozen workload: one full forward of the two-layer mean GraphSAGE encoder over
the standard seeded directed multigraph (4096 nodes / 65536 edges), FP32, TF32
disabled. Semantics mirror inputs/source/infer.py and model.py. The model
definition is inlined verbatim and config constants are embedded because the
worker's reference-source directory does not preserve the intake sandbox layout
(previous failure: workloads.json not found next to the copied adapter).
"""
import torch
from torch import nn

_MODEL_CONFIG = {"input_features": 128, "hidden_features": 256, "output_features": 64}
_SEED = 20261021


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
    device = context["device"]
    workload = request["workload"]
    torch.set_num_threads(4)
    torch.manual_seed(_SEED)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    model = GraphEncoder(**_MODEL_CONFIG).to(device).eval()
    generator = torch.Generator().manual_seed(_SEED + 1)
    features = torch.randn(workload["nodes"], _MODEL_CONFIG["input_features"], generator=generator)
    sources = torch.randint(workload["nodes"], (workload["edges"],), generator=generator)
    destinations = torch.randint(workload["nodes"], (workload["edges"],), generator=generator)
    degree = torch.bincount(destinations, minlength=workload["nodes"]).float()
    inputs = tuple(t.to(device) for t in (features, sources, destinations, degree))
    return {"model": model, "inputs": inputs, "device": device}


def reset(state):
    pass


def run(state):
    with torch.inference_mode():
        return state["model"](*state["inputs"])


def observe(state, result):
    return {"output": result.detach().cpu()}


def metrics(state, result, elapsed_ms):
    values = {
        "latency_ms": float(elapsed_ms),
        "graphs_per_second": 1000.0 / float(elapsed_ms),
    }
    try:
        values["peak_allocated_bytes"] = float(torch.cuda.max_memory_allocated(state["device"]))
    except Exception:
        pass
    return values


def models(state):
    return {"model": state["model"]}
