import json
from pathlib import Path

import torch
from torch import nn

_FALLBACK_CONFIG = {
    "model": {"input_features": 128, "hidden_features": 256, "output_features": 64},
    "quality": {"atol": 1e-05, "rtol": 1e-04},
    "seed": 20261021,
    "dtype": "float32",
}


def _source_config(request):
    candidates = []
    source_root = request.get("source_root")
    if source_root:
        candidates.append(Path(source_root) / "workloads.json")
    here = Path(__file__).resolve().parent
    candidates.append(here / "inputs" / "source" / "workloads.json")
    candidates.append(here / "source" / "workloads.json")
    candidates.append(here / "workloads.json")
    for path in candidates:
        try:
            if path.is_file():
                return json.loads(path.read_text())
        except OSError:
            continue
    return _FALLBACK_CONFIG


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
    config = _source_config(request)
    workload = request["workload"]
    torch.set_num_threads(4)
    torch.manual_seed(config["seed"])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    model = GraphEncoder(**config["model"]).to(device).eval()
    generator = torch.Generator().manual_seed(config["seed"] + 1)
    features = torch.randn(
        workload["nodes"], config["model"]["input_features"], generator=generator
    )
    sources = torch.randint(workload["nodes"], (workload["edges"],), generator=generator)
    destinations = torch.randint(
        workload["nodes"], (workload["edges"],), generator=generator
    )
    degree = torch.bincount(destinations, minlength=workload["nodes"]).float()
    inputs = tuple(t.to(device) for t in (features, sources, destinations, degree))
    return {
        "model": model,
        "inputs": inputs,
        "device": device,
        "config": config,
        "workload": workload,
        "mode": "measure",
    }


def reset(state):
    return None


@torch.inference_mode()
def run(state):
    return state["model"](*state["inputs"])


def observe(state, result):
    return {"output": result.detach().to("cpu", copy=True)}


def metrics(state, result, elapsed_ms):
    latency_ms = float(elapsed_ms)
    values = {
        "latency_ms": latency_ms,
        "graphs_per_second": 1000.0 / latency_ms,
        "nodes": float(state["workload"]["nodes"]),
        "edges": float(state["workload"]["edges"]),
    }
    if torch.cuda.is_available():
        values["peak_allocated_bytes"] = float(torch.cuda.max_memory_allocated())
    return values


def models(state):
    return {"model": state["model"]}


def set_mode(state, mode):
    state["mode"] = mode
    return None
