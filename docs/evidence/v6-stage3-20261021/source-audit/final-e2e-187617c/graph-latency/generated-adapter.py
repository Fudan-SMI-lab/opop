"""Adapter for the graph inference validation project (frozen 'standard' workload).

Self-contained replica of the source semantics (infer.py, model.py,
workloads.json).  A previous attempt failed inside the measured worker with
FileNotFoundError because it read workloads.json from an intake source-root
path that is not shipped to the worker; every frozen value is therefore
embedded here and the adapter reads no external files at runtime.
"""
import torch
from torch import nn

# Frozen configuration copied from workloads.json / request.json.
SEED = 20261021
INPUT_FEATURES = 128
HIDDEN_FEATURES = 256
OUTPUT_FEATURES = 64
NODES = 4096
EDGES = 65536


class MeanSAGE(nn.Module):
    """Verbatim from model.py: mean-aggregation SAGE layer."""

    def __init__(self, input_features, output_features):
        super().__init__()
        self.project = nn.Linear(2 * input_features, output_features)

    def forward(self, features, sources, destinations, degree):
        messages = features.index_select(0, sources)
        neighbors = torch.zeros_like(features).index_add_(0, destinations, messages)
        neighbors = neighbors / degree.clamp_min(1).unsqueeze(1)
        return self.project(torch.cat((features, neighbors), dim=1))


class GraphEncoder(nn.Module):
    """Verbatim from model.py: two-layer mean GraphSAGE with ReLU between."""

    def __init__(self, input_features=128, hidden_features=256, output_features=64):
        super().__init__()
        self.first = MeanSAGE(input_features, hidden_features)
        self.second = MeanSAGE(hidden_features, output_features)

    def forward(self, features, sources, destinations, degree):
        hidden = torch.relu(self.first(features, sources, destinations, degree))
        return self.second(hidden, sources, destinations, degree)


def prepare(context, request):
    device = context["device"]
    # Numerical policies and thread count exactly as infer.py:13-17.
    torch.set_num_threads(4)
    torch.manual_seed(SEED)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    model = GraphEncoder(INPUT_FEATURES, HIDDEN_FEATURES, OUTPUT_FEATURES).to(device).eval()
    # CPU RNG gives an independently reconstructible input fixture (infer.py:20-24).
    generator = torch.Generator().manual_seed(SEED + 1)
    features = torch.randn(NODES, INPUT_FEATURES, generator=generator)
    sources = torch.randint(NODES, (EDGES,), generator=generator)
    destinations = torch.randint(NODES, (EDGES,), generator=generator)
    degree = torch.bincount(destinations, minlength=NODES).float()
    inputs = tuple(t.to(device) for t in (features, sources, destinations, degree))
    return {"model": model, "inputs": inputs, "device": device}


def reset(state):
    # Inference has no cross-invocation state (README.md:19); run() performs an
    # independent forward over unchanged GPU-resident inputs. Nothing to reset.
    return None


@torch.inference_mode()
def run(state):
    model, inputs = state["model"], state["inputs"]
    return model(*inputs)


def observe(state, result):
    return {"output": result.detach()}


def metrics(state, result, elapsed_ms):
    values = {
        "latency_ms": float(elapsed_ms),
        "graphs_per_second": 1000.0 / float(elapsed_ms),
    }
    device = state["device"]
    if isinstance(device, str) and device.startswith("cuda"):
        values["peak_allocated_bytes"] = float(
            torch.cuda.max_memory_allocated(torch.device(device))
        )
    return values


def models(state):
    return {"encoder": state["model"]}
