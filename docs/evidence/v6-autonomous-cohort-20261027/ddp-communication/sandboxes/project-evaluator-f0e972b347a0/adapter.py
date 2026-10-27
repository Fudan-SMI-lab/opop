from contextlib import contextmanager, nullcontext
from datetime import timedelta

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import GPT2Config, GPT2LMHeadModel

try:
    from kernel_optimizer.project.discovery import phase as _discovery_phase
except Exception:
    _discovery_phase = None

ARCHITECTURE = {"n_layer": 12, "n_embd": 768, "n_head": 12, "vocab_size": 50257, "n_positions": 1024}
INIT_SEED = 20261020
INPUT_SEED_BASE = 31000
LEARNING_RATE = 0.001


@contextmanager
def _phase(name):
    if _discovery_phase is None:
        yield
    else:
        with _discovery_phase(name):
            yield


def _configure(device):
    torch.set_num_threads(4)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    torch.manual_seed(INIT_SEED)
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision("highest")


def prepare(context, request):
    workload = request["workload"]
    rank = int(context["rank"])
    world_size = int(context["world_size"])
    device = torch.device(context["device"])
    if world_size != int(workload["world_size"]):
        raise ValueError("This workload must launch the declared number of ranks")
    micro_batch = int(workload["micro_batch"])
    sequence_length = int(workload["sequence_length"])
    accumulation_steps = int(workload["accumulation_steps"])
    _configure(device)
    control = dist.new_group(backend="gloo", timeout=timedelta(seconds=120))
    torch.manual_seed(INIT_SEED)
    model_config = GPT2Config(
        **ARCHITECTURE,
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        attn_pdrop=0.0,
        use_cache=False,
        attn_implementation="eager",
    )
    raw_model = GPT2LMHeadModel(model_config).float().to(device).train()
    local_index = device.index if device.index is not None else torch.cuda.current_device()
    model = DDP(
        raw_model,
        device_ids=[local_index],
        broadcast_buffers=False,
        bucket_cap_mb=25,
        gradient_as_bucket_view=True,
    )
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.0, weight_decay=0.0, foreach=False)
    generator = torch.Generator().manual_seed(INPUT_SEED_BASE + rank)
    inputs = [
        torch.randint(0, ARCHITECTURE["vocab_size"], (micro_batch, sequence_length), generator=generator).to(device)
        for _ in range(accumulation_steps)
    ]
    initial = {name: value.detach().cpu().clone() for name, value in raw_model.state_dict().items()}
    return {
        "raw_model": raw_model,
        "model": model,
        "optimizer": optimizer,
        "inputs": inputs,
        "initial": initial,
        "control_group": control,
        "device": device,
        "rank": rank,
        "world_size": world_size,
        "micro_batch": micro_batch,
        "sequence_length": sequence_length,
        "accumulation_steps": accumulation_steps,
        "effective_batch": world_size * micro_batch * accumulation_steps,
        "tokens_per_step": world_size * micro_batch * sequence_length * accumulation_steps,
        "parameter_count": sum(p.numel() for p in raw_model.parameters()),
        "quality": request.get("quality", {}),
        "mode": "measure",
    }


def reset(state):
    state["raw_model"].load_state_dict(state["initial"])


def before_run(state):
    control = state.get("control_group")
    if control is not None:
        dist.barrier(group=control)


def set_mode(state, mode):
    state["mode"] = mode


def run(state):
    model = state["model"]
    optimizer = state["optimizer"]
    inputs = state["inputs"]
    total = len(inputs)
    optimizer.zero_grad(set_to_none=True)
    losses = []
    with _phase("gradient_accumulation"):
        for index, tokens in enumerate(inputs):
            context = model.no_sync() if isinstance(model, DDP) and index + 1 < total else nullcontext()
            with context:
                loss = model(input_ids=tokens, labels=tokens, use_cache=False).loss / total
                loss.backward()
                losses.append(loss.detach())
        step_loss = torch.stack(losses).sum()
    with _phase("optimizer_step"):
        optimizer.step()
        torch.cuda.synchronize()
    return {"loss": step_loss, "microbatch_losses": torch.stack(losses)}


def observe(state, result):
    quality = state["quality"]
    raw_model = state["raw_model"]
    rank = state["rank"]
    grad_atol = float(quality.get("grad_atol", 1e-6))
    grad_rtol = float(quality.get("grad_rtol", 1e-4))
    loss = result["loss"].detach()
    finite = bool(torch.isfinite(loss).all())
    observations = {"finite_loss": finite, "loss": loss.cpu(), "microbatch_losses": result["microbatch_losses"].detach().cpu()}
    if quality.get("require_finite_loss", True) and not finite:
        raise AssertionError("Accumulated loss is not finite")
    gradients = []
    parameters = []
    max_replica_diff = 0.0
    for parameter in raw_model.parameters():
        gradient = parameter.grad
        if gradient is None:
            raise RuntimeError(f"Missing gradient for parameter with shape {tuple(parameter.shape)}")
        replica = gradient.detach().clone() if rank == 0 else torch.empty_like(gradient)
        dist.broadcast(replica, src=0)
        torch.testing.assert_close(gradient, replica, atol=0.0, rtol=0.0)
        max_replica_diff = max(max_replica_diff, (gradient.detach() - replica).abs().max().item())
        gradients.append(gradient.detach().cpu())
        parameters.append(parameter.detach().cpu())
        del replica
    observations["gradients"] = gradients
    observations["parameters"] = parameters
    observations["exact_gradient_replica_check"] = True
    observations["exact_gradient_replica_max_abs_diff"] = max_replica_diff
    probe = gradients[0].flatten()[:4096]
    try:
        torch.testing.assert_close(probe + 1, probe, atol=grad_atol, rtol=grad_rtol)
        rejected = False
    except AssertionError:
        rejected = True
    observations["corrupted_output_rejected"] = bool(rejected)
    if not rejected:
        raise AssertionError("Correctness control did not reject +1 corruption")
    return observations


def metrics(state, result, elapsed_ms):
    elapsed = float(elapsed_ms)
    tokens_per_step = int(state["tokens_per_step"])
    return {
        "step_ms": elapsed,
        "tokens_per_second": tokens_per_step * 1000.0 / elapsed,
        "tokens_per_step": float(tokens_per_step),
        "effective_batch": float(state["effective_batch"]),
        "micro_batches_per_step": float(state["accumulation_steps"]),
        "micro_batch": float(state["micro_batch"]),
        "sequence_length": float(state["sequence_length"]),
        "world_size": float(state["world_size"]),
        "parameter_count": float(state["parameter_count"]),
        "peak_allocated_bytes": float(torch.cuda.max_memory_allocated()),
    }


def models(state):
    return {"model": state["raw_model"]}
