"""Adapter for the frozen qwen_inference 'interactive' workload.

Reproduces the semantics of inputs/source/qwen_inference.py for exactly one
workload: batch=1, prompt_tokens=256, new_tokens=32 greedy generation with a
BF16 SDPA Qwen3-4B loaded from request['assets']. The objective (ttft_ms) is
measured inside run() with the source's own completion boundary: an explicit
torch.cuda.synchronize() immediately after the first greedy token is ready.
"""
from __future__ import annotations

import contextlib
import time

import torch
from transformers import AutoTokenizer, Qwen3ForCausalLM

try:  # optional phase annotation; harmless if unavailable
    from kernel_optimizer.project.discovery import phase as _discovery_phase
except Exception:  # pragma: no cover
    _discovery_phase = None

# Fixed source data (workloads.json -> qwen_inference.prompt), embedded so the
# adapter does not depend on the source tree at runtime.
PROMPT = (
    "Explain how a computer executes a neural network and why memory access, "
    "arithmetic and communication affect performance. Use concrete examples."
)
SEED = 20261020


def _phase(name):
    if _discovery_phase is not None:
        return _discovery_phase(name)
    return contextlib.nullcontext()


def _configure(device):
    # Mirrors common.configure() from the source.
    torch.set_num_threads(4)
    with contextlib.suppress(RuntimeError):
        torch.set_num_interop_threads(1)
    torch.manual_seed(SEED)
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision("highest")


def prepare(context, request):
    device = torch.device(context["device"])
    _configure(device)

    workload = request["workload"]
    batch = int(workload["batch"])
    prompt_tokens = int(workload["prompt_tokens"])
    new_tokens = int(workload["new_tokens"])
    assets = request["assets"]["assets"]

    tokenizer = AutoTokenizer.from_pretrained(assets, local_files_only=True)
    model = (
        Qwen3ForCausalLM.from_pretrained(
            assets,
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
            local_files_only=True,
        )
        .to(device)
        .eval()
    )

    # Input construction identical to the source main loop.
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": PROMPT}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    base = tokenizer.encode(text, add_special_tokens=False)
    ids = (base * (prompt_tokens // len(base) + 1))[:prompt_tokens]
    inputs = torch.tensor([ids] * batch, device=device)

    state = {
        "model": model,
        "tokenizer": tokenizer,
        "inputs": inputs,
        "batch": batch,
        "prompt_tokens": prompt_tokens,
        "new_tokens": new_tokens,
        "device": device,
        "mode": "quality",
        "parameter_count": sum(p.numel() for p in model.parameters()),
    }
    return state


def set_mode(state, mode):
    state["mode"] = mode


def reset(state):
    # Stateless inference: every run() builds its own request-local KV cache
    # and the eval-mode model carries no mutable weights/gradients. Only the
    # source's per-repeat peak-memory protocol is restored (outside timing).
    if torch.cuda.is_available():
        with contextlib.suppress(RuntimeError):
            torch.cuda.reset_peak_memory_stats(state["device"])


@torch.inference_mode()
def run(state):
    model = state["model"]
    inputs = state["inputs"]
    new_tokens = state["new_tokens"]
    capture = state.get("mode", "quality") == "quality"

    torch.cuda.synchronize()
    start = time.perf_counter_ns()
    with _phase("prefill"):
        output = model(input_ids=inputs, use_cache=True, logits_to_keep=1)
        token = output.logits[:, -1].argmax(dim=-1, keepdim=True)
        generated = [token]
        logits = [output.logits[:, -1]] if capture else []
        cache = output.past_key_values
    torch.cuda.synchronize()  # explicit first-token-ready boundary
    first = time.perf_counter_ns()
    with _phase("decode"):
        for _ in range(new_tokens - 1):
            output = model(
                input_ids=token,
                past_key_values=cache,
                use_cache=True,
                logits_to_keep=1,
            )
            cache = output.past_key_values
            token = output.logits[:, -1].argmax(dim=-1, keepdim=True)
            generated.append(token)
            if capture:
                logits.append(output.logits[:, -1])
        tokens = torch.cat(generated, dim=1)
    torch.cuda.synchronize()
    end = time.perf_counter_ns()

    batch = inputs.shape[0]
    prefill_ns = first - start
    decode_ns = end - first
    metrics = {
        "ttft_ms": prefill_ns / 1e6,
        "decode_ms": decode_ns / 1e6,
        "end_to_end_ms": (end - start) / 1e6,
        "decode_tokens_per_second": (
            batch * (new_tokens - 1) * 1e9 / decode_ns if decode_ns > 0 else 0.0
        ),
        "generated_tokens_per_second": (
            batch * new_tokens * 1e9 / (end - start) if end > start else 0.0
        ),
    }
    result = {
        "tokens": tokens,
        "logits": torch.stack(logits, dim=1) if capture else None,
        "metrics": metrics,
    }
    return result


def observe(state, result):
    observations = {"tokens": result["tokens"].detach().cpu()}
    if result.get("logits") is not None:
        observations["logits"] = result["logits"].detach().cpu()
    return observations


def metrics(state, result, elapsed_ms):
    values = {key: float(value) for key, value in result["metrics"].items()}
    batch = state["batch"]
    new_tokens = state["new_tokens"]
    elapsed = float(elapsed_ms)
    values["elapsed_ms"] = elapsed
    values["batch"] = float(batch)
    values["prompt_tokens"] = float(state["prompt_tokens"])
    values["new_tokens"] = float(new_tokens)
    values["generated_tokens"] = float(batch * new_tokens)
    values["decode_tokens"] = float(batch * (new_tokens - 1))
    values["throughput_tokens_per_second"] = (
        float(batch * new_tokens) / (elapsed / 1000.0) if elapsed > 0 else 0.0
    )
    if torch.cuda.is_available():
        with contextlib.suppress(RuntimeError):
            values["peak_allocated_bytes"] = float(
                torch.cuda.max_memory_allocated(state["device"])
            )
    return values


def models(state):
    return {"model": state["model"]}
