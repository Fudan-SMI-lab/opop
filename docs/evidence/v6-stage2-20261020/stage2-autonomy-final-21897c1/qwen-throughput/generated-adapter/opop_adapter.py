"""Adapter for the frozen qwen_inference 'interactive' workload.

Source: examples/v6_validation/qwen_inference.py (Qwen3-4B, BF16, SDPA, greedy).
Frozen workload: batch=1, prompt_tokens=256, new_tokens=32, objective
generated_tokens_per_second (maximize). The adapter reproduces the source's
numerical policies, prompt construction, greedy decode loop, KV-cache reuse and
its explicit CUDA-synchronize completion boundaries.
"""
from __future__ import annotations

import time
from contextlib import contextmanager

import torch
from transformers import AutoTokenizer, Qwen3ForCausalLM

try:  # optional discovery annotations; never required for correctness
    from kernel_optimizer.project.discovery import phase as _phase
except Exception:  # pragma: no cover - fallback when the package is absent
    @contextmanager
    def _phase(name):  # noqa: D401 - simple no-op
        yield


SEED = 20261020
PROMPT = (
    "Explain how a computer executes a neural network and why memory access, "
    "arithmetic and communication affect performance. Use concrete examples."
)


def _configure(device):
    """Mirror common.configure() with the worker-bound device."""
    torch.set_num_threads(4)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass  # already initialized in this process; numerical policies below still apply
    torch.manual_seed(SEED)
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision("highest")


def prepare(context, request):
    device = context["device"]
    _configure(device)
    assets = request["assets"]["assets"]
    workload = request["workload"]
    tokenizer = AutoTokenizer.from_pretrained(assets, local_files_only=True)
    model = (
        Qwen3ForCausalLM.from_pretrained(
            assets, dtype=torch.bfloat16, attn_implementation="sdpa", local_files_only=True
        )
        .to(device)
        .eval()
    )
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": PROMPT}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    base = tokenizer.encode(text, add_special_tokens=False)
    ids = (base * (workload["prompt_tokens"] // len(base) + 1))[: workload["prompt_tokens"]]
    inputs = torch.tensor([ids] * workload["batch"], device=device)
    return {
        "model": model,
        "inputs": inputs,
        "batch": workload["batch"],
        "prompt_tokens": workload["prompt_tokens"],
        "new_tokens": workload["new_tokens"],
        "device": device,
        "mode": "quality",
    }


def reset(state):
    """Inference is stateless across runs: run() builds its own request-local
    KV cache from the prefill output. Only the RNG and eval mode are restored."""
    torch.manual_seed(SEED)
    state["model"].eval()


def set_mode(state, mode):
    state["mode"] = mode


@torch.inference_mode()
def _generate(model, inputs, new_tokens, capture):
    batch = inputs.shape[0]
    torch.cuda.synchronize()
    start = time.perf_counter_ns()
    generated = []
    logits = []
    with _phase("prefill"):
        output = model(input_ids=inputs, use_cache=True, logits_to_keep=1)
        token = output.logits[:, -1].argmax(dim=-1, keepdim=True)
        generated.append(token)
        if capture:
            logits.append(output.logits[:, -1])
        cache = output.past_key_values
    torch.cuda.synchronize()  # explicit first-token-ready boundary (source TTFT)
    first = time.perf_counter_ns()
    with _phase("decode"):
        for _ in range(new_tokens - 1):
            output = model(input_ids=token, past_key_values=cache, use_cache=True, logits_to_keep=1)
            cache = output.past_key_values
            token = output.logits[:, -1].argmax(dim=-1, keepdim=True)
            generated.append(token)
            if capture:
                logits.append(output.logits[:, -1])
        tokens = torch.cat(generated, dim=1)
    torch.cuda.synchronize()
    end = time.perf_counter_ns()
    metrics = {
        "ttft_ms": (first - start) / 1e6,
        "decode_ms": (end - first) / 1e6,
        "end_to_end_ms": (end - start) / 1e6,
        "decode_tokens_per_second": batch * (new_tokens - 1) * 1e9 / (end - first),
        "generated_tokens_per_second": batch * new_tokens * 1e9 / (end - start),
    }
    return tokens, (torch.stack(logits, dim=1) if capture else None), metrics


def run(state):
    capture = state.get("mode", "quality") == "quality"
    tokens, logits, phase_metrics = _generate(state["model"], state["inputs"], state["new_tokens"], capture)
    return {"tokens": tokens, "logits": logits, "phase_metrics": phase_metrics}


def observe(state, result):
    observation = {"tokens": result["tokens"].detach().to("cpu")}
    logits = result.get("logits")
    if logits is not None:
        observation["logits"] = logits.detach().to("cpu")
    return observation


def metrics(state, result, elapsed_ms):
    work = state["batch"] * state["new_tokens"]
    phase = result["phase_metrics"]
    return {
        "generated_tokens_per_second": work * 1000.0 / elapsed_ms,
        "ttft_ms": float(phase["ttft_ms"]),
        "decode_ms": float(phase["decode_ms"]),
        "end_to_end_ms": float(phase["end_to_end_ms"]),
        "decode_tokens_per_second": float(phase["decode_tokens_per_second"]),
        "batch": float(state["batch"]),
        "prompt_tokens": float(state["prompt_tokens"]),
        "new_tokens": float(state["new_tokens"]),
        "generated_tokens": float(work),
    }


def models(state):
    return {"model": state["model"]}
