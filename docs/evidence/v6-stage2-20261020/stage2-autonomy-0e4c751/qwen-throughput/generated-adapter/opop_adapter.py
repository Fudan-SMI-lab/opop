"""Adapter for the frozen qwen_inference 'interactive' workload (batch=1, prompt=256, new_tokens=32).

Mirrors inputs/source/qwen_inference.py: BF16 SDPA Qwen3-4B greedy decoding with a KV cache,
TF32 disabled, seed 20261020, fixed chat-template prompt tiled to the requested token count.
The full 32-token generation with CUDA synchronization at both boundaries is the workload.
"""
from __future__ import annotations

import time

import torch
from transformers import AutoTokenizer, Qwen3ForCausalLM

try:
    from kernel_optimizer.project.discovery import phase
except Exception:  # diagnostics-only annotation; never affect execution
    from contextlib import contextmanager

    @contextmanager
    def phase(name):  # noqa: D401 - no-op fallback
        yield


SEED = 20261020
PROMPT = (
    "Explain how a computer executes a neural network and why memory access, arithmetic "
    "and communication affect performance. Use concrete examples."
)


def _configure(device):
    torch.set_num_threads(4)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    torch.manual_seed(SEED)
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision("highest")


def _build_inputs(tokenizer, workload, device):
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": PROMPT}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    base = tokenizer.encode(text, add_special_tokens=False)
    ids = (base * (workload["prompt_tokens"] // len(base) + 1))[: workload["prompt_tokens"]]
    return torch.tensor([ids] * workload["batch"], device=device)


def prepare(context, request):
    device = torch.device(context["device"])
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
    inputs = _build_inputs(tokenizer, workload, device)
    state = {
        "model": model,
        "inputs": inputs,
        "batch": workload["batch"],
        "new_tokens": workload["new_tokens"],
        "prompt_tokens": workload["prompt_tokens"],
        "device": device,
        "mode": "quality",
        "parameter_count": sum(p.numel() for p in model.parameters()),
    }
    return state


def set_mode(state, mode):
    state["mode"] = mode


def before_run(state):
    torch.cuda.reset_peak_memory_stats()


def reset(state):
    torch.manual_seed(SEED)
    if state["device"].type == "cuda":
        torch.cuda.reset_peak_memory_stats()


@torch.inference_mode()
def run(state):
    model = state["model"]
    inputs = state["inputs"]
    new_tokens = state["new_tokens"]
    capture = state.get("mode", "quality") == "quality"
    torch.cuda.synchronize()
    start = time.perf_counter_ns()
    with phase("prefill"):
        output = model(input_ids=inputs, use_cache=True, logits_to_keep=1)
    token = output.logits[:, -1].argmax(dim=-1, keepdim=True)
    generated = [token]
    logits = [output.logits[:, -1]] if capture else []
    cache = output.past_key_values
    torch.cuda.synchronize()  # explicit first-token-ready boundary
    first = time.perf_counter_ns()
    with phase("decode"):
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
    batch = state["batch"]
    timing = {
        "ttft_ms": (first - start) / 1e6,
        "decode_ms": (end - first) / 1e6,
        "end_to_end_ms": (end - start) / 1e6,
        "decode_tokens_per_second": batch * (new_tokens - 1) * 1e9 / (end - first),
        "generated_tokens_per_second": batch * new_tokens * 1e9 / (end - start),
    }
    result = {"tokens": tokens, "timing": timing, "peak_allocated_bytes": float(torch.cuda.max_memory_allocated())}
    if capture:
        result["logits"] = torch.stack(logits, dim=1)
    return result


def observe(state, result):
    observation = {"tokens": result["tokens"].detach()}
    if "logits" in result:
        observation["logits"] = result["logits"].detach()
    return observation


def metrics(state, result, elapsed_ms):
    batch = state["batch"]
    new_tokens = state["new_tokens"]
    timing = result.get("timing", {})
    values = {
        "generated_tokens_per_second": batch * new_tokens / (elapsed_ms / 1e3),
        "new_tokens": float(new_tokens),
        "batch": float(batch),
        "prompt_tokens": float(state["prompt_tokens"]),
        "parameter_count": float(state["parameter_count"]),
        "peak_allocated_bytes": float(result.get("peak_allocated_bytes", float("nan"))),
    }
    for key in ("ttft_ms", "decode_ms", "end_to_end_ms", "decode_tokens_per_second"):
        if key in timing:
            values[key] = float(timing[key])
    return {key: value for key, value in values.items() if value == value and abs(value) != float("inf")}


def models(state):
    return {"model": state["model"]}
