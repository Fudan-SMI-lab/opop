"""Adapter for the frozen qwen_inference interactive workload.

Mirrors inputs/source/qwen_inference.py: Qwen3-4B BF16 SDPA greedy generation of
exactly new_tokens tokens (EOS early stopping disabled) over a chat-templated
prompt repeated/truncated to prompt_tokens, batch-replicated on the bound device.
The measured unit is one complete infer() execution: prefill with use_cache=True
and logits_to_keep=1, argmax first token, then new_tokens-1 KV-cached decode
steps under torch.inference_mode(), bracketed by the source's CUDA-sync timing
boundaries.
"""
from __future__ import annotations

import time
from contextlib import contextmanager

import torch
from transformers import AutoTokenizer, Qwen3ForCausalLM

try:  # optional discovery annotation; phases are task-level, not target selection
    from kernel_optimizer.project.discovery import phase as _phase
except Exception:  # pragma: no cover - keeps the adapter importable anywhere

    @contextmanager
    def _phase(name):
        yield


# Prompt text copied verbatim from the source's qwen_inference config in
# inputs/source/workloads.json; the request workload row carries only sizes.
SOURCE_PROMPT = (
    "Explain how a computer executes a neural network and why memory access, "
    "arithmetic and communication affect performance. Use concrete examples."
)
SEED = 20261020


def _configure(device):
    """Source numerical policies (common.py:configure) on the bound device."""
    torch.set_num_threads(4)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass  # already initialized by the host process; the cap is then unchanged
    torch.manual_seed(SEED)
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision("highest")


def prepare(context, request):
    device = context["device"]
    _configure(device)

    workload = request["workload"]
    assets = request["assets"]["qwen3_4b_weights"]
    batch = int(workload["batch"])
    prompt_tokens = int(workload["prompt_tokens"])
    new_tokens = int(workload["new_tokens"])

    tokenizer = AutoTokenizer.from_pretrained(assets, local_files_only=True)
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": SOURCE_PROMPT}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    base = tokenizer.encode(text, add_special_tokens=False)
    ids = (base * (prompt_tokens // len(base) + 1))[:prompt_tokens]
    inputs = torch.tensor([ids] * batch, device=device)

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

    state = {
        "model": model,
        "inputs": inputs,
        "batch": batch,
        "prompt_tokens": prompt_tokens,
        "new_tokens": new_tokens,
        "device": device,
        "mode": "quality",  # full capture by default; set_mode switches it
    }
    return state


def set_mode(state, mode):
    state["mode"] = mode


def reset(state):
    # Stateless inference: each run builds its own request-local KV cache under
    # inference_mode. Only the RNG needs restoring to the source's seed policy;
    # nothing is torn down and no process group is destroyed.
    torch.manual_seed(SEED)


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

    batch = state["batch"]
    timing = {
        "ttft_ms": (first - start) / 1e6,
        "decode_ms": (end - first) / 1e6,
        "end_to_end_ms": (end - start) / 1e6,
        "decode_tokens_per_second": batch * (new_tokens - 1) * 1e9 / (end - first),
        "internal_generated_tokens_per_second": batch * new_tokens * 1e9 / (end - start),
    }
    result = {
        "tokens": tokens,
        "logits": torch.stack(logits, dim=1) if capture else None,
        "timing": timing,
    }
    return result


def observe(state, result):
    observation = {"tokens": result["tokens"].detach()}
    if result["logits"] is not None:
        observation["logits"] = result["logits"].detach()
    return observation


def metrics(state, result, elapsed_ms):
    batch = state["batch"]
    new_tokens = state["new_tokens"]
    timing = result["timing"]
    values = {
        # Frozen objective: full-request throughput over the end-to-end window,
        # including prefill/first token, derived from the unchanged work amount.
        "generated_tokens_per_second": batch * new_tokens * 1000.0 / elapsed_ms,
        "ttft_ms": float(timing["ttft_ms"]),
        "decode_ms": float(timing["decode_ms"]),
        "end_to_end_ms": float(timing["end_to_end_ms"]),
        "decode_tokens_per_second": float(timing["decode_tokens_per_second"]),
        "internal_generated_tokens_per_second": float(timing["internal_generated_tokens_per_second"]),
        "generated_tokens": float(batch * new_tokens),
        "batch": float(batch),
        "prompt_tokens": float(state["prompt_tokens"]),
        "new_tokens": float(new_tokens),
        "peak_allocated_bytes": float(torch.cuda.max_memory_allocated(state["device"])),
    }
    return values


def models(state):
    return {"model": state["model"]}
