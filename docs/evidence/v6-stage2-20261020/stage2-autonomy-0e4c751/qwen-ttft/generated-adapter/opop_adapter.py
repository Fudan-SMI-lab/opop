"""Adapter for the frozen qwen_inference 'interactive' workload (Qwen3-4B, BF16 SDPA, greedy).

Mirrors inputs/source/qwen_inference.py semantics: same configure() numerical policies,
same asset loading, same tiled 256-token prompt input, same greedy decode loop with
use_cache=True / logits_to_keep=1, and the same TTFT first-token-ready boundary
(prefill forward + argmax + torch.cuda.synchronize).
"""
from __future__ import annotations

import contextlib
import time

import torch
from transformers import AutoTokenizer, Qwen3ForCausalLM

try:
    from kernel_optimizer.project.discovery import phase as _phase
except Exception:  # annotation is optional; never affect execution
    def _phase(_name):
        return contextlib.nullcontext()

# Frozen source values (common.py configure() and workloads.json qwen_inference config).
SEED = 20261020
PROMPT = ("Explain how a computer executes a neural network and why memory access, "
          "arithmetic and communication affect performance. Use concrete examples.")


def _configure(device):
    torch.set_num_threads(4)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass  # already initialized by the host process
    torch.manual_seed(SEED)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision("highest")
    if str(device).startswith("cuda"):
        torch.cuda.set_device(device)


def prepare(context, request):
    device = torch.device(context["device"])
    _configure(device)
    assets = request["assets"]["assets"]
    workload = request["workload"]
    tokenizer = AutoTokenizer.from_pretrained(assets, local_files_only=True)
    model = Qwen3ForCausalLM.from_pretrained(
        assets, dtype=torch.bfloat16, attn_implementation="sdpa", local_files_only=True
    ).to(device).eval()
    # Source input construction: chat template (thinking disabled), encode without
    # special tokens, tile+truncate to exactly prompt_tokens, replicate per batch row.
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": PROMPT}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )
    base = tokenizer.encode(text, add_special_tokens=False)
    prompt_tokens = int(workload["prompt_tokens"])
    batch = int(workload["batch"])
    ids = (base * (prompt_tokens // len(base) + 1))[:prompt_tokens]
    inputs = torch.tensor([ids] * batch, device=device)
    state = {
        "model": model,
        "inputs": inputs,
        "batch": batch,
        "prompt_tokens": prompt_tokens,
        "new_tokens": int(workload["new_tokens"]),
        "device": device,
        "mode": "quality",
        "capture": True,
    }
    return state


def set_mode(state, mode):
    state["mode"] = mode
    # Only the source's capture path retains per-step logits; measure/profile keep
    # the clean timing path. Execution semantics and token count are unchanged.
    state["capture"] = (mode == "quality")


def reset(state):
    # Stateless greedy inference: the KV cache is created inside run() per call and
    # no parameters, gradients or RNG drive the decode path. Restore the frozen
    # starting point: eval mode and the source seed policy.
    state["model"].eval()
    torch.manual_seed(SEED)
    state["mode"] = state.get("mode", "quality")
    state["capture"] = state["mode"] == "quality"


def run(state):
    model = state["model"]
    inputs = state["inputs"]
    new_tokens = state["new_tokens"]
    device = state["device"]
    capture = state["capture"]
    with torch.inference_mode():
        torch.cuda.synchronize(device)
        start = time.perf_counter_ns()
        with _phase("prefill"):
            output = model(input_ids=inputs, use_cache=True, logits_to_keep=1)
            token = output.logits[:, -1].argmax(dim=-1, keepdim=True)
        generated = [token]
        logits = [output.logits[:, -1]] if capture else []
        cache = output.past_key_values
        torch.cuda.synchronize(device)  # explicit first-token-ready boundary
        first = time.perf_counter_ns()
        with _phase("decode"):
            for _ in range(new_tokens - 1):
                output = model(input_ids=token, past_key_values=cache,
                               use_cache=True, logits_to_keep=1)
                cache = output.past_key_values
                token = output.logits[:, -1].argmax(dim=-1, keepdim=True)
                generated.append(token)
                if capture:
                    logits.append(output.logits[:, -1])
        tokens = torch.cat(generated, dim=1)
        torch.cuda.synchronize(device)
        end = time.perf_counter_ns()
    stacked = torch.stack(logits, dim=1) if capture else None
    batch = inputs.shape[0]
    timing = {
        "ttft_ms": (first - start) / 1e6,
        "decode_ms": (end - first) / 1e6,
        "end_to_end_ms": (end - start) / 1e6,
        "decode_tokens_per_second": batch * (new_tokens - 1) * 1e9 / (end - first),
        "generated_tokens_per_second": batch * new_tokens * 1e9 / (end - start),
    }
    return {"tokens": tokens, "logits": stacked, "timing": timing}


def observe(state, result):
    observed = {"tokens": result["tokens"].detach().clone()}
    if result.get("logits") is not None:
        observed["logits"] = result["logits"].detach().clone()
    return observed


def metrics(state, result, elapsed_ms):
    timing = result["timing"]
    batch = state["batch"]
    new_tokens = state["new_tokens"]
    metrics = {
        "ttft_ms": float(timing["ttft_ms"]),
        "decode_ms": float(timing["decode_ms"]),
        "end_to_end_ms": float(timing["end_to_end_ms"]),
        "decode_tokens_per_second": float(timing["decode_tokens_per_second"]),
        "generated_tokens_per_second": float(timing["generated_tokens_per_second"]),
        "batch": float(batch),
        "prompt_tokens": float(state["prompt_tokens"]),
        "new_tokens": float(new_tokens),
        "generated_tokens": float(batch * new_tokens),
        "decode_tokens": float(batch * (new_tokens - 1)),
        "framework_elapsed_ms": float(elapsed_ms),
    }
    return metrics


def models(state):
    return {"model": state["model"]}
