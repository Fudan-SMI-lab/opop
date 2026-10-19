"""Adapter for the qwen_inference project.

Reproduces the semantics of inputs/source/qwen_inference.py (infer) and
inputs/source/common.py (configure) for the single frozen workload in
request.json: batch=1, prompt_tokens=256, new_tokens=32, greedy decoding on
pretrained Qwen3-4B in BF16 with SDPA attention.

Timing semantics (from the source):
  - ttft_ms is measured across the prefill forward and an explicit
    torch.cuda.synchronize() that marks the first-token-ready boundary.
  - The full workload additionally runs new_tokens-1 greedy decode steps and
    ends with a final torch.cuda.synchronize().
"""
from __future__ import annotations

import time

import torch
from transformers import AutoTokenizer, Qwen3ForCausalLM

# Generation prompt, verbatim from inputs/source/workloads.json
# (config["qwen_inference"]["prompt"]); request.workload does not carry it.
PROMPT = (
    "Explain how a computer executes a neural network and why memory access, "
    "arithmetic and communication affect performance. Use concrete examples."
)

SEED = 20261020

try:  # optional phase annotation; absent in plain environments
    from kernel_optimizer.project.discovery import phase as _discovery_phase
except Exception:  # pragma: no cover - optional dependency
    _discovery_phase = None


class _NullPhase:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _phase(name):
    if _discovery_phase is None:
        return _NullPhase()
    try:
        cm = _discovery_phase(name)
        if hasattr(cm, "__enter__") and hasattr(cm, "__exit__"):
            return cm
    except Exception:
        pass
    return _NullPhase()


def _configure(device):
    # Mirrors common.configure(), bound to the worker-assigned device.
    torch.set_num_threads(4)
    try:
        torch.set_num_interop_threads(1)
    except Exception:
        pass
    torch.manual_seed(SEED)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision("highest")


def prepare(context, request):
    device = torch.device(context.get("device") or "cuda:0")
    _configure(device)

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

    workload = request["workload"]
    # Real data path from the source: chat-templated prompt, tiled/truncated
    # to the frozen prompt token count, batched and GPU-resident.
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": PROMPT}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    base = tokenizer.encode(text, add_special_tokens=False)
    ids = (base * (workload["prompt_tokens"] // len(base) + 1))[: workload["prompt_tokens"]]
    inputs = torch.tensor([ids] * workload["batch"], device=device)

    state = {
        "model": model,
        "tokenizer": tokenizer,
        "inputs": inputs,
        "batch": workload["batch"],
        "prompt_tokens": workload["prompt_tokens"],
        "new_tokens": workload["new_tokens"],
        "device": device,
        "mode": "measure",
    }
    return state


def reset(state):
    # Stateless inference: every run builds its own request-local KV cache, so
    # there is no model/optimizer state to restore beyond the seed policy.
    torch.manual_seed(SEED)
    state["model"].eval()
    device = state["device"]
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)


def before_run(state):
    # Outside the timer; matches the source resetting peak stats before a sample.
    device = state["device"]
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def set_mode(state, mode):
    state["mode"] = mode


@torch.inference_mode()
def run(state):
    model = state["model"]
    inputs = state["inputs"]
    new_tokens = state["new_tokens"]
    batch = state["batch"]
    capture = state.get("mode") == "quality"

    torch.cuda.synchronize()
    start = time.perf_counter_ns()
    with _phase("prefill"):
        output = model(input_ids=inputs, use_cache=True, logits_to_keep=1)
        token = output.logits[:, -1].argmax(dim=-1, keepdim=True)
        generated = [token]
        logits = [output.logits[:, -1].detach()] if capture else []
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
                logits.append(output.logits[:, -1].detach())
        tokens = torch.cat(generated, dim=1)
    torch.cuda.synchronize()
    end = time.perf_counter_ns()

    return {
        "tokens": tokens,
        "logits": torch.stack(logits, dim=1) if capture else None,
        "timing_ns": {"start": start, "first": first, "end": end},
        "batch": batch,
        "new_tokens": new_tokens,
    }


def observe(state, result):
    observation = {"tokens": result["tokens"].detach().to("cpu")}
    if result.get("logits") is not None:
        observation["logits"] = result["logits"].detach().to("cpu")
    return observation


def metrics(state, result, elapsed_ms):
    timing = result["timing_ns"]
    prefill_ns = max(timing["first"] - timing["start"], 1)
    decode_ns = max(timing["end"] - timing["first"], 1)
    total_ns = max(timing["end"] - timing["start"], 1)
    batch = result["batch"]
    new_tokens = result["new_tokens"]
    values = {
        "ttft_ms": prefill_ns / 1e6,
        "decode_ms": decode_ns / 1e6,
        "end_to_end_ms": total_ns / 1e6,
        "decode_tokens_per_second": batch * (new_tokens - 1) * 1e9 / decode_ns,
        "generated_tokens_per_second": batch * new_tokens * 1e9 / total_ns,
        "generated_tokens_per_second_wall": batch * new_tokens * 1000.0 / max(elapsed_ms, 1e-9),
        "elapsed_ms": float(elapsed_ms),
        "generated_tokens": float(batch * new_tokens),
        "peak_allocated_bytes": float(torch.cuda.max_memory_allocated()),
    }
    return values


def models(state):
    return {"model": state["model"]}
