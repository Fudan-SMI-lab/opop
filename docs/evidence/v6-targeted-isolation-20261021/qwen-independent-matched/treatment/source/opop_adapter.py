"""Adapter for the Qwen3-4B interactive greedy-inference project.

Source reference: qwen_inference.py / common.py (frozen project inputs).
One run() equals exactly one source infer(model, inputs, new_tokens) call:
prefill over the prompt (use_cache=True, logits_to_keep=1), argmax first
token, explicit first-token-ready CUDA synchronize (the ttft_ms boundary),
then new_tokens-1 KV-cached greedy decode steps, final synchronize.
"""
from __future__ import annotations

import contextlib
import json
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer, Qwen3ForCausalLM

SEED = 20261020

# Frozen task data from workloads.json ("qwen_inference.prompt"); used only as
# a last-resort fallback when the workloads file cannot be located.
_FALLBACK_PROMPT = (
    "Explain how a computer executes a neural network and why memory access, "
    "arithmetic and communication affect performance. Use concrete examples."
)


def _phase(name):
    """Optional phase annotation hook; no-op when the API is unavailable."""
    try:
        from kernel_optimizer.project.discovery import phase

        return phase(name)
    except Exception:
        return contextlib.nullcontext()


def _configure(device):
    """Mirror common.configure() numerics/seed policies (common.py:34-42)."""
    torch.set_num_threads(4)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass  # already initialized in this process
    torch.manual_seed(SEED)
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision("highest")


def _load_prompt_text(request):
    here = Path(__file__).resolve().parent
    candidates = [
        here / "inputs" / "source" / "workloads.json",
        Path(str(request.get("source_root", ""))) / "workloads.json"
        if request.get("source_root")
        else None,
        here / "workloads.json",
    ]
    for candidate in candidates:
        if candidate and candidate.is_file():
            config = json.loads(candidate.read_text())["qwen_inference"]
            return config["prompt"]
    return _FALLBACK_PROMPT


def prepare(context, request):
    device = context["device"]
    _configure(device)

    assets = request["assets"]["assets"]
    workload = request["workload"]
    batch = int(workload["batch"])
    prompt_tokens = int(workload["prompt_tokens"])
    new_tokens = int(workload["new_tokens"])

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

    # Exact source input construction (qwen_inference.py:55-58).
    prompt = _load_prompt_text(request)
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
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
        "last_result": None,
    }
    return state


def set_mode(state, mode):
    """'quality' retains per-step logits (source capture path); 'measure' and
    'profile' keep the source's clean non-capture timing path."""
    state["mode"] = mode


def reset(state):
    # Stateless greedy inference: run() builds its own fresh KV cache inside
    # the timed invocation exactly as the source does, and model weights are
    # never mutated, so there is no persistent cache to clear here. Re-seed
    # so any incidental host randomness is identical across repeats.
    torch.manual_seed(SEED)
    state["last_result"] = None


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

    result = {
        "tokens": tokens,
        "logits": torch.stack(logits, dim=1) if capture else None,
        "timings_ns": (start, first, end),
        "new_tokens": new_tokens,
        "batch": state["batch"],
    }
    state["last_result"] = result
    return result


def observe(state, result):
    with torch.no_grad():
        observation = {
            "input_ids": state["inputs"].detach().to("cpu"),
            "tokens": result["tokens"].detach().to("cpu"),
        }
        if result.get("logits") is not None:
            observation["logits"] = result["logits"].detach().to("cpu")
        return observation


def metrics(state, result, elapsed_ms):
    start, first, end = result["timings_ns"]
    batch = state["batch"]
    new_tokens = state["new_tokens"]
    metrics_out = {
        "ttft_ms": (first - start) / 1e6,
        "decode_ms": (end - first) / 1e6,
        "end_to_end_ms": (end - start) / 1e6,
        "decode_tokens_per_second": batch * (new_tokens - 1) * 1e9 / (end - first),
        "generated_tokens_per_second": batch * new_tokens * 1e9 / (end - start),
        "generated_tokens_per_second_wall": batch * new_tokens * 1000.0 / elapsed_ms,
        "generated_tokens": batch * new_tokens,
        "elapsed_ms": float(elapsed_ms),
    }
    try:
        metrics_out["peak_allocated_bytes"] = float(torch.cuda.max_memory_allocated())
    except RuntimeError:
        pass
    return metrics_out


def models(state):
    return {"model": state["model"]}
