"""Adapter for the qwen_inference project: Qwen3-4B BF16 SDPA fixed-length greedy generation.

Mirrors qwen_inference.infer (qwen_inference.py:15-41) and the configure() policies from
common.py:34-42, with the device taken from the worker context instead of physical GPU 0.
"""
from __future__ import annotations

import time
from contextlib import contextmanager

import torch
from transformers import AutoTokenizer, Qwen3ForCausalLM

try:
    from kernel_optimizer.project.discovery import phase as _phase
except Exception:  # pragma: no cover - annotation is optional for the framework
    @contextmanager
    def _phase(name):
        yield


SEED = 20261020


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _build_inputs(tokenizer, workload, device):
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": workload["prompt"]}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    base = tokenizer.encode(text, add_special_tokens=False)
    prompt_tokens = int(workload["prompt_tokens"])
    ids = (base * (prompt_tokens // len(base) + 1))[:prompt_tokens]
    return torch.tensor([ids] * int(workload["batch"]), device=device)


def _configure(device):
    torch.set_num_threads(4)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    torch.manual_seed(SEED)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision("highest")


def prepare(context, request):
    device = torch.device(context["device"])
    workload = request["workload"]
    assets = request["assets"]["qwen3_4b"]
    _configure(device)
    tokenizer = AutoTokenizer.from_pretrained(assets, local_files_only=True)
    model = (
        Qwen3ForCausalLM.from_pretrained(
            assets, dtype=torch.bfloat16, attn_implementation="sdpa", local_files_only=True
        )
        .to(device)
        .eval()
    )
    text_inputs = _build_inputs(tokenizer, workload, device)
    return {
        "model": model,
        "tokenizer": tokenizer,
        "inputs": text_inputs,
        "device": device,
        "batch": int(workload["batch"]),
        "prompt_tokens": int(workload["prompt_tokens"]),
        "new_tokens": int(workload["new_tokens"]),
        "assets": assets,
        "mode": "measure",
    }


def set_mode(state, mode):
    state["mode"] = mode


def reset(state):
    return None


@torch.inference_mode()
def _infer(model, inputs, new_tokens, capture, device):
    _sync(device)
    start = time.perf_counter_ns()
    with _phase("prefill"):
        output = model(input_ids=inputs, use_cache=True, logits_to_keep=1)
        token = output.logits[:, -1].argmax(dim=-1, keepdim=True)
    generated = [token]
    logits = [output.logits[:, -1]] if capture else []
    cache = output.past_key_values
    _sync(device)  # explicit first-token-ready boundary
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
    _sync(device)
    end = time.perf_counter_ns()
    batch = inputs.shape[0]
    source_metrics = {
        "ttft_ms": (first - start) / 1e6,
        "decode_ms": (end - first) / 1e6,
        "end_to_end_ms": (end - start) / 1e6,
        "decode_tokens_per_second": batch * (new_tokens - 1) * 1e9 / (end - first),
        "generated_tokens_per_second": batch * new_tokens * 1e9 / (end - start),
    }
    stacked = torch.stack(logits, dim=1) if capture else None
    return tokens, stacked, source_metrics


def run(state):
    capture = state.get("mode") == "quality"
    tokens, logits, source_metrics = _infer(
        state["model"], state["inputs"], state["new_tokens"], capture, state["device"]
    )
    result = {"tokens": tokens.detach(), "metrics": source_metrics}
    if logits is not None:
        result["logits"] = logits.detach()
    return result


def observe(state, result):
    observed = {"tokens": result["tokens"].detach().cpu()}
    if result.get("logits") is not None:
        observed["logits"] = result["logits"].detach().cpu()
    return observed


def metrics(state, result, elapsed_ms):
    values = {key: float(value) for key, value in result["metrics"].items()}
    values["generated_tokens_per_second_wall"] = (
        state["batch"] * state["new_tokens"] / (elapsed_ms / 1000.0)
    )
    if state["device"].type == "cuda" and torch.cuda.is_available():
        values["peak_allocated_bytes"] = float(torch.cuda.max_memory_allocated(state["device"]))
    return values


def models(state):
    return {"qwen3_4b": state["model"]}
