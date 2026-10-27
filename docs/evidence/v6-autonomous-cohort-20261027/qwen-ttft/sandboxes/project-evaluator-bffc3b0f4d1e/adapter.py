"""Baseline adapter reproducing one timed qwen_inference.infer() invocation (Qwen3-4B, BF16, SDPA, greedy)."""
from __future__ import annotations

import json
import time
from contextlib import contextmanager
from pathlib import Path

import torch
from transformers import AutoTokenizer, Qwen3ForCausalLM

try:
    from kernel_optimizer.project.discovery import phase as _phase
except Exception:  # pragma: no cover - annotation is optional
    @contextmanager
    def _phase(name):
        yield


_ROOT = Path(__file__).resolve().parent
_SOURCE_WORKLOADS = _ROOT / "inputs" / "source" / "workloads.json"
_FALLBACK_PROMPT = (
    "Explain how a computer executes a neural network and why memory access, "
    "arithmetic and communication affect performance. Use concrete examples."
)
_SEED = 20261020


def _configure(device):
    torch.set_num_threads(4)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    torch.manual_seed(_SEED)
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision("highest")


def _prompt_text(request):
    candidates = [_SOURCE_WORKLOADS]
    source_root = request.get("source_root")
    if source_root:
        candidates.append(Path(source_root) / "workloads.json")
    for path in candidates:
        try:
            return json.loads(path.read_text())["qwen_inference"]["prompt"]
        except Exception:
            continue
    return _FALLBACK_PROMPT


def _build_inputs(tokenizer, prompt, prompt_tokens, batch, device):
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    base = tokenizer.encode(text, add_special_tokens=False)
    ids = (base * (prompt_tokens // len(base) + 1))[:prompt_tokens]
    return torch.tensor([ids] * batch, device=device)


@torch.inference_mode()
def _infer(model, inputs, new_tokens, *, capture=False, teacher=None):
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
        for i in range(new_tokens - 1):
            current = teacher[:, i : i + 1] if teacher is not None else token
            output = model(input_ids=current, past_key_values=cache, use_cache=True, logits_to_keep=1)
            cache = output.past_key_values
            token = output.logits[:, -1].argmax(dim=-1, keepdim=True)
            generated.append(token)
            if capture:
                logits.append(output.logits[:, -1])
        tokens = torch.cat(generated, dim=1)
    torch.cuda.synchronize()
    end = time.perf_counter_ns()
    batch = inputs.shape[0]
    metrics = {
        "ttft_ms": (first - start) / 1e6,
        "decode_ms": (end - first) / 1e6,
        "end_to_end_ms": (end - start) / 1e6,
        "decode_tokens_per_second": batch * (new_tokens - 1) * 1e9 / (end - first),
        "generated_tokens_per_second": batch * new_tokens * 1e9 / (end - start),
    }
    return tokens, (torch.stack(logits, dim=1) if capture else None), metrics


def prepare(context, request):
    device = context["device"]
    _configure(device)
    assets = request["assets"]["assets"] if isinstance(request["assets"], dict) else request["assets"]
    workload = request["workload"]
    tokenizer = AutoTokenizer.from_pretrained(assets, local_files_only=True)
    model = (
        Qwen3ForCausalLM.from_pretrained(
            assets, dtype=torch.bfloat16, attn_implementation="sdpa", local_files_only=True
        )
        .to(device)
        .eval()
    )
    inputs = _build_inputs(
        tokenizer,
        _prompt_text(request),
        int(workload["prompt_tokens"]),
        int(workload["batch"]),
        device,
    )
    return {
        "model": model,
        "tokenizer": tokenizer,
        "inputs": inputs,
        "device": device,
        "workload": workload,
        "quality": request["quality"],
        "mode": "measure",
        "parameter_count": sum(p.numel() for p in model.parameters()),
    }


def reset(state):
    state.pop("capture", None)
    torch.cuda.synchronize()


def set_mode(state, mode):
    state["mode"] = mode


def run(state):
    capture = state.get("mode") == "quality"
    tokens, logits, metrics = _infer(
        state["model"], state["inputs"], int(state["workload"]["new_tokens"]), capture=capture
    )
    if capture:
        state["capture"] = (tokens, logits)
    return {"tokens": tokens, "logits": logits, "metrics": metrics}


def observe(state, result):
    quality = state.get("quality") or {}
    atol = float(quality.get("logits_atol", quality.get("atol", 0.02)))
    rtol = float(quality.get("logits_rtol", quality.get("rtol", 0.02)))
    model, inputs = state["model"], state["inputs"]
    new_tokens = int(state["workload"]["new_tokens"])
    tokens = result["tokens"]
    logits = result.get("logits")
    if logits is None:
        tokens, logits, _ = _infer(model, inputs, new_tokens, capture=True)
    actual_tokens, actual_logits, _ = _infer(model, inputs, new_tokens, capture=True, teacher=tokens)
    exact_greedy_tokens = bool(torch.equal(actual_tokens, tokens))
    expected = logits.float()
    actual = actual_logits.float()
    logits_close = bool(torch.allclose(actual, expected, atol=atol, rtol=rtol))
    logits_max_abs_error = (actual - expected).abs().max().item()
    corrupted_output_rejected = not torch.allclose(expected + 1.0, expected, atol=atol, rtol=rtol)
    if not (exact_greedy_tokens and logits_close and corrupted_output_rejected):
        raise AssertionError(
            {
                "exact_greedy_tokens": exact_greedy_tokens,
                "logits_close": logits_close,
                "corrupted_output_rejected": corrupted_output_rejected,
                "logits_max_abs_error": logits_max_abs_error,
                "atol": atol,
                "rtol": rtol,
            }
        )
    return {
        "tokens": tokens.detach().cpu(),
        "logits": logits.detach().cpu(),
        "exact_greedy_tokens": int(exact_greedy_tokens),
        "logits_close": int(logits_close),
        "logits_max_abs_error": float(logits_max_abs_error),
        "corrupted_output_rejected": int(corrupted_output_rejected),
    }


def metrics(state, result, elapsed_ms):
    internal = (result or {}).get("metrics") or {}
    workload = state["workload"]
    batch = int(workload["batch"])
    new_tokens = int(workload["new_tokens"])
    out = {}
    for key in (
        "ttft_ms",
        "decode_ms",
        "end_to_end_ms",
        "decode_tokens_per_second",
        "generated_tokens_per_second",
    ):
        if key in internal:
            out[key] = float(internal[key])
    out["wall_elapsed_ms"] = float(elapsed_ms)
    out["wall_generated_tokens_per_second"] = batch * new_tokens * 1000.0 / float(elapsed_ms)
    out["batch"] = float(batch)
    out["prompt_tokens"] = float(workload["prompt_tokens"])
    out["new_tokens"] = float(new_tokens)
    try:
        out["peak_allocated_bytes"] = float(torch.cuda.max_memory_allocated())
    except Exception:
        pass
    if state.get("parameter_count"):
        out["parameter_count"] = float(state["parameter_count"])
    return out


def models(state):
    return {"model": state["model"]}
