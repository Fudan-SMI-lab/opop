"""Adapter for the Qwen3-4B interactive greedy-generation project (TTFT objective).

Mirrors inputs/source/qwen_inference.py for the frozen 'interactive' workload
(batch=1, prompt_tokens=256, new_tokens=32): BF16 SDPA model loaded from the
request assets, chat-templated prompt replicated to 256 tokens, greedy argmax
decoding under torch.inference_mode with a KV cache, and the source's explicit
first-token-ready CUDA synchronize as the ttft_ms completion boundary.
"""
from __future__ import annotations

import contextlib
import time

import torch
from transformers import AutoTokenizer, Qwen3ForCausalLM

try:  # optional discovery annotation; never required for correctness
    from kernel_optimizer.project.discovery import phase as _discovery_phase
except Exception:  # pragma: no cover - keep the adapter importable anywhere

    @contextlib.contextmanager
    def _discovery_phase(name):  # noqa: D103
        yield


SEED = 20261020
ATTN_IMPLEMENTATION = "sdpa"
# Prompt text from inputs/source/workloads.json ("qwen_inference"."prompt"),
# which parameterizes the input construction in qwen_inference.py:55-58.
PROMPT = (
    "Explain how a computer executes a neural network and why memory access, "
    "arithmetic and communication affect performance. Use concrete examples."
)


def _configure(device) -> None:
    """Numerical policy of common.configure(), bound to the worker device."""
    torch.set_num_threads(4)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass  # already initialized by the worker process
    torch.manual_seed(SEED)
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision("highest")


def _build_inputs(tokenizer, workload, device):
    """Exact input construction of qwen_inference.py:55-58 for one workload row."""
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": PROMPT}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    base = tokenizer.encode(text, add_special_tokens=False)
    prompt_tokens = workload["prompt_tokens"]
    ids = (base * (prompt_tokens // len(base) + 1))[:prompt_tokens]
    return torch.tensor([ids] * workload["batch"], device=device)


def prepare(context, request):
    """Load model/data (untimed) and apply the source's numerical policy."""
    device = context["device"]
    _configure(device)
    workload = request["workload"]
    assets = request["assets"]["assets"]
    tokenizer = AutoTokenizer.from_pretrained(assets, local_files_only=True)
    model = (
        Qwen3ForCausalLM.from_pretrained(
            assets,
            dtype=torch.bfloat16,
            attn_implementation=ATTN_IMPLEMENTATION,
            local_files_only=True,
        )
        .to(device)
        .eval()
    )
    inputs = _build_inputs(tokenizer, workload, device)
    state = {
        "model": model,
        "tokenizer": tokenizer,
        "inputs": inputs,
        "device": device,
        "batch": workload["batch"],
        "prompt_tokens": workload["prompt_tokens"],
        "new_tokens": workload["new_tokens"],
        "seed": SEED,
        "mode": "measure",
        "last": None,
    }
    return state


def set_mode(state, mode):
    """Select the source's correctness-capture path in quality mode only."""
    state["mode"] = mode


def reset(state):
    """Restore the starting point for a repeat of the frozen workload.

    Inference is stateless here: the KV cache is created inside run() and the
    model is a frozen eval module, so only the RNG seed and stored results of
    the previous invocation are reset.
    """
    torch.manual_seed(state["seed"])
    state["last"] = None
    state["model"].eval()


def run(state):
    """One full infer() invocation (qwen_inference.py:14-41) with source timing.

    Includes the internal first-token-ready synchronize (qwen_inference.py:23)
    that defines the ttft_ms completion boundary, then all remaining greedy
    decode steps and the final synchronize. Returns tokens, optional per-step
    logits (quality mode) and the raw phase timestamps in nanoseconds.
    """
    capture = state["mode"] == "quality"
    model = state["model"]
    inputs = state["inputs"]
    new_tokens = state["new_tokens"]
    with torch.inference_mode():
        torch.cuda.synchronize()
        start = time.perf_counter_ns()
        with _discovery_phase("prefill"):
            output = model(input_ids=inputs, use_cache=True, logits_to_keep=1)
            token = output.logits[:, -1].argmax(dim=-1, keepdim=True)
            generated = [token]
            logits = [output.logits[:, -1]] if capture else []
            cache = output.past_key_values
        torch.cuda.synchronize()  # explicit first-token-ready boundary
        first = time.perf_counter_ns()
        with _discovery_phase("decode"):
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
        "timing_ns": {"start": start, "first": first, "end": end},
    }
    state["last"] = result
    return result


def observe(state, result):
    """Correctness observations: greedy tokens and, in quality mode, full logits."""
    observations = {"tokens": result["tokens"].detach()}
    logits = result.get("logits")
    if logits is not None:
        observations["logits"] = logits.detach()
    return observations


def metrics(state, result, elapsed_ms):
    """Source metric keys in native units; ttft_ms uses the source boundary."""
    timing = result["timing_ns"]
    batch = state["batch"]
    new_tokens = state["new_tokens"]
    decode_ns = timing["end"] - timing["first"]
    total_ns = timing["end"] - timing["start"]
    values = {
        "ttft_ms": (timing["first"] - timing["start"]) / 1e6,
        "decode_ms": decode_ns / 1e6,
        "end_to_end_ms": total_ns / 1e6,
        "decode_tokens_per_second": batch * (new_tokens - 1) * 1e9 / decode_ns,
        "generated_tokens_per_second": batch * new_tokens * 1e9 / total_ns,
        "wall_elapsed_ms": float(elapsed_ms),
        "batch": float(batch),
        "prompt_tokens": float(state["prompt_tokens"]),
        "new_tokens": float(new_tokens),
    }
    try:  # framework resets peak stats immediately before its timer
        values["peak_allocated_bytes"] = float(torch.cuda.max_memory_allocated())
    except RuntimeError:
        pass
    return values


def models(state):
    """Root model for automatic discovery and controlled replacement."""
    return {"model": state["model"]}
