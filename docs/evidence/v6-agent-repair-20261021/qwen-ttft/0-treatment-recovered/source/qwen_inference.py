"""Project input: real pretrained autoregressive inference, without predefined groups."""
from __future__ import annotations

import gc
import json
import time

import torch
from transformers import AutoTokenizer, Qwen3ForCausalLM

from common import arguments, close_check, configure, metadata, negative_check, profile_once, save, summarize


@torch.inference_mode()
def infer(model, inputs, new_tokens, *, capture=False, teacher=None):
    torch.cuda.synchronize()
    start = time.perf_counter_ns()
    output = model(input_ids=inputs, use_cache=True, logits_to_keep=1)
    token = output.logits[:, -1].argmax(dim=-1, keepdim=True)
    generated = [token]
    logits = [output.logits[:, -1]] if capture else []
    cache = output.past_key_values
    torch.cuda.synchronize()  # explicit first-token-ready boundary
    first = time.perf_counter_ns()
    for i in range(new_tokens - 1):
        current = teacher[:, i:i + 1] if teacher is not None else token
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
    metrics = {"ttft_ms": (first - start) / 1e6, "decode_ms": (end - first) / 1e6,
               "end_to_end_ms": (end - start) / 1e6,
               "decode_tokens_per_second": batch * (new_tokens - 1) * 1e9 / (end - first),
               "generated_tokens_per_second": batch * new_tokens * 1e9 / (end - start)}
    return tokens, torch.stack(logits, dim=1) if capture else None, metrics


def main():
    args = arguments(__doc__)
    configure()
    config = json.loads(args.workloads.read_text())["qwen_inference"]
    tokenizer = AutoTokenizer.from_pretrained(args.assets, local_files_only=True)
    model = Qwen3ForCausalLM.from_pretrained(args.assets, dtype=torch.bfloat16, attn_implementation="sdpa", local_files_only=True).cuda().eval()
    record = metadata(args, "qwen_inference")
    record.update({"model_assets": str(args.assets), "model_revision": args.assets.name, "parameter_count": sum(p.numel() for p in model.parameters()), "workloads": []})
    quality = config["correctness"]
    for workload in config["workloads"]:
        # Fixed token count, with EOS stopping disabled by the explicit loop.
        text = tokenizer.apply_chat_template([{"role": "user", "content": config["prompt"]}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
        base = tokenizer.encode(text, add_special_tokens=False)
        ids = (base * (workload["prompt_tokens"] // len(base) + 1))[:workload["prompt_tokens"]]
        inputs = torch.tensor([ids] * workload["batch"], device="cuda")
        tokens, reference, _ = infer(model, inputs, workload["new_tokens"], capture=True)
        actual_tokens, actual, _ = infer(model, inputs, workload["new_tokens"], capture=True, teacher=tokens)
        checks = close_check(actual, reference, atol=quality["logits_atol"], rtol=quality["logits_rtol"])
        assert torch.equal(actual_tokens, tokens)
        checks.update(negative_check(reference, atol=quality["logits_atol"], rtol=quality["logits_rtol"]))
        checks["exact_greedy_tokens"] = True
        torch.save({"input_ids": inputs.cpu(), "tokens": tokens.cpu(), "logits": reference.cpu()}, args.output / f"{workload['id']}-reference.pt")
        del reference, actual, tokens, actual_tokens
        gc.collect()
        torch.cuda.empty_cache()
        for _ in range(args.warmup):
            infer(model, inputs, workload["new_tokens"])
        samples = []
        for _ in range(args.repeats):
            torch.cuda.reset_peak_memory_stats()
            _, _, metrics = infer(model, inputs, workload["new_tokens"])
            metrics["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
            samples.append(metrics)
        row = {"workload": workload, "correctness": checks, "samples": samples, "summary": summarize(samples)}
        if args.profile:
            row["diagnostic"] = profile_once(lambda: infer(model, inputs, workload["new_tokens"]), args.output / f"{workload['id']}-trace.json")
        record["workloads"].append(row)
        save(args.output / "result.json", record)
        print(workload["id"], json.dumps(row["summary"]), flush=True)


if __name__ == "__main__":
    main()
