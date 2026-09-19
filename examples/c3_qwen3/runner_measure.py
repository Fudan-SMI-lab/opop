from dataclasses import dataclass
from statistics import median
from typing import assert_never

from kernel_optimizer.evaluation.task_eval import TaskEvaluation

from .binding_runtime import ModelBinding
from .runner_backend import ModelBackend
from .runner_records import GoalSpec, Prompt, RunnerError, WaveRecord


@dataclass(frozen=True, slots=True)
class WaveSpec:
    prompts: tuple[Prompt, ...]
    output_tokens: int
    warmup: bool = False


def execute_wave(backend: ModelBackend, binding: ModelBinding, spec: WaveSpec) -> WaveRecord:
    rows = tuple(p.input_ids for p in spec.prompts)
    if not rows or len({len(row) for row in rows}) != 1 or spec.output_tokens < 1:
        raise RunnerError("wave must contain equal nonempty sequences and a positive output length")
    backend.reset()
    backend.synchronize()
    ready: list[float] = []
    lengths: list[int] = []
    tokens: list[list[int]] = [[] for _ in rows]
    initial_calls = backend.forward_calls
    detail = None
    with backend.inference():
        accepted = backend.clock()
        try:
            backend.begin(rows)
            current = rows
            for step in range(spec.output_tokens):
                binding.phase = "prefill" if step == 0 else "decode"
                logits = backend.forward(current)
                next_tokens = backend.greedy(logits)
                backend.synchronize()
                ready.append(backend.clock())
                if len(next_tokens) != len(rows):
                    raise RunnerError("generation returned an incomplete batch")
                for output, token in zip(tokens, next_tokens, strict=True):
                    output.append(token)
                length = backend.cache_length()
                lengths.append(length)
                if length != len(rows[0]) + step:
                    raise RunnerError("KV did not grow from a fresh request position")
                current = tuple((token,) for token in next_tokens)
        except (RuntimeError, ValueError, ArithmeticError, TypeError, AssertionError) as exc:
            detail = f"{type(exc).__name__}: {exc}"
        finally:
            finished = ready[-1] if ready and detail is None else backend.clock()
            backend.reset()
    return WaveRecord(prompt_ids=tuple(p.id for p in spec.prompts), input_tokens=tuple(map(len, rows)),
                      tokens=tuple(tuple(row) for row in tokens), accepted_s=accepted,
                      token_ready_s=tuple(ready), finished_s=finished, cache_lengths=tuple(lengths),
                      warmup=spec.warmup, complete=detail is None, detail=detail,
                      forward_calls=backend.forward_calls - initial_calls)


def score_waves(goal: GoalSpec, waves: tuple[WaveRecord, ...]) -> TaskEvaluation:
    scored = tuple(w for w in waves if not w.warmup)
    count = sum(len(w.tokens) for w in scored)
    outputs = sum(len(row) for w in scored for row in w.tokens)
    if (count != goal.requests or outputs != goal.requests * goal.output_tokens
            or any(not w.complete or len(w.token_ready_s) != goal.output_tokens for w in scored)
            or any(len(row) != goal.output_tokens for w in scored for row in w.tokens)
            or any(n != goal.input_tokens for w in scored for n in w.input_tokens)):
        raise RunnerError("incomplete generation or incorrect input/output counts")
    totals = [w.finished_s - w.accepted_s for w in scored]
    if any(t <= 0 for t in totals):
        raise RunnerError("nonpositive request wall duration")
    metrics = {"completed_requests": float(count), "output_tokens": float(outputs),
               "total_wall_ms": sum(totals) * 1000}
    match goal.id:
        case "ttft":
            if len(scored) != 4 or any(len(w.tokens) != 1 for w in scored):
                raise RunnerError("TTFT requires four independent requests")
            score = median((w.token_ready_s[0] - w.accepted_s) * 1000 for w in scored)
        case "single":
            if len(scored) != 2 or any(len(w.tokens) != 1 for w in scored):
                raise RunnerError("single requires two independent requests")
            decode = [w.token_ready_s[-1] - w.token_ready_s[0] for w in scored]
            if any(t <= 0 for t in decode):
                raise RunnerError("nonpositive decode duration")
            score = median(127 / t for t in decode)
            metrics.update(ttft_ms=median((w.token_ready_s[0] - w.accepted_s) * 1000 for w in scored),
                           decode_wall_ms=median(decode) * 1000)
        case "multi":
            if len(scored) != 1 or len(scored[0].tokens) != 8:
                raise RunnerError("multi requires one real batch of eight requests")
            score = 1024 / totals[0]
        case unreachable:
            assert_never(unreachable)
    return TaskEvaluation(score=score, metrics=metrics)
