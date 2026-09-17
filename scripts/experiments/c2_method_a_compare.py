"""CPU-only own-card ready normalization; terminal measurements cannot backfill it."""

from statistics import median
from typing import Literal

from scripts.experiments.c2_local_inputs import InputError, Strict
from scripts.experiments.c2_method_a import AResult
from scripts.experiments.c2_method_gates import block_values
from kernel_optimizer.models.core import TrialRecord


class Normalized(Strict):
    ratio: float
    span: tuple[float, float]


def normalize(records: list[TrialRecord], baseline: list[float]) -> Normalized | None:
    values = block_values(records)
    if len(baseline) != 3 or not values:
        return None
    return Normalized(ratio=median(values) / median(baseline),
                      span=(min(values) / max(baseline), max(values) / min(baseline)))


class ReadyComparison(Strict):
    task: str
    h_time_s: float
    g0: Normalized | None
    h: Normalized | None
    status: Literal["pass", "fail", "inconclusive"]
    source_fidelity: Literal["failed"] = "failed"
    execution_kind: Literal["bounded_exploratory_confirmation"] = "bounded_exploratory_confirmation"


def compare_ready(g0: AResult, h: AResult) -> ReadyComparison:
    if g0.arm != "G0" or h.arm != "H" or g0.task != h.task or g0.original_shared_id != h.original_shared_id:
        raise InputError("ready comparison requires matched G0/H original-parent trajectories")
    cutoff = min(g0.loop.elapsed_total_s, h.loop.elapsed_total_s)
    normalized: list[Normalized | None] = []
    complete = True
    for result in (g0, h):
        baseline = block_values(result.loop.initial_parent_finals)
        ready = [r for r in result.loop.rounds if r.elapsed_ready_s is not None and r.elapsed_ready_s <= cutoff]
        normalized.append(normalize(ready[-1].fresh_finals, baseline) if ready else
                          Normalized(ratio=1, span=(1, 1)) if baseline else None)
        complete = complete and len(result.loop.rounds) == 2 and all(
            r.status != "censored" and r.elapsed_ready_s is not None for r in result.loop.rounds)
    a, b = normalized
    status = "inconclusive"
    if complete and a is not None and b is not None:
        status = "pass" if b.ratio <= a.ratio * 0.98 and b.span[1] < a.span[0] else "fail"
    return ReadyComparison(task=g0.task, h_time_s=cutoff, g0=a, h=b, status=status)
