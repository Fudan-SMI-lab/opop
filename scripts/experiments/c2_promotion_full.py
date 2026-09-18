"""Pure full-block selection; callers own measurement order and deadline handling."""

from dataclasses import dataclass
from math import isfinite
from statistics import median
from typing import Literal

from kernel_optimizer.models.core import TrialRecord


@dataclass(frozen=True, slots=True)
class ConfirmationPair:
    """Arm-labelled records regardless of execution order; None means missing."""

    parent: TrialRecord | None
    child: TrialRecord | None


@dataclass(frozen=True, slots=True)
class PromotionEvidence:
    """Original three-block screens, native quick latencies, and ordered pairs."""

    parent_screen: tuple[TrialRecord | None, ...]
    child_screen: tuple[TrialRecord | None, ...]
    parent_quick_ms: float | None = None
    child_quick_ms: float | None = None
    confirmations: tuple[ConfirmationPair, ...] = ()


@dataclass(frozen=True, slots=True)
class PromotionDecision:
    """A terminal result or a 1-based pair request (1: P,C; 2: C,P)."""

    outcome: Literal["request_pair", "parent", "child", "failed"]
    next_pair_index: Literal[1, 2] | None = None


def decide_promotion(evidence: PromotionEvidence) -> PromotionDecision:
    """Apply approved section 5 without measurement, I/O, or quick re-ranking."""
    pairs = evidence.confirmations
    if len(evidence.parent_screen) != 3 or len(evidence.child_screen) != 3 or len(pairs) > 2:
        return PromotionDecision("failed")
    parent = _block_medians(evidence.parent_screen + tuple(pair.parent for pair in pairs))
    child = _block_medians(evidence.child_screen + tuple(pair.child for pair in pairs))
    if parent is None or child is None:
        return PromotionDecision("failed")
    parent_median, child_median = median(parent), median(child)
    if not pairs:
        overlap = max(min(parent), min(child)) <= min(max(parent), max(child))
        parent_quick, child_quick = evidence.parent_quick_ms, evidence.child_quick_ms
        conflict = False
        if (parent_quick is not None and child_quick is not None
                and isfinite(parent_quick) and isfinite(child_quick)
                and parent_quick > 0 and child_quick > 0):
            conflict = ((parent_quick < child_quick and parent_median > child_median)
                        or (parent_quick > child_quick and parent_median < child_median))
        if overlap or conflict:
            return PromotionDecision("request_pair", 1)
        return PromotionDecision("child" if child_median < parent_median else "parent")
    if len(pairs) == 1:
        if parent[3] <= child[3] and parent_median <= child_median:
            return PromotionDecision("parent")
        return PromotionDecision("request_pair", 2)
    promotes = child[3] < parent[3] and child[4] < parent[4] and child_median < parent_median
    return PromotionDecision("child" if promotes else "parent")


def _block_medians(records: tuple[TrialRecord | None, ...]) -> tuple[float, ...] | None:
    values: list[float] = []
    for record in records:
        if record is None or record.status != "complete" or record.latency_ms is None:
            return None
        latency = record.latency_ms
        value = latency.median
        if latency.n_samples != 100 or value is None or not isfinite(value) or value <= 0:
            return None
        values.append(value)
    return tuple(values)
