"""Exact-case measurement selection and overflow-safe scalar reductions."""

from dataclasses import dataclass
from fractions import Fraction
from math import fsum, isfinite
from typing import Literal, assert_never

from kernel_optimizer.models.objective import MetricValue


@dataclass(frozen=True, slots=True)
class Selection:
    values: tuple[MetricValue, ...]
    missing: tuple[str, ...]
    errors: tuple[str, ...]


def select_values(records: tuple[MetricValue, ...], cases: tuple[str, ...], unit: str) -> Selection:
    """Records are already name/entity filtered; identical repeats are idempotent."""
    values: list[MetricValue] = []
    missing: list[str] = []
    errors: list[str] = []
    for case in cases:
        matching = tuple(record for record in records if record.case_id == case)
        if not matching:
            missing.append(f"missing:{case}")
        elif any(record.unit != unit for record in matching):
            errors.append(f"unit_mismatch:{case}")
        elif len({record.value for record in matching}) != 1:
            errors.append(f"conflicting_measurements:{case}")
        else:
            values.append(matching[0])
    return Selection(tuple(values), tuple(missing), tuple(errors))


def reduce_values(values: tuple[float, ...], kind: Literal["sum", "weighted_mean", "max"],
                  weights: tuple[float, ...] = ()) -> float | None:
    """Return None on unrepresentable output; rational fallback avoids intermediate overflow."""
    try:
        match kind:
            case "sum":
                try:
                    result = fsum(values)
                except OverflowError:
                    result = float(sum((Fraction(value) for value in values), Fraction()))
            case "weighted_mean":
                total = sum((Fraction(weight) for weight in weights), Fraction())
                numerator = sum((Fraction(value) * Fraction(weight)
                                 for value, weight in zip(values, weights, strict=True)), Fraction())
                result = float(numerator / total)
            case "max":
                result = max(values)
            case _:
                assert_never(kind)
    except OverflowError:
        return None
    return result if isfinite(result) else None
