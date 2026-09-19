import math
import operator
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .runner_records import RunnerError


@dataclass(frozen=True, slots=True)
class LogitComparison:
    candidate_nll: float
    reference_nll: float
    difference2: float
    norm2: float


def _finite_vector(logits: Sequence[float], target: int) -> NDArray[np.float64]:
    values = np.asarray(logits, dtype=np.float64)
    if values.ndim != 1 or not values.size or not 0 <= target < values.size or not np.isfinite(values).all():
        raise RunnerError("invalid logits/target in teacher-forced quality")
    return values


def _nll(values: NDArray[np.float64], target: int) -> float:
    maximum = float(np.max(values))
    # Preserve the original double-precision, max-shifted formula; only reductions move into NumPy.
    return maximum + math.log(float(np.sum(np.exp(values - maximum), dtype=np.float64))) - float(values[operator.index(target)])


def nll(logits: Sequence[float], target: int) -> float:
    return _nll(_finite_vector(logits, target), target)


def compare_logits(candidate: Sequence[float], reference: Sequence[float], target: int) -> LogitComparison:
    if len(candidate) != len(reference):
        raise RunnerError("logit shape mismatch")
    candidate_values = _finite_vector(candidate, target)
    reference_values = _finite_vector(reference, target)
    difference = candidate_values - reference_values
    return LogitComparison(
        _nll(candidate_values, target), _nll(reference_values, target),
        float(np.sum(np.square(difference), dtype=np.float64)),
        float(np.sum(np.square(reference_values), dtype=np.float64)),
    )
