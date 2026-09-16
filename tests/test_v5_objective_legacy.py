"""Characterize the unchanged latency estimator before adapting it."""

import pytest

from kernel_optimizer.models.core import LatencyStats


@pytest.mark.parametrize(("median", "expected"), [(2.0, 2.0), (None, 7.0), (0.0, 7.0), (-1.0, 7.0)])
def test_legacy_robust_positive_median_else_mean(median: float | None, expected: float) -> None:
    # Given
    latency = LatencyStats(mean=7, std=0, min=1, max=9, n_samples=3, median=median)
    # When
    value = latency.robust_ms
    # Then
    assert value == expected


def test_legacy_negative_sentinel_remains_negative() -> None:
    # Given
    latency = LatencyStats(mean=-1, std=0, min=-1, max=-1, n_samples=0)
    # When
    value = latency.robust_ms
    # Then
    assert value == -1.0


@pytest.mark.parametrize(("mean", "median", "expected"), [
    (7.0, 2.0, 2.0), (7.0, None, 7.0), (-1.0, None, None), (7.0, 0.0, 7.0),
    (7.0, -1.0, 7.0), (0.0, None, None), (float("inf"), None, None), (float("nan"), None, None),
])
def test_adapter_preserves_legacy_estimator(mean: float, median: float | None, expected: float | None) -> None:
    from kernel_optimizer.evaluation.legacy_objective import LegacyContext, from_legacy_latency

    # Given
    latency = LatencyStats(mean=mean, std=0, min=mean, max=mean, n_samples=3, median=median)
    # When
    result = from_legacy_latency(latency, LegacyContext(protocol_id="legacy:task-17:policy-1"))
    # Then
    assert result.native_j == expected
    assert result.quality == "unknown"
    assert result.protocol_id == "legacy:task-17:policy-1"
