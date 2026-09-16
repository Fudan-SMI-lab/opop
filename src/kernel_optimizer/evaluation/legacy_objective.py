"""Explicit legacy latency boundary; negative sentinels never become signed native J."""

from dataclasses import dataclass
from math import isfinite

from kernel_optimizer.models.core import LatencyStats
from kernel_optimizer.models.evaluation_bundle import EvalResult, QualityStatus
from kernel_optimizer.models.objective import MetricValue


@dataclass(frozen=True, slots=True)
class LegacyContext:
    """Caller binds task/policy identity and an independently established quality verdict."""

    protocol_id: str
    quality: QualityStatus = "unknown"
    case_id: str = "legacy"


def from_legacy_latency(latency: LatencyStats, context: LegacyContext) -> EvalResult:
    """Preserve robust_ms; only finite positive latency is a usable legacy measurement."""
    value = latency.robust_ms
    if not isfinite(value) or value <= 0:
        return EvalResult(measurement_validity="invalid", quality=context.quality, feasibility="feasible",
                          objective_unit="ms", direction="minimize", protocol_id=context.protocol_id,
                          reasons=("legacy_latency_unavailable",))
    metric = MetricValue(name="latency", value=value, unit="ms", case_id=context.case_id)
    return EvalResult(measurement_validity="valid", quality=context.quality, feasibility="feasible",
                      case_metrics=(metric,), native_j=value, objective_unit="ms",
                      direction="minimize", protocol_id=context.protocol_id)
