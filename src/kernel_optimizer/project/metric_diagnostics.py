"""Explicit secondary diagnostic sampling through registered metric collectors."""
import hashlib
import math
from pathlib import Path
import statistics
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class DiagnosticMetric(BaseModel):
    model_config = ConfigDict(extra="forbid")
    source: dict[str, Any] = Field(description="Installed metric collector and field, with optional provider settings")
    unit: str
    rank_aggregation: Literal["max", "min", "sum", "mean"] = "max"
    aggregation_order: Literal["samples_then_ranks", "ranks_then_samples"] = "samples_then_ranks"


DESCRIPTION = (
    "Optional named secondary metrics sampled by registered collectors after all clean measurements, "
    "without changing the primary objective or adding acceptance constraints. Use deployment.metric_collectors "
    "and source/environment evidence to select applicable collector/field pairs and source-grounded rank/sample "
    "aggregation. required_observations only checks discovery capabilities; it does not request evaluator "
    "metric samples. Leave empty when no secondary metric is needed or applicable. An explicitly requested "
    "metric must be observed; unavailable/failed collection is an error, never a fabricated zero. "
    "Use a distinct metric name, not an adapter metric or the objective name. "
    "Discovery requirements, output aliases, and sampling configuration are different interfaces. "
    "required_observations uses exact field names from the discovery/capability manifest, not "
    "collector-qualified paths or evaluator metric aliases; a numeric collector field required in "
    "discovery uses its field name without inventing a collector prefix. source is forwarded unchanged "
    "to the provider and is not framework configuration. The evaluator obtains the number of profiled "
    "repetitions from deployment.profiling.metric_repeats (default 3), not source.profiling. "
    "Preserve the source repetition count; if it differs from available sampling configuration, report "
    "the missing setup rather than declaring an ignored nested override."
)


def validate_sources(specs, profiling, objective):
    if not specs:
        return
    from kernel_optimizer.project.collectors import configured
    registry = configured(profiling)
    for name, raw in specs.items():
        if name == objective["name"]:
            raise ValueError("secondary metric collides with objective: " + name)
        spec = DiagnosticMetric.model_validate(raw)
        provider = registry.resolve(spec.source.get("collector"))
        if not callable(getattr(provider, "metric", None)):
            raise ValueError("secondary source must select a metric-capable collector")
        fields = getattr(provider, "metric_fields", None)
        if fields is not None and spec.source.get("field") not in fields:
            raise ValueError("secondary collector field unavailable: " + str(spec.source.get("field")))


def collect(request, adapter, state, operation, synchronize, context, metrics, metric_samples):
    from kernel_optimizer.project.collectors import configured
    specs = request.get("diagnostic_metrics", {})
    validate_sources(specs, request.get("profiling", {}), request["objective"])
    collisions = set(specs) & (set(metrics) | set(metric_samples))
    if collisions:
        raise ValueError("secondary metric collides with existing metric: " + str(sorted(collisions)))
    registry = configured(request.get("profiling", {})); observations = {}
    repeats = int(request.get("profiling", {}).get("metric_repeats", 3))
    for name in sorted(specs):
        spec = DiagnosticMetric.model_validate(specs[name])
        output = Path(context["output"])/"secondary-metrics"/hashlib.sha256(name.encode()).hexdigest()[:16]
        output.mkdir(parents=True, exist_ok=True)
        sampled = registry.resolve(spec.source["collector"]).metric(
            adapter, state, operation, synchronize, {**context, "output":str(output)}, spec.source, repeats)
        values = sampled.get("values", [])
        if len(values) != repeats or not all(isinstance(v, (int,float)) and not isinstance(v,bool) and math.isfinite(v) for v in values):
            raise ValueError("secondary metric has missing/nonfinite samples: " + name)
        metric_samples[name] = values
        metrics[name] = statistics.median(values)
        observations[name] = {"status":"observed", "specification":spec.model_dump(), **sampled}
    return observations
