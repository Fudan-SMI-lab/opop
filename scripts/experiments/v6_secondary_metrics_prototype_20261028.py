"""Build an isolated generic secondary-metric prototype; never edit production sources."""
import hashlib
from pathlib import Path
import shutil


MODULE = '''"""Explicit secondary diagnostic sampling through registered metric collectors."""
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
    "Use a distinct metric name, not an adapter metric or the objective name."
)


def validate_sources(specs, profiling, objective):
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
'''


def prepare_overlay(repo, output):
    """Copy once, require exact edit anchors, and record every changed source hash."""
    source = repo/'src/kernel_optimizer'; target = output/'kernel_optimizer'
    shutil.copytree(source, target, ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    changes = []
    def edit(name, old, new):
        path = target/name; text = path.read_text()
        assert text.count(old)==1, (name,old)
        path.write_text(text.replace(old,new))
    (target/'project/metric_diagnostics.py').write_text(MODULE)
    edit('project/request.py','from kernel_optimizer.models.contract_common import Name',
         'from kernel_optimizer.models.contract_common import Name\nfrom kernel_optimizer.project.metric_diagnostics import DiagnosticMetric')
    edit('project/request.py','    required_observations: list[str] = Field(default_factory=list)',
         '    required_observations: list[str] = Field(default_factory=list)\n    diagnostic_metrics: dict[str, DiagnosticMetric] = Field(default_factory=dict)')
    edit('project/intake.py','class Description(BaseModel):',
         'from kernel_optimizer.project.metric_diagnostics import DiagnosticMetric, DESCRIPTION, validate_sources\n\n\nclass Description(BaseModel):')
    edit('project/intake.py','    required_observations: list[str] = Field(default_factory=list)',
         '    required_observations: list[str] = Field(default_factory=list)\n    diagnostic_metrics: dict[str, DiagnosticMetric] = Field(default_factory=dict, description=DESCRIPTION)')
    edit('project/intake.py','    return ProjectRequest(source_root=source,',
         '    validate_sources({k:v.model_dump() for k,v in d.diagnostic_metrics.items()}, environment["profiling"], d.objective.model_dump())\n    return ProjectRequest(source_root=source,')
    edit('project/intake.py','        communication=d.communication, required_observations=d.required_observations,',
         '        communication=d.communication, required_observations=d.required_observations, diagnostic_metrics=d.diagnostic_metrics,')
    edit('project/evaluation.py','        if request["objective"]["name"] not in metrics:',
         '        secondary = {}\n        if request.get("diagnostic_metrics"):\n            from kernel_optimizer.project.metric_diagnostics import collect\n            mode("profile")\n            secondary = collect(request, adapter, state, operation, synchronize, context, metrics, metric_samples)\n        if request["objective"]["name"] not in metrics:')
    edit('project/evaluation.py','        return {"metrics": metrics, "metric_samples": metric_samples, "diagnostics": diagnostics}',
         '        return {"metrics": metrics, "metric_samples": metric_samples, "diagnostics": diagnostics,\n                **({"diagnostic_metrics": secondary} if secondary else {})}')
    edit('project/inner_loop.py','        constraints = constraints_for_runs([run], objective, self.coordinator.prepared["evaluation"]) if valid else []',
         '        if valid:\n            for name, spec in getattr(request, "diagnostic_metrics", {}).items():\n                metrics[name] = metric(run, name, spec.rank_aggregation, spec.aggregation_order)\n        constraints = constraints_for_runs([run], objective, self.coordinator.prepared["evaluation"]) if valid else []')
    for path in target.rglob('*.py'):
        original = source/path.relative_to(target)
        raw = path.read_bytes()
        if not original.exists() or original.read_bytes()!=raw:
            compile(raw,str(path),'exec')
            changes.append({'path':path.relative_to(output).as_posix(),
                'before_sha256':hashlib.sha256(original.read_bytes()).hexdigest() if original.exists() else None,
                'after_sha256':hashlib.sha256(raw).hexdigest()})
    return changes
