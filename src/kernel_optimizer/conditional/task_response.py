"""Fresh one-axis contrasts. Diagnostic probes never enter incumbent selection."""

from __future__ import annotations

from math import isfinite
from typing import ClassVar, Literal, assert_never

from pydantic import BaseModel, ConfigDict, Field

from kernel_optimizer.evaluation.task_eval import TaskEvaluation
from kernel_optimizer.models.core import ParamSet
from kernel_optimizer.tuning.objective import Objective


class TaskResponse(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)
    candidate_id: str
    axis: str
    a_params: ParamSet
    b_params: ParamSet | None = None
    a: TaskEvaluation | None = None
    b: TaskEvaluation | None = None
    delta_j: float | None = None
    gain: float | None = None
    parameter_slope: float | None = None
    resource_deltas: dict[str, float] = Field(default_factory=dict)
    resource_slopes: dict[str, float] = Field(default_factory=dict)
    unknown_resources: list[str] = Field(default_factory=list)
    resource_status: Literal["observed", "unknown"] = "unknown"
    reason: str | None = None


def complete_response(response: TaskResponse, objective: Objective, numeric: bool) -> TaskResponse:
    a, b = response.a, response.b
    if a is None or b is None or response.b_params is None:
        return response.model_copy(update={"reason": "wall_budget"})
    if not a.valid or not b.valid or a.score is None or b.score is None:
        return response.model_copy(update={"reason": "invalid_endpoint"})
    delta = b.score - a.score
    if not isfinite(delta):
        return response.model_copy(update={"reason": "nonfinite_difference"})
    parameter_slope = None
    left, right = response.a_params.values[response.axis], response.b_params.values[response.axis]
    if numeric:
        match left, right:
            case (int() | float()), (int() | float()):
                distance = right - left
                if distance and isfinite(distance) and isfinite(delta / distance):
                    parameter_slope = delta / distance
            case (str(), _) | (_, str()):
                # Legacy domains permit mismatched kinds; such axes remain contrasts.
                parameter_slope = None
            case unreachable:
                assert_never(unreachable)
    deltas: dict[str, float] = {}
    slopes: dict[str, float] = {}
    unknown: list[str] = []
    for name in response.unknown_resources:
        if name not in a.metrics or name not in b.metrics:
            unknown.append(name)
            continue
        change = b.metrics[name] - a.metrics[name]
        if not isfinite(change):
            unknown.append(name)
            continue
        deltas[name] = change
        if change and isfinite(delta / change):
            slopes[name] = delta / change
    return response.model_copy(update={
        "delta_j": delta, "gain": objective.gain(a.score, b.score),
        "parameter_slope": parameter_slope, "resource_deltas": deltas,
        "resource_slopes": slopes, "unknown_resources": unknown,
        "resource_status": "observed" if deltas else "unknown",
    })
