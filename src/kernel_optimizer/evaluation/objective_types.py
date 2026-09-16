"""Pure scoring dependencies and comparison outcomes; no module loading."""

from dataclasses import dataclass
from typing import Literal, Protocol

from kernel_optimizer.models.objective import MetricValue, ObjectiveSpec
from kernel_optimizer.models.task_bundle import TaskDefinition


class BundleAggregation(Protocol):
    """Consume validated required values in metric declaration/workload case order."""

    def __call__(self, objective: ObjectiveSpec, values: tuple[MetricValue, ...], /) -> float: ...


@dataclass(frozen=True, slots=True)
class ScoringContext:
    """Task 6 supplies the resolved frozen callable and scientific protocol identity."""

    task: TaskDefinition
    protocol_id: str
    aggregator: BundleAggregation | None = None


@dataclass(frozen=True, slots=True)
class Incompatibility:
    reasons: tuple[str, ...]
    kind: Literal["incompatible"] = "incompatible"


@dataclass(frozen=True, slots=True)
class Comparison:
    winner: Literal["candidate", "incumbent"]
    improvement: float
    kind: Literal["comparable"] = "comparable"


type ComparisonResult = Comparison | Incompatibility
type ImprovementResult = float | Incompatibility
