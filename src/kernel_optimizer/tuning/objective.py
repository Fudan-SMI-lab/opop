"""Native objective comparison, independent of progress thresholds."""

import math
from typing import ClassVar, Literal, assert_never

from pydantic import BaseModel, ConfigDict

from kernel_optimizer.models.core import BestRecord, TrialRecord


class Objective(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)

    direction: Literal["minimize", "maximize"]
    label: str = "J"
    unit: str | None = None

    def gain(self, previous: float, current: float) -> float:
        match self.direction:
            case "minimize":
                return previous - current
            case "maximize":
                return current - previous
            case unreachable:
                assert_never(unreachable)

    def rank(self, value: float) -> float:
        return -self.gain(0.0, value)


def objective_value(record: TrialRecord | BestRecord, objective: Objective | None) -> float | None:
    if isinstance(record, TrialRecord) and record.status != "complete":
        return None
    if objective is not None:
        evaluation = record.task_evaluation
        value = evaluation.score if evaluation is not None and evaluation.valid else None
    else:
        match record:
            case TrialRecord():
                value = record.latency_ms.robust_ms if record.latency_ms else None
            case BestRecord():
                value = record.latency_ms
            case unreachable:
                assert_never(unreachable)
    return value if value is not None and math.isfinite(value) else None


def rank_record(record: TrialRecord | BestRecord, objective: Objective | None) -> float:
    value = objective_value(record, objective)
    if value is None:
        return math.inf
    return objective.rank(value) if objective else value
