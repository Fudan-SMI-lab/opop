from collections.abc import Mapping
from pathlib import Path
from runpy import run_path
from typing import Protocol, runtime_checkable

from pydantic import JsonValue, TypeAdapter

from kernel_optimizer.evaluation.task_eval import TaskEvaluation, TaskEvaluationError


@runtime_checkable
class SortPayload(Protocol):
    def __call__(self, data: list[int], width: int) -> tuple[list[int], bytes]: ...


def assess(path: Path, params: Mapping[str, JsonValue], context: Mapping[str, JsonValue]) -> TaskEvaluation:
    data = TypeAdapter(list[int]).validate_python(context["data"])
    width = TypeAdapter(int).validate_python(params["width"])
    run = run_path(str(path))["run"]
    if not isinstance(run, SortPayload):
        raise TaskEvaluationError("candidate must export run(data, width)")
    output, payload = run(data, width)
    if output != sorted(data):
        return TaskEvaluation(score=-1000.0, valid=False, detail="incorrect sorted output")
    return TaskEvaluation(score=float(len(payload) - 2))
