"""Direct, in-process evaluation of trusted user Python code.

Use ``with TaskEvaluator(eval_path, callable_name) as evaluator`` and call
``evaluator.evaluate(candidate_path, params, context)``. The exported synchronous
callable receives exactly ``(Path, Mapping[str, JsonValue], Mapping[str, T])``
and returns an int/float J or TaskEvaluation. It owns candidate execution,
reference checks and scoring; the core never infers correctness from timing.
Scores remain native: min/max direction and units belong to the calling run.
"""

import importlib.util
import sys
from collections.abc import Mapping
from pathlib import Path
from types import TracebackType
from typing import ClassVar, Protocol, Self, assert_never, runtime_checkable
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator


class TaskEvaluationError(ValueError):
    """Configuration/lifecycle error; callback failures instead return invalid results."""

    def __init__(self, detail: str) -> None:
        self.detail: str = detail
        super().__init__(detail)


class TaskEvaluation(BaseModel):
    """Native finite objective plus optional task-named numeric measurements."""

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, strict=True, allow_inf_nan=False)

    score: float | None = None
    valid: bool = True
    metrics: dict[str, float] = Field(default_factory=dict)
    detail: str | None = None

    @model_validator(mode="after")
    def require_valid_score(self) -> Self:
        if self.valid and self.score is None:
            raise TaskEvaluationError("valid evaluation requires a finite score")
        return self


@runtime_checkable
class _EvaluationFunction(Protocol):
    def __call__[T](
        self, candidate_path: Path, params: Mapping[str, JsonValue], context: Mapping[str, T],
    ) -> TaskEvaluation | int | float: ...


class TaskEvaluator:
    """Load once; reuse callback state until close/context exit.

    Relative imports (including lazy imports) resolve beside eval_path under a
    private package name. No sys.path/cwd changes or project __init__ execution.
    close() removes this package and its helpers, not unrelated global imports.
    This is trusted synchronous code, not a sandbox or a worker pool.
    """

    def __init__(self, eval_path: str | Path, callable_name: str) -> None:
        self._namespace: str = "_task_eval_" + uuid4().hex
        self._function: _EvaluationFunction | None = None
        path = Path(eval_path).resolve()
        try:
            spec = importlib.util.spec_from_file_location(
                self._namespace, path, submodule_search_locations=[str(path.parent)],
            )
            if spec is None or spec.loader is None:
                raise TaskEvaluationError(f"cannot load evaluation file: {path}")
            module = importlib.util.module_from_spec(spec)
            sys.modules[self._namespace] = module
            spec.loader.exec_module(module)
            function = getattr(module, callable_name, None)
            if not isinstance(function, _EvaluationFunction):
                raise TaskEvaluationError(f"{path}: {callable_name!r} is not callable")
            self._function = function
        except Exception as exc:  # noqa: BLE001  # noqa: BROAD_EXCEPT_OK - user module boundary
            raise TaskEvaluationError(f"{path}: {type(exc).__name__}: {exc}") from exc
        finally:
            if self._function is None:
                self.close()

    def __enter__(self) -> Self:
        if self._function is None:
            raise TaskEvaluationError("evaluator is closed")
        return self

    def __exit__(
        self, exc_type: type[BaseException] | None,
        exc_value: BaseException | None, traceback: TracebackType | None,
    ) -> None:
        self.close()

    def close(self) -> None:
        """Release callback state and all modules in this evaluator's namespace."""
        self._function = None
        for name in tuple(sys.modules):
            if name == self._namespace or name.startswith(self._namespace + "."):
                del sys.modules[name]

    def evaluate[T](
        self, candidate_path: str | Path, params: Mapping[str, JsonValue], context: Mapping[str, T],
    ) -> TaskEvaluation:
        """Execute the callback; failure produces valid=False, score=None, detail.

        Context is passed through untouched, including task-specific Python
        objects. Parameters must be JSON-compatible. KeyboardInterrupt and
        SystemExit propagate; callers should use the context manager for cleanup.
        """
        if self._function is None:
            raise TaskEvaluationError("evaluator is closed")
        try:
            result = self._function(Path(candidate_path), params, context)
            match result:
                case TaskEvaluation():
                    return result
                case int() | float():
                    return TaskEvaluation(score=result)
                case _:
                    assert_never(result)
        except Exception as exc:  # noqa: BLE001  # noqa: BROAD_EXCEPT_OK - callback boundary
            return TaskEvaluation(valid=False, detail=f"{type(exc).__name__}: {exc}"[:1000])
