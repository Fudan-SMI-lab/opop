"""Generate task-owned Python evaluators through the existing agent retry loop."""

from collections.abc import Callable
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, Final, Literal, override

from pydantic import BaseModel, ConfigDict, Field

from kernel_optimizer.agents.base import AgentModule, AgentOutcome
from kernel_optimizer.agents.sandbox import Sandbox
from kernel_optimizer.evaluation.task_eval import TaskEvaluation, TaskEvaluationError

_PROBE: Final[ContextVar[Callable[[Path, str], TaskEvaluation] | None]] = ContextVar(
    "eval_builder_probe", default=None,
)


@dataclass(frozen=True, slots=True)
class EvalBuilderInputs:
    """Readable task inputs; probe executes on the caller's chosen target."""

    project_root: Path
    source_paths: tuple[Path, ...]
    goal: str
    task_context: str = ""
    probe: Callable[[Path, str], TaskEvaluation] | None = None


class EvalBuilderResult(BaseModel):
    """eval_file is relative to AgentOutcome.sandbox.root; J stays native."""

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid")

    eval_file: str = Field(min_length=1)
    callable_name: str = Field(min_length=1)
    direction: Literal["minimize", "maximize"]
    unit: str | None = None
    label: str | None = None


class EvalBuilderAgent(AgentModule[EvalBuilderInputs, EvalBuilderResult]):
    name: str = "eval_builder"
    output_model: type[BaseModel] = EvalBuilderResult

    @override
    def invoke(self, inputs: EvalBuilderInputs) -> AgentOutcome[EvalBuilderResult]:
        """Scope the target probe to this invocation, including nested/concurrent calls."""
        token = _PROBE.set(inputs.probe)
        try:
            return super().invoke(inputs)
        finally:
            _PROBE.reset(token)

    @override
    def seed_sandbox(self, inputs: EvalBuilderInputs, sb: Sandbox) -> None:
        root = inputs.project_root.resolve()
        _ = sb.write_input("task/goal.md", inputs.goal)
        _ = sb.write_input("task/context.md", inputs.task_context)
        sources: list[str] = []
        for source in inputs.source_paths:
            path = (root / source).resolve()
            relative = path.relative_to(root)
            destination = f"project/{relative.as_posix()}"
            _ = sb.write_input(destination, path.read_text(encoding="utf-8"))
            sources.append(f"- {destination} (original: {path})")
        _ = sb.write_input("task/project.md", f"Project root: {root}\n\n" + "\n".join(sources))

    @override
    def render_prompt(self, inputs: EvalBuilderInputs, sb: Sandbox) -> str:
        return f"""Write a runnable Python evaluation program for this project and goal:
{inputs.goal}

Read task/goal.md, task/context.md, task/project.md and the listed project sources
before deciding the evaluation logic. Derive the workload, independent reference,
quality checks and native objective from this actual task, not a task registry.
Write the Python file to this workspace; declare its relative eval_file path and
callable_name in your structured response. You may choose any callable name.

The synchronous callable receives (candidate_path, params, context): candidate_path
is a pathlib.Path, params is a mapping of tunable values, and context is a mapping
that may contain Python objects, input data or project paths as described in
task/context.md. It must execute that candidate with those parameters and inputs,
check its outputs against an independent reference computed from the inputs, and
only then assign a measured native J. Never load the candidate as its own reference
or infer correctness from a performance claim. Do not select a built-in evaluator.

Return a finite int/float J, or import TaskEvaluation from
kernel_optimizer.evaluation.task_eval and return TaskEvaluation(score=J,
metrics={{"your_measurement_name": numeric_value}}). Incorrect output must return
TaskEvaluation(valid=False, detail="reason") with no score. Measure the objective
the goal actually requests; do not fabricate timing or resource measurements.
Declare direction as minimize or maximize, plus optional unit and label strings.
Keep J in native units: never negate it to implement direction.

The goal, input workload, reference semantics and quality requirements are fixed
run settings across every repair. Repair code, not the task: do not shrink inputs,
relax tolerances, replace the reference or change the objective to make a failing
candidate valid. Leave task/ and project/ inputs unchanged. Keep any generated
helpers beside the evaluator and use relative imports; do not depend on cwd.
Syntax checking is available on the host. Target execution occurs only through a
supplied probe; do not import or run GPU code on a host lacking its dependencies.
Probe success is a code smoke check, not evidence of model or GPU performance.
"""

    @override
    def check_output(self, output: EvalBuilderResult, sb: Sandbox) -> str | None:
        try:
            source = sb.read_output(output.eval_file)
            _ = compile(source, output.eval_file, "exec")
        except (OSError, ValueError, SyntaxError) as exc:
            return f"Evaluator file check failed: {type(exc).__name__}: {exc}"
        probe = _PROBE.get()
        if probe is not None:
            try:
                result = probe((sb.root / output.eval_file).resolve(), output.callable_name)
            except TaskEvaluationError as exc:
                return f"Evaluator probe could not load the program: {exc}"
            if not result.valid:
                return f"Evaluator probe failed: {result.detail or 'reference was rejected'}"
        return None
