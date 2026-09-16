"""Generated evaluator code crosses the real agent and CPU evaluation boundaries."""

from collections.abc import Iterator
from contextlib import closing
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Final, Literal, override

import pytest
from pydantic import JsonValue

from kernel_optimizer.agents.eval_builder import EvalBuilderAgent, EvalBuilderInputs
from kernel_optimizer.agents.runtime import AgentCallError, OpencodeClient, PromptResult
from kernel_optimizer.agents.sandbox import SandboxFactory
from kernel_optimizer.config import AgentModuleConfig
from kernel_optimizer.evaluation.task_eval import TaskEvaluation, TaskEvaluator
from kernel_optimizer.store.run_store import RunStore

GENERATED: Final = '''
from runpy import run_path
from kernel_optimizer.evaluation.task_eval import TaskEvaluation

def quasar_work(candidate_path, params, context):
    transform = run_path(str(candidate_path))["transform"]
    output, work = transform(context["values"], params["repeat"])
    expected = [x * x + 7 for x in context["values"]]
    if output != expected:
        return TaskEvaluation(valid=False, detail="reference mismatch")
    return TaskEvaluation(score=work, metrics={"quasar_steps": work})

def orchid_yield(candidate_path, params, context):
    result = quasar_work(candidate_path, params, context)
    if not result.valid:
        return result
    return len(context["values"]) / result.score
'''


class WritingClient(OpencodeClient):
    """Only the provider is fake; each turn writes code into its real workspace."""

    def __init__(self) -> None:
        super().__init__("http://127.0.0.1:0")
        self.versions: list[str | None] = [GENERATED]
        self.calls: int = 0
        self.callable_name: str = "quasar_work"
        self.direction: Literal["minimize", "maximize"] = "minimize"
        self.unit: str = "quasar_steps"
        self.prompts: list[str] = []
        self.goals: list[str] = []

    @override
    def create_session(self, directory: Path, title: str) -> str:
        return "local-session"

    # The full signature is required by the existing provider interface.
    @override
    def prompt(
        self, session_id: str, text: str, *, model: str, agent: str = "build",
        schema: dict[str, JsonValue] | None = None, directory: Path | None = None,
        system: str | None = None,
    ) -> PromptResult:
        assert directory is not None
        self.goals.append((directory / "task/goal.md").read_text(encoding="utf-8"))
        self.prompts.append(text)
        source = self.versions[min(self.calls, len(self.versions) - 1)]
        self.calls += 1
        if source is not None:
            _ = (directory / "generated.py").write_text(source, encoding="utf-8")
        return PromptResult(text="", session_id=session_id, structured={
            "eval_file": "generated.py", "callable_name": self.callable_name,
            "direction": self.direction, "unit": self.unit, "label": "quasar objective",
        })


@dataclass(frozen=True, slots=True)
class Setup:
    builder: EvalBuilderAgent
    client: WritingClient
    inputs: EvalBuilderInputs


@pytest.fixture
def setup(tmp_path: Path) -> Iterator[Setup]:
    project = tmp_path / "project"
    project.mkdir()
    for name, repetitions in (("reference", 1), ("slower", 3)):
        _ = (project / f"{name}.py").write_text(
            f'''def transform(values, repeat):
    output, work = [], 0
    for x in values:
        for _ in range(repeat * {repetitions}):
            y = x ** 2 + 7
            work += 1
        output.append(y)
    return output, work
''', encoding="utf-8",
        )
    _ = (project / "wrong.py").write_text(
        "def transform(values, repeat):\n    return values, 0\n", encoding="utf-8",
    )

    def probe(path: Path, name: str) -> TaskEvaluation:
        with TaskEvaluator(path, name) as evaluator:
            return evaluator.evaluate(project / "reference.py", {"repeat": 1}, {"values": [2, -3]})

    inputs = EvalBuilderInputs(
        project_root=project, source_paths=(Path("reference.py"),),
        goal="Preserve x squared plus seven; minimize executed arithmetic steps.",
        task_context="transform(values, repeat) returns output and measured work; values=[2,-3]",
        probe=probe,
    )
    with closing(WritingClient()) as client:
        yield Setup(EvalBuilderAgent(
            client, SandboxFactory(tmp_path / "sandboxes"), RunStore(tmp_path),
            AgentModuleConfig(max_retries=1),
        ), client, inputs)


@pytest.mark.parametrize("maximize", [False, True])
def test_native_objective_when_generated_code_runs(setup: Setup, maximize: bool) -> None:
    # Given: an unseen task and independently computed expected outputs.
    inputs = setup.inputs
    if maximize:
        setup.client.callable_name = "orchid_yield"
        setup.client.direction = "maximize"
        setup.client.unit = "items/quasar_step"
        inputs = replace(inputs, goal="Preserve outputs; maximize items per arithmetic step.")
    # When: the real agent module generates and smoke-tests an evaluator.
    outcome = setup.builder.invoke(inputs)
    with TaskEvaluator(outcome.sandbox.root / outcome.output.eval_file,
                       outcome.output.callable_name) as evaluator:
        results = [evaluator.evaluate(inputs.project_root / file, {"repeat": repeat},
                                      {"values": [2, -3]})
                   for file, repeat in (("reference.py", 1), ("slower.py", 1),
                                        ("reference.py", 2), ("wrong.py", 1))]
    # Then: actual work determines J; incorrect output cannot receive a score.
    assert [result.score for result in results[:3]] == pytest.approx(
        [1, 1 / 3, 1 / 2] if maximize else [2, 6, 4],
    )
    assert all(result.valid for result in results[:3])
    assert not results[3].valid and results[3].score is None
    assert outcome.output.direction == setup.client.direction
    assert outcome.output.unit == ("items/quasar_step" if maximize else "quasar_steps")
    assert outcome.attempts == setup.client.calls == 1
    assert (outcome.sandbox.root / "project/reference.py").read_text(encoding="utf-8") == (
        inputs.project_root / "reference.py").read_text(encoding="utf-8")


@pytest.mark.parametrize("broken", [
    "def quasar_work(:\n", None,
    "def quasar_work(candidate, params, context):\n    return 1 / 0\n",
    "raise RuntimeError('load failed')\n", "unrelated = 1\n",
])
def test_repaired_when_first_version_fails(setup: Setup, broken: str | None) -> None:
    # Given
    setup.client.versions = [broken, GENERATED]
    # When
    outcome = setup.builder.invoke(setup.inputs)
    # Then: one existing retry, with the same goal and diagnostic feedback.
    assert outcome.attempts == setup.client.calls == 2
    assert setup.client.goals == [setup.inputs.goal, setup.inputs.goal]
    assert setup.client.prompts[0] != setup.client.prompts[1]


def test_no_result_when_attempt_budget_exhausted(setup: Setup) -> None:
    # Given
    setup.client.versions = ["def quasar_work(a, b, c):\n    return 1 / 0\n"]
    # When / Then
    with pytest.raises(AgentCallError, match="ZeroDivisionError"):
        _ = setup.builder.invoke(setup.inputs)
    assert setup.client.calls == 2


def test_compile_only_when_target_probe_absent(setup: Setup) -> None:
    # Given: importing this evaluator on the host would fail.
    setup.client.versions = ["import unavailable_target_gpu_library\n" + GENERATED]
    # When
    outcome = setup.builder.invoke(replace(setup.inputs, probe=None))
    # Then: syntax checking does not import target-only dependencies.
    assert outcome.attempts == setup.client.calls == 1
