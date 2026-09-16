"""Direct evaluation exercises actual CPU candidate functions, not preset scores."""

import subprocess
import sys
from pathlib import Path
from typing import Final

import pytest

from kernel_optimizer.evaluation.task_eval import TaskEvaluator, TaskEvaluationError


EVALUATOR: Final = '''
from runpy import run_path
from kernel_optimizer.evaluation.task_eval import TaskEvaluation
from .reference import expected

calls = 0

def dragonfruit_cost(candidate_path, params, context):
    global calls
    from .lazy_helper import offset
    calls += 1
    transform = run_path(str(candidate_path))["transform"]
    output, operations = transform(context["values"], params["repeat"])
    if output != expected(context["values"]):
        return TaskEvaluation(valid=False, detail="reference mismatch")
    return TaskEvaluation(score=operations + offset, valid=True,
                          metrics={"novel_work_units": operations, "calls": calls})

def nebula_yield(candidate_path, params, context):
    result = dragonfruit_cost(candidate_path, params, context)
    if not result.valid:
        return result
    return len(context["values"]) / result.score
'''


@pytest.fixture
def cpu_project(tmp_path: Path) -> Path:
    _ = (tmp_path / "unusual eval-name.py").write_text(EVALUATOR, encoding="utf-8")
    _ = (tmp_path / "reference.py").write_text(
        "def expected(values):\n    return [value + value for value in values]\n",
        encoding="utf-8",
    )
    _ = (tmp_path / "lazy_helper.py").write_text("offset = 0\n", encoding="utf-8")
    for name, work in (("fast", 1), ("slow", 3)):
        _ = (tmp_path / f"{name}.py").write_text(
            f'''def transform(values, repeat):
    operations = 0
    output = []
    for value in values:
        for _ in range(repeat * {work}):
            doubled = value * 2
            operations += 1
        output.append(doubled)
    return output, operations
''',
            encoding="utf-8",
        )
    _ = (tmp_path / "wrong.py").write_text(
        "def transform(values, repeat):\n    return values, 0\n", encoding="utf-8",
    )
    return tmp_path


@pytest.mark.parametrize("name", ["dragonfruit_cost", "nebula_yield"])
def test_objective_changes_when_candidate_or_parameters_change(cpu_project: Path, name: str) -> None:
    # Given: independent reference, real candidates, arbitrary task context.
    context = {"values": [1, -2, 4], "project_root": cpu_project}
    with TaskEvaluator(cpu_project / "unusual eval-name.py", name) as evaluator:
        # When: evaluate each actual candidate/parameter combination.
        results = [evaluator.evaluate(cpu_project / file, {"repeat": repeat}, context)
                   for file, repeat in (("fast.py", 1), ("slow.py", 1), ("fast.py", 2))]
    # Then: native min-cost and max-yield objectives are computed without sign flipping.
    expected = [3, 9, 6] if name == "dragonfruit_cost" else [1, 1 / 3, 1 / 2]
    assert all(result.valid for result in results)
    assert [result.score for result in results] == pytest.approx(expected)


def test_invalid_when_real_candidate_fails_reference(cpu_project: Path) -> None:
    # Given
    with TaskEvaluator(cpu_project / "unusual eval-name.py", "dragonfruit_cost") as evaluator:
        # When
        result = evaluator.evaluate(cpu_project / "wrong.py", {"repeat": 1}, {"values": [3]})
    # Then
    assert not result.valid and result.score is None
    assert result.detail == "reference mismatch"


def test_helpers_and_state_persist_until_context_closes(cpu_project: Path) -> None:
    # Given
    before = set(sys.modules)
    original_path = sys.path.copy()
    with TaskEvaluator(cpu_project / "unusual eval-name.py", "dragonfruit_cost") as evaluator:
        # When
        results = [evaluator.evaluate(cpu_project / "fast.py", {"repeat": 1}, {"values": [2]})
                   for _ in range(2)]
        loaded = set(sys.modules) - before
    # Then: module and lazily imported helpers are removed, state was reused.
    assert [result.metrics["calls"] for result in results] == [1, 2]
    assert any(name.endswith(".lazy_helper") for name in loaded)
    assert not any(name.startswith("_task_eval_") for name in loaded & set(sys.modules))
    assert sys.path == original_path


@pytest.mark.parametrize("expression", ["True", "float('nan')", "float('inf')", "float('-inf')", "'3'"])
def test_invalid_when_callback_returns_bad_scalar(tmp_path: Path, expression: str) -> None:
    # Given
    path = tmp_path / "score.py"
    _ = path.write_text(f"def arbitrary_name(candidate, params, context):\n    return {expression}\n",
                    encoding="utf-8")
    with TaskEvaluator(path, "arbitrary_name") as evaluator:
        # When
        result = evaluator.evaluate(tmp_path / "candidate.py", {}, {})
    # Then
    assert not result.valid and result.score is None and result.detail


def test_failure_when_callback_raises(tmp_path: Path) -> None:
    # Given
    path = tmp_path / "score.py"
    _ = path.write_text("def explode(candidate, params, context):\n    raise RuntimeError('CPU failed')\n",
                    encoding="utf-8")
    with TaskEvaluator(path, "explode") as evaluator:
        # When
        result = evaluator.evaluate(tmp_path / "candidate.py", {}, {})
    # Then
    assert not result.valid and result.score is None
    assert result.detail == "RuntimeError: CPU failed"


@pytest.mark.parametrize("source", ["other = 1\n", "wanted = 2\n", "raise RuntimeError('load failed')\n"])
def test_load_failure_cleans_namespace(tmp_path: Path, source: str) -> None:
    # Given
    path = tmp_path / "score.py"
    _ = path.write_text(source, encoding="utf-8")
    before = set(sys.modules)
    # When
    with pytest.raises(TaskEvaluationError):
        _ = TaskEvaluator(path, "wanted")
    # Then
    assert not any(name.startswith("_task_eval_") for name in set(sys.modules) - before)


def test_closed_evaluator_rejects_calls(cpu_project: Path) -> None:
    # Given
    with TaskEvaluator(cpu_project / "unusual eval-name.py", "dragonfruit_cost") as evaluator:
        pass
    # When / Then
    with pytest.raises(TaskEvaluationError, match="closed"):
        _ = evaluator.evaluate(cpu_project / "fast.py", {"repeat": 1}, {"values": [1]})


def test_real_evaluation_in_fresh_cpu_process(cpu_project: Path) -> None:
    # Given: the public API in a clean process, with no GPU imports or harness setup.
    program = '''
import sys
from pathlib import Path
from kernel_optimizer.evaluation.task_eval import TaskEvaluator
root = Path(sys.argv[1])
with TaskEvaluator(root / "unusual eval-name.py", "dragonfruit_cost") as evaluator:
    good = evaluator.evaluate(root / "fast.py", {"repeat": 2}, {"values": [2, 3]})
    bad = evaluator.evaluate(root / "wrong.py", {"repeat": 1}, {"values": [2, 3]})
assert good.valid and good.score == 4
assert not bad.valid and bad.score is None
assert "torch" not in sys.modules
assert not any(name.startswith("kernel_optimizer.tasks") for name in sys.modules)
'''
    # When
    completed = subprocess.run(
        [sys.executable, "-c", program, str(cpu_project)],
        capture_output=True, text=True, timeout=20, check=False,
    )
    # Then
    assert completed.returncode == 0, completed.stderr
