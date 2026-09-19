"""CPU contracts for the manual task; no model, GPU, provider or Git execution."""

import json
import shutil
import sys
from collections.abc import Mapping
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path

import pytest
from pydantic import JsonValue, TypeAdapter

from examples.c3_qwen3.model_binding import BundleSpec
from examples.c3_qwen3.runner_records import (
    BindingReceipt as Receipt,
    GoalSpec,
    MeasurementReport as Measurement,
    Prompt,
    QualityReport as Quality,
)
from kernel_optimizer import task_cli, wiring
from kernel_optimizer.agents.runtime import AgentCallError
from kernel_optimizer.agents.task_rewriter import TaskRewriteInputs
from kernel_optimizer.cli import main
from kernel_optimizer.evaluation.task_eval import TaskEvaluation, TaskEvaluator

ROOT = Path(__file__).resolve().parents[1]
MANUAL = ROOT / "examples/c3_qwen3/manual_task.py"
RAW = ROOT / "results/c3-qwen3-4b-operator-goals"


@dataclass(slots=True)  # noqa: MUTABLE_OK - admission deliberately consumes counters
class Budget:
    """Mutable injected admission counter; failed calls retain their consumed slot."""
    remaining: int = 20
    used: int = 0

    def admit(self) -> bool:
        if self.remaining <= 0:
            return False
        self.remaining -= 1
        self.used += 1
        return True


@dataclass(slots=True)  # noqa: MUTABLE_OK - fake records calls and injected faults
class Runner:
    """CPU runner fake records the actual bind/reset/quality/measure interaction."""
    root: Path
    fault: str = ""
    calls: list[str] = field(default_factory=list)
    quality_ids: tuple[str, ...] = ()
    effective: Mapping[str, JsonValue] = field(default_factory=dict)

    def bind(self, bundle: BundleSpec) -> Receipt:
        self.calls.append("bind")
        self.effective = {"params": dict(bundle.params)}
        return Receipt(bundle.bundle_sha256, bundle.source_hashes, bundle.params_sha256, (),
                       baseline_restored=self.fault != "binding")

    def reset(self, specs: tuple[Prompt, ...]) -> None:
        self.calls.append("reset")
        assert specs

    def quality(self, ids: tuple[str, ...], refs: Path) -> Quality:
        self.calls.append("quality")
        assert refs.is_file()
        self.quality_ids = ids
        metrics = {"local_checks_passed": 1.0, "logits_relative_l2_max": 0.005,
                   "paired_mean_nll_delta_nat_per_token": 0.01}
        changes = {"wrong": ("local_checks_passed", 0.0), "logits": ("logits_relative_l2_max", .011),
                   "nll": ("paired_mean_nll_delta_nat_per_token", .021)}
        if self.fault in changes:
            name, value = changes[self.fault]
            metrics[name] = value
        if self.fault == "missing_quality":
            metrics.pop("local_checks_passed")
        if self.fault == "quality_mutation":
            (self.root / "contract.json").write_text("{}")
        return Quality(self.fault != "skip", metrics, self.root / "quality.json")

    def measure(self, goal: GoalSpec, ids: tuple[str, ...]) -> Measurement:
        self.calls.append("measure")
        counts = {"ttft": 4, "single": 256, "multi": 1024}
        metrics = {"completed_requests": float(len(ids)), "output_tokens": float(counts[goal.id]),
                   "total_wall_ms": 1000.0, "ttft_ms": 1.0, "decode_wall_ms": 999.0}
        if self.fault == "short":
            metrics["output_tokens"] -= 1
        if self.fault == "mutation":
            (self.root / "contract.json").write_text("{}")
        effective = TypeAdapter(dict[str, float]).validate_python(self.effective["params"])
        return Measurement(TaskEvaluation(score=effective.get("width", 8.0), metrics=metrics), self.root / "measurement.json")


type TaskFixture = tuple[Path, dict[str, JsonValue | Path | Runner | Budget]]


@pytest.fixture
def task(tmp_path: Path) -> TaskFixture:
    for name in ("contract.json", "interface.md", "approval.json", "proposed-prompts.json", "preflight.json"):
        shutil.copyfile(RAW / name, tmp_path / name)
    candidate = tmp_path / "operators.py"
    candidate.write_text("def replacement(x): return x + 0\n")
    bundle = {"entry": "operators.py", "sites": [], "files": ["operators.py"], "helpers": [],
              "params": {"width": 99}, "space": {"params": [], "constraints": []}, "parent_bundle": None,
              "cumulative_from_original_baseline": True, "source_body_rewrite_required": True}
    (tmp_path / "bundle.json").write_text(json.dumps(bundle))
    ids = ("calibration-calibration-00", "calibration-calibration-01")
    oracle = tmp_path / "oracle.json"
    oracle.write_text(json.dumps({"contract_sha256": "9d1ef6a79bf99eb6ac0e639d82e894227b403124764ba4018531c9ddf179eaf8",
        "corpus_sha256": "d60e5c108e79c8327a74199d5c093b117f9e64661cff4356e33a723dceb30a02", "baseline": True,
        "prompts": [{"prompt_id": pid, "input_ids": [1], "tokens": [2], "logits_file": "logits.json",
                     "logits_sha256": "cpu-boundary", "vocab_size": 3, "nll": [0.1]} for pid in ids]}))
    return candidate, {"contract": tmp_path / "contract.json", "assets_manifest": tmp_path / "preflight.json",
                       "bundle_path": tmp_path / "bundle.json", "output_dir": tmp_path,
                       "goal_id": "ttft", "split": "search",
                       "prompt_ids": [f"search-ttft-{i:02}" for i in range(4)],
                       "resident_runner": Runner(tmp_path), "call_budget": Budget(),
                       "oracle_refs": oracle}


@pytest.mark.parametrize(("direction", "winner"), [("minimize", 1), ("maximize", 3)])
def test_provided_eval_path_characterization(tmp_path: Path, direction: str, winner: int) -> None:
    # Given: a provided CPU evaluator and a native positive objective.
    (tmp_path / "candidate.py").write_text("def run(width): return width\n")
    (tmp_path / "eval.py").write_text(
        "from runpy import run_path\ndef evaluate(path, params, context):\n"
        "    return float(run_path(str(path))['run'](params['width']))\n")
    (tmp_path / "space.json").write_text(json.dumps({"params": [
        {"name": "width", "kind": "int", "choices": [1, 3]}]}))
    # When: the real CLI provided-eval branch executes search and final evaluation.
    code = main(["optimize-task", "--project", str(tmp_path), "--candidate", "candidate.py",
                 "--space", "space.json", "--eval-file", "eval.py", "--direction", direction,
                 "--goal", "preserve the model outputs", "--trials", "2",
                 "--output", str(tmp_path / "result")])
    # Then: native direction, not sign inversion, determines the actual winner.
    summary = json.loads((tmp_path / "result/summary.json").read_text())
    assert code == 0 and summary["final_execution"]["params"]["values"] == {"width": winner}


def test_manual_entry_exists_and_loads_without_model() -> None:
    # Given: the task-local manual entry, deliberately not an EvalBuilder output.
    assert MANUAL.is_file(), "manual evaluator has not been implemented"
    # When: TaskEvaluator loads the real callback without model dependencies.
    with TaskEvaluator(MANUAL, "evaluate") as evaluator:
        result = evaluator.evaluate(MANUAL, {}, {})
    # Then: missing runtime state is invalid, not a fabricated performance score.
    assert not result.valid and result.score is None
    assert "torch" not in sys.modules


@pytest.mark.parametrize("case", [("ttft", 4, 4), ("single", 2, 256), ("multi", 8, 1024)])
def test_shared_manual_metrics(task: TaskFixture, case: tuple[str, int, int]) -> None:
    # Given: one runner and the frozen workload selected by context, not operator strategy.
    path, context = task
    goal, requests, tokens = case
    context.update(goal_id=goal, prompt_ids=[f"search-{goal}-{i:02}" for i in range(requests)])
    runner, budget = context["resident_runner"], context["call_budget"]
    assert isinstance(runner, Runner) and isinstance(budget, Budget)
    before = (path.parent / "bundle.json").read_bytes()
    # When: the real TaskEvaluator invokes the manual entry.
    with TaskEvaluator(MANUAL, "evaluate") as evaluator:
        result = evaluator.evaluate(path, {"width": 3}, context)
    # Then: quality precedes measurement, effective params do not mutate the bundle.
    assert result.valid and result.score == 3 and result.metrics["output_tokens"] == tokens
    assert runner.calls == ["bind", "reset", "quality", "reset", "measure"] and budget.used == 1
    assert runner.effective["params"] == {"width": 3} and (path.parent / "bundle.json").read_bytes() == before
    assert result.metrics["local_checks_passed"] == 1 and "quality.json" in (result.detail or "")


@pytest.mark.parametrize("fault", ["wrong", "skip", "logits", "nll", "missing_quality", "binding", "quality_mutation"])
def test_missing_quality_or_wrong_binding_never_measures(task: TaskFixture, fault: str) -> None:
    # Given: a runner that fails one independent binding or quality obligation.
    path, context = task
    runner, budget = context["resident_runner"], context["call_budget"]
    assert isinstance(runner, Runner) and isinstance(budget, Budget)
    runner.fault = fault
    # When
    with TaskEvaluator(MANUAL, "evaluate") as evaluator:
        result = evaluator.evaluate(path, {}, context)
    # Then: failure consumes its slot but cannot acquire a performance score.
    assert not result.valid and result.score is None and "measure" not in runner.calls and budget.used == 1
    assert "bind" in runner.calls and (fault == "binding" or "quality" in runner.calls)


@pytest.mark.parametrize("fault", ["short", "mutation"])
def test_wrong_metric_or_frozen_change_rejected(task: TaskFixture, fault: str) -> None:
    # Given
    path, context = task
    runner = context["resident_runner"]
    assert isinstance(runner, Runner)
    runner.fault = fault
    # When
    with TaskEvaluator(MANUAL, "evaluate") as evaluator:
        result = evaluator.evaluate(path, {}, context)
    # Then
    assert not result.valid and result.score is None and "measure" in runner.calls


@pytest.mark.parametrize("field", ["assets_manifest", "oracle_refs", "resident_runner", "call_budget", "prompt_ids"])
def test_missing_assets_block_execution(task: TaskFixture, field: str) -> None:
    # Given
    path, context = task
    context.pop(field)
    # When
    with TaskEvaluator(MANUAL, "evaluate") as evaluator:
        result = evaluator.evaluate(path, {}, context)
    # Then
    assert not result.valid and result.score is None


def test_budget_denial_prevents_binding(task: TaskFixture) -> None:
    # Given
    path, context = task
    runner, budget = context["resident_runner"], context["call_budget"]
    assert isinstance(runner, Runner) and isinstance(budget, Budget)
    budget.remaining = 0
    # When
    with TaskEvaluator(MANUAL, "evaluate") as evaluator:
        result = evaluator.evaluate(path, {}, context)
    # Then
    assert not result.valid and result.score is None and runner.calls == [] and budget.used == 0


@pytest.mark.parametrize("case",
                         [("ttft", "minimize", 1, 4), ("single", "maximize", 3, 2), ("multi", "maximize", 3, 8)])
def test_provided_eval_path_goal_context_reaches_rewriter(task: TaskFixture, monkeypatch: pytest.MonkeyPatch,
                                                        case: tuple[str, str, int, int]) -> None:
    # Given: only external model/runtime/agent boundaries are fake; CLI/search/evaluator are real.
    path, runtime_context = task
    goal, direction, winner, count = case
    runtime_context.update(goal_id=goal, prompt_ids=[f"search-{goal}-{i:02}" for i in range(count)])
    captured: list[TaskRewriteInputs] = []
    class InjectedEvaluator(TaskEvaluator):
        def evaluate[T](self, candidate_path: Path, params: Mapping[str, JsonValue], context: Mapping[str, T]) -> TaskEvaluation:
            return super().evaluate(candidate_path, params, {**context, **runtime_context})
    class AgentBoundary:
        def invoke(self, inputs: TaskRewriteInputs) -> None:
            captured.append(inputs)
            raise AgentCallError("controlled CPU agent boundary: no child")
    monkeypatch.setattr(task_cli, "TaskEvaluator", InjectedEvaluator)
    monkeypatch.setattr(wiring, "Runtime", lambda *args, **kwargs: nullcontext(AgentBoundary()))
    monkeypatch.setattr(wiring, "build_task_rewriter", lambda *args: AgentBoundary())
    monkeypatch.setattr(wiring, "build_eval_builder", lambda *args: pytest.fail("provided evaluation invoked builder"))
    (path.parent / "space.json").write_text(json.dumps({"params": [
        {"name": "width", "kind": "int", "choices": [1, 3]}]}))
    requested_goal = f"Optimize {goal}; preserve frozen quality"
    unit = {"ttft": "ms", "single": "tokens/s", "multi": "tokens/s"}[goal]
    # When: --eval-file bypasses EvalBuilder and the actual CLI supplies the user goal to rewriting.
    code = main(["optimize-task", "--project", str(path.parent), "--candidate", path.name,
                 "--space", "space.json", "--eval-file", str(MANUAL), "--direction", direction,
                 "--goal", requested_goal, "--unit", unit, "--trials", "2", "--rewrite-rounds", "1", "--probe-budget", "0",
                 "--output", str(path.parent / "result")])
    # Then: selected params and final evaluation retain native direction and task-goal context.
    summary = json.loads((path.parent / "result/summary.json").read_text())
    assert code == 0 and summary["best"]["params"]["values"] == {"width": winner}
    assert summary["final_execution"]["task_evaluation"]["score"] == winner
    assert len(captured) == 1 and captured[0].goal == requested_goal
    assert captured[0].objective.direction == direction
    assert captured[0].objective.unit == unit
    budget = runtime_context["call_budget"]
    assert isinstance(budget, Budget) and budget.used == 3


@pytest.mark.parametrize("fault", ["prompt_ids", "goal_id", "split", "contract", "oracle_refs", "entry"])
def test_invalid_identity_consumes_slot_without_binding(task: TaskFixture, fault: str) -> None:
    # Given
    path, context = task
    changes = {"prompt_ids": ["heldout-ttft-00"], "goal_id": "unknown", "split": "unknown",
               "contract": path, "oracle_refs": path}
    if fault in changes:
        context[fault] = changes[fault]
    else:
        path = path.parent / "other.py"
    runner, budget = context["resident_runner"], context["call_budget"]
    assert isinstance(runner, Runner) and isinstance(budget, Budget)
    # When
    with TaskEvaluator(MANUAL, "evaluate") as evaluator:
        result = evaluator.evaluate(path, {}, context)
    # Then
    assert not result.valid and result.score is None and runner.calls == [] and budget.used == 1


def test_heldout_uses_selected_group_for_quality(task: TaskFixture) -> None:
    # Given
    path, context = task
    ids = [f"heldout-ttft-{i:02}" for i in range(4)]
    context.update(split="heldout", prompt_ids=ids)
    oracle = path.parent / "oracle.json"
    data = json.loads(oracle.read_text())
    data["prompts"] = [{**data["prompts"][0], "prompt_id": pid} for pid in ids]
    oracle.write_text(json.dumps(data))
    # When
    with TaskEvaluator(MANUAL, "evaluate") as evaluator:
        result = evaluator.evaluate(path, {}, context)
    # Then
    runner = context["resident_runner"]
    assert isinstance(runner, Runner) and result.valid and runner.quality_ids == tuple(ids)


def test_evaluator_identity_persists_between_calls(task: TaskFixture) -> None:
    # Given: a copied evaluator isolates the deliberate corruption from product sources.
    path, context = task
    copied = path.parent / "evaluator"
    copied.mkdir()
    for name in ("manual_task.py", "manual_data.py", "manual_types.py"):
        shutil.copyfile(MANUAL.with_name(name), copied / name)
    with TaskEvaluator(copied / MANUAL.name, "evaluate") as evaluator:
        assert evaluator.evaluate(path, {}, context).valid
        (copied / MANUAL.name).write_text("# changed frozen evaluator\n")
        # When
        result = evaluator.evaluate(path, {}, context)
    # Then
    assert not result.valid and result.score is None
