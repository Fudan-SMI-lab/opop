"""Real CPU evaluation through the command-line search surface."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from kernel_optimizer.config import BudgetConfig
from kernel_optimizer.control.convergence import ConvergencePolicy
from kernel_optimizer.control.direct_task import TaskSearch, TaskSpace
from kernel_optimizer.evaluation.task_eval import TaskEvaluation, TaskEvaluator
from kernel_optimizer.models.core import Family, LatencyStats, ParamDomain, ParameterSpace, ParamSet, TrialRecord
from kernel_optimizer.tuning.objective import Objective
from kernel_optimizer.tuning.tpe import OptunaTPETuner


@pytest.fixture
def task_project(tmp_path: Path) -> Path:
    # Given: a candidate whose parameters change actual execution and output size.
    example = Path(__file__).resolve().parents[1] / "examples" / "direct_task"
    for name in ("candidate.py", "eval.py", "space.json", "context.json"):
        (tmp_path / name).write_text((example / name).read_text(encoding="utf-8"), encoding="utf-8")
    return tmp_path


@pytest.mark.parametrize(("direction", "winner", "score"), [
    ("minimize", 1, -1.0), ("maximize", 3, 1.0),
])
def test_cli_searches_native_objective_when_provided(
    task_project: Path, direction: str, winner: int, score: float,
) -> None:
    # Given: an independent evaluator and no PARAMS/ModelNew/GPU requirements.
    command = [sys.executable, "-m", "kernel_optimizer.cli", "optimize-task",
               "--project", str(task_project), "--candidate", "candidate.py",
               "--space", "space.json", "--eval-file", "eval.py",
               "--eval-function", "assess", "--context", "context.json",
               "--direction", direction, "--label", "payload delta", "--unit", "bytes",
               "--trials", "4", "--output", str(task_project / "result")]
    # When: the actual parser launches a real Optuna search in a fresh process.
    result = subprocess.run(command, capture_output=True, text=True, timeout=30)
    # Then: invalid self-reported fast output loses; native signed scores survive.
    assert result.returncode == 0, result.stderr
    summary = json.loads((task_project / "result" / "summary.json").read_text())
    assert summary["best"]["params"]["values"] == {"width": winner}
    assert summary["best"]["task_evaluation"]["score"] == score
    assert summary["best"]["latency_ms"] is None
    assert (summary["valid_count"], summary["invalid_count"]) == (3, 1)
    assert summary["objective"] == {"direction": direction, "label": "payload delta", "unit": "bytes"}
    trials = json.loads((task_project / "result" / "trials.json").read_text())
    assert sorted(t["task_evaluation"]["score"] for t in trials if t["status"] == "complete") == [-1, 0, 1]
    assert "J=" in result.stdout


@pytest.mark.parametrize("direction", ["minimize", "maximize"])
def test_family_and_stats_follow_native_winner_when_multiple_candidates(task_project: Path, direction: str) -> None:
    # Given: two independently executed implementations with different output sizes.
    second = task_project / "second.py"
    second.write_text("def run(data, width):\n    return sorted(data), bytes(width + 10)\n")
    objective = Objective.model_validate({"direction": direction})
    space = TaskSpace.model_validate_json((task_project / "space.json").read_text())
    with TaskEvaluator(task_project / "eval.py", "assess") as evaluator:
        search = TaskSearch(evaluator, objective, BudgetConfig(trials_per_space=4))
        # When: both candidates go through the real tuner and existing family manager.
        first = search.evaluate_candidate(task_project / "candidate.py", space, {"data": [3, 1, 2]})
        other = search.evaluate_candidate(second, space, {"data": [3, 1, 2]})
    # Then: family tie-breaking, stats, and global best choose the same native winner.
    winner = first if direction == "minimize" else other
    assert winner is not None
    assert search.best() == winner
    assert search.families.active_families()[0].best.candidate_id == winner.candidate_id
    assert search.stats[0].best == first
    assert search.stats[1].best == other
    assert search.stats[0].param_stats[0].latency_by_value == {}
    assert search.stats[0].param_stats[0].best_trial_value == (1 if direction == "minimize" else 3)


@pytest.mark.parametrize(("direction", "history"), [
    ("minimize", [0.0, -0.1, -0.2]), ("maximize", [-0.2, -0.1, 0.0]),
])
def test_convergence_uses_absolute_gain_when_scores_cross_zero(direction: str, history: list[float]) -> None:
    # Given: signed native progress, with a legacy percentage threshold that would stall.
    family = Family(family_id="f", anchor_candidate_id="c", best_history=history)
    policy = ConvergencePolicy(BudgetConfig(), Objective.model_validate({"direction": direction}))
    # When: existing convergence policy decides whether to continue.
    verdict = policy.family_verdict(family)
    # Then: real absolute progress is not discarded by division at zero.
    assert verdict.verdict == "continue"


def test_tuner_excludes_bad_scores_and_retains_ties_when_native() -> None:
    # Given: real Optuna with explicit anchors, including invalid and absent scores.
    space = ParameterSpace(space_id="s", candidate_id="c", source_sha="x", domains=[
        ParamDomain(name="x", kind="int", choices=list(range(5))),
    ])
    tuner = OptunaTPETuner(space, lambda _: True, 5, objective=Objective(direction="maximize"),
                           anchors=tuple(ParamSet(values={"x": n}) for n in range(5)))
    results = [None, TaskEvaluation(valid=False, score=999.0),
               TaskEvaluation.model_construct(valid=True, score=float("nan")),
               TaskEvaluation(score=0.0), TaskEvaluation(score=0.0)]
    records = []
    # When: missing, invalid, nonfinite and tied results are told to the tuner.
    for result in results:
        trial_id, params = tuner.ask()
        record = TrialRecord(trial_id=trial_id, candidate_id="c", space_id="s", params=params,
                             status="complete", task_evaluation=result)
        records.append(record)
        tuner.tell(trial_id, record)
    # Then: only usable values reach Optuna and strict ties retain the first incumbent.
    assert tuner.best() == records[3]
    assert tuner.study.best_value == 0.0
    assert [t.value for t in tuner.study.trials] == [None, None, None, 0.0, 0.0]


def test_legacy_tuner_uses_median_when_mean_ranks_oppositely() -> None:
    # Given: latency-only records with opposite mean/median ordering.
    space = ParameterSpace(space_id="s", candidate_id="c", source_sha="x", domains=[
        ParamDomain(name="x", kind="int", choices=[1, 2]),
    ])
    tuner = OptunaTPETuner(space, lambda _: True, 2)
    # When: both actual legacy record shapes are consumed.
    for mean, median in [(1.0, 3.0), (5.0, 2.0)]:
        trial_id, params = tuner.ask()
        tuner.tell(trial_id, TrialRecord(trial_id=trial_id, candidate_id="c", space_id="s",
            params=params, status="complete", latency_ms=LatencyStats(
                mean=mean, median=median, std=0.0, min=1.0, max=6.0, n_samples=5)))
    # Then: legacy optimization still chooses robust latency, not mean.
    assert tuner.best().latency_ms.mean == 5.0
    assert tuner.study.best_value == 2.0


def test_generated_cli_executes_written_evaluator(task_project: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from contextlib import contextmanager
    from kernel_optimizer import wiring
    from kernel_optimizer.cli import main
    from test_v5_direct_eval_builder import WritingClient
    runtime_type = wiring.Runtime

    # Given: only the provider/host is replaced; the builder writes real executable code.
    @contextmanager
    def local_runtime(cfg, log_dir):
        client = WritingClient()
        client.versions = [(task_project / "eval.py").read_text()]
        client.callable_name = "assess"
        client.direction = "maximize"
        runtime = runtime_type(cfg, log_dir)
        runtime.client = client
        try:
            yield runtime
        finally:
            client.close()

    monkeypatch.setattr(wiring, "Runtime", local_runtime)
    # When: the real CLI builds then executes the generated evaluation in search.
    code = main(["optimize-task", "--project", str(task_project), "--candidate", "candidate.py",
                 "--space", "space.json", "--context", "context.json", "--trials", "4",
                 "--goal", "maximize payload delta while sorting correctly",
                 "--output", str(task_project / "generated")])
    # Then: generated native maximize J produces the known winner, not canned scores.
    assert code == 0
    summary = json.loads((task_project / "generated" / "summary.json").read_text())
    assert summary["best"]["params"]["values"] == {"width": 3}
    assert summary["best"]["task_evaluation"]["score"] == 1.0


def test_cli_reports_failure_when_every_output_is_wrong(task_project: Path) -> None:
    # Given: every parameter choice produces an incorrect result.
    (task_project / "candidate.py").write_text("def run(data, width):\n    return [], b''\n")
    # When: the real command searches the invalid candidate.
    result = subprocess.run([sys.executable, "-m", "kernel_optimizer.cli", "optimize-task",
        "--project", str(task_project), "--candidate", "candidate.py", "--space", "space.json",
        "--context", "context.json", "--eval-file", "eval.py", "--eval-function", "assess",
        "--direction", "minimize", "--trials", "4", "--output", str(task_project / "bad")],
        capture_output=True, text=True, timeout=30)
    # Then: no incumbent is fabricated and a saved failure summary accompanies exit 1.
    assert result.returncode == 1, result.stderr
    summary = json.loads((task_project / "bad" / "summary.json").read_text())
    assert summary["best"] is None
    assert (summary["valid_count"], summary["invalid_count"]) == (0, 4)


def test_child_search_keeps_small_improvement_separate_from_stopping(task_project: Path) -> None:
    # Given: native gains smaller than the stopping threshold still beat the incumbent.
    space = TaskSpace(params=[ParamDomain(name="width", kind="int", choices=[2])])
    child = task_project / "child.py"
    child.write_text("def run(data, width):\n    return list(sorted(data)), bytes(width + 1)\n")
    with TaskEvaluator(task_project / "eval.py", "assess") as evaluator:
        search = TaskSearch(evaluator, Objective(direction="maximize"),
                            BudgetConfig(trials_per_space=1, no_improve_rounds=1))
        search.convergence.min_absolute_gain = 2.0
        parent = search.evaluate_candidate(task_project / "candidate.py", space, {"data": [2, 1]})
        assert parent is not None
        # When: Task5's child-path seam is evaluated under the same J and policies.
        winner = search.evaluate_candidate(child, space, {"data": [2, 1]}, parent_id=parent.candidate_id)
    # Then: stopping never changes the winner or its native history.
    assert winner is not None
    assert search.best() == winner
    family = next(iter(search.families.families.values()))
    assert family.best.candidate_id == winner.candidate_id
    assert family.best_history == [0.0, 1.0]
    assert family.status == "frozen_converged"
