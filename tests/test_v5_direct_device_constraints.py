"""CPU callback regressions for published constraints referencing device constants."""

import argparse
import json
from pathlib import Path

import pytest

from kernel_optimizer.config import BudgetConfig
from kernel_optimizer.control.direct_task import TaskSearch, TaskSpace
from kernel_optimizer.control.task_rewrite import probe_responses
from kernel_optimizer.evaluation.task_eval import TaskEvaluation, TaskEvaluator
from kernel_optimizer.models.core import Constraint, DeviceLimits, ParamDomain, ParamSet, ParamValue, TrialRecord
from kernel_optimizer.paramspace.guard import ConstraintError
from kernel_optimizer.task_cli import cmd_optimize_task
from kernel_optimizer.tuning.objective import Objective


@pytest.fixture
def cpu_project(tmp_path: Path) -> Path:
    (tmp_path / "candidate.py").write_text(
        "def run(params):\n    return params['NUM_WARPS'] * params['PARTNER']\n", encoding="utf-8")
    (tmp_path / "eval.py").write_text(
        "from runpy import run_path\n"
        "def evaluate(path, params, context):\n"
        "    context.setdefault('calls', []).append(dict(params))\n"
        "    return float(run_path(str(path))['run'](params))\n", encoding="utf-8")
    return tmp_path


@pytest.fixture
def space() -> TaskSpace:
    return TaskSpace(params=[
        ParamDomain(name="NUM_WARPS", kind="int", choices=[1, 2, 4, 8]),
        ParamDomain(name="PARTNER", kind="int", choices=[3, 5]),
    ], constraints=[Constraint(expr="NUM_WARPS * 32 <= MAX_THREADS_PER_BLOCK")])


def test_published_constraint_runs_with_optional_default_device(cpu_project: Path, space: TaskSpace) -> None:
    # Given: an existing caller using the original three positional arguments.
    with TaskEvaluator(cpu_project / "eval.py", "evaluate") as evaluator:
        search = TaskSearch(evaluator, Objective(direction="maximize"), BudgetConfig(trials_per_space=8))
        # When: tuning encounters a published-style hardware constraint.
        best = search.evaluate_candidate(cpu_project / "candidate.py", space, {})
    # Then: valid choices execute rather than raising unknown-name ConstraintError.
    assert best is not None and best.task_evaluation is not None
    assert best.task_evaluation.score == 40.0


@pytest.mark.parametrize("threads", [64, 128])
def test_tuning_uses_nondefault_device_limits(cpu_project: Path, space: TaskSpace, threads: int) -> None:
    # Given: limits deliberately different from the default 1024 threads.
    calls: list[dict[str, ParamValue]] = []
    with TaskEvaluator(cpu_project / "eval.py", "evaluate") as evaluator:
        search = TaskSearch(evaluator, Objective(direction="maximize"), BudgetConfig(trials_per_space=8),
                            device=DeviceLimits(max_threads_per_block=threads))
        # When: native TPE evaluates only constraint-admitted choices.
        best = search.evaluate_candidate(cpu_project / "candidate.py", space, {"calls": calls})
    # Then: valid configurations execute, and larger violating choices never reach the callback.
    assert calls and best is not None
    assert {p["NUM_WARPS"] for p in calls} == ({1, 2} if threads == 64 else {1, 2, 4})
    assert best.params.values == {"NUM_WARPS": threads // 32, "PARTNER": 5}


@pytest.mark.parametrize("threads", [64, 128])
def test_probes_use_same_device_limits_and_keep_fixed_partners(cpu_project: Path, space: TaskSpace, threads: int) -> None:
    # Given: a fixed parent registered without retuning, as in the thin acquisition driver.
    parent = TrialRecord(trial_id="parent", candidate_id="parent", space_id="s",
                         params=ParamSet(values={"NUM_WARPS": 2, "PARTNER": 3}), status="complete",
                         task_evaluation=TaskEvaluation(score=6.0))
    calls: list[dict[str, ParamValue]] = []
    with TaskEvaluator(cpu_project / "eval.py", "evaluate") as evaluator:
        search = TaskSearch(evaluator, Objective(direction="maximize"),
                            device=DeviceLimits(max_threads_per_block=threads))
        search.paths[parent.candidate_id] = cpu_project / "candidate.py"
        search.spaces[parent.candidate_id] = space
        search.probe_budget = 4
        # When: fresh native response probes evaluate legal endpoints.
        responses = probe_responses(search, parent, {"calls": calls})
    # Then: both axes are measured; invalid warps are excluded and partners stay fixed.
    assert search.probe_calls == 4
    assert calls == [{"NUM_WARPS": 1, "PARTNER": 3},
                     {"NUM_WARPS": threads // 32, "PARTNER": 3},
                     {"NUM_WARPS": 2, "PARTNER": 3}, {"NUM_WARPS": 2, "PARTNER": 5}]
    assert all(r.a is not None and r.b is not None and r.reason is None for r in responses)
    assert search.trials == []


def test_plain_parameter_constraints_still_work_without_device(cpu_project: Path, space: TaskSpace) -> None:
    # Given: a hardware-independent caller and ordinary parameter-only constraints.
    plain = space.model_copy(update={"constraints": [Constraint(expr="NUM_WARPS <= 2")]})
    with TaskEvaluator(cpu_project / "eval.py", "evaluate") as evaluator:
        search = TaskSearch(evaluator, Objective(direction="maximize"), BudgetConfig(trials_per_space=8))
        # When: the unchanged positional API runs native tuning.
        best = search.evaluate_candidate(cpu_project / "candidate.py", plain, {})
    # Then: parameter-only admission and objective selection are preserved.
    assert best is not None and best.params.values == {"NUM_WARPS": 2, "PARTNER": 5}


def test_unknown_constraints_remain_rejected(cpu_project: Path, space: TaskSpace) -> None:
    # Given: a bad constraint, not a missing supported device constant.
    bad = space.model_copy(update={"constraints": [Constraint(expr="NUM_WARPS <= UNKNOWN_LIMIT")]})
    with TaskEvaluator(cpu_project / "eval.py", "evaluate") as evaluator:
        search = TaskSearch(evaluator, Objective(direction="maximize"),
                            device=DeviceLimits(max_threads_per_block=64))
        # When / Then: native tuning must not admit an unknown constraint.
        with pytest.raises(ConstraintError, match="UNKNOWN_LIMIT"):
            search.evaluate_candidate(cpu_project / "candidate.py", bad, {})
        assert search.trials == []


def test_direct_cli_passes_configured_device(cpu_project: Path, space: TaskSpace) -> None:
    # Given: a nondefault configured device and the real direct CLI handler.
    (cpu_project / "space.json").write_text(space.model_dump_json(), encoding="utf-8")
    args = argparse.Namespace(project=cpu_project, candidate=Path("candidate.py"), space=Path("space.json"),
                              eval_file=Path("eval.py"), direction="maximize", trials=8,
                              output=cpu_project / "result", override=["device.max_threads_per_block=64"])
    # When: CLI configuration flows into TaskSearch and its final execution.
    code = cmd_optimize_task(args)
    # Then: the selected candidate respects 64, not a silently substituted default limit.
    result = json.loads((cpu_project / "result" / "summary.json").read_text(encoding="utf-8"))
    assert code == 0 and result["best"]["params"]["values"] == {"NUM_WARPS": 2, "PARTNER": 5}
