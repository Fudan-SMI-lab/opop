"""Operator search composes the real manual evaluator, runner and native sampler on CPU."""

import importlib
import json
from contextlib import closing
from pathlib import Path

import pytest

from tests.test_c3_model_runner import runner


def search_module(name="operator_search"):
    path = Path(__file__).parents[1] / "examples/c3_qwen3" / f"{name}.py"
    assert path.is_file(), f"missing task-local search surface: {name}"
    return importlib.import_module(f"examples.c3_qwen3.{name}")


def test_empty_space_is_one_native_evaluation(tmp_path):
    # Given: an actual CPU callback and no tunable coordinates.
    from kernel_optimizer.control.direct_task import TaskSearch, TaskSpace
    from kernel_optimizer.evaluation.task_eval import TaskEvaluator
    from kernel_optimizer.config import BudgetConfig
    from kernel_optimizer.tuning.objective import Objective
    callback = tmp_path / "eval.py"
    callback.write_text("def evaluate(path, params, context):\n    context['calls'].append(dict(params))\n    return 3.0\n")
    calls = []
    # When
    with TaskEvaluator(callback, "evaluate") as evaluator:
        search = TaskSearch(evaluator, Objective(direction="maximize"), BudgetConfig(trials_per_space=8))
        result = search.evaluate_candidate(callback, TaskSpace(params=[]), {"calls": calls})
    # Then
    assert calls == [{}] and result.task_evaluation.score == 3.0


def test_bundle_structure_detects_helpers_but_ignores_params(tmp_path):
    # Given: complete baseline-relative source bundles.
    module = search_module("search_bundle")
    baseline = module.baseline_bundle(tmp_path / "baseline")
    folder = tmp_path / "candidate"
    folder.mkdir()
    (folder / "operators.py").write_text("from .helper import body\nPARAMS={'x':1}\ndef replace(site, call, params): return body(call.args[0])\n")
    (folder / "helper.py").write_text("def body(x): return x * 2\n")
    data = json.loads(baseline.read_text())
    data.update(sites=[{"site_id": "scale", "replacement_callable": "replace"}], helpers=["helper.py"])
    path = folder / "bundle.json"
    path.write_text(json.dumps(data))
    original = module.bundle_structure(path)
    # When: only PARAMS changes, then the helper computation changes.
    (folder / "operators.py").write_text((folder / "operators.py").read_text().replace("'x':1", "'x':2"))
    assert module.bundle_structure(path) == original
    (folder / "operators.py").write_text((folder / "operators.py").read_text().replace("'x':2", "'x':2, 'unused':7"))
    assert module.bundle_structure(path) == original
    (folder / "helper.py").write_text("def body(x): return x + x\n")
    # Then
    assert module.bundle_structure(path) != original


def test_session_injects_real_runner_and_shared_budget(tmp_path):
    # Given: actual numerical oracles and the frozen contract, with no model clone.
    module = search_module("search_session")
    bundles = search_module("search_bundle")
    resident, backend, prepared = runner(tmp_path / "raw")
    baseline = bundles.baseline_bundle(tmp_path / "baseline")
    oracle = resident.create_oracles(("calibration-calibration-00", "calibration-calibration-01"))
    goal = prepared.contract.goals[0]
    # When
    with module.OperatorSession(resident, goal, oracle, tmp_path / "session") as session:
        session.begin_opportunity(baseline, {}, deadline_unix_s=10**12)
        result = session.measure_baseline(baseline)
    # Then
    assert result.valid and result.score == pytest.approx(11.0)
    assert backend.loads == 1 and not resident.closed
    resident.close()


def test_native_startup_two_and_anchor_budget(tmp_path):
    # Given: the same native TaskSearch, with explicit pilot options.
    from kernel_optimizer.control.direct_task import TaskSearch, TaskSpace
    from kernel_optimizer.evaluation.task_eval import TaskEvaluator
    from kernel_optimizer.config import BudgetConfig
    from kernel_optimizer.models.core import ParamSet
    from kernel_optimizer.tuning.objective import Objective
    callback = tmp_path / "eval.py"
    callback.write_text("def evaluate(path, params, context):\n    context['calls'].append(params['x'])\n    return float(params['x'])\n")
    space = TaskSpace.model_validate({"params": [{"name": "x", "kind": "int", "choices": list(range(20))}]})
    calls = []
    # When
    with TaskEvaluator(callback, "evaluate") as evaluator:
        search = TaskSearch(evaluator, Objective(direction="maximize"), BudgetConfig(trials_per_space=8))
        best = search.evaluate_candidate(callback, space, {"calls": calls}, startup_trials=2,
            anchors=(ParamSet(values={"x": 3}), ParamSet(values={"x": 17}), ParamSet(values={"x": 18})))
    # Then
    assert calls[:3] == [3, 17, 18] and len(calls) == 8
    assert best.task_evaluation.score == max(calls)


@pytest.mark.parametrize("goal_id", ["ttft", "single", "multi"])
def test_helper_only_roundtrip_two_rounds_and_native_direction(tmp_path, goal_id):
    # Given: a real TaskRewriter and numerical resident backend, not a hand-authored winner path.
    from tests.c3_search_fakes import search_case
    session, provider, agent, inputs = search_case(tmp_path, goal_id)
    module = search_module()
    # When
    with session, closing(provider):
        result = module.optimize_goal(session, agent, inputs)
    # Then
    assert len(provider.requests) == 2 and len(result.opportunities) == 2
    assert all(r["goal"] == inputs.goal for r in provider.requests)
    assert all(r["responses"] == [] for r in provider.requests)
    assert all(o.status == "accepted" and o.slots_used <= 8 for o in result.opportunities)
    assert result.selected.evaluation.valid and result.selected.params.values["speed"] >= 19
    assert provider.requests[1]["bundle_document"]["sites"] == [{"site_id": "scale", "replacement_callable": "replace"}]
    assert "helper.py" in provider.requests[1]["bundle_sources"]
    assert result.selected.source_hashes["helper.py"] != result.baseline.source_hashes.get("helper.py")
    assert len([r for r in session.attempts if r["purpose"] == "baseline"]) == 1
    assert session.runner.backend.loads == 1
    session.runner.close()
