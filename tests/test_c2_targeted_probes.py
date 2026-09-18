"""Targeted requests through a real CPU evaluator, without historical-score reuse."""

import importlib
from pathlib import Path

import pytest

from kernel_optimizer.config import BudgetConfig
from kernel_optimizer.control.direct_task import TaskSearch, TaskSpace
from kernel_optimizer.evaluation.task_eval import TaskEvaluator
from kernel_optimizer.models.core import ParamSet, TrialRecord
from kernel_optimizer.models.reports import ProbeRequest
from kernel_optimizer.tuning.objective import Objective


@pytest.fixture
def probe_case(tmp_path: Path):
    (tmp_path / "candidate.py").write_text("PARAMS = {}\n", encoding="utf-8")
    (tmp_path / "evaluate.py").write_text(
        "from kernel_optimizer.evaluation.task_eval import TaskEvaluation\n"
        "def evaluate(path, params, context):\n"
        "    context['calls'].append(dict(params))\n"
        "    return TaskEvaluation(valid=False, detail='failure') if params['early'] == 4 else "
        "TaskEvaluation(score=float(params['late']), metrics={'slots': float(params['early'])})\n",
        encoding="utf-8")
    space = TaskSpace.model_validate({"params": [
        {"name": "early", "kind": "int", "choices": [0, 2, 4]},
        {"name": "late", "kind": "int", "choices": [1, 3, 5]},
        {"name": "mode", "kind": "str", "choices": ["a", "b"]},
    ], "constraints": [{"expr": "early + late <= 8"}]})
    parent = TrialRecord(trial_id="p", candidate_id="p", space_id="s", status="complete",
                         params=ParamSet(values={"early": 2, "late": 3, "mode": "a"}))
    return tmp_path, space, parent


def execute(case, requests, budget=12, allowed=lambda: True):
    module = importlib.import_module("kernel_optimizer.control.targeted_probes")
    path, space, parent = case
    calls, snapshots = [], []
    with TaskEvaluator(path / "evaluate.py", "evaluate") as evaluator:
        search = TaskSearch(evaluator, Objective(direction="minimize"), BudgetConfig())
        search.paths["p"], search.spaces["p"] = path / "candidate.py", space
        search.probe_budget = budget
        evidence = module.targeted_probes(search, parent, module.ProbeRun(
            requests=tuple(requests), context={"calls": calls}, allowed=allowed, checkpoint=snapshots.append))
    return evidence, calls, snapshots


def test_late_interior_and_partners_precede_fallback(probe_case):
    # Given: a late axis, interior endpoint, and a non-default fixed partner.
    request = ProbeRequest(axis="late", a_value=1, b_value=3, partners={"early": 0})
    # When
    evidence, calls, _ = execute(probe_case, [request])
    # Then
    assert calls[:2] == [{"early": 0, "late": 1, "mode": "a"}, {"early": 0, "late": 3, "mode": "a"}]
    assert evidence.contrasts[0].fixed.values == {"early": 0, "mode": "a"}
    assert evidence.contrasts[0].response.parameter_slope == 1.0
    assert len(calls) == len(evidence.observations) <= 12


def test_duplicate_endpoints_and_failures_count_once(probe_case):
    # Given: repeated contrasts with one failing endpoint.
    request = ProbeRequest(axis="early", a_value=0, b_value=4)
    # When
    evidence, calls, _ = execute(probe_case, [request, request], budget=3)
    # Then
    assert len(calls) == 3 and len({tuple(sorted(p.items())) for p in calls}) == 3
    assert evidence.contrasts[0].response == evidence.contrasts[1].response
    assert evidence.contrasts[0].response.b.valid is False
    assert evidence.contrasts[0].response.delta_j is None


@pytest.mark.parametrize("payload,reason", [
    ({"axis": "missing", "a_value": 1, "b_value": 3}, "unknown_axis"),
    ({"axis": "late", "a_value": 2, "b_value": 3}, "not_in_choices"),
    ({"axis": "late", "a_value": 1, "b_value": 3, "partners": {"missing": 0}}, "unknown_param"),
    ({"axis": "late", "a_value": 1, "b_value": 5, "partners": {"early": 4}}, "constraint_violated"),
    ({"axis": "late", "a_value": 3, "b_value": 3}, "identical_endpoints"),
])
def test_invalid_request_is_recorded_without_execution(probe_case, payload, reason):
    # Given / When
    evidence, calls, _ = execute(probe_case, [ProbeRequest.model_validate(payload)], budget=2)
    # Then: only the existing first-axis fallback runs.
    assert evidence.contrasts[0].rejection == reason
    assert evidence.contrasts[0].response is None
    assert [p["early"] for p in calls] == [0, 4]


def test_nominal_contrast_has_no_numeric_slope(probe_case):
    # Given / When
    evidence, _, _ = execute(probe_case, [ProbeRequest(axis="mode", a_value="a", b_value="b")], budget=2)
    # Then
    response = evidence.contrasts[0].response
    assert response.delta_j == 0.0 and response.parameter_slope is None


def test_cutoff_preserves_partial_contrast(probe_case):
    # Given: admit precisely one endpoint.
    admissions = []
    def allowed():
        admissions.append(True)
        return len(admissions) == 1
    # When
    evidence, calls, snapshots = execute(probe_case, [ProbeRequest(axis="late", a_value=1, b_value=3)], allowed=allowed)
    # Then
    assert len(calls) == 1 and len(evidence.observations) == 1
    assert evidence.contrasts[0].response.b is None
    assert evidence.contrasts[0].response.reason == "cutoff"
    assert snapshots[-1] == evidence


def test_interruption_keeps_completed_and_inflight_attempts(probe_case):
    # Given: the actual evaluation callback interrupts on the second endpoint.
    path, space, parent = probe_case
    (path / "evaluate.py").write_text(
        "from kernel_optimizer.evaluation.task_eval import TaskEvaluation\n"
        "def evaluate(path, params, context):\n"
        "    if params['late'] == 3: raise KeyboardInterrupt()\n"
        "    return TaskEvaluation(score=1.0)\n", encoding="utf-8")
    module = importlib.import_module("kernel_optimizer.control.targeted_probes")
    snapshots = []
    with TaskEvaluator(path / "evaluate.py", "evaluate") as evaluator:
        search = TaskSearch(evaluator, Objective(direction="minimize"), BudgetConfig())
        search.paths["p"], search.spaces["p"] = path / "candidate.py", space
        search.probe_budget = 12
        # When / Then
        with pytest.raises(KeyboardInterrupt):
            module.targeted_probes(search, parent, module.ProbeRun(
                requests=(ProbeRequest(axis="late", a_value=1, b_value=3),), context={},
                allowed=lambda: True, checkpoint=snapshots.append))
    assert search.probe_calls == 2
    assert snapshots[-1].observations[0].evaluation.score == 1.0
    assert snapshots[-1].observations[1].evaluation is None
    assert snapshots[-1].contrasts[0].response.a.score == 1.0


def test_budget_is_capped_at_twelve_even_if_search_allows_more(probe_case):
    # Given: more than twelve unique fallback endpoints.
    path, _, parent = probe_case
    values = {"early": 2, "late": 2, "mode": "a", **{f"extra{i}": 2 for i in range(8)}}
    space = TaskSpace.model_validate({"params": [
        {"name": name, "kind": "str" if name == "mode" else "int",
         "choices": ["a", "b"] if name == "mode" else [0, 2, 4]} for name in values]})
    parent = parent.model_copy(update={"params": ParamSet(values=values)})
    # When
    evidence, calls, _ = execute((path, space, parent), [], budget=99)
    # Then
    assert len(calls) == len(evidence.observations) == 12
