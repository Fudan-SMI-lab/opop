"""Real CPU responses and structural children; only the provider is replaced."""

import json
from collections.abc import Iterator
from contextlib import closing, contextmanager
from pathlib import Path
from typing import override

import pytest
from pydantic import JsonValue

from kernel_optimizer.agents.runtime import OpencodeClient, PromptResult
from kernel_optimizer.config import AppConfig, BudgetConfig
from kernel_optimizer.control.direct_task import TaskSearch, TaskSpace
from kernel_optimizer.evaluation.task_eval import TaskEvaluator
from kernel_optimizer.models.core import ParamSet
from kernel_optimizer.tuning.objective import Objective


@pytest.fixture
def project(tmp_path: Path) -> Path:
    (tmp_path / "candidate.py").write_text(
        "def run(data, width, mode):\n"
        "    scratch = [0] * (width * len(data))\n"
        "    return [x*x+7 for x in data], scratch\n", encoding="utf-8")
    (tmp_path / "eval.py").write_text(
        "from runpy import run_path\n"
        "from kernel_optimizer.evaluation.task_eval import TaskEvaluation\n"
        "calls = 0\n"
        "def inspect_work(path, params, context):\n"
        "    global calls\n"
        "    calls += 1\n"
        "    out, scratch = run_path(str(path))['run'](context['data'], **params)\n"
        "    if out != [x*x+7 for x in context['data']]:\n"
        "        return TaskEvaluation(valid=False, detail='wrong output')\n"
        "    return TaskEvaluation(score=float(len(scratch)-8), metrics={\n"
        "        context['resource']: float(len(scratch)), 'calls': float(calls), 'constant': 2.0})\n",
        encoding="utf-8")
    (tmp_path / "space.json").write_text(json.dumps({"params": [
        {"name": "width", "kind": "int", "choices": [1, 3]},
        {"name": "mode", "kind": "str", "choices": ["plain", "other"]},
    ]}), encoding="utf-8")
    (tmp_path / "context.json").write_text(json.dumps({"data": [1, -2, 3, 4], "resource": "slots"}))
    return tmp_path


@pytest.mark.parametrize("direction", ["minimize", "maximize"])
def test_fresh_controlled_responses_when_native_scores(project: Path, direction: str) -> None:
    # Given: parameter search has already evaluated the endpoints.
    from kernel_optimizer.control.task_rewrite import probe_responses

    space = TaskSpace.model_validate_json((project / "space.json").read_text())
    with TaskEvaluator(project / "eval.py", "inspect_work") as evaluator:
        search = TaskSearch(evaluator, Objective.model_validate({"direction": direction}),
                            BudgetConfig(trials_per_space=4))
        parent = search.evaluate_candidate(project / "candidate.py", space, {"data": [1, 2, 3, 4], "resource": "slots"})
        assert parent is not None
        search.resource_metrics = ("slots", "constant", "missing")
        search.probe_budget = 4
        # When: each axis is remeasured with partners held at the incumbent.
        responses = probe_responses(search, parent, {"data": [1, 2, 3, 4], "resource": "slots"})
    # Then: numeric distance is two, nominal distance is absent, and no resource is invented.
    numeric, nominal = responses
    assert numeric.a is not None and numeric.b is not None
    assert numeric.b_params is not None and nominal.b_params is not None
    assert numeric.delta_j == 8.0
    assert numeric.parameter_slope == 4.0
    assert numeric.gain == (-8.0 if direction == "minimize" else 8.0)
    assert numeric.resource_slopes == {"slots": 1.0}
    assert numeric.resource_deltas == {"slots": 8.0, "constant": 0.0}
    assert numeric.unknown_resources == ["missing"]
    assert numeric.a_params.values["mode"] == numeric.b_params.values["mode"]
    assert nominal.parameter_slope is None
    assert nominal.a_params.values["width"] == nominal.b_params.values["width"]
    assert numeric.a.metrics["calls"] == 5.0 and numeric.b.metrics["calls"] == 6.0
    assert search.probe_calls == 4
    assert len(search.trials) == 4
    assert search.best() == parent


class ResponseWriter(OpencodeClient):
    def __init__(self, bad: bool = False) -> None:
        super().__init__("http://127.0.0.1:0")
        self.bad = bad
        self.inputs: list[dict[str, JsonValue]] = []

    @override
    def create_session(self, directory: Path, title: str) -> str:
        return "local"

    @override
    def prompt(self, session_id: str, text: str, *, model: str, agent: str = "build",
               schema: dict[str, JsonValue] | None = None, directory: Path | None = None,
               system: str | None = None) -> PromptResult:
        assert directory is not None
        request = json.loads((directory / "analysis/task_response.json").read_text())
        self.inputs.append(request)
        measured = any(r["resource_slopes"] for r in request["responses"])
        maximize = request["objective"]["direction"] == "maximize"
        scratch = "list(range(32))" if maximize and measured else "[]"
        output = "[]" if self.bad else "list(map(lambda x: x*x+7, data))"
        (directory / "child.py").write_text(
            f"def run(data, width, mode):\n    return {output}, {scratch}\n", encoding="utf-8")
        return PromptResult(text="", session_id=session_id, structured={
            "candidate_file": "child.py", "space": {"params": [
                {"name": "width", "kind": "int", "choices": [2, 4]},
                {"name": "mode", "kind": "str", "choices": ["plain", "other"]},
            ]},
        })


@pytest.mark.parametrize(("direction", "bad", "budget", "resource", "expected"), [
    ("minimize", False, 4, "slots", -8.0),
    ("maximize", False, 4, "unseen_allocations", 24.0),
    ("minimize", True, 4, "slots", -4.0),
    ("minimize", False, 0, "slots", -8.0),
])
def test_cli_rewrites_from_actual_responses(project: Path, monkeypatch: pytest.MonkeyPatch,
                                          direction: str, bad: bool, budget: int,
                                          resource: str, expected: float) -> None:
    # Given: the existing runtime/provider seam writes a real child from structured observations.
    from kernel_optimizer import wiring
    from kernel_optimizer.cli import main

    runtime_type = wiring.Runtime
    clients: list[ResponseWriter] = []

    @contextmanager
    def local_runtime(cfg: AppConfig, log_dir: Path) -> Iterator[wiring.Runtime]:
        with closing(ResponseWriter(bad)) as client:
            clients.append(client)
            runtime = runtime_type(cfg, log_dir)
            runtime.client = client
            yield runtime

    monkeypatch.setattr(wiring, "Runtime", local_runtime)
    (project / "context.json").write_text(json.dumps({"data": [1, -2, 3, 4], "resource": resource}))
    # When: the real parser executes TPE -> fresh response -> agent -> child TPE.
    code = main(["optimize-task", "--project", str(project), "--candidate", "candidate.py",
        "--space", "space.json", "--eval-file", "eval.py", "--eval-function", "inspect_work",
        "--context", "context.json", "--direction", direction, "--trials", "4",
        "--goal", "preserve x squared plus seven; optimize scratch allocation",
        "--rewrite-rounds", "1", "--probe-budget", str(budget), "--resource-metric", resource,
        "--output", str(project / "result")])
    # Then: quality-checked child execution, not provider claims, decides the winner.
    assert code == 0
    summary = json.loads((project / "result/summary.json").read_text())
    assert summary["best"]["task_evaluation"]["score"] == expected
    assert len(clients) == 1 and len(clients[0].inputs) == 1
    assert summary["probe_calls"] == budget
    assert len(summary["rewrites"]) == 1
    winner = Path(summary["best_candidate"])
    assert (winner == project / "candidate.py") == bad
    if not bad:
        assert summary["best"]["params"]["values"]["width"] in (2, 4)
    with TaskEvaluator(project / "eval.py", "inspect_work") as evaluator:
        actual = evaluator.evaluate(winner, ParamSet.model_validate(summary["best"]["params"]).values,
                                    {"data": [7, 8, 9, 10], "resource": resource})
    assert actual.valid and actual.score == expected
