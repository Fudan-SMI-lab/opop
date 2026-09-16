"""Unavailable contrasts, budgets and the shared generated-evaluator lifecycle."""

import json
from collections.abc import Iterator
from contextlib import closing, contextmanager
from pathlib import Path
from typing import override

import pytest
from pydantic import JsonValue

from kernel_optimizer.agents.runtime import PromptResult
from kernel_optimizer.config import AppConfig, BudgetConfig
from kernel_optimizer.control.task_rewrite import probe_responses
from kernel_optimizer.control.direct_task import TaskSearch, TaskSpace
from kernel_optimizer.evaluation.task_eval import TaskEvaluator
from kernel_optimizer.models.core import Constraint
from kernel_optimizer.tuning.objective import Objective
from tests.test_v5_direct_resource_response import ResponseWriter, project as project


@pytest.mark.parametrize("mode", ["invalid", "constraint", "budget", "wall", "no_resources"])
def test_unknown_response_never_fabricates_measurements(project: Path, mode: str) -> None:
    # Given: a tuned parent, then a missing measurement or insufficient budget.
    space = TaskSpace.model_validate_json((project / "space.json").read_text())
    with TaskEvaluator(project / "eval.py", "inspect_work") as evaluator:
        search = TaskSearch(evaluator, Objective(direction="minimize"), BudgetConfig(trials_per_space=4))
        parent = search.evaluate_candidate(project / "candidate.py", space, {"data": [1, 2], "resource": "slots"})
        assert parent is not None
        search.resource_metrics = ("slots",)
        search.probe_budget = 2
        if mode == "invalid":
            (project / "candidate.py").write_text(
                "def run(data, width, mode):\n    return ([] if width == 3 else [x*x+7 for x in data]), []\n")
        elif mode == "constraint":
            search.spaces[parent.candidate_id] = space.model_copy(update={
                "constraints": [Constraint(expr="width == 1 and mode == 'plain'", rationale="fixture")],
            })
        elif mode == "budget":
            search.probe_budget = 1
        elif mode == "wall":
            search.started -= search.budgets.wall_clock_hours * 3600 + 1
        else:
            search.resource_metrics = ()
        # When: fresh controlled probing is requested.
        responses = probe_responses(search, parent, {"data": [1, 2], "resource": "slots"})
    # Then: unknown data do not become zero slopes, and no hidden trials enter selection.
    first = responses[0]
    assert search.best() == parent and len(search.trials) == 4
    if mode == "no_resources":
        assert first.delta_j == 4.0 and first.resource_status == "unknown"
    else:
        assert first.delta_j is None and first.resource_slopes == {}
    assert search.probe_calls == (2 if mode in ("invalid", "no_resources") else 0)


class GeneratedWriter(ResponseWriter):
    def __init__(self, source: str) -> None:
        super().__init__()
        self.source = source
        self.builder_calls = 0

    @override
    def prompt(self, session_id: str, text: str, *, model: str, agent: str = "build",
               schema: dict[str, JsonValue] | None = None, directory: Path | None = None,
               system: str | None = None) -> PromptResult:
        assert directory is not None
        if (directory / "task/goal.md").exists():
            self.builder_calls += 1
            (directory / "generated.py").write_text(self.source)
            return PromptResult(text="", session_id=session_id, structured={
                "eval_file": "generated.py", "callable_name": "inspect_work",
                "direction": "minimize", "label": "scratch", "unit": "slots",
            })
        return super().prompt(session_id, text, model=model, agent=agent, schema=schema,
                              directory=directory, system=system)


def test_generated_eval_and_rewriter_share_runtime(project: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Given: a provider which writes both an evaluator and a response-driven child.
    from kernel_optimizer import wiring
    from kernel_optimizer.cli import main

    runtime_type = wiring.Runtime
    clients: list[GeneratedWriter] = []

    @contextmanager
    def local_runtime(cfg: AppConfig, log_dir: Path) -> Iterator[wiring.Runtime]:
        with closing(GeneratedWriter((project / "eval.py").read_text())) as client:
            clients.append(client)
            runtime = runtime_type(cfg, log_dir)
            runtime.client = client
            yield runtime

    monkeypatch.setattr(wiring, "Runtime", local_runtime)
    # When: generated evaluation and one structural round go through the actual parser.
    result = main(["optimize-task", "--project", str(project), "--candidate", "candidate.py",
        "--space", "space.json", "--context", "context.json", "--goal", "minimize scratch; preserve output",
        "--trials", "4", "--rewrite-rounds", "1", "--resource-metric", "slots",
        "--probe-budget", "2", "--output", str(project / "result")])
    # Then: one runtime serves both agents, and generated code computes the child score.
    assert result == 0
    assert len(clients) == 1 and clients[0].builder_calls == 1 and len(clients[0].inputs) == 1
    summary = json.loads((project / "result/summary.json").read_text())
    assert summary["best"]["task_evaluation"]["score"] == -8.0
    assert summary["probe_calls"] == 2


def test_parameter_only_edit_is_refused_by_agent(project: Path) -> None:
    # Given: a provider output that only changes PARAMS, not the candidate structure.
    from kernel_optimizer.agents.sandbox import Sandbox, SandboxFactory
    from kernel_optimizer.agents.task_rewriter import TaskRewriterAgent, TaskRewriteResult
    from kernel_optimizer.config import AgentModuleConfig
    from kernel_optimizer.store.run_store import RunStore

    source = (project / "candidate.py").read_text()
    sandbox = Sandbox(project / "workspace")
    sandbox.write_input("candidate/current.py", "PARAMS = {'width': 1}\n" + source)
    sandbox.write_input("child.py", "PARAMS = {'width': 3}\n" + source)
    with closing(ResponseWriter()) as client:
        agent = TaskRewriterAgent(client, SandboxFactory(project / "sandboxes"),
                                  RunStore(project), AgentModuleConfig(max_retries=0))
        # When: the actual structural agent checks the proposed artifact.
        problem = agent.check_output(TaskRewriteResult(candidate_file="child.py"), sandbox)
    # Then: it is not accepted as a structural rewrite.
    assert problem is not None


def test_wall_deadline_between_endpoints_prevents_second_call(project: Path) -> None:
    # Given: the real evaluator consumes the remaining run budget during endpoint A.
    space = TaskSpace.model_validate_json((project / "space.json").read_text())
    with TaskEvaluator(project / "eval.py", "inspect_work") as evaluator:
        search = TaskSearch(evaluator, Objective(direction="minimize"), BudgetConfig(trials_per_space=4))
        parent = search.evaluate_candidate(project / "candidate.py", space, {"data": [1, 2], "resource": "slots"})
        assert parent is not None
        (project / "deadline.py").write_text(
            "from runpy import run_path\n"
            "def inspect_work(path, params, context):\n"
            "    result, scratch = run_path(str(path))['run'](context['data'], **params)\n"
            "    assert result == [x*x+7 for x in context['data']]\n"
            "    context['search'].started -= context['search'].budgets.wall_clock_hours*3600+1\n"
            "    return float(len(scratch)-8)\n")
        with TaskEvaluator(project / "deadline.py", "inspect_work") as deadline_eval:
            search.evaluator = deadline_eval
            search.probe_budget = 4
            # When: the two-endpoint operation runs without sleeps or wall-clock races.
            responses = probe_responses(search, parent, {"data": [1, 2], "search": search})
    # Then: the measured A remains visible, but B and the slope stay unknown.
    assert search.probe_calls == 1
    assert responses[0].a is not None
    assert responses[0].a.score == -6.0
    assert responses[0].b is None and responses[0].delta_j is None
    assert responses[0].reason == "wall_budget"
