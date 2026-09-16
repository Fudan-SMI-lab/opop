"""Full direct CLI execution; fake provider, real evaluator and candidate programs."""

import json
import subprocess
import sys
from collections.abc import Iterator
from contextlib import closing, contextmanager
from pathlib import Path
from typing import Literal, override

import pytest
from pydantic import JsonValue

from kernel_optimizer import wiring
from kernel_optimizer.agents.runtime import OpencodeClient, PromptResult
from kernel_optimizer.cli import main
from kernel_optimizer.config import AppConfig


@pytest.fixture
def aurora(tmp_path: Path) -> Path:
    (tmp_path / "aurora.py").write_text(
        "def transform(items, copies):\n"
        "    scratch = list(range(copies * len(items)))\n"
        "    return [x*3-2 for x in items], scratch\n", encoding="utf-8")
    (tmp_path / "judge.py").write_text(
        "import json\nfrom pathlib import Path\nfrom runpy import run_path\n"
        "from kernel_optimizer.evaluation.task_eval import TaskEvaluation\n"
        "def measure_aurora(path, params, context):\n"
        "    output, scratch = run_path(str(path))['transform'](context['items'], **params)\n"
        "    valid = output == [x*3-2 for x in context['items']]\n"
        "    with Path(context['trace']).open('a', encoding='utf-8') as stream:\n"
        "        stream.write(json.dumps({'path': str(path), 'params': dict(params), 'output': output})+'\\n')\n"
        "    return TaskEvaluation(score=float(len(scratch)-6), valid=valid,\n"
        "                          metrics={context['resource']: float(len(scratch))})\n",
        encoding="utf-8")
    (tmp_path / "space.json").write_text(json.dumps({"params": [
        {"name": "copies", "kind": "int", "choices": [1, 4]},
    ]}), encoding="utf-8")
    (tmp_path / "context.json").write_text(json.dumps({
        "items": [1, 2, 3], "resource": "aurora_transient_cells", "trace": str(tmp_path / "calls.jsonl"),
    }), encoding="utf-8")
    return tmp_path


class AuroraProvider(OpencodeClient):
    def __init__(self, source: str, direction: Literal["minimize", "maximize"]) -> None:
        super().__init__("http://127.0.0.1:0")
        self.source = source
        self.direction = direction
        self.builder_calls = 0
        self.rewriter_calls = 0

    @override
    def create_session(self, directory: Path, title: str) -> str:
        return "cpu-provider"

    @override
    def prompt(self, session_id: str, text: str, *, model: str, agent: str = "build",
               schema: dict[str, JsonValue] | None = None, directory: Path | None = None,
               system: str | None = None) -> PromptResult:
        assert directory is not None
        if (directory / "task/goal.md").exists():
            self.builder_calls += 1
            (directory / "generated_judge.py").write_text(self.source, encoding="utf-8")
            return PromptResult(text="", session_id=session_id, structured={
                "eval_file": "generated_judge.py", "callable_name": "measure_aurora",
                "direction": self.direction, "label": "aurora native delta", "unit": "cells",
            })
        self.rewriter_calls += 1
        request = json.loads((directory / "analysis/task_response.json").read_text())
        response = request["responses"][0]
        assert response["parameter_slope"] == 3.0
        assert response["resource_slopes"] == {"aurora_transient_cells": 1.0}
        scratch = "range((copies+5)*len(items))" if request["objective"]["direction"] == "maximize" else "()"
        (directory / "aurora_child.py").write_text(
            f"def transform(items, copies):\n    return list(map(lambda x: 3*x-2, items)), {scratch}\n",
            encoding="utf-8")
        return PromptResult(text="", session_id=session_id, structured={"candidate_file": "aurora_child.py"})


@pytest.mark.parametrize("generated", [False, True])
@pytest.mark.parametrize("direction", ["minimize", "maximize"])
def test_complete_chain_executes_winner_after_rewrite(
    aurora: Path, monkeypatch: pytest.MonkeyPatch, generated: bool,
    direction: Literal["minimize", "maximize"],
) -> None:
    # Given: an unseen task/callable/resource and only a fake model-provider boundary.
    runtime_type = wiring.Runtime
    providers: list[AuroraProvider] = []
    calls_at_close: list[int] = []

    @contextmanager
    def local_runtime(cfg: AppConfig, log_dir: Path) -> Iterator[wiring.Runtime]:
        with closing(AuroraProvider((aurora / "judge.py").read_text(), direction)) as client:
            providers.append(client)
            runtime = runtime_type(cfg, log_dir)
            runtime.client = client
            yield runtime
            calls_at_close.append(len((aurora / "calls.jsonl").read_text().splitlines()))

    monkeypatch.setattr(wiring, "Runtime", local_runtime)
    args = ["optimize-task", "--project", str(aurora), "--candidate", "aurora.py",
            "--space", "space.json", "--context", "context.json", "--direction", direction,
            "--goal", "preserve the complete affine output; optimize transient cells",
            "--trials", "2", "--rewrite-rounds", "1", "--probe-budget", "2",
            "--resource-metric", "aurora_transient_cells", "--output", str(aurora / "result")]
    if not generated:
        args.extend(["--eval-file", "judge.py", "--eval-function", "measure_aurora"])
    # When: the real CLI finishes the entire direct method.
    code = main(args)
    # Then: the seventh actual candidate call executes the selected child after all search calls.
    calls = [json.loads(line) for line in (aurora / "calls.jsonl").read_text().splitlines()]
    assert code == 0
    assert len(calls) == 7 and calls_at_close == [7]
    summary = json.loads((aurora / "result/summary.json").read_text())
    final = summary["final_execution"]
    assert final["candidate_id"] == summary["best"]["candidate_id"]
    assert final["params"] == summary["best"]["params"]
    assert final["status"] == "complete"
    assert final["task_evaluation"]["score"] == (-6.0 if direction == "minimize" else 21.0)
    assert calls[-1] == {"path": summary["best_candidate"],
                         "params": final["params"]["values"], "output": [1, 4, 7]}
    assert len(providers) == 1
    assert providers[0].builder_calls == int(generated) and providers[0].rewriter_calls == 1
    assert summary["valid_count"] == 4 and summary["probe_calls"] == 2


@pytest.mark.parametrize("failure", ["wrong_output", "nan_score", "missing_score"])
def test_cli_fails_when_final_execution_is_invalid(aurora: Path, failure: str) -> None:
    # Given: a real candidate which only stops satisfying the contract on its final call.
    (aurora / "aurora.py").write_text(
        "from pathlib import Path\n"
        "def transform(items, copies):\n"
        "    marker = Path(__file__).with_suffix('.calls')\n"
        "    calls = int(marker.read_text())+1 if marker.exists() else 1\n"
        "    marker.write_text(str(calls))\n"
        "    return ([] if calls == 3 else [x*3-2 for x in items]), list(range(copies*len(items)))\n")
    if failure != "wrong_output":
        judge = (aurora / "judge.py").read_text()
        value = "float('nan')" if failure == "nan_score" else "None"
        (aurora / "judge.py").write_text(judge.replace(
            "return TaskEvaluation(score=float(len(scratch)-6), valid=valid,",
            f"return TaskEvaluation(score=(float(len(scratch)-6) if valid else {value}), valid=True,",
        ))
    # When: parameter-only optimization finishes through the actual process entry point.
    result = subprocess.run([sys.executable, "-m", "kernel_optimizer.cli", "optimize-task",
        "--project", str(aurora), "--candidate", "aurora.py", "--space", "space.json",
        "--context", "context.json", "--eval-file", "judge.py", "--eval-function", "measure_aurora",
        "--direction", "minimize", "--trials", "2", "--output", str(aurora / "result")],
        capture_output=True, text=True, timeout=30)
    # Then: historical best remains distinct from the failed final execution; no false success.
    assert result.returncode == 1, result.stdout + result.stderr
    summary = json.loads((aurora / "result/summary.json").read_text())
    assert summary["best"]["task_evaluation"]["valid"]
    assert summary["final_execution"]["status"] == "fail"
    assert not summary["final_execution"]["task_evaluation"]["valid"]
    assert len((aurora / "calls.jsonl").read_text().splitlines()) == 3
