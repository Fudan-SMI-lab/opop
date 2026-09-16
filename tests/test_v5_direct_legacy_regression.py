"""Default no-agent execution and legacy latency-ranking compatibility."""

import json
from pathlib import Path

import pytest

from kernel_optimizer import wiring
from kernel_optimizer.cli import main
from kernel_optimizer.config import AppConfig
from kernel_optimizer.evaluation.task_eval import TaskEvaluator
from kernel_optimizer.models.core import LatencyStats, ParamDomain, ParameterSpace, ParamSet, TrialRecord
from kernel_optimizer.reporting.report import _reconstruct_summary
from kernel_optimizer.tuning.tpe import OptunaTPETuner
from tests.test_v5_direct_end_to_end import aurora as aurora


@pytest.mark.parametrize(("direction", "choices", "expected"), [
    ("minimize", [1, 4], -3.0), ("maximize", [1, 4], 6.0), ("minimize", [2], 0.0),
])
def test_provided_default_executes_final_without_agent(
    aurora: Path, monkeypatch: pytest.MonkeyPatch, direction: str, choices: list[int], expected: float,
) -> None:
    # Given: CPU evaluation only; starting any agent Runtime is an error.
    def forbidden_runtime(cfg: AppConfig, log_dir: Path) -> wiring.Runtime:
        raise AssertionError("provided parameter-only execution must not start Runtime")

    monkeypatch.setattr(wiring, "Runtime", forbidden_runtime)
    (aurora / "space.json").write_text(json.dumps({"params": [
        {"name": "copies", "kind": "int", "choices": choices},
    ]}))
    # When: the default command tunes and finalizes, including a zero native J.
    code = main(["optimize-task", "--project", str(aurora), "--candidate", "aurora.py",
        "--space", "space.json", "--context", "context.json", "--eval-file", "judge.py",
        "--eval-function", "measure_aurora", "--direction", direction,
        "--trials", str(len(choices)), "--output", str(aurora / "result")])
    # Then: all calls execute real candidates and final execution is not a tuning trial.
    summary = json.loads((aurora / "result/summary.json").read_text())
    assert code == 0
    assert summary["final_execution"]["task_evaluation"]["score"] == expected
    assert summary["final_execution"]["latency_ms"] is None
    assert len((aurora / "calls.jsonl").read_text().splitlines()) == len(choices) + 1
    assert summary["valid_count"] == len(choices) and summary["probe_calls"] == 0


def test_latency_default_and_legacy_report_keep_robust_ranking(aurora: Path) -> None:
    # Given: actual CPU evaluation plus fixed millisecond summaries for estimator characterization.
    space = ParameterSpace(space_id="s", candidate_id="c", source_sha="", domains=[
        ParamDomain(name="copies", kind="int", choices=[1, 4]),
    ])
    tuner = OptunaTPETuner(space, lambda _: True, 2, anchors=(
        ParamSet(values={"copies": 4}), ParamSet(values={"copies": 1}),
    ))
    records: list[TrialRecord] = []
    with TaskEvaluator(aurora / "judge.py", "measure_aurora") as evaluator:
        # When: legacy TPE consumes records whose mean and robust latency disagree.
        for mean, median in ((2.0, 0.4), (0.8, None)):
            asked = tuner.ask()
            assert asked is not None
            trial_id, params = asked
            actual = evaluator.evaluate(aurora / "aurora.py", dict(params.values), {
                "items": [1, 2, 3], "resource": "aurora_transient_cells", "trace": str(aurora / "calls.jsonl"),
            })
            assert actual.valid
            record = TrialRecord(trial_id=trial_id, candidate_id="c", space_id="s", params=params,
                status="complete", task_evaluation=actual, latency_ms=LatencyStats(
                    mean=mean, median=median, min=0.1, max=5.0, std=1.0, n_samples=5))
            records.append(record)
            tuner.tell(trial_id, record)
    # Then: default min latency ignores native J; absent median falls back to mean in the report too.
    assert tuner.best() == records[0]
    assert tuner.study.best_value == 0.4
    assert tuner.study.trials[1].value == 0.8
    summary = _reconstruct_summary([], {}, [r.model_dump(mode="json") for r in records])
    assert summary["best"]["params"]["values"] == {"copies": 4}
    assert summary["best"]["tuned_ms"] == 0.4
