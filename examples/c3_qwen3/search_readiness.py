"""Operator-owned preparation: original oracles, phase profiles and baseline A/A checks."""

from pathlib import Path
from time import time
from collections.abc import Iterator
from contextlib import contextmanager
import json

from pydantic import Field

from kernel_optimizer.evaluation.task_eval import TaskEvaluation
from .manual_data import CALIBRATION_IDS
from .model_binding import load_bundle
from .model_runner import ResidentRunner
from .runner_records import FrozenRecord, GoalId
from .search_bundle import baseline_bundle


class Readiness(FrozenRecord):
    contract: Path
    assets_manifest: Path
    contract_sha256: str
    model_revision: str
    baseline_bundle: Path
    valid: bool = False
    profiles: dict[GoalId, Path] = Field(default_factory=dict)
    oracles: dict[str, Path] = Field(default_factory=dict)
    checks: dict[GoalId, tuple[TaskEvaluation, ...]] = Field(default_factory=dict)
    forward_calls: int = 0
    wall_s: float = 0


@contextmanager
def preparation_cost(runner: ResidentRunner, output: Path, kind: str) -> Iterator[None]:
    started, before = time(), runner.backend.forward_calls
    try:
        yield
    finally:
        with (output / "preparation.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"kind": kind, "started_unix_s": started, "ended_unix_s": time(),
                "forward_calls": runner.backend.forward_calls - before,
                "bundle_sha256": runner.receipt.bundle_sha256}) + "\n")


def prepare_readiness(runner: ResidentRunner, output: Path) -> Readiness:
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    runner.output_dir = output / "raw"
    baseline = baseline_bundle(output / "baseline")
    runner.binding.restore()
    runner.bind(load_bundle(baseline, {}))
    prepared = runner.prepared
    started, before = time(), runner.backend.forward_calls
    report = Readiness(contract=prepared.asset_spec.contract_path, assets_manifest=prepared.asset_spec.assets_manifest,
        contract_sha256=prepared.contract_sha256, model_revision=prepared.asset_spec.revision, baseline_bundle=baseline)
    path = output / "readiness.json"
    path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    with preparation_cost(runner, output, "oracle:calibration"):
        oracles = {"calibration": runner.create_oracles(CALIBRATION_IDS)}
    profiles: dict[GoalId, Path] = {}
    checks: dict[GoalId, tuple[TaskEvaluation, ...]] = {}
    try:
        for goal in prepared.contract.goals:
            with preparation_cost(runner, output, f"profile:{goal.id}"):
                profiles[goal.id] = runner.profile(goal, goal.search_prompt_ids)
            runner.bind(load_bundle(baseline, {}))
            results = []
            for _ in range(2):
                with preparation_cost(runner, output, f"aa-quality:{goal.id}"):
                    quality = runner.quality(CALIBRATION_IDS, oracles["calibration"])
                with preparation_cost(runner, output, f"aa-measurement:{goal.id}"):
                    evaluation = (runner.measure(goal, goal.search_prompt_ids).evaluation if quality.valid else
                                  TaskEvaluation(valid=False, detail=quality.detail or "baseline quality failed"))
                results.append(evaluation)
                if not evaluation.valid:
                    break
            checks[goal.id] = tuple(results)
            if len(results) != 2 or not all(r.valid for r in results):
                break
            for block, ids in enumerate(goal.final_prompt_groups):
                with preparation_cost(runner, output, f"oracle:{goal.id}:{block}"):
                    oracles[f"{goal.id}:{block}"] = runner.create_oracles(ids)
    finally:
        report = report.model_copy(update={"oracles": oracles, "profiles": profiles, "checks": checks,
            "valid": len(checks) == 3 and len(oracles) == 10 and all(len(pair) == 2 and all(r.valid for r in pair) for pair in checks.values()),
            "forward_calls": runner.backend.forward_calls - before, "wall_s": time() - started})
        path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    return report
