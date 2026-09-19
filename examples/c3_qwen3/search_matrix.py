"""Fresh same-runner final matrix; frozen selections never change in response to heldout."""

from collections.abc import Mapping
from pathlib import Path
from time import time
from typing import Literal

from pydantic import Field

from kernel_optimizer.evaluation.task_eval import TaskEvaluation
from kernel_optimizer.models.core import ParamSet
from .manual_data import CALIBRATION_IDS, fingerprint
from .model_binding import Site, load_bundle
from .model_runner import ResidentRunner
from .runner_records import FrozenRecord, GoalId, OracleManifest, RunnerError
from .search_bundle import export_selection
from .search_readiness import Readiness
from .search_records import EvaluationRequest, SearchClock, Selection, SlotBudget
from .search_session import OperatorSession, ProfileDocument

type Row = Literal["baseline", "ttft", "single", "multi"]


class MatrixCell(FrozenRecord):
    row: Row
    goal_id: GoalId
    block: int
    status: Literal["valid", "invalid", "censored"]
    evaluation: TaskEvaluation
    selection_sha256: str | None = None
    execution_bundle_sha256: str | None = None
    params: ParamSet | None = None


class MatrixResult(FrozenRecord):
    clock: SearchClock
    cells: list[MatrixCell] = Field(default_factory=list)
    slots_used: int = 0
    selection_unchanged: bool = True
    preparation_error: str | None = None
    drain_s: float = 0


def expired_matrix(clock: SearchClock, output: Path) -> MatrixResult:
    orders: tuple[tuple[Row, ...], ...] = (("baseline", "ttft", "single", "multi"),
        ("multi", "single", "ttft", "baseline"), ("single", "multi", "baseline", "ttft"))
    goals: tuple[GoalId, ...] = ("ttft", "single", "multi")
    result = MatrixResult(clock=clock, cells=[MatrixCell(row=row, goal_id=goal, block=block, status="censored",
        evaluation=TaskEvaluation(valid=False, detail="final cutoff before model admission"))
        for goal in goals for block, order in enumerate(orders) for row in order])
    output.mkdir(parents=True, exist_ok=False)
    (output / "matrix.json").write_text(result.model_dump_json(indent=2), encoding="utf-8")
    return result


def load_selection(path: Path) -> Selection:
    selected = Selection.model_validate_json(path.read_bytes())
    relocated = path.parent / "bundle.json"
    bundle = load_bundle(relocated, selected.params.values)
    if (bundle.bundle_sha256 != selected.bundle_sha256 or dict(bundle.source_hashes) != selected.source_hashes
            or bundle.params_sha256 != selected.params_sha256 or selected.baseline_fallback != (not bundle.document.sites)):
        raise RunnerError("frozen winner source/params identity mismatch")
    return selected.model_copy(update={"bundle": relocated.resolve()})


def run_matrix(runner: ResidentRunner, ready: Readiness, selections: Mapping[GoalId, Path], *,
               clock: SearchClock, output: Path) -> MatrixResult:
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    if not ready.valid or ready.contract_sha256 != runner.prepared.contract_sha256 or ready.model_revision != runner.prepared.asset_spec.revision:
        raise RunnerError("heldout requires matching valid readiness and frozen model")
    baseline = load_bundle(ready.baseline_bundle, {})
    rows: dict[Row, Selection] = {"baseline": Selection(bundle=baseline.path, bundle_sha256=baseline.bundle_sha256,
        source_hashes=dict(baseline.source_hashes), params=ParamSet(values={}), params_sha256=baseline.params_sha256,
        site_groups={}, evaluation=ready.checks["ttft"][0], contract_sha256=ready.contract_sha256,
        model_revision=ready.model_revision, baseline_fallback=True)}
    errors: dict[str, str] = {}
    pins = {p: fingerprint(p) for p in selections.values() if p.is_file()}
    for goal in ("ttft", "single", "multi"):
        try:
            selected = load_selection(selections[goal])
            if selected.contract_sha256 != ready.contract_sha256 or selected.model_revision != ready.model_revision:
                raise RunnerError("winner belongs to another frozen task/model")
            if (selected.clock is not None and selected.clock != clock
                    or selected.goal_id != goal and not selected.baseline_fallback):
                raise RunnerError("winner belongs to another goal or campaign clock")
            rows[goal] = selected
        except (KeyError, OSError, ValueError, RuntimeError) as exc:
            errors[goal] = str(exc)
    budget = SlotBudget(36, clock.final_deadline_unix_s)
    cells: list[MatrixCell] = []
    preparation_error = None
    execution: dict[str, Path] = {}
    with OperatorSession(runner, runner.prepared.contract.goals[0], ready.oracles["calibration"], output / "evaluation") as session:
        paths = {p for selected in rows.values() for group in selected.site_groups.values() for p in group}
        try:
            if budget.available():
                traced = {str(row["module_path"]) for path in ready.profiles.values()
                    for row in ProfileDocument.model_validate_json(path.read_bytes()).data.trace}
                if not paths <= traced:
                    raise RunnerError("final instrumentation contains an untraced module")
                runner.binding.restore()
                for path in sorted(paths):
                    runner.binding.register_site(path, (path,))
                runner.bind(load_bundle(session.baseline, {}))
                if paths:
                    before = runner.backend.forward_calls
                    fixture = runner.capture_fixtures(CALIBRATION_IDS)
                    session._write("preparation.jsonl", {"kind": "final_selected_site_fixtures", "raw": str(fixture),
                        "forward_calls": runner.backend.forward_calls - before})
            for row, selected in rows.items():
                frozen = export_selection(selected, output / "rows" / row)
                original = load_bundle(frozen.bundle, frozen.params.values)
                sites = tuple(Site(site_id=path, replacement_callable=site.replacement_callable)
                    for site in original.document.sites for path in selected.site_groups[site.site_id])
                resolved = original.document.model_copy(update={"sites": sites})
                path = frozen.bundle.with_name("execution-bundle.json")
                path.write_text(resolved.model_dump_json(indent=2), encoding="utf-8")
                execution[row] = path
        except (OSError, ValueError, RuntimeError) as exc:
            preparation_error = str(exc)
        orders: tuple[tuple[Row, ...], ...] = (("baseline", "ttft", "single", "multi"),
            ("multi", "single", "ttft", "baseline"), ("single", "multi", "baseline", "ttft"))
        for goal in runner.prepared.contract.goals:
            session.goal = goal
            session.set_profile(ready.profiles[goal.id])
            for block, order in enumerate(orders):
                oracle = ready.oracles[f"{goal.id}:{block}"]
                manifest = OracleManifest.model_validate_json(oracle.read_bytes())
                session.frozen.update({oracle: fingerprint(oracle),
                    **{oracle.parent / p.logits_file: p.logits_sha256 for p in manifest.prompts}})
                for row in order:
                    selected = rows.get(row)
                    value = TaskEvaluation(valid=False, detail=errors.get(row) or preparation_error or "final cutoff")
                    status = "censored" if not budget.available() else "invalid"
                    identity = None
                    if selected is not None and preparation_error is None and budget.available():
                        path = execution[row]
                        identity = load_bundle(path, selected.params.values).bundle_sha256
                        value = session._evaluate(EvaluationRequest(bundle=path, params=selected.params,
                            site_groups={p: (p,) for p in paths}), "heldout", structural=False, budget=budget,
                            split="heldout", prompt_ids=goal.final_prompt_groups[block], oracle=oracle)
                        status = "valid" if value.valid else "invalid"
                    cells.append(MatrixCell(row=row, goal_id=goal.id, block=block, status=status, evaluation=value,
                        selection_sha256=selected.bundle_sha256 if selected else None, execution_bundle_sha256=identity,
                        params=selected.params if selected else None))
                    result = MatrixResult(clock=clock, cells=cells, slots_used=budget.used,
                        selection_unchanged=all(fingerprint(p) == sha for p, sha in pins.items()),
                        preparation_error=preparation_error, drain_s=max(0, time() - clock.final_deadline_unix_s) if budget.used else 0)
                    (output / "matrix.json").write_text(result.model_dump_json(indent=2), encoding="utf-8")
    return result
