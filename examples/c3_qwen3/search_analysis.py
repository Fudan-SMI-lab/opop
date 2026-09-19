"""Descriptive fresh-cell analysis and explicit private-GPU audit status, without reselection."""

from pathlib import Path
from statistics import median

from pydantic import Field, TypeAdapter

from .runner_records import FrozenRecord
from .search_matrix import MatrixResult


class ToolUse(FrozenRecord):
    session_id: str
    command: str
    gpu_execution: bool
    via_resident_helper: bool


class CellSummary(FrozenRecord):
    row: str
    goal_id: str
    values: list[float]
    median: float | None = None
    improvement_pct: float | None = None


class Analysis(FrozenRecord):
    complete: bool
    cells: list[CellSummary]
    private_gpu_audit: str
    protocol_violations: list[str] = Field(default_factory=list)
    selection_unchanged: bool


def analyze_matrix(path: Path, usage_audit: Path | None = None) -> Analysis:
    matrix = MatrixResult.model_validate_json(path.read_bytes())
    summaries = []
    for goal in ("ttft", "single", "multi"):
        baseline = [c.evaluation.score for c in matrix.cells if c.row == "baseline" and c.goal_id == goal
                    and c.evaluation.valid and c.evaluation.score is not None]
        for row in ("baseline", "ttft", "single", "multi"):
            values = [c.evaluation.score for c in matrix.cells if c.row == row and c.goal_id == goal
                      and c.evaluation.valid and c.evaluation.score is not None]
            value = median(values) if len(values) == 3 else None
            improvement = None
            if len(baseline) == 3 and value is not None:
                base = median(baseline)
                improvement = 100 * ((base - value) if goal == "ttft" else (value - base)) / base
            summaries.append(CellSummary(row=row, goal_id=goal, values=values, median=value, improvement_pct=improvement))
    violations = []
    if usage_audit is not None:
        uses = TypeAdapter(list[ToolUse]).validate_json(usage_audit.read_bytes())
        violations = [f"{use.session_id}: private GPU bypass: {use.command}" for use in uses
                      if use.gpu_execution and not use.via_resident_helper]
    identities = {(c.row, c.goal_id, c.block) for c in matrix.cells}
    return Analysis(complete=len(matrix.cells) == len(identities) == 36 and all(c.evaluation.valid for c in matrix.cells),
        cells=summaries, private_gpu_audit="reviewed" if usage_audit is not None else "pending",
        protocol_violations=violations, selection_unchanged=matrix.selection_unchanged)
