"""Agent-facing views of one validated acquisition; ordinary evidence is untouched."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal, assert_never

from kernel_optimizer.conditional.task_response import TaskResponse
from kernel_optimizer.config import AppConfig
from kernel_optimizer.models.core import ParamSet, TrialRecord
from scripts.experiments.c2_local_costs import Costs
from scripts.experiments.c2_local_inputs import Responses, Shared, Strict
from scripts.experiments.c2_retune import RetuneResult

if TYPE_CHECKING:
    from scripts.experiments.c2_method_protocol import Deadline

Group = Literal["G0", "G1", "G2", "H"]


@dataclass(frozen=True, slots=True)
class InformationRun:
    cfg: AppConfig
    group: Group
    run_dir: Path
    sampler_seed: int = 0
    deadline: Deadline | None = None
    helpers: tuple[Path, ...] = ()
    helper_root: Path | None = None
    require_b40: bool = False


class InformationResult(Strict):
    group: Group
    state: str
    selected: Literal["parent", "child"]
    selected_artifact: str
    parent_baseline_ms: float
    parent_params: ParamSet
    parent_finals: list[TrialRecord]
    child: RetuneResult | None
    error: str | None
    generation_costs: Costs
    acquisition_costs: Costs
    parent_costs: Costs
    wall_s: float
    status: Literal["valid", "failed", "censored"] = "valid"
    asked: int | None = None


def group_responses(shared: Shared, acquisition: Responses, group: Group) -> list[TaskResponse]:
    acquisition.validate_for(shared)
    match group:
        case "G0":
            return []
        case "G1":
            return [response.model_copy(deep=True, update={
                "a": response.a.model_copy(deep=True, update={"metrics": {}, "detail": None}) if response.a else None,
                "b": response.b.model_copy(deep=True, update={"metrics": {}, "detail": None}) if response.b else None,
                "resource_deltas": {}, "resource_slopes": {}, "unknown_resources": [],
                "resource_status": "unknown", "reason": "endpoint_resource_information_withheld",
            }) for response in acquisition.responses]
        case "G2" | "H":
            return [response.model_copy(deep=True) for response in acquisition.responses]
        case unreachable:
            assert_never(unreachable)
