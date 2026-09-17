"""Typed experiment inputs and common, pre-decision evidence (never credentials)."""

import json
import re
from pathlib import Path
from typing import ClassVar, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

from kernel_optimizer.agents.task_rewriter import TaskRewriteInputs
from kernel_optimizer.conditional.task_response import TaskResponse
from kernel_optimizer.config import AppConfig
from kernel_optimizer.control.direct_task import TaskSpace
from kernel_optimizer.models.core import DeviceLimits, ParameterSpace, TrialRecord, sha256_text
from kernel_optimizer.paramspace.materializer import extract_defaults, materialize
from kernel_optimizer.store.run_store import RunStore
from kernel_optimizer.tuning.objective import Objective
from kernel_optimizer.tuning.stats import TuningStatsAnalyzer
from scripts.experiments.c2_local_costs import Costs


class InputError(ValueError):
    def __init__(self, detail: str) -> None:
        self.detail = detail
        super().__init__(detail)


class Strict(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid")


class Selection(Strict):
    task: str
    historical_run: Path
    parent_id: str
    parent_trial_id: str
    parent_source: Path
    space: Path
    cutoff_seq: int = Field(ge=0)
    state: Literal["off", "conditional_on_treated_state"]
    semantics: Path
    backend: Literal["triton", "cuda"] = "triton"


def protocol(cfg: AppConfig) -> str:
    modules = {name: cfg.agents.module(name).model_dump(mode="json")
               for name in ("analyst", "rewriter", "parameterizer")}
    modules["rewriter"]["n_candidates"] = 1
    source = Path(__file__).resolve().parents[2]
    code = [(p.relative_to(source).as_posix(), sha256_text(p.read_text(encoding="utf-8")))
            for p in sorted([*(source / "src" / "kernel_optimizer").rglob("*.py"),
                             *(source / "scripts" / "experiments").glob("c2_local*.py")])]
    return sha256_text(json.dumps({"seed": cfg.run.seed, "agents": modules, "code": code,
                                  "wall_hours": cfg.budgets.wall_clock_hours,
                                  "evaluation": cfg.evaluation.model_dump(mode="json"),
                                  "agent": cfg.opencode.agent,
                                  "sandbox_extra_config_hash": sha256_text(json.dumps(cfg.opencode.sandbox_extra_config, sort_keys=True)),
                                  "request_timeout_s": cfg.opencode.request_timeout_s,
                                  "memory_abort_frac": cfg.opencode.memory_abort_frac,
                                  "resource_poll_s": cfg.opencode.resource_poll_s,
                                  "idle_abort_frac": cfg.opencode.idle_abort_frac}, sort_keys=True))


class Shared(Strict):
    task: str
    state: Literal["off", "conditional_on_treated_state"]
    source: str
    reference_source: str
    parent: TrialRecord
    space: TaskSpace
    trials: list[TrialRecord]
    semantics: dict[str, JsonValue]
    device: DeviceLimits
    evaluation: dict[str, JsonValue] = Field(default_factory=dict)
    backend: Literal["triton", "cuda"] = "triton"
    resource_metrics: tuple[str, ...] = ("n_regs", "n_spills", "shared_bytes")
    protocol_id: str = ""
    cutoff_seq: int | None = None
    failed_hypotheses: list[dict[str, JsonValue]] = Field(default_factory=list)

    def identity(self) -> str:
        return sha256_text(self.model_dump_json(
            exclude={"failed_hypotheses"} if not self.failed_hypotheses else None))


class Responses(Strict):
    shared_id: str
    responses: list[TaskResponse] = Field(min_length=1)
    probe_calls: int = Field(ge=0)
    costs: Costs

    def validate_for(self, shared: Shared) -> None:
        if self.shared_id != shared.identity() or self.costs.worker_attempts != self.probe_calls:
            raise InputError("response identity or acquisition accounting mismatch")
        if [r.axis for r in self.responses] != [d.name for d in shared.space.params]:
            raise InputError("responses do not cover the declared parent axes in order")
        actual = sum(int(r.a is not None) + int(r.b is not None) for r in self.responses)
        if actual != self.probe_calls:
            raise InputError("response observations differ from acquisition count")
        for response in self.responses:
            if response.candidate_id != shared.parent.candidate_id:
                raise InputError("response belongs to another candidate")
            for endpoint in (response.a_params, response.b_params):
                if endpoint is None:
                    continue
                if set(endpoint.values) != set(shared.parent.params.values):
                    raise InputError("response endpoint keys differ from parent")
                for domain in shared.space.params:
                    value = endpoint.values[domain.name]
                    if value not in domain.choices or (domain.name != response.axis and value != shared.parent.params.values[domain.name]):
                        raise InputError("response changed a fixed partner or left the parent domain")


class RunOptions(Strict):
    arm: Literal["A", "B"]
    path: Literal["direct", "legacy"]
    final_blocks: int = Field(default=3, ge=3)
    acquisition: Responses | None = None

    @model_validator(mode="after")
    def require_treatment(self) -> Self:
        if self.arm == "B" and self.acquisition is None:
            raise InputError("B requires a baked acquisition envelope")
        return self


class HistoryManifest(BaseModel):
    task: str
    config: AppConfig


def prepare(selection: Selection, cfg: AppConfig, reference: Path) -> Shared:
    history = RunStore.open(selection.historical_run)
    events = history.iter_events()
    prefix = [e for e in events if e.seq < selection.cutoff_seq]
    scans = {e.payload["trial_id"] for e in events if e.type == "SCAN_POINT_DONE"}
    trials = [TrialRecord.model_validate(e.payload["trial"]) for e in prefix
              if e.type == "TRIAL_DONE" and e.payload["trial"]["candidate_id"] == selection.parent_id
              and e.payload["trial"]["trial_id"] not in scans
              and not e.payload.get("reused_measurement")]
    parents = [t for t in trials if t.trial_id == selection.parent_trial_id]
    if len(parents) != 1 or parents[0].status != "complete" or parents[0].latency_ms is None:
        raise InputError("parent must be one successful ordinary pre-cutoff trial")
    parent = parents[0]
    source = selection.parent_source.read_text(encoding="utf-8")
    historical = selection.historical_run / "candidates" / selection.parent_id / "trials" / f"{parent.trial_id}.py"
    if sha256_text(source) != sha256_text(historical.read_text(encoding="utf-8")):
        raise InputError("parent source differs from historical tuned trial artifact")
    if extract_defaults(source) != parent.params.values:
        raise InputError("parent source PARAMS differ from selected tuned trial")
    manifest = HistoryManifest.model_validate_json((selection.historical_run / "manifest.json").read_text(encoding="utf-8"))
    mode = manifest.config.v4.conditional_scan.mode
    if manifest.task != reference.stem or manifest.config.evaluation != cfg.evaluation:
        raise InputError("historical task or evaluation differs from current task/config")
    if selection.state == "off" and mode != "off":
        raise InputError("off state requires historical manifest with conditional_scan.mode=off")
    semantics = json.loads(selection.semantics.read_text(encoding="utf-8"))
    historical_semantics = [e.payload.get("semantics", {}) for e in prefix if e.type == "SEMANTICS_PROBED"]
    if historical_semantics and semantics != historical_semantics[-1]:
        raise InputError("semantics differ from the pre-cutoff historical reference probe")
    registered = [e.payload["candidate"] for e in prefix if e.type == "CANDIDATE_REGISTERED"
                  and e.payload["candidate"]["candidate_id"] == selection.parent_id]
    if registered and registered[-1]["backend"] != selection.backend:
        raise InputError("backend differs from the historical parent registration")
    space = TaskSpace.model_validate_json(selection.space.read_text(encoding="utf-8"))
    published = [ParameterSpace.model_validate(e.payload["space"]) for e in prefix
                 if e.type == "SPACE_PUBLISHED" and e.payload["space"]["space_id"] == parent.space_id]
    if len(published) != 1 or published[0].domains != space.params or published[0].constraints != space.constraints:
        raise InputError("supplied space differs from the historical published space")
    if set(parent.params.values) != {d.name for d in space.params}:
        raise InputError("parent space keys differ from tuned parameters")
    if any(parent.params.values[d.name] not in d.choices for d in space.params):
        raise InputError("tuned parent is outside the declared original space")
    safe_trials = [t.model_copy(update={"failure_detail": re.sub(r"(?:[A-Za-z]:[\\/]|/)[^\s\"'<>]+", "<path>", t.failure_detail)}) for t in trials]
    return Shared(task=selection.task, state=selection.state, source=source,
                  reference_source=reference.read_text(encoding="utf-8"), parent=parent,
                  space=space, trials=safe_trials, device=cfg.device, protocol_id=protocol(cfg), cutoff_seq=selection.cutoff_seq,
                  semantics=semantics,
                  evaluation=cfg.evaluation.model_dump(mode="json"), backend=selection.backend)


def stage_inputs(shared: Shared, directory: Path, responses: list[TaskResponse]) -> TaskRewriteInputs:
    directory.mkdir(parents=True, exist_ok=False)
    parameter_space = ParameterSpace(space_id=shared.parent.space_id,
                                    candidate_id=shared.parent.candidate_id,
                                    source_sha=sha256_text(shared.source),
                                    domains=shared.space.params, constraints=shared.space.constraints)
    stats = TuningStatsAnalyzer(shared.device).analyze(parameter_space, shared.trials)
    files = {
        "parent.py": materialize(shared.source, shared.parent.params),
        "reference.py": shared.reference_source,
        "ordinary.json": json.dumps([t.model_dump(mode="json") for t in shared.trials]),
        "stats.json": stats.model_dump_json(indent=2),
        "semantics.json": json.dumps(shared.semantics),
        "device.json": shared.device.model_dump_json(indent=2),
        "evaluation.json": json.dumps(shared.evaluation),
    }
    for name, text in files.items():
        (directory / name).write_text(text, encoding="utf-8")
    profile = shared.parent.profile.model_dump(mode="json") if shared.parent.profile else {}
    resources = {name: float(value) for name, value in profile.items()
                 if name in shared.resource_metrics and type(value) in (int, float)}
    return TaskRewriteInputs(
        project_root=Path("."), candidate_path=Path("parent.py"),
        source_paths=tuple(Path(n) for n in files if n != "parent.py"),
        candidate_id=shared.parent.candidate_id,
        goal="Reduce full-task robust latency in ms; preserve reference outputs and ModelNew interface. "
             "This worker adapter materializes a module-level literal PARAMS dict using the supplied space. "
             "Read project/ordinary.json, stats.json, semantics.json, device.json, evaluation.json and reference.py.",
        context={"task": shared.task, "comparison": "fixed-parent information increment", "state": shared.state},
        objective=Objective(direction="minimize", label="full-task robust latency", unit="ms"),
        params=shared.parent.params.model_copy(deep=True), space=shared.space.model_copy(deep=True),
        resource_metrics=shared.resource_metrics, resources=resources,
        responses=[r.model_copy(deep=True) for r in responses],
    )
