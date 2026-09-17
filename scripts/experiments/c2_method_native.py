"""Adapt native retuning and selection without owning a tuning or witness loop."""

from dataclasses import dataclass
from pathlib import Path

from pydantic import TypeAdapter

from kernel_optimizer.config import AppConfig
from kernel_optimizer.control.direct_task import TaskSpace
from kernel_optimizer.control.orchestrator import boundary_knobs_to_expand
from kernel_optimizer.models.core import Backend, ParameterSpace, sha256_text
from kernel_optimizer.paramspace.materializer import extract_defaults
from kernel_optimizer.store.run_store import RunStore
from kernel_optimizer.tuning.stats import TuningStatsAnalyzer
from scripts.experiments.c2_local_inputs import InputError, Shared
from scripts.experiments.c2_method_generation import Generated
from scripts.experiments.c2_method_protocol import Deadline, MethodInputs, Opportunity, SchedulingExpired, Status
from scripts.experiments.c2_retune import RetuneInputs, RetuneResult, retune


def native_status(result: RetuneResult, asked: int) -> Status:
    if result.rejection:
        return "invalid"
    if result.status == "artifact_error":
        return "failed"
    return "complete" if result.status == "complete" and asked == 40 else "censored"


@dataclass(frozen=True, slots=True)
class NativeContext:
    shared: Shared
    cfg: AppConfig
    inputs: MethodInputs
    root: Path
    deadline: Deadline


def tune_opportunity(row: Opportunity, generated: Generated, context: NativeContext) -> Opportunity:
    row = row.model_copy(update={"proposal": generated.proposal, "status": generated.status, "error": generated.error})
    if generated.child is None:
        return row
    try:
        context.deadline.check()
        child = generated.child
        source = child.path.read_text(encoding="utf-8")
        space = ParameterSpace(space_id=f"{row.arm}-space", candidate_id=row.arm,
                               source_sha=sha256_text(source), domains=child.space.params,
                               constraints=child.space.constraints)
        space_path = context.root / f"{row.arm}-space.json"
        space_path.write_text(space.model_dump_json(indent=2), encoding="utf-8")
        output = context.root / row.arm / "retune"
        for helper in context.inputs.helpers:
            target = child.path.parent / helper.resolve().relative_to(context.inputs.reference.resolve().parent)
            data = helper.read_bytes()
            if target.exists() and target.read_bytes() != data:
                raise InputError(f"generated helper conflicts with original helper: {target.name}")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
        helpers = tuple(p for p in child.path.parent.rglob("*.py") if p != child.path)
        spec = RetuneInputs.model_validate({
            "task": context.shared.task, "source": child.path, "space": space_path,
            "reference": context.inputs.reference, "helpers": helpers, "sampler_seed": 0,
            "evaluation_seed": 0, "output": output, "backend": child.backend, "final_blocks": 0,
            "space_expansions_per_candidate": 0,
        })
        cfg = context.cfg.model_copy(deep=True)
        cfg.budgets.wall_clock_hours = context.deadline.remaining() / 3600
        result = retune(spec, cfg)
        events = RunStore.open(output).iter_events()
        published = [ParameterSpace.model_validate(e.payload["space"]) for e in events if e.type == "SPACE_PUBLISHED"]
        if published:
            space_path = context.root / f"{row.arm}-native-space.json"
            space_path.write_text(published[-1].model_dump_json(indent=2), encoding="utf-8")
        snapshots = [e.payload["snapshot"] for e in events if e.type == "TUNING_DONE" and "snapshot" in e.payload]
        asked = int(snapshots[-1].get("asked", 0)) if snapshots else 0
        status = "censored" if context.deadline.remaining() <= 0 else native_status(result, asked)
        row = row.model_copy(update={"space": space_path, "retune": result, "asked": asked,
                                     "status": status, "error": result.rejection.detail if result.rejection else
                                      result.error or (None if status == "complete" else f"retune_{result.status}: asked={asked}")})
        parent_latency = context.shared.parent.latency_ms
        best = result.selected
        if result.status == "complete" and best is not None and parent_latency is not None and best.latency_ms < parent_latency.robust_ms:
            if not published:
                raise InputError("native selected candidate has no published space")
            artifact = output / "report" / "selected.py"
            if extract_defaults(artifact.read_text(encoding="utf-8")) != best.params.values:
                raise InputError("native selected artifact differs from native parameters")
            row = row.model_copy(update={"selected": "child", "artifact": artifact, "params": best.params,
                                         "selected_space": TaskSpace(params=published[-1].domains, constraints=published[-1].constraints),
                                         "backend": TypeAdapter(Backend).validate_python(child.backend)})
        if best is not None and published and status == "complete":
            stats = TuningStatsAnalyzer(context.shared.device).analyze(published[-1], result.trials)
            budgets = context.cfg.budgets
            eligible = bool(boundary_knobs_to_expand(stats, budgets.space_expansion_idle_frac, published[-1],
                min_effect_pct=budgets.min_improvement_pct, max_edge_failure_frac=budgets.max_edge_failure_frac))
            row = row.model_copy(update={"native_expansion_eligible": eligible})
        return row
    except SchedulingExpired as exc:
        return row.model_copy(update={"status": "censored", "error": str(exc)})
    except (OSError, ValueError, RuntimeError) as exc:
        return row.model_copy(update={"status": "failed", "error": f"{type(exc).__name__}: {exc}"})
