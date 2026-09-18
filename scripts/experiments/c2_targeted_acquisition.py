"""Persist request-first acquisition checkpoints around the existing formal evaluator."""

from dataclasses import dataclass
from pathlib import Path

from kernel_optimizer.config import AppConfig
from kernel_optimizer.control.direct_task import TaskSearch
from kernel_optimizer.control.targeted_probes import ProbeEvidence, ProbeRun, targeted_probes
from kernel_optimizer.evaluation.task_eval import TaskEvaluator
from kernel_optimizer.models.reports import BottleneckReport
from kernel_optimizer.store.run_store import RunStore
from scripts.experiments import c2_local_adapter
from scripts.experiments.c2_local_adapter import GpuAdapter
from scripts.experiments.c2_local_costs import costs
from scripts.experiments.c2_local_inputs import Shared, stage_inputs
from scripts.experiments.c2_method_protocol import Deadline
from scripts.experiments.c2_targeted_inputs import TargetedResponses


@dataclass(frozen=True, slots=True)
class TargetedRun:
    cfg: AppConfig
    store: RunStore
    report: BottleneckReport
    deadline: Deadline | None = None


def acquire_targeted(shared: Shared, run: TargetedRun) -> TargetedResponses:
    store = run.store
    (store.run_dir / "preliminary-report.json").write_text(run.report.model_dump_json(indent=2), encoding="utf-8")
    (store.run_dir / "shared.json").write_text(shared.model_dump_json(indent=2), encoding="utf-8")
    directory = store.run_dir / "common"
    inputs = stage_inputs(shared, directory, [])
    adapter = GpuAdapter(shared, run.cfg, store)
    adapter.phase = "response_acquisition"
    latest = ProbeEvidence()

    def envelope(evidence: ProbeEvidence, *, complete: bool = False) -> TargetedResponses:
        return TargetedResponses(shared_id=shared.identity(), source=shared.source, selected_params=shared.parent.params,
            preliminary_report=run.report, evidence=evidence, probe_calls=len(evidence.observations),
            responses=[c.response for c in evidence.contrasts if c.response is not None], costs=costs(store), complete=complete)

    def checkpoint(evidence: ProbeEvidence) -> None:
        nonlocal latest
        latest = evidence
        (store.run_dir / "responses.json").write_text(envelope(evidence).model_dump_json(indent=2), encoding="utf-8")

    def allowed() -> bool:
        return run.deadline is None or run.deadline.remaining() > 0

    try:
        with TaskEvaluator(Path(c2_local_adapter.__file__), "evaluate") as evaluator:
            search = TaskSearch(evaluator, inputs.objective, run.cfg.budgets, device=run.cfg.device)
            cid = shared.parent.candidate_id
            search.paths[cid], search.spaces[cid] = directory / "parent.py", shared.space
            search.resource_metrics, search.probe_budget = shared.resource_metrics, 12
            evidence = targeted_probes(search, shared.parent, ProbeRun(
                requests=tuple(run.report.probe_requests), context={"adapter": adapter}, allowed=allowed, checkpoint=checkpoint))
        result = envelope(evidence, complete=True)
        result.validate_for(shared)
        (store.run_dir / "responses.json").write_text(result.model_dump_json(indent=2), encoding="utf-8")
        store.append("RUN_FINISHED", {"probe_calls": result.probe_calls})
        return result
    finally:
        store.append("TARGETED_ACQUISITION_STOPPED", {"probe_calls": len(latest.observations),
            "worker_attempts": costs(store).worker_attempts})
