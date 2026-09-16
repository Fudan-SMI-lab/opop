"""Full KernelBench worker bridge for TaskEvaluator; no in-process candidate execution."""

from collections.abc import Mapping
from pathlib import Path
from time import monotonic
from uuid import uuid4

from pydantic import JsonValue

from kernel_optimizer.config import AppConfig
from kernel_optimizer.evaluation.correctness import latency_from_result
from kernel_optimizer.evaluation.task_eval import TaskEvaluation
from kernel_optimizer.models.core import ParamSet, TaskSpec, TrialRecord, sha256_text
from kernel_optimizer.paramspace.materializer import MaterializeError, materialize
from kernel_optimizer.tasks.kernelbench import parse_task_arg
from kernel_optimizer.wiring import build_gpu_stack
from kernel_optimizer.store.run_store import RunStore
from scripts.experiments.c2_local_inputs import Shared


class GpuAdapter:
    """Own per-run worker state; record each materialization and evaluation attempt."""

    def __init__(self, shared: Shared, cfg: AppConfig, store: RunStore) -> None:
        self.shared = shared
        self.store = store
        self.backend = shared.backend
        self.full = False
        self.phase = "tuning"
        self.records: list[TrialRecord] = []
        reference = store.run_dir / "reference.py"
        reference.write_text(shared.reference_source, encoding="utf-8")
        level, problem_id = parse_task_arg(shared.task)
        self.task = TaskSpec(level=level, problem_id=problem_id, name=shared.task,
                             ref_path=reference.resolve(), ref_src_sha=sha256_text(shared.reference_source))
        _, self.correctness, self.benchmarker, self.profiler = build_gpu_stack(cfg, store)

    def measure(self, candidate: Path, params: ParamSet) -> TrialRecord:
        started = monotonic()
        trial_id = f"local-{uuid4().hex[:12]}"
        self.store.append("LOCAL_EVAL_STARTED", {"trial_id": trial_id, "phase": self.phase,
                                               "params": params.model_dump(mode="json")})
        raw = {}
        try:
            source = materialize(candidate.read_text(encoding="utf-8"), params)
            path = candidate.parent / f"{trial_id}.py"
            path.write_text(source, encoding="utf-8")
            raw = (self.benchmarker.final_reeval(self.task, path, self.backend) if self.full else
                   self.correctness.quick_test(self.task, path, trial_id, self.backend))
            latency = latency_from_result(raw)
            valid = bool(raw.get("ok")) and latency is not None
            profile = self.profiler.extract(raw)
            metrics: dict[str, float] = {}
            for name in self.shared.resource_metrics:
                match profile.model_dump().get(name):
                    case int() | float() as value:
                        metrics[name] = float(value)
                    case _:
                        continue
            evaluation = TaskEvaluation(score=latency.robust_ms if valid and latency else None,
                                        valid=valid, metrics=metrics,
                                        detail=None if valid else str(raw.get("log_tail", "worker rejected candidate")))
            record = TrialRecord.model_validate({
                "trial_id": trial_id, "candidate_id": "local", "space_id": "local",
                "params": params, "status": "complete" if valid else "fail",
                "latency_ms": latency, "profile": profile, "task_evaluation": evaluation,
                "failure_kind": None if valid else raw.get("failure_kind", "runtime_error"),
                "failure_detail": evaluation.detail or "",
            })
        except (MaterializeError, OSError, ValueError, RuntimeError) as exc:
            record = TrialRecord(trial_id=trial_id, candidate_id="local", space_id="local", params=params,
                                 status="fail", failure_detail=f"{type(exc).__name__}: {exc}",
                                 task_evaluation=TaskEvaluation(valid=False, detail=str(exc)))
        self.records.append(record)
        self.store.append("LOCAL_EVAL_DONE", {"phase": self.phase, "wall_s": monotonic() - started,
                                             "trial": record.model_dump(mode="json"), "worker": raw})
        return record


def evaluate(candidate: Path, params: Mapping[str, JsonValue],
             context: Mapping[str, GpuAdapter]) -> TaskEvaluation:
    record = context["adapter"].measure(candidate, ParamSet.model_validate({"values": dict(params)}))
    if record.task_evaluation is None:
        return TaskEvaluation(valid=False, detail="worker bridge returned no task evaluation")
    return record.task_evaluation
