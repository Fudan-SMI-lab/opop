"""Existing agent calls with shared evidence and an optional B-only response brief."""

import csv
import io
from dataclasses import dataclass
from pathlib import Path

from kernel_optimizer.agents.modules import AnalystInputs, ParameterizerInputs, RewriterInputs, _detect_backend
from kernel_optimizer.agents.task_rewriter import TaskRewriteInputs
from kernel_optimizer.config import AppConfig
from kernel_optimizer.control.direct_task import TaskSpace
from kernel_optimizer.models.core import ParameterSpace, TaskSpec, sha256_text
from kernel_optimizer.paramspace.materializer import extract_defaults
from kernel_optimizer.store.run_store import RunStore
from kernel_optimizer.tuning.stats import TuningStatsAnalyzer
from kernel_optimizer.tasks.kernelbench import parse_task_arg
from kernel_optimizer.wiring import Runtime, build_orchestrator, build_task_rewriter
from scripts.experiments.c2_local_inputs import InputError, Shared


@dataclass(frozen=True, slots=True)
class Services:
    cfg: AppConfig
    store: RunStore
    runtime: Runtime


@dataclass(frozen=True, slots=True)
class Child:
    path: Path
    space: TaskSpace
    backend: str


def direct_child(inputs: TaskRewriteInputs, services: Services) -> Child:
    agent = build_task_rewriter(services.cfg, services.store, services.runtime)
    outcome = agent.invoke(inputs)
    path = (outcome.sandbox.root / outcome.output.candidate_file).resolve()
    return Child(path, outcome.output.space or inputs.space, _detect_backend(path.read_text(encoding="utf-8")))


def legacy_child(shared: Shared, services: Services, inputs: TaskRewriteInputs) -> Child:
    level, problem_id = parse_task_arg(shared.task)
    task = TaskSpec(level=level, problem_id=problem_id, name=shared.task, ref_path=Path("reference.py"),
                    ref_src_sha=sha256_text(shared.reference_source))
    deps = build_orchestrator(services.cfg, services.store, task, services.runtime).deps
    space = ParameterSpace(space_id=shared.parent.space_id, candidate_id=shared.parent.candidate_id,
                           source_sha=sha256_text(shared.source), domains=shared.space.params,
                           constraints=shared.space.constraints)
    stats = TuningStatsAnalyzer(shared.device).analyze(space, shared.trials)
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(["trial_id", "params", "status", "failure_kind", "failure_detail", "latency_ms", "profile"])
    for trial in shared.trials:
        writer.writerow([trial.trial_id, trial.params.model_dump_json(), trial.status,
                         trial.failure_kind, trial.failure_detail,
                         trial.latency_ms.model_dump_json() if trial.latency_ms else "",
                         trial.profile.model_dump_json() if trial.profile else ""])
    brief = "\n".join(response.model_dump_json() for response in inputs.responses)
    common = "Full task reference source:\n```python\n" + shared.reference_source + "\n```\n"
    common += "Evaluation contract:\n" + (Path("evaluation.json").read_text(encoding="utf-8"))
    report = deps.analyst.invoke(AnalystInputs(
        task=task, candidate_source=shared.source, stats=stats, trials_csv=buffer.getvalue(),
        device=shared.device, candidate_id=shared.parent.candidate_id,
        eval_semantics=shared.semantics, profile=shared.parent.profile,
        digest_text=common + ("\nFresh conditional responses:\n" + brief if brief else ""),
    )).output
    outcome = deps.rewriter.invoke(RewriterInputs(
        task=task, best_source=shared.source, report=report, failed_hypotheses=[], device=shared.device,
        n_candidates=1, eval_semantics=shared.semantics, wall_text=brief or None,
    ))
    if len(outcome.output.candidates) != 1:
        raise InputError("legacy rewriter must return exactly one child for this diagnostic")
    proposed = outcome.output.candidates[0]
    source = outcome.sandbox.read_output(proposed.file)
    parameterized = deps.parameterizer.invoke(ParameterizerInputs(
        task=task, candidate_source=source, device=shared.device, candidate_id="child",
    ))
    path = (parameterized.sandbox.root / parameterized.output.file).resolve()
    child_space = TaskSpace.model_validate(parameterized.output.space.model_dump())
    if set(extract_defaults(path.read_text(encoding="utf-8"))) != {d.name for d in child_space.params}:
        raise InputError("child PARAMS keys and published space disagree")
    return Child(path, child_space, proposed.backend)
