"""Existing agent calls with shared evidence and an optional B-only response brief."""

import csv
import io
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import JsonValue

from kernel_optimizer.agents.modules import AnalystInputs, ParameterizerInputs, RewriterInputs, _detect_backend
from kernel_optimizer.agents.task_rewriter import TaskRewriteInputs
from kernel_optimizer.config import AppConfig
from kernel_optimizer.control.direct_task import TaskSpace
from kernel_optimizer.models.core import Backend, ParameterSpace, TaskSpec, sha256_text
from kernel_optimizer.models.reports import BottleneckReport
from kernel_optimizer.paramspace.materializer import extract_defaults, find_params_span, materialize
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
    rewrite_intent: str | None = None


@dataclass(frozen=True, slots=True)
class LegacyProposal:
    source: str
    backend: Backend
    change_summary: str
    hypothesis_id: str | None
    report: BottleneckReport


def direct_child(inputs: TaskRewriteInputs, services: Services) -> Child:
    agent = build_task_rewriter(services.cfg, services.store, services.runtime)
    outcome = agent.invoke(inputs)
    path = (outcome.sandbox.root / outcome.output.candidate_file).resolve()
    return Child(path, outcome.output.space or inputs.space, _detect_backend(path.read_text(encoding="utf-8")))


def _task(shared: Shared) -> TaskSpec:
    level, problem_id = parse_task_arg(shared.task)
    return TaskSpec(level=level, problem_id=problem_id, name=shared.task, ref_path=Path("reference.py"),
                    ref_src_sha=sha256_text(shared.reference_source))


def _materialized_parent(shared: Shared, inputs: TaskRewriteInputs) -> str:
    path = inputs.candidate_path
    if not path.is_absolute():
        path = inputs.project_root / path
    try:
        source = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return materialize(shared.source, shared.parent.params)
    selected_span = find_params_span(source)
    raw_span = find_params_span(shared.source)
    if selected_span.defaults != shared.parent.params.values:
        raise InputError("staged parent PARAMS differ from selected parameters")
    if (source[:selected_span.start] != shared.source[:raw_span.start]
            or source[selected_span.end:] != shared.source[raw_span.end:]):
        raise InputError("staged parent is not a materialization of this Shared source")
    return source


def generate_legacy_proposal(
    shared: Shared, services: Services, inputs: TaskRewriteInputs, *,
    reasoning_mode: Literal["whole_task", "legacy_local"] = "whole_task",
    failed_hypotheses: list[dict[str, JsonValue]] | None = None,
) -> LegacyProposal:
    parent_source = _materialized_parent(shared, inputs)
    task = _task(shared)
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
    brief = "\n".join(response.model_dump_json() for response in inputs.responses
                      if not (response.reason == "not_acquired" and response.a is None
                              and response.b is None and response.b_params is None))
    common = "Evaluation contract:\n" + json.dumps(shared.evaluation)
    report = deps.analyst.invoke(AnalystInputs(
        task=task, candidate_source=shared.source, stats=stats, trials_csv=buffer.getvalue(),
        device=shared.device, candidate_id=shared.parent.candidate_id,
        eval_semantics=shared.semantics, profile=shared.parent.profile,
        digest_text=common, reasoning_mode=reasoning_mode, reference_source=shared.reference_source,
        conditional_response_text=brief or None, selected_params=shared.parent.params,
    )).output
    outcome = deps.rewriter.invoke(RewriterInputs(
        task=task, best_source=parent_source, report=report,
        failed_hypotheses=shared.failed_hypotheses if failed_hypotheses is None else failed_hypotheses,
        device=shared.device, n_candidates=1, eval_semantics=shared.semantics,
        reasoning_mode=reasoning_mode, reference_source=shared.reference_source,
        conditional_response_text=brief or None, selected_params=shared.parent.params, source_materialized=True,
    ))
    if len(outcome.output.candidates) != 1:
        raise InputError("legacy rewriter must return exactly one child for this diagnostic")
    proposed = outcome.output.candidates[0]
    source = outcome.sandbox.read_output(proposed.file)
    services.store.append("LEGACY_PROPOSAL_GENERATED", {
        "parent_candidate_id": shared.parent.candidate_id, "hypothesis_id": proposed.hypothesis_id or None,
        "change_summary": proposed.change_summary, "backend": proposed.backend,
    })
    return LegacyProposal(source, proposed.backend, proposed.change_summary, proposed.hypothesis_id or None, report)


def parameterize_legacy_proposal(
    shared: Shared, services: Services, proposal: LegacyProposal, *, pass_intent: bool = True,
) -> Child:
    task = _task(shared)
    deps = build_orchestrator(services.cfg, services.store, task, services.runtime).deps
    intent = proposal.change_summary if pass_intent else None
    parameterized = deps.parameterizer.invoke(ParameterizerInputs(
        task=task, candidate_source=proposal.source, device=shared.device, candidate_id="child",
        rewrite_intent=intent,
    ))
    path = (parameterized.sandbox.root / parameterized.output.file).resolve()
    child_space = TaskSpace.model_validate(parameterized.output.space.model_dump())
    if set(extract_defaults(path.read_text(encoding="utf-8"))) != {d.name for d in child_space.params}:
        raise InputError("child PARAMS keys and published space disagree")
    return Child(path, child_space, proposal.backend, rewrite_intent=intent)


def legacy_child(shared: Shared, services: Services, inputs: TaskRewriteInputs) -> Child:
    proposal = generate_legacy_proposal(shared, services, inputs)
    return parameterize_legacy_proposal(shared, services, proposal)
