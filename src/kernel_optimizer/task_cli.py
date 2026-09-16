"""Direct-task CLI; parameter-only provided evaluation needs no agent runtime."""

import argparse
import json
from contextlib import ExitStack
from pathlib import Path
from typing import ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, TypeAdapter

from kernel_optimizer.config import load_config
from kernel_optimizer.control.direct_task import TaskSearch, TaskSpace
from kernel_optimizer.evaluation.task_eval import TaskEvaluationError, TaskEvaluator
from kernel_optimizer.reporting.report import ReportGenerator
from kernel_optimizer.tuning.objective import Objective, objective_value


class TaskOptions(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)
    project: Path
    candidate: Path
    space: Path
    eval_file: Path | None = None
    eval_function: str = "evaluate"
    goal: str | None = None
    direction: Literal["minimize", "maximize"] | None = None
    context: Path | None = None
    label: str | None = None
    unit: str | None = None
    trials: int = Field(default=40, gt=0)
    output: Path
    source: list[Path] = Field(default_factory=list)
    config: Path | None = None
    override: list[str] | None = None
    rewrite_rounds: int = Field(default=0, ge=0)
    probe_budget: int = Field(default=8, ge=0)
    resource_metric: list[str] = Field(default_factory=list)


def add_task_parser(parser: argparse.ArgumentParser) -> None:
    for name in ("project", "candidate", "space", "output"):
        _ = parser.add_argument(f"--{name}", required=True)
    _ = parser.add_argument("--eval-file")
    _ = parser.add_argument("--eval-function", default="evaluate")
    _ = parser.add_argument("--goal")
    _ = parser.add_argument("--direction", choices=["minimize", "maximize"])
    _ = parser.add_argument("--context")
    _ = parser.add_argument("--label")
    _ = parser.add_argument("--unit")
    _ = parser.add_argument("--source", action="append", default=[])
    _ = parser.add_argument("--trials", type=int, default=40)
    _ = parser.add_argument("--rewrite-rounds", type=int, default=0)
    _ = parser.add_argument("--probe-budget", type=int, default=8,
                            help="total fresh diagnostic evaluator calls across rewrite rounds")
    _ = parser.add_argument("--resource-metric", action="append", default=[],
                            help="task-declared resource metric; repeat for multiple names")


def cmd_optimize_task(args: argparse.Namespace) -> int:
    options = TaskOptions.model_validate(vars(args))
    root = options.project.resolve()
    candidate = (root / options.candidate).resolve()
    space = TaskSpace.model_validate_json((root / options.space).read_text(encoding="utf-8"))
    context = TypeAdapter(dict[str, JsonValue]).validate_json(
        (root / options.context).read_text(encoding="utf-8") if options.context else "{}",
    )
    cfg = load_config(options.config, options.override or [])
    output = options.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if options.eval_file is None and not options.goal:
        raise TaskEvaluationError("provide --eval-file or --goal")
    if options.eval_file is not None and options.direction is None:
        raise TaskEvaluationError("--direction is required with --eval-file")
    with ExitStack() as stack:
        runtime = None
        if options.eval_file is None or options.rewrite_rounds:
            from kernel_optimizer.wiring import Runtime

            runtime = stack.enter_context(Runtime(cfg, log_dir=output))
        if options.eval_file is not None:
            if options.direction is None:
                raise TaskEvaluationError("--direction is required with --eval-file")
            eval_file, function = root / options.eval_file, options.eval_function
            objective = Objective(direction=options.direction, label=options.label or "J", unit=options.unit)
        else:
            if not options.goal or runtime is None:
                raise TaskEvaluationError("provide --eval-file or --goal")
            from kernel_optimizer.agents.eval_builder import EvalBuilderInputs
            from kernel_optimizer.store.run_store import RunStore
            from kernel_optimizer.wiring import build_eval_builder

            builder = build_eval_builder(cfg, RunStore(output), runtime)
            outcome = builder.invoke(EvalBuilderInputs(
                project_root=root, source_paths=(candidate, *options.source),
                goal=options.goal,
                task_context=json.dumps({"context": context, "space": space.model_dump(),
                                         "direction": options.direction}),
            ))
            result = outcome.output
            if options.direction is not None and options.direction != result.direction:
                raise TaskEvaluationError("generated direction conflicts with --direction")
            eval_file, function = outcome.sandbox.root / result.eval_file, result.callable_name
            objective = Objective(direction=result.direction, label=options.label or result.label or "J",
                                  unit=options.unit or result.unit)
        evaluator = stack.enter_context(TaskEvaluator(eval_file, function))
        search = TaskSearch(evaluator, objective, cfg.budgets.model_copy(
            update={"trials_per_space": options.trials},
        ), device=cfg.device)
        _ = search.evaluate_candidate(candidate, space, context, seed=cfg.run.seed)
        if options.rewrite_rounds and runtime is not None:
            from kernel_optimizer.control.task_rewrite import RewriteRun, run_rewrites
            from kernel_optimizer.store.run_store import RunStore
            from kernel_optimizer.wiring import build_task_rewriter

            search.probe_budget = options.probe_budget
            search.resource_metrics = tuple(dict.fromkeys(options.resource_metric))
            run_rewrites(search, build_task_rewriter(cfg, RunStore(output), runtime), RewriteRun(
                rounds=options.rewrite_rounds, project_root=root,
                goal=options.goal or f"Optimize {objective.label}; preserve all evaluator-checked outputs.",
                context=context, source_paths=tuple(options.source), seed=cfg.run.seed,
            ))
        final = search.execute_best(context)
    report = ReportGenerator().generate_task(search, output)
    for trial in search.trials:
        value = objective_value(trial, objective)
        print(f"{trial.trial_id}: J={value} {objective.unit or ''} {trial.status} {trial.params.values}")
    best = search.best()
    print(f"objective: {objective.label} ({objective.direction}), unit={objective.unit or 'unspecified'}")
    if best is not None:
        print(f"best: {search.paths[best.candidate_id]} J={objective_value(best, objective)} params={best.params.values}")
    if final is not None:
        print(f"final execution: {final.status} J={objective_value(final, objective)} "
              f"{objective.unit or ''} {final.failure_detail}")
    else:
        print("final execution: not run (no valid search winner)")
    print(f"valid={sum(t.status == 'complete' for t in search.trials)} "
          f"invalid={sum(t.status == 'fail' for t in search.trials)}; report: {report}")
    print(f"structural rounds={len(search.rewrites)}; fresh diagnostic probes={search.probe_calls}")
    return 0 if final is not None and final.status == "complete" else 1
