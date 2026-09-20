# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["kernel-optimizer"]
# ///
# Existing model-host interpreter: python -B -m examples.c3_qwen3.operator_cli --help
"""C3 operator commands; only explicit readiness/search/heldout commands load the resident model."""

import argparse
import json
from pathlib import Path
import sys
from time import time
from typing import Literal, assert_never

from kernel_optimizer.config import load_config
from kernel_optimizer.evaluation.task_eval import TaskEvaluation
from kernel_optimizer.models.core import ParamSet
from kernel_optimizer.store.run_store import RunStore
from kernel_optimizer.wiring import Runtime, build_task_rewriter
from .model_binding import load_bundle
from .model_runner import load, prepare
from .operator_search import optimize_goal
from .runner_records import FrozenRecord, GoalId, RunnerError
from .search_analysis import analyze_matrix
from .search_bundle import baseline_bundle, export_selection
from .search_matrix import expired_matrix, run_matrix
from .search_readiness import Readiness, prepare_readiness
from .search_records import Opportunity, SearchClock, SearchInputs, SearchResult, Selection
from .search_session import OperatorSession


class MatrixInputs(FrozenRecord):
    readiness: Path
    selections: dict[GoalId, Path]
    clock: SearchClock
    output: Path


class Options(FrozenRecord):
    mode: Literal["readiness", "optimize-goal", "search", "heldout", "analyze"]
    inputs: Path | None = None
    config: Path | None = None
    device: str = "cuda:0"
    contract: Path | None = None
    assets: Path | None = None
    output: Path | None = None
    usage_audit: Path | None = None
    clock_from_stdin: bool = False
    eval_file: Path | None = None
    execution_profile: Literal["existing_generic", "model_operator"] = "existing_generic"


def expired_search(inputs: SearchInputs) -> SearchResult:
    prepared = prepare(inputs.contract, inputs.assets_manifest)
    inputs.output.mkdir(parents=True, exist_ok=False)
    path = baseline_bundle(inputs.output / "baseline")
    bundle = load_bundle(path, {})
    baseline = Selection(bundle=path, bundle_sha256=bundle.bundle_sha256, source_hashes=dict(bundle.source_hashes),
        params=ParamSet(values={}), params_sha256=bundle.params_sha256, site_groups={},
        evaluation=TaskEvaluation(valid=False, detail="search cutoff before model admission"),
        contract_sha256=prepared.contract_sha256, model_revision=prepared.asset_spec.revision,
        baseline_fallback=True, goal_id=inputs.goal_id, clock=inputs.clock)
    selected = export_selection(baseline, inputs.output / "selected")
    result = SearchResult(goal_id=inputs.goal_id, goal=inputs.goal, clock=inputs.clock, started_unix_s=time(),
        deadline_unix_s=inputs.clock.search_cutoff_unix_s, finished_unix_s=time(), drain_s=0,
        baseline=baseline, selected=selected, opportunities=[Opportunity(number=i, parent_bundle_sha256=bundle.bundle_sha256,
            status="censored", slots_used=0, model_calls=0, repair_calls=0) for i in (1, 2)])
    (inputs.output / "result.json").write_text(result.model_dump_json(indent=2), encoding="utf-8")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    readiness = sub.add_parser("readiness")
    for name in ("contract", "assets", "output"):
        readiness.add_argument(f"--{name}", type=Path, required=True)
    readiness.add_argument("--device", default="cuda:0")
    for name in ("optimize-goal", "search", "heldout"):
        command = sub.add_parser(name)
        command.add_argument("--inputs", type=Path, required=True)
        command.add_argument("--device", default="cuda:0")
        if name != "heldout":
            command.add_argument("--config", type=Path, required=True)
            command.add_argument("--clock-from-stdin", action="store_true")
            command.add_argument("--eval-file", type=Path)
            command.add_argument("--execution-profile", choices=["existing_generic", "model_operator"],
                                 default="existing_generic")
    analyze = sub.add_parser("analyze")
    analyze.add_argument("--inputs", type=Path, required=True)
    analyze.add_argument("--output", type=Path, required=True)
    analyze.add_argument("--usage-audit", type=Path)
    try:
        options = Options.model_validate(vars(parser.parse_args(argv)))
        match options.mode:
            case "analyze":
                if options.inputs is None or options.output is None:
                    raise RunnerError("analysis requires input/output")
                report = analyze_matrix(options.inputs, options.usage_audit)
                options.output.write_text(report.model_dump_json(indent=2), encoding="utf-8")
                return 0 if report.complete else 1
            case "readiness":
                if options.contract is None or options.assets is None or options.output is None:
                    raise RunnerError("readiness requires contract/assets/output")
                prepared = prepare(options.contract, options.assets)
                resident = load(prepared.asset_spec, options.device)
                try:
                    return 0 if prepare_readiness(resident, options.output).valid else 1
                finally:
                    resident.close()
            case "optimize-goal" | "search":
                if options.inputs is None or options.config is None:
                    raise RunnerError("search requires inputs/config")
                inputs = SearchInputs.model_validate_json(options.inputs.read_bytes())
                if options.eval_file is not None:
                    inputs = inputs.model_copy(update={"eval_file": options.eval_file.resolve()})
                if not options.clock_from_stdin and time() < inputs.clock.campaign_started_unix_s:
                    raise RunnerError("search launch precedes declared common S")
                if not options.clock_from_stdin and time() >= inputs.clock.search_cutoff_unix_s:
                    expired_search(inputs)
                    return 1
                prepared = prepare(inputs.contract, inputs.assets_manifest)
                goal = next(g for g in prepared.contract.goals if g.id == inputs.goal_id)
                cfg = load_config(options.config)
                store = RunStore.create(inputs.output.parent, inputs.output.name + "-agents", {
                    "goal": inputs.goal, "execution_profile": options.execution_profile,
                    "task_kind": {"existing_generic": "existing_generic",
                                  "model_operator": "model_project_operator"}[options.execution_profile]})
                cfg.opencode.launch_cwd = store.run_dir
                resident = load(prepared.asset_spec, options.device)
                try:
                    if options.clock_from_stdin:
                        print(json.dumps({"resident_ready": True, "goal_id": inputs.goal_id,
                            "model_revision": prepared.asset_spec.revision}), flush=True)
                        declared = SearchClock.model_validate_json(sys.stdin.readline())
                        inputs = inputs.model_copy(update={"clock": declared})
                    with OperatorSession(resident, goal, inputs.oracle_refs, inputs.output.with_name(inputs.output.name + "-evaluation"),
                                         eval_file=inputs.eval_file) as session, Runtime(cfg, store.run_dir) as runtime:
                        session.device = cfg.device
                        result = optimize_goal(session, build_task_rewriter(cfg, store, runtime,
                            execution_profile=options.execution_profile), inputs)
                        return 0 if all(o.status != "censored" for o in result.opportunities) else 1
                finally:
                    resident.close()
            case "heldout":
                if options.inputs is None:
                    raise RunnerError("heldout requires inputs")
                spec = MatrixInputs.model_validate_json(options.inputs.read_bytes())
                if time() >= spec.clock.final_deadline_unix_s:
                    expired_matrix(spec.clock, spec.output)
                    return 1
                ready = Readiness.model_validate_json(spec.readiness.read_bytes())
                prepared = prepare(ready.contract, ready.assets_manifest)
                resident = load(prepared.asset_spec, options.device)
                try:
                    result = run_matrix(resident, ready, spec.selections, clock=spec.clock, output=spec.output)
                    return 0 if all(c.evaluation.valid for c in result.cells) else 1
                finally:
                    resident.close()
            case unreachable:
                assert_never(unreachable)
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
