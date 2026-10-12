# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["kernel-optimizer"]
# ///
# Existing interpreter: python -B -m examples.c3_qwen3.fast_cli --help
"""One development epoch, then STOP; publication and framework diagnosis remain operator-owned."""

import argparse
import json
from pathlib import Path
import sys
from typing import Literal, assert_never

from pydantic import TypeAdapter

from kernel_optimizer.config import load_config
from .fast_artifacts import framework_id, identify_framework
from .fast_development import develop, resume_development
from .fast_formal import FormalInputs, freeze, run_formal_goal
from .fast_matrix import MatrixInputs, run_matrix
from .fast_prepare import FastAccess, prepare_fast
from .fast_records import Framework, FrameworkReview, ProbeResult, TargetSpec
from .fast_runtime import RunInputs, open_evaluator, open_generator
from .fast_target import build_target
from .fast_workflow import scoped_config
from .runner_records import FrozenRecord, GoalId, RunnerError


class Options(FrozenRecord):
    mode: Literal["seal", "target", "develop", "probe", "resume", "freeze", "formal", "matrix"]
    inputs: Path | None = None
    contract: Path | None = None
    assets: Path | None = None
    agent_config: Path | None = None
    revision: str | None = None
    profile: Path | None = None
    module: str | None = None
    output: Path | None = None
    state: Path | None = None
    review: Path | None = None
    framework: Path | None = None
    targets: Path | None = None
    compatibility: Path | None = None
    frozen: Path | None = None
    budget: Path | None = None
    results: Path | None = None
    started_unix_s: float | None = None
    deadline_unix_s: float | None = None
    device: str = "cuda:0"
    allow_composite: bool = False


def execute_development(inputs: RunInputs) -> ProbeResult:
    if inputs.development_state is None:
        raise RunnerError("develop requires a persistent development_state path")
    with open_evaluator(inputs, "development", "development") as engine:
        with open_generator(inputs, engine) as generator:
            return develop(generator, inputs.development_state, inputs.output)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    seal = sub.add_parser("seal", help="CPU-only framework identity for the tested code/config")
    for name in ("contract", "assets", "agent-config", "output"):
        seal.add_argument(f"--{name}", type=Path, required=True)
    seal.add_argument("--revision", required=True)
    target = sub.add_parser("target", help="CPU-only pack from an existing native CUDA profile")
    target.add_argument("--profile", type=Path, required=True)
    target.add_argument("--module", required=True)
    target.add_argument("--output", type=Path, required=True)
    # Opt-in, off by default: admit a WRAPPER module (one with observed children) as the site, so a
    # candidate may fuse across operator KINDS instead of only across one family's repeated stages.
    # Purity is still required -- a subtree receiving the KV cache is refused, because it could never
    # be captured as a fixture and so could never carry a device proof.
    target.add_argument("--allow-composite", action="store_true",
                        help="allow a composite (wrapper) site so the replacement subsumes its subtree")
    for name in ("develop", "probe"):
        sub.add_parser(name, help="one bounded F_r epoch, never an automatic framework loop").add_argument("--inputs", type=Path, required=True)
    resume = sub.add_parser("resume", help="record parent diagnosis without changing D or counters")
    for name in ("state", "review", "framework"):
        resume.add_argument(f"--{name}", type=Path, required=True)
    # freeze/formal/matrix complete the chain from one accepted development epoch to the final
    # cross-goal table. They stay three separate commands, each writing its own file, because the
    # steps between them are operator decisions: what to freeze, when the shared formal clock
    # starts, and whether the lanes finished. A single command would hide those.
    freeze = sub.add_parser("freeze", help="freeze F-star from one DEV_GO plus three declared targets")
    for name in ("contract", "assets", "state", "targets", "output"):
        freeze.add_argument(f"--{name}", type=Path, required=True)
    # Optional: a C2 regression receipt is recorded and hash-pinned when supplied, but its absence
    # does not block a freeze. C2 isolation is structural (opt-in switches), not a file's existence.
    freeze.add_argument("--compatibility", type=Path, default=None)
    formal = sub.add_parser("formal", help="one goal's formal lane: fresh generation from the original baseline")
    formal.add_argument("--inputs", type=Path, required=True)
    formal.add_argument("--frozen", type=Path, required=True)
    matrix = sub.add_parser("matrix", help="4 rows x 3 goals x 3 blocks on the sealed held-out corpus")
    for name in ("contract", "assets", "frozen", "results", "budget", "output"):
        matrix.add_argument(f"--{name}", type=Path, required=True)
    matrix.add_argument("--started-unix-s", type=float, required=True)
    matrix.add_argument("--deadline-unix-s", type=float, required=True)
    matrix.add_argument("--device", default="cuda:0")
    try:
        options = Options.model_validate(vars(parser.parse_args(argv)))
        match options.mode:
            case "seal":
                if options.contract is None or options.assets is None or options.agent_config is None or options.revision is None or options.output is None:
                    raise RunnerError("seal arguments are incomplete")
                prepared = prepare_fast(options.contract, options.assets, FastAccess(profile="c3_fast_device", phase="development"))
                frame = identify_framework(options.revision, scoped_config(load_config(options.agent_config)), prepared)
                with options.output.open("x", encoding="utf-8") as stream:
                    stream.write(frame.model_dump_json(indent=2))
                print(f"framework_id={framework_id(frame)}; phase=development; model_loaded=false")
                return 0
            case "target":
                if options.profile is None or options.module is None or options.output is None:
                    raise RunnerError("target arguments are incomplete")
                target = build_target(options.profile, options.module,
                                      allow_composite=options.allow_composite)
                with options.output.open("x", encoding="utf-8") as stream:
                    stream.write(target.model_dump_json(indent=2))
                print(f"site_id={target.brief.site_id}; modules={len(target.brief.module_paths)}; "
                      f"composite={options.allow_composite}; "
                      f"representatives={len(target.representatives)}; device_proof=not_run")
                return 0
            case "develop" | "probe":
                if options.inputs is None:
                    raise RunnerError("develop requires inputs")
                result = execute_development(RunInputs.model_validate_json(options.inputs.read_bytes()))
                print(result.model_dump_json(indent=2))
                return 0 if result.status == "DEV_GO" else 1
            case "resume":
                if options.state is None or options.review is None or options.framework is None:
                    raise RunnerError("resume arguments are incomplete")
                state = resume_development(options.state, FrameworkReview.model_validate_json(options.review.read_bytes()),
                                           Framework.model_validate_json(options.framework.read_bytes()))
                print(state.model_dump_json(indent=2))
                return 0
            case "freeze":
                for field in (options.contract, options.assets, options.state, options.targets, options.output):
                    if field is None:
                        raise RunnerError("freeze arguments are incomplete")
                # --targets maps each goal to its own TargetSpec file. All three must be declared
                # BEFORE the formal lanes run, so a goal cannot have its operator chosen after
                # seeing how another goal scored.
                paths = TypeAdapter(dict[GoalId, Path]).validate_json(options.targets.read_bytes())
                targets = {goal: TargetSpec.model_validate_json(path.read_bytes()) for goal, path in paths.items()}
                frozen = freeze(options.state, targets, options.compatibility,
                                options.contract, options.assets, options.output)
                print(f"framework_id={framework_id(frozen.framework)}; goals={sorted(frozen.targets)}")
                return 0
            case "formal":
                if options.inputs is None or options.frozen is None:
                    raise RunnerError("formal arguments are incomplete")
                spec = RunInputs.model_validate_json(options.inputs.read_bytes())
                if spec.clock is None:
                    raise RunnerError("a formal lane needs the campaign's shared clock in its inputs")
                result = run_formal_goal(FormalInputs(inputs=spec, frozen=options.frozen, clock=spec.clock))
                print(result.model_dump_json(indent=2))
                return 0 if result.status == "accepted" else 1
            case "matrix":
                for field in (options.contract, options.assets, options.frozen, options.results,
                              options.budget, options.output):
                    if field is None:
                        raise RunnerError("matrix arguments are incomplete")
                if options.started_unix_s is None or options.deadline_unix_s is None:
                    raise RunnerError("the final lane needs an explicitly recorded start and deadline")
                results = TypeAdapter(dict[GoalId, Path]).validate_json(options.results.read_bytes())
                report = run_matrix(MatrixInputs(contract=options.contract, assets_manifest=options.assets,
                    frozen_framework=options.frozen, formal_results=results, budget=options.budget,
                    output=options.output, started_unix_s=options.started_unix_s,
                    deadline_unix_s=options.deadline_unix_s, device=options.device))
                print(json.dumps({"complete": report.complete, "cells": len(report.cells),
                    "valid": sum(1 for c in report.cells if c.status == "valid"),
                    "device_proven_rows": sorted(r.row for r in report.rows if r.device_proven),
                    "summary": report.summary}, indent=2))
                return 0 if report.complete else 1
            case unreachable:
                assert_never(unreachable)
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
