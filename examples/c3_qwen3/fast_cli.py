# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["kernel-optimizer"]
# ///
# Existing interpreter: python -B -m examples.c3_qwen3.fast_cli --help
"""One development epoch, then STOP; publication and framework diagnosis remain operator-owned."""

import argparse
from pathlib import Path
import sys
from typing import Literal, assert_never

from kernel_optimizer.config import load_config
from .fast_artifacts import framework_id, identify_framework
from .fast_development import develop, resume_development
from .fast_prepare import FastAccess, prepare_fast
from .fast_records import Framework, FrameworkReview, ProbeResult
from .fast_runtime import RunInputs, open_evaluator, open_generator
from .fast_target import build_target
from .fast_workflow import scoped_config
from .runner_records import FrozenRecord, RunnerError


class Options(FrozenRecord):
    mode: Literal["seal", "target", "develop", "probe", "resume"]
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
    for name in ("develop", "probe"):
        sub.add_parser(name, help="one bounded F_r epoch, never an automatic framework loop").add_argument("--inputs", type=Path, required=True)
    resume = sub.add_parser("resume", help="record parent diagnosis without changing D or counters")
    for name in ("state", "review", "framework"):
        resume.add_argument(f"--{name}", type=Path, required=True)
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
                target = build_target(options.profile, options.module)
                with options.output.open("x", encoding="utf-8") as stream:
                    stream.write(target.model_dump_json(indent=2))
                print(f"site_id={target.brief.site_id}; representatives={len(target.representatives)}; device_proof=not_run")
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
            case unreachable:
                assert_never(unreachable)
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
