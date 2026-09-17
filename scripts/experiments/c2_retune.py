# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["kernel-optimizer"]
# ///
# Existing v5 environment: python -m scripts.experiments.c2_retune --config CONFIG --input INPUT
"""Retune one existing structure via native accepted-candidate continuation."""

import argparse
import os
import shutil
import sys
from contextlib import ExitStack, closing
from pathlib import Path
from time import monotonic
from typing import ClassVar, Literal, assert_never

from pydantic import BaseModel, ConfigDict, Field

from kernel_optimizer.agents.runtime import OpencodeClient
from kernel_optimizer.config import AppConfig, load_config
from kernel_optimizer.control.orchestrator import Orchestrator
from kernel_optimizer.evaluation.correctness import latency_from_result
from kernel_optimizer.gpu.worker_client import to_wsl_path
from kernel_optimizer.models.core import Backend, BestRecord, ParameterSpace, TaskSpec, TrialRecord, sha256_text
from kernel_optimizer.models.reports import ParameterizationResult
from kernel_optimizer.paramspace.validation import SpaceAccepted, SpaceRejection
from kernel_optimizer.store.run_store import RunStore
from kernel_optimizer.tasks.kernelbench import parse_task_arg
from kernel_optimizer.wiring import Runtime, build_orchestrator
from scripts.experiments.c2_local_runner import isolated_config, worker_environment
from scripts.experiments.c2_local_inputs import InputError


class RetuneInputs(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid")
    task: str
    source: Path
    space: Path
    reference: Path
    sampler_seed: int = Field(ge=0)
    evaluation_seed: int = Field(ge=0)
    output: Path
    backend: Backend = "triton"
    helpers: tuple[Path, ...] = ()
    final_blocks: Literal[0, 3] = 3
    space_expansions_per_candidate: int = Field(default=0, ge=0)
    rewrite_intent: str | None = None


class RetuneResult(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)
    status: Literal["complete", "rejected", "no_best", "final_failed", "artifact_error"]
    rejection: SpaceRejection | None = None
    selected: BestRecord | None = None
    trials: list[TrialRecord] = Field(default_factory=list)
    finals: list[TrialRecord] = Field(default_factory=list)
    selected_trial: TrialRecord | None = None
    final_blocks_requested: Literal[0, 3] = 3
    error: str | None = None


def tune_existing(orch: Orchestrator, inputs: RetuneInputs) -> RetuneResult:
    source = (orch.store.run_dir / "inputs" / "source.py").read_text(encoding="utf-8")
    published = ParameterSpace.model_validate_json(inputs.space.read_text(encoding="utf-8"))
    proposal = ParameterizationResult.model_validate({"file": "source.py", "space": {
        "params": [p.model_dump() for p in published.domains],
        "constraints": [c.model_dump() for c in published.constraints],
    }})
    candidate = orch._register(source, "rewrite" if inputs.rewrite_intent else "seed", [], inputs.backend,
                               inputs.rewrite_intent or "existing structure; retuning only")
    if candidate is None:
        return RetuneResult(status="rejected", rejection=SpaceRejection(reason="registration_refused", detail="no candidate"),
                            final_blocks_requested=inputs.final_blocks)
    crun = orch.runs[candidate.candidate_id]
    accepted = orch.deps.validator.validate_and_publish(
        candidate, source, proposal, orch.task, orch.store.candidate_dir(candidate.candidate_id))
    orch.store.append("RETUNE_VALIDATION", {"result": accepted.model_dump(mode="json")})
    match accepted:
        case SpaceRejection():
            candidate.status = "dropped"
            return RetuneResult(status="rejected", rejection=accepted, final_blocks_requested=inputs.final_blocks)
        case SpaceAccepted():
            orch._continue_accepted_candidate(crun, accepted, run_analysis=False)
        case unreachable:
            assert_never(unreachable)
    selected = orch.deps.families.families[candidate.family_id].best
    if selected is None:
        return RetuneResult(status="no_best", trials=crun.trials, final_blocks_requested=inputs.final_blocks)
    selected_trial = next((t for t in crun.trials if t.status == "complete" and t.latency_ms is not None
                           and t.candidate_id == selected.candidate_id and t.params == selected.params
                           and t.latency_ms.robust_ms == selected.latency_ms), None)
    selected_path = orch.store.run_dir / "report" / "selected.py"
    try:
        if selected_trial is None:
            raise InputError("native selection has no matching measured trial")
        measured = orch.store.candidate_dir(selected_trial.candidate_id) / "trials" / f"{selected_trial.trial_id}.py"
        shutil.copyfile(measured, selected_path)
    except (OSError, InputError) as exc:
        orch.store.append("RETUNE_ARTIFACT_ERROR", {"error": str(exc)})
        return RetuneResult(status="artifact_error", selected=selected, selected_trial=selected_trial,
                            trials=crun.trials, error=str(exc), final_blocks_requested=inputs.final_blocks)
    orch.store.append("RETUNE_SELECTED", {"selected": selected.model_dump(mode="json"),
                                          "selected_trial": selected_trial.model_dump(mode="json"),
                                          "artifact": "report/selected.py"})
    finals: list[TrialRecord] = []
    for block in range(inputs.final_blocks):
        started = monotonic()
        orch.store.append("RETUNE_FINAL_STARTED", {"block": block})
        try:
            raw = orch.deps.benchmarker.final_reeval(orch.task, selected_path, inputs.backend)
        except (OSError, RuntimeError) as exc:
            raw = {"ok": False, "failure_kind": "runtime_error", "log_tail": str(exc)}
        latency = latency_from_result(raw)
        valid = bool(raw.get("ok")) and latency is not None
        record = TrialRecord.model_validate({
            "trial_id": f"final-{block}", "candidate_id": candidate.candidate_id,
            "space_id": selected_trial.space_id, "params": selected.params,
            "status": "complete" if valid else "fail", "latency_ms": latency,
            "profile": orch.deps.profiler.extract(raw),
            "failure_kind": None if valid else raw.get("failure_kind") or "runtime_error",
            "failure_detail": "" if valid else str(raw.get("log_tail", "no full measurement")),
        })
        finals.append(record)
        orch.store.append("RETUNE_FINAL_DONE", {"block": block, "worker": raw,
                                               "trial": record.model_dump(mode="json"), "wall_s": monotonic() - started})
    return RetuneResult(status="complete" if all(t.status == "complete" for t in finals) else "final_failed",
                        selected=selected, selected_trial=selected_trial, trials=crun.trials, finals=finals,
                        final_blocks_requested=inputs.final_blocks)


def retune(inputs: RetuneInputs, cfg: AppConfig, *, runtime: Runtime | None = None) -> RetuneResult:
    if inputs.space_expansions_per_candidate and (runtime is None or runtime.client is None):
        raise InputError("native expansion requires a caller-owned Runtime with a client")
    started = monotonic()
    root = inputs.output.resolve()
    store = RunStore.create(root.parent, root.name, {
        "inputs": inputs.model_dump(mode="json"), "evaluation": cfg.evaluation.model_dump(mode="json"),
        "device": cfg.device.model_dump(mode="json"), "gpu": cfg.gpu.model_dump(mode="json"),
        "search": cfg.v3.search.model_dump(mode="json"), "ordered_categoricals": cfg.v3.ordered_categoricals.model_dump(),
        "budget": 40, "conditional_scan": "off", "slope_guide": False,
    })
    staged = root / "inputs"
    staged.mkdir()
    (staged / "source.py").write_text(inputs.source.read_text(encoding="utf-8"), encoding="utf-8")
    reference = staged / "reference.py"
    reference.write_text(inputs.reference.read_text(encoding="utf-8"), encoding="utf-8")
    for helper in inputs.helpers:
        destination = staged / helper.resolve().relative_to(inputs.source.resolve().parent)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(helper, destination)
    local = isolated_config(cfg, root)
    local.run.seed = inputs.sampler_seed
    local.budgets.space_expansions_per_candidate = inputs.space_expansions_per_candidate
    local.v4.conditional_scan.mode = "off"
    local.v3.slope_guide.enabled = False
    local.wsl.extra_pythonpath = f"{to_wsl_path(staged)}:{local.wsl.extra_pythonpath}"
    level, problem = parse_task_arg(inputs.task)
    task = TaskSpec(level=level, problem_id=problem, name=inputs.task, ref_path=reference,
                    ref_src_sha=sha256_text(reference.read_text(encoding="utf-8")))
    previous_seed = os.environ.get("C2_RETUNE_EVALUATION_SEED")
    try:
        os.environ["C2_RETUNE_EVALUATION_SEED"] = str(inputs.evaluation_seed)
        with worker_environment(local), ExitStack() as stack:
            active_runtime = runtime
            if active_runtime is None:
                active_runtime = Runtime(local)
                active_runtime.client = stack.enter_context(closing(OpencodeClient("http://127.0.0.1:1")))
            orch = build_orchestrator(local, store, task, active_runtime)
            orch.deps.evaluator.seed = inputs.evaluation_seed
            orch.deps.validator.seed = inputs.evaluation_seed
            orch.deps.evaluator.worker.worker_main_path = Path(__file__).with_name("c2_retune_worker.py")
            result = tune_existing(orch, inputs)
    finally:
        if previous_seed is None:
            os.environ.pop("C2_RETUNE_EVALUATION_SEED", None)
        else:
            os.environ["C2_RETUNE_EVALUATION_SEED"] = previous_seed
    (root / "result.json").write_text(result.model_dump_json(indent=2), encoding="utf-8")
    store.append("RUN_FINISHED", {"status": result.status, "wall_s": monotonic() - started})
    return result


class Arguments(argparse.Namespace):
    config: Path = Path()
    input: Path = Path()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    args = Arguments()
    parser.parse_args(argv, namespace=args)
    try:
        inputs = RetuneInputs.model_validate_json(args.input.read_text(encoding="utf-8"))
        result = retune(inputs, load_config(args.config))
        print(result.status)
        return 0 if result.status == "complete" else 1
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
