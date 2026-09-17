# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["kernel-optimizer"]
# ///
# Existing v5 environment: python -m scripts.experiments.c2_retune --config CONFIG --input INPUT
"""Retune one existing structure through normal v5 validation and Orchestrator._tune."""

import argparse
import os
import shutil
import sys
from contextlib import closing
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
from kernel_optimizer.paramspace.materializer import materialize
from kernel_optimizer.paramspace.validation import SpaceAccepted, SpaceRejection
from kernel_optimizer.store.run_store import RunStore
from kernel_optimizer.tasks.kernelbench import parse_task_arg
from kernel_optimizer.wiring import Runtime, build_orchestrator
from scripts.experiments.c2_local_runner import isolated_config, worker_environment


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


class RetuneResult(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)
    status: Literal["complete", "rejected", "no_best", "final_failed"]
    rejection: SpaceRejection | None = None
    selected: BestRecord | None = None
    trials: list[TrialRecord] = Field(default_factory=list)
    finals: list[TrialRecord] = Field(default_factory=list)


def tune_existing(orch: Orchestrator, inputs: RetuneInputs) -> RetuneResult:
    source = (orch.store.run_dir / "inputs" / "source.py").read_text(encoding="utf-8")
    published = ParameterSpace.model_validate_json(inputs.space.read_text(encoding="utf-8"))
    proposal = ParameterizationResult.model_validate({"file": "source.py", "space": {
        "params": [p.model_dump() for p in published.domains],
        "constraints": [c.model_dump() for c in published.constraints],
    }})
    candidate = orch._register(source, "seed", [], inputs.backend, "existing structure; retuning only")
    if candidate is None:
        return RetuneResult(status="rejected", rejection=SpaceRejection(reason="registration_refused", detail="no candidate"))
    crun = orch.runs[candidate.candidate_id]
    accepted = orch.deps.validator.validate_and_publish(
        candidate, source, proposal, orch.task, orch.store.candidate_dir(candidate.candidate_id))
    orch.store.append("RETUNE_VALIDATION", {"result": accepted.model_dump(mode="json")})
    match accepted:
        case SpaceRejection():
            candidate.status = "dropped"
            return RetuneResult(status="rejected", rejection=accepted)
        case SpaceAccepted():
            crun.space = accepted.space
        case unreachable:
            assert_never(unreachable)
    orch.store.append("SPACE_PUBLISHED", {"space": accepted.space.model_dump(mode="json")})
    anchors = tuple(w.params for w in accepted.witnesses)
    measured_cache: dict[str, TrialRecord] = {}
    for witness in accepted.witnesses:
        if witness.latency_mean_ms is not None:
            measured_cache[witness.params.key()] = TrialRecord(
                trial_id=f"wit-{witness.params.key()}", candidate_id=candidate.candidate_id,
                space_id=accepted.space.space_id, params=witness.params, status="complete",
                latency_ms=latency_from_result(witness.worker_result),
                profile=orch.deps.profiler.extract(witness.worker_result),
            )
    orch._tune(crun, anchors, measured_cache)
    candidate.status = "tuned"
    selected = orch.deps.families.families[candidate.family_id].best
    if selected is None:
        return RetuneResult(status="no_best", trials=crun.trials)
    selected_path = orch.store.run_dir / "report" / "selected.py"
    selected_path.write_text(materialize(source, selected.params), encoding="utf-8")
    orch.store.append("RETUNE_SELECTED", {"selected": selected.model_dump(mode="json"),
                                          "artifact": "report/selected.py"})
    finals: list[TrialRecord] = []
    for block in range(3):
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
            "space_id": accepted.space.space_id, "params": selected.params,
            "status": "complete" if valid else "fail", "latency_ms": latency,
            "profile": orch.deps.profiler.extract(raw),
            "failure_kind": None if valid else raw.get("failure_kind") or "runtime_error",
            "failure_detail": "" if valid else str(raw.get("log_tail", "no full measurement")),
        })
        finals.append(record)
        orch.store.append("RETUNE_FINAL_DONE", {"block": block, "worker": raw,
                                               "trial": record.model_dump(mode="json"), "wall_s": monotonic() - started})
    return RetuneResult(status="complete" if all(t.status == "complete" for t in finals) else "final_failed",
                        selected=selected, trials=crun.trials, finals=finals)


def retune(inputs: RetuneInputs, cfg: AppConfig) -> RetuneResult:
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
    local.v4.conditional_scan.mode = "off"
    local.v3.slope_guide.enabled = False
    local.wsl.extra_pythonpath = f"{to_wsl_path(staged)}:{local.wsl.extra_pythonpath}"
    level, problem = parse_task_arg(inputs.task)
    task = TaskSpec(level=level, problem_id=problem, name=inputs.task, ref_path=reference,
                    ref_src_sha=sha256_text(reference.read_text(encoding="utf-8")))
    previous_seed = os.environ.get("C2_RETUNE_EVALUATION_SEED")
    try:
        os.environ["C2_RETUNE_EVALUATION_SEED"] = str(inputs.evaluation_seed)
        with worker_environment(local), closing(OpencodeClient("http://127.0.0.1:1")) as client:
            runtime = Runtime(local)
            runtime.client = client
            orch = build_orchestrator(local, store, task, runtime)
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
