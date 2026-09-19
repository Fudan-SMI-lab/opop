"""Voluntary formal evaluation CLI; no model calls, search, or alternative timer."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys
import time
from typing import ClassVar, Literal, assert_never, override

from pydantic import BaseModel, ConfigDict, JsonValue, TypeAdapter

from kernel_optimizer.agents.self_test_artifacts import copy_dependencies, stage_candidate
from kernel_optimizer.agents.self_test_context import SelfTestContext
from kernel_optimizer.evaluation.benchmark import Benchmarker
from kernel_optimizer.evaluation.correctness import CorrectnessEvaluator
from kernel_optimizer.gpu.worker_client import WslGpuWorker, to_wsl_path
from kernel_optimizer.models.core import Backend, ParamSet
from kernel_optimizer.paramspace.materializer import MaterializeError


class Request(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid")

    context: Path
    candidate: Path
    params: Path | None = None
    mode: Literal["quick", "full"]
    output: Path
    backend: Backend = "triton"


class WorkerVerdict(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="ignore")

    ok: bool = False
    compiled: bool = False
    correct: bool = False
    latency_ms: dict[str, JsonValue] | None = None
    fp64_gate_enabled: bool | None = None
    fp64_rescued_trials: int | None = None


class RecordedWorker(WslGpuWorker):
    """Record only actual argv and evaluation environment, never inherited credentials."""

    def __init__(self, context: SelfTestContext, record_dir: Path):
        super().__init__(context.wsl, context.concurrency, context.jobs_dir,
                         worker_main_path=context.worker_main_path)
        self.record_dir = record_dir
        self.cutoff = context.formal_cutoff_unix_s
        self.admitted_submissions = 0

    @override
    def run_job(self, job: dict[str, JsonValue], timeout_s: float, tag: str,
                lock_mode: str = "exclusive") -> dict[str, JsonValue]:
        started = time.time()
        admitted = self.cutoff is None or started < self.cutoff
        try:
            if not admitted:
                return {"ok": False, "compiled": False, "correct": False,
                        "failure_kind": "pilot_cutoff", "log_tail": "formal helper admission cutoff reached"}
            self.admitted_submissions += 1
            return super().run_job(job, timeout_s, tag, lock_mode)
        finally:
            ended = time.time()
            record = {"tag": tag, "admitted": admitted, "started_unix_s": started, "ended_unix_s": ended,
                "formal_cutoff_unix_s": self.cutoff,
                "drain_s": max(0, ended - self.cutoff) if admitted and self.cutoff is not None else 0}
            with (self.record_dir / "submissions.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record) + "\n")

    @override
    def _build_command(self, job_path: Path, out_path: Path) -> tuple[list[str], dict[str, str]]:
        argv, env = super()._build_command(job_path, out_path)
        record = {"argv": argv, "job": str(job_path), "out": str(out_path),
                  "environment": {k: env[k] for k in ("PYTHONPATH", "TRITON_CACHE_DIR") if k in env}}
        with (self.record_dir / "commands.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record) + "\n")
        return argv, env

def run(request: Request) -> bool:
    """Evaluate an exact fresh artifact through the existing worker and shared lock."""
    started = time.monotonic()
    started_unix_s = time.time()
    context = SelfTestContext.model_validate_json(request.context.read_text(encoding="utf-8"))
    (request.output / "context.json").write_text(context.model_dump_json(indent=2), encoding="utf-8")
    params = (ParamSet.model_validate_json(request.params.read_text(encoding="utf-8"))
              if request.params else None)
    candidate, selected = stage_candidate(request.candidate, request.output / "source", params)
    source_sha = hashlib.sha256(candidate.read_bytes()).hexdigest()
    reference = copy_dependencies(context.task.ref_path, request.output / "reference")
    shutil.copytree(context.reference_dependencies, request.output / "reference_helpers")
    reference_sha = hashlib.sha256(reference.read_bytes()).hexdigest()
    if reference_sha != context.task.ref_src_sha:
        raise MaterializeError("REFERENCE_CHANGED", "reference snapshot differs from context hash")
    task = context.task.model_copy(update={"ref_path": reference})
    import_paths = [to_wsl_path(candidate.parent), to_wsl_path(request.output / "source"),
                    to_wsl_path(reference.parent), to_wsl_path(request.output / "reference_helpers"),
                    context.wsl.extra_pythonpath]
    wsl = context.wsl.model_copy(update={"extra_pythonpath": ":".join(filter(None, import_paths))})
    worker = RecordedWorker(context.model_copy(update={"wsl": wsl}), request.output)
    evaluator = CorrectnessEvaluator(worker, context.evaluation, context.concurrency, seed=context.seed)
    match request.mode:
        case "quick":
            raw = evaluator.quick_test(task, candidate, "agent-self-test", backend=request.backend)
        case "full":
            raw = Benchmarker(worker, evaluator, context.evaluation).final_reeval(
                task, candidate, backend=request.backend)
        case unreachable:
            assert_never(unreachable)
    (request.output / "raw_worker.json").write_text(json.dumps(raw, indent=2), encoding="utf-8")
    verdict = WorkerVerdict.model_validate(raw)
    latency = verdict.latency_ms or {}
    median = latency.get("median")
    score = median if median is not None else latency.get("mean")
    valid_number = isinstance(score, (int, float)) and not isinstance(score, bool) and math.isfinite(score) and score > 0
    valid = verdict.ok and verdict.compiled and verdict.correct and valid_number
    result: dict[str, JsonValue] = {
        "label": "agent_self_test", "mode": request.mode, "seed": context.seed,
        "compiled": verdict.compiled, "correct": verdict.correct, "formal_ok": verdict.ok,
        "result_valid": valid, "score_ms": score if valid else None,
        "raw_latency_ms": verdict.latency_ms,
        "correctness_mode": context.evaluation.correctness_mode,
        "fp64_gate_enabled": verdict.fp64_gate_enabled,
        "fp64_rescued_trials": verdict.fp64_rescued_trials,
        "materialized_source_sha256": source_sha,
        "reference_sha256": reference_sha, "params": selected.model_dump(),
        "configured_worker_python": context.configured_worker_python,
        "evaluation": context.evaluation.model_dump(), "wsl": wsl.model_dump(),
        "runtime_versions": "unknown; no environment probe performed",
        "warmup_contract": "owned by configured worker/KernelBench source, not an EvalConfig field",
        "source_path": context.source_path, "wall_time_s": time.monotonic() - started,
        "formal_cutoff_unix_s": context.formal_cutoff_unix_s,
        "started_unix_s": started_unix_s, "ended_unix_s": time.time(),
        "admitted_submissions": worker.admitted_submissions,
        "drain_s": max(0, time.time() - context.formal_cutoff_unix_s)
            if worker.admitted_submissions and context.formal_cutoff_unix_s is not None else 0,
    }
    serialized = TypeAdapter(dict[str, JsonValue]).dump_json(result, indent=2).decode()
    (request.output / "result.json").write_text(serialized, encoding="utf-8")
    print(serialized)
    return valid


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ("context", "candidate", "mode", "output"):
        parser.add_argument(f"--{option}", required=True)
    parser.add_argument("--params")
    parser.add_argument("--backend", default="triton", choices=("triton", "cuda"))
    args = parser.parse_args(argv)
    output: Path | None = None
    try:
        request = Request.model_validate(vars(args))
        request.output.mkdir(parents=True, exist_ok=False)
        output = request.output
        return 0 if run(request) else 1
    except (OSError, ValueError, SyntaxError, MaterializeError) as exc:
        error = {"label": "agent_self_test", "result_valid": False, "score_ms": None,
                 "error": str(exc)}
        if output is not None:
            (output / "result.json").write_text(json.dumps(error, indent=2), encoding="utf-8")
        print(json.dumps(error))
        return 1


if __name__ == "__main__":
    sys.exit(main())
