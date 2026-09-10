"""Baselines (eager + torch.compile) and independent final re-evaluation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from kernel_optimizer.config import EvalConfig
from kernel_optimizer.evaluation.correctness import CorrectnessEvaluator, latency_from_result
from kernel_optimizer.evaluation.task_cost import TaskCost, cost_from_worker
from kernel_optimizer.gpu.jobs import (
    make_baseline_job,
    make_probe_semantics_job,
    make_task_cost_job,
)
from kernel_optimizer.gpu.worker_client import WslGpuWorker
from kernel_optimizer.models.core import Baseline, LatencyStats, TaskSpec


def environment_defect(log_tail: str) -> str | None:
    """Is this worker failure the BOX's fault rather than the kernel's? Returns what to fix.

    A missing dependency and a wrong kernel arrive through the same channel -- `failure_kind:
    runtime_error` plus a traceback -- and they read identically to an operator and to a repair
    agent. That confusion has a measured cost: `kernelbench`'s package __init__ imports a chain
    reaching dotenv -> openai -> litellm, and on a venv where those are absent EVERY candidate came
    back `runtime_error`, so a 6-call re-run produced 12 usable candidates and 0 correct ones. "0
    correct" reads exactly like "the model wrote bad kernels" (G29).

    It recurred on a venv that had never evaluated anything before: box 1's own config pointed
    `wsl.venv` at an interpreter holding torch, triton and optuna but not kernelbench's import
    chain, and the run died at its eager baseline with the SAME ModuleNotFoundError on the SAME
    line. Installing packages on one box does not generalise; recognising the signature does.

    Deliberately narrow. It matches only an import failure of a module the harness never asks a
    CANDIDATE to import -- so a candidate that itself imports something absent is still the
    candidate's problem, which is a real case (an agent reaching for CUTLASS or TileLang, neither
    installed, must keep getting `compile_error` and a repair attempt). Widening this to any
    ModuleNotFoundError would silently reclassify those as box defects and stop the repair loop
    from ever seeing them.
    """
    if not log_tail:
        return None
    if "ModuleNotFoundError" not in log_tail and "ImportError" not in log_tail:
        return None
    # Only when the failing import happened INSIDE the evaluation library's own import chain.
    # `kernelbench/__init__` or `kernelbench/utils` in the frames is what distinguishes "this box
    # cannot load the evaluator" from "this kernel imports something odd".
    if "kernelbench/__init__" not in log_tail and "kernelbench/utils" not in log_tail:
        return None
    missing = ""
    for line in log_tail.splitlines():
        line = line.strip()
        if line.startswith(("ModuleNotFoundError", "ImportError")):
            missing = line
            break
    return (
        "THIS IS AN ENVIRONMENT DEFECT ON THIS BOX, NOT A PROBLEM WITH ANY KERNEL. The evaluation "
        "library (`kernelbench`) could not be imported by the worker interpreter, so nothing here "
        "can be evaluated and every candidate would come back `runtime_error` -- which reads "
        "exactly like the model writing bad kernels. %s. Fix: install the missing package into the "
        "venv named by `wsl.venv` in this run's config (NOT the interpreter that launched the CLI "
        "-- on some boxes those are deliberately different), then re-run "
        "`kernel-opt --config <this config> doctor`, whose `kernelbench importable in WSL` check "
        "covers exactly this, and `scripts/verify_box_can_evaluate.py` for an end-to-end proof."
        % (missing or "an import in kernelbench's own __init__ chain failed")
    )


class Benchmarker:
    def __init__(self, worker: WslGpuWorker, evaluator: CorrectnessEvaluator, cfg: EvalConfig):
        self.worker = worker
        self.evaluator = evaluator
        self.cfg = cfg

    def measure_task_cost(self, task: TaskSpec) -> TaskCost:
        """Step 3: how much arithmetic and traffic this TASK requires, from its reference.

        Shared/advisory like `probe_semantics`: a failure returns an empty TaskCost whose
        `summary_line()` says "not measured", so a box where this cannot run loses the
        denominators and nothing else. Never fatal.
        """
        job = make_task_cost_job(str(task.ref_path))
        try:
            result = self.worker.run_job(
                job, self.cfg.eval_timeout_s, "task-cost", lock_mode="shared")
        except Exception as exc:  # noqa: BLE001 — advisory, never fatal
            return TaskCost(notes=[f"task cost job failed: {type(exc).__name__}: {exc}"[:300]])
        if not result.get("ok"):
            return TaskCost(notes=[
                f"task cost job failed: {result.get('failure_kind')}: "
                f"{str(result.get('log_tail'))[-300:]}"])
        return cost_from_worker(result)

    def probe_semantics(self, task: TaskSpec) -> dict[str, Any]:
        """Improvement J: probe the reference's runtime eval semantics (train/eval
        mode + norm-layer flags). Best-effort — a failure returns an empty dict and
        the agent contract degrades gracefully (no forced assumption)."""
        job = make_probe_semantics_job(str(task.ref_path))
        try:
            result = self.worker.run_job(
                job, self.cfg.eval_timeout_s, "probe-semantics", lock_mode="shared")
        except Exception:  # noqa: BLE001 — probe is advisory, never fatal
            return {}
        if not result.get("ok"):
            return {}
        return {
            "training": result.get("training"),
            "norm_layers": result.get("norm_layers", []),
        }

    def measure_baseline(self, task: TaskSpec) -> list[Baseline]:
        baselines: list[Baseline] = []
        # Decision 2: under the dual-witness mode also record tf32-matmul baselines so
        # the speedup denominator is explicit (kernels may be timed against either).
        # ieee is always the primary, honest fp32 baseline.
        matmul_modes = [("ieee", "ieee")]
        if self.cfg.correctness_mode == "dual_witness_relaxed":
            matmul_modes.append(("tf32", "tf32"))
        for kind, use_compile in (("eager", False), ("torch_compile", True)):
            for mm_label, mm_mode in matmul_modes:
                suffix = "" if mm_mode == "ieee" else f"_{mm_label}"
                job = make_baseline_job(
                    str(task.ref_path),
                    num_trials=self.cfg.perf_trials,
                    timing_method=self.cfg.timing_method,
                    precision=self.cfg.precision,
                    use_torch_compile=use_compile,
                    matmul_precision=mm_mode,
                )
                timeout = self.cfg.eval_timeout_s + (
                    self.cfg.build_timeout_s if use_compile else 0)
                result = self.worker.run_job(job, timeout, f"baseline-{kind}{suffix}",
                                             lock_mode="exclusive")
                lat = latency_from_result(result)
                bkind = f"{kind}{suffix}"
                if lat is None:
                    if kind == "eager" and mm_mode == "ieee":
                        tail = str(result.get("log_tail", ""))
                        env = environment_defect(tail)
                        raise RuntimeError(
                            f"eager baseline failed for {task.name}: "
                            f"{result.get('failure_kind')}: {tail[:500]}"
                            + (f"\n\n{env}" if env else "")
                        )
                    baselines.append(
                        Baseline(
                            kind=bkind,
                            latency_ms=LatencyStats(mean=-1, std=0, min=-1, max=-1,
                                                    n_samples=0),
                            note=f"failed: {result.get('failure_kind')}",
                        )
                    )
                    continue
                note = "" if mm_mode == "ieee" else "tf32 matmul reference"
                baselines.append(Baseline(kind=bkind, latency_ms=lat, note=note))
        return baselines

    def final_reeval(self, task: TaskSpec, kernel_src_path: Path,
                     backend: str = "triton") -> dict[str, Any]:
        """Independent re-eval of theta_best: fresh process, full trials."""
        return self.evaluator.full_eval(task, kernel_src_path, tag="final-reeval",
                                        backend=backend)
