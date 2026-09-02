"""Baselines (eager + torch.compile) and independent final re-evaluation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from kernel_optimizer.config import EvalConfig
from kernel_optimizer.evaluation.correctness import CorrectnessEvaluator, latency_from_result
from kernel_optimizer.gpu.jobs import make_baseline_job
from kernel_optimizer.gpu.worker_client import WslGpuWorker
from kernel_optimizer.models.core import Baseline, LatencyStats, TaskSpec


class Benchmarker:
    def __init__(self, worker: WslGpuWorker, evaluator: CorrectnessEvaluator, cfg: EvalConfig):
        self.worker = worker
        self.evaluator = evaluator
        self.cfg = cfg

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
                        raise RuntimeError(
                            f"eager baseline failed for {task.name}: "
                            f"{result.get('failure_kind')}: {result.get('log_tail', '')[:500]}"
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
                note = None if mm_mode == "ieee" else "tf32 matmul reference"
                baselines.append(Baseline(kind=bkind, latency_ms=lat, note=note))
        return baselines

    def final_reeval(self, task: TaskSpec, kernel_src_path: Path,
                     backend: str = "triton") -> dict[str, Any]:
        """Independent re-eval of theta_best: fresh process, full trials."""
        return self.evaluator.full_eval(task, kernel_src_path, tag="final-reeval",
                                        backend=backend)
