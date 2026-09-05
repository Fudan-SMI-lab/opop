"""Correctness-before-timing evaluation built on the WSL worker."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from kernel_optimizer.config import EvalConfig, GpuConcurrencyConfig
from kernel_optimizer.gpu.jobs import (
    make_eval_job,
    make_relaxed_correctness_job,
    make_static_check_job,
)
from kernel_optimizer.gpu.worker_client import WslGpuWorker
from kernel_optimizer.models.core import LatencyStats, TaskSpec


def latency_from_result(result: dict[str, Any]) -> LatencyStats | None:
    lat = result.get("latency_ms")
    if not lat or lat.get("mean", -1) < 0:
        return None
    return LatencyStats(
        mean=lat["mean"], std=lat["std"], min=lat["min"], max=lat["max"], n_samples=lat["n"]
    )


class CorrectnessEvaluator:
    """quick_test / full_eval: static check (cached per source) -> one merged
    eval job (correctness-before-timing inside eval_kernel_against_ref).

    The merged job runs in the exclusive lane (it times). Screening-only
    correctness jobs (`screen`) run in the shared lane and may be concurrent.
    """

    def __init__(
        self,
        worker: WslGpuWorker,
        cfg: EvalConfig,
        conc: GpuConcurrencyConfig,
        seed: int = 42,
    ):
        self.worker = worker
        self.cfg = cfg
        self.conc = conc
        self.seed = seed
        self._static_cache: dict[str, dict[str, Any]] = {}

    def _static_check(self, task: TaskSpec, kernel_src_path: Path, backend: str,
                      tag: str) -> dict[str, Any]:
        src = Path(kernel_src_path).read_text(encoding="utf-8")
        # Cache on the structure: PARAMS literal values never change check results.
        import re

        normalized = re.sub(r"PARAMS\s*=\s*\{[^}]*\}", "PARAMS={}", src, count=1)
        key = f"{backend}:{hash(normalized)}"
        if key in self._static_cache:
            return self._static_cache[key]
        job = make_static_check_job(str(kernel_src_path), backend, self.cfg.precision)
        result = self.worker.run_job(job, self.cfg.eval_timeout_s, f"{tag}-static",
                                     lock_mode="shared")
        result["phase"] = "static_check"
        self._static_cache[key] = result
        return result

    def screen(self, task: TaskSpec, kernel_src_path: Path, tag: str,
               backend: str = "triton") -> dict[str, Any]:
        """Correctness-only screening (shared lane, concurrent-safe)."""
        static = self._static_check(task, kernel_src_path, backend, tag)
        if not static.get("ok"):
            return static
        if self.cfg.correctness_mode == "dual_witness_relaxed":
            job = make_relaxed_correctness_job(
                str(task.ref_path), str(kernel_src_path),
                num_correct_trials=self.cfg.quick_correctness_trials,
                backend=backend,
                precision=self.cfg.precision,
                seed=self.seed,
                collect_triton_metadata=(backend == "triton"),
                relaxed_elem_tol=self.cfg.relaxed_elem_tol,
                relaxed_pass_frac=self.cfg.relaxed_pass_frac,
                cosine_min=self.cfg.cosine_min,
                fp64_relative_gate=self.cfg.fp64_relative_gate,
                fp64_rel_multiplier=self.cfg.fp64_rel_multiplier,
                fp64_rel_multiplier_lowp=self.cfg.fp64_rel_multiplier_lowp,
            )  # num_perf_trials defaults to 0 -> correctness only
        else:
            job = make_eval_job(
                str(task.ref_path), str(kernel_src_path),
                measure_performance=False,
                num_correct_trials=self.cfg.quick_correctness_trials,
                num_perf_trials=0,
                timing_method=self.cfg.timing_method,
                backend=backend,
                precision=self.cfg.precision,
                seed=self.seed,
                build_dir=None,
                collect_triton_metadata=(backend == "triton"),
                excessive_speedup_threshold=self.cfg.excessive_speedup,
            )
        result = self.worker.run_job(job, self.cfg.build_timeout_s + self.cfg.eval_timeout_s,
                                     f"{tag}-screen", lock_mode="shared")
        if result.get("failure_kind") == "oom" and self.conc.enabled:
            result = self.worker.run_job(job, self.cfg.build_timeout_s + self.cfg.eval_timeout_s,
                                         f"{tag}-screen-retry", lock_mode="exclusive")
        result["phase"] = "screen"
        result["static_warnings"] = static.get("warnings", [])
        return result

    def _run(self, task: TaskSpec, kernel_src_path: Path, backend: str, tag: str,
             correct_trials: int, perf_trials: int) -> dict[str, Any]:
        static = self._static_check(task, kernel_src_path, backend, tag)
        if not static.get("ok"):
            return static

        if self.cfg.correctness_mode == "dual_witness_relaxed":
            job = make_relaxed_correctness_job(
                str(task.ref_path), str(kernel_src_path),
                num_correct_trials=correct_trials,
                backend=backend,
                precision=self.cfg.precision,
                seed=self.seed,
                collect_triton_metadata=(backend == "triton"),
                relaxed_elem_tol=self.cfg.relaxed_elem_tol,
                relaxed_pass_frac=self.cfg.relaxed_pass_frac,
                cosine_min=self.cfg.cosine_min,
                fp64_relative_gate=self.cfg.fp64_relative_gate,
                fp64_rel_multiplier=self.cfg.fp64_rel_multiplier,
                fp64_rel_multiplier_lowp=self.cfg.fp64_rel_multiplier_lowp,
            )
            job["num_perf_trials"] = perf_trials
        else:
            job = make_eval_job(
                str(task.ref_path), str(kernel_src_path),
                measure_performance=True,
                num_correct_trials=correct_trials,
                num_perf_trials=perf_trials,
                timing_method=self.cfg.timing_method,
                backend=backend,
                precision=self.cfg.precision,
                seed=self.seed,
                build_dir=None,
                collect_triton_metadata=(backend == "triton"),
                excessive_speedup_threshold=self.cfg.excessive_speedup,
            )
        result = self.worker.run_job(job, self.cfg.build_timeout_s + self.cfg.eval_timeout_s,
                                     f"{tag}-eval", lock_mode="exclusive")
        result["phase"] = "eval"
        result["static_warnings"] = static.get("warnings", [])
        return result

    def quick_test(self, task: TaskSpec, kernel_src_path: Path, tag: str,
                   backend: str = "triton") -> dict[str, Any]:
        return self._run(task, kernel_src_path, backend, tag,
                         self.cfg.quick_correctness_trials, self.cfg.quick_perf_trials)

    def full_eval(self, task: TaskSpec, kernel_src_path: Path, tag: str,
                  backend: str = "triton") -> dict[str, Any]:
        return self._run(task, kernel_src_path, backend, tag,
                         self.cfg.correctness_trials, self.cfg.perf_trials)
