"""Correctness-before-timing evaluation built on the WSL worker."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from kernel_optimizer.config import EvalConfig, GpuConcurrencyConfig
from kernel_optimizer.gpu.jobs import (
    make_compile_probe_job,
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
    samples = lat.get("samples")
    return LatencyStats(
        mean=lat["mean"], std=lat["std"], min=lat["min"], max=lat["max"], n_samples=lat["n"],
        median=lat.get("median"),
        samples=tuple(samples) if samples else None,
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
        self._screen_cache: dict[str, dict[str, Any]] = {}

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

    def prescreen_batch(self, task: TaskSpec, kernel_src_paths: list[Path], tag: str,
                        backend: str) -> None:
        """Probe several materialized configurations in ONE worker process, into the cache.

        This is what makes the shared-memory screen affordable at SAMPLING time rather than
        after materializing a trial. Measured on box 2: a probe in its own worker process
        costs a median 16.7 s (11.9-25.6), essentially all of it process start plus
        torch/CUDA/KernelBench import, against the 18.6 s that the wasted trial it replaces
        already cost -- so probing one configuration per process saves nothing at all, and
        putting it in the `ask()` reject loop would let one `ask()` stall for
        64 x 16.7 s = 17.8 minutes. Forty-eight configurations in one process took 11.02 s
        total, a marginal 7 ms each: 73x cheaper.

        Exhaustive pre-screening is still out of reach -- the shared-affecting subgrid of a
        real space reaches 600,000 points, which is 70 minutes at 7 ms -- but
        `trials_per_space` is 40, so the screen only ever needs the corner the sampler
        actually visits. Hence: on demand, in batches, cached.

        Populates `_screen_cache` and returns nothing; `compile_screen` reads it. Silent on
        every failure by design -- a screen that cannot answer must leave the decision to a
        real trial.
        """
        if not kernel_src_paths:
            return
        pending: list[tuple[str, Path]] = []
        for path in kernel_src_paths:
            try:
                src = Path(path).read_text(encoding="utf-8")
            except OSError:
                continue
            key = f"{backend}:{hash(src)}"
            if key not in self._screen_cache:
                pending.append((key, Path(path)))
        if not pending:
            return
        first = pending[0][1]
        job = make_compile_probe_job(str(task.ref_path), str(first), backend=backend,
                                     extra_kernel_src_paths=[str(p) for _, p in pending[1:]])
        try:
            probe = self.worker.run_job(job, self.cfg.build_timeout_s,
                                        f"{tag}-prescreen", lock_mode="shared")
        except Exception as exc:  # noqa: BLE001 — a screen failure is never a verdict
            probe = {"ok": False, "reason": f"{type(exc).__name__}: {exc}"[:200]}
        per_path = probe.get("results") or {}
        for key, path in pending:
            # Fall back to the top-level verdict for the primary path, which is what a
            # single-variant job returns. A variant the worker could not answer for keeps
            # its own `ok: False`, so it falls through to a real trial on its own rather
            # than being suppressed by a sibling's failure.
            entry = per_path.get(str(path))
            if entry is None:
                entry = probe if path == first else {
                    "ok": False, "reason": "not answered by the batch probe"}
            if not entry.get("ok"):
                # DO NOT CACHE A NON-ANSWER. Measured on box 3's run-l3-48-20260911-052647:
                # one batch of 40 was killed mid-probe when the run took a SIGTERM, and
                # because the failure was written into the cache for all 40 keys, every one
                # of them was unscreenable for the rest of the run -- `compile_screen` reads
                # this same cache and returns early on a non-ok entry without re-probing. Three
                # configurations then reached a real trial and hit Triton's own
                # `out of resource: shared memory`, one of them burning 2531 s of a 12 h budget
                # (0.71 h across the three) on a launch the compiler could have refused in 7 ms.
                #
                # An "ok" verdict is a fact about the source and caching it is the whole point.
                # A FAILURE is a fact about that one probe attempt -- the worker, the timeout,
                # the signal -- so caching it converts a transient into a permanent blind spot.
                # Leaving the key absent costs at most one re-probe (7 ms marginal in a batch,
                # ~16.7 s alone) and restores the screen for every later trial.
                continue
            self._screen_cache[key] = entry

    def cached_shared_verdict(self, kernel_src: str, backend: str,
                             max_shared_bytes: int | None) -> bool | None:
        """Is this materialized source known-infeasible? True = fits, False = cannot launch,
        None = not screened, so the caller must NOT draw a conclusion.

        Reads the cache only; never talks to the worker. That is what lets it be called from
        the sampler's guard, where a GPU round-trip is unaffordable (see `prescreen_batch`
        for the measured costs). Three-valued on purpose: a boolean would force "unknown" to
        collapse into one of the verdicts, and either collapse is a defect -- treating
        unknown as infeasible silently shrinks the search space, treating it as feasible is
        merely the status quo.
        """
        if not max_shared_bytes:
            return None
        probe = self._screen_cache.get(f"{backend}:{hash(kernel_src)}")
        if probe is None or not probe.get("ok"):
            return None
        max_shared = probe.get("max_shared")
        if max_shared is None:
            return None
        return max_shared <= max_shared_bytes

    def compile_screen(self, task: TaskSpec, kernel_src_path: Path, tag: str,
                       backend: str, max_shared_bytes: int | None) -> dict[str, Any] | None:
        """Compile-only feasibility screen. Returns a refusal dict, or None to proceed.

        WHY. On run-l3-43-20260908-053708, 180 of 1004 tuning trials (18% of the budget,
        0.93 h of 11.73 h) failed with Triton's `out of resource: shared memory` -- and every
        one was decidable before the launch, because the compiler fills `metadata.shared` and
        the launch only compares it against the device limit. Probed on box 2 across 6
        configurations of a real pipelined matmul, the compile-time figure equals the runtime
        `Required:` value byte-for-byte.

        Not expressible as a guard constraint: audited over all 16 L3:43 candidates,
        `BLOCK_M * BLOCK_N * stages` has failing-min BELOW passing-max in 15 of 15, so the
        feasible and infeasible sets overlap in any such product.

        REFUSES ONLY on the compiler's own figure exceeding the device's own limit. Every
        other outcome -- probe unavailable, no Triton kernel reached, probe raised, figure
        within the limit -- returns None so the real trial decides. This screen must never be
        the thing that rejects a candidate.

        Cached on the materialized source, since a re-tune after a space expansion re-asks
        configurations it has already screened. ONLY answers are cached: see `prescreen_batch`
        for the measured cost of caching a non-answer.
        """
        if not max_shared_bytes:
            return None
        src = Path(kernel_src_path).read_text(encoding="utf-8")
        key = f"{backend}:{hash(src)}"
        probe = self._screen_cache.get(key)
        if probe is None:
            job = make_compile_probe_job(str(task.ref_path), str(kernel_src_path),
                                         backend=backend)
            try:
                probe = self.worker.run_job(job, self.cfg.build_timeout_s,
                                           f"{tag}-compile-screen", lock_mode="shared")
            except Exception as exc:  # noqa: BLE001 — a screen failure is never a verdict
                probe = {"ok": False, "reason": f"{type(exc).__name__}: {exc}"[:200]}
            if probe.get("ok"):
                # Same rule as the batch: an ANSWER is a fact about the source and belongs in
                # the cache; a FAILURE is a fact about this one probe attempt (a worker timeout,
                # a signal, a transient) and caching it makes a transient permanent. The re-tune
                # this cache exists for re-asks the same configurations, so a cached failure
                # would blind the screen for exactly the configurations it is meant to catch.
                self._screen_cache[key] = probe
        if not probe.get("ok"):
            return None
        max_shared = probe.get("max_shared")
        if max_shared is None or max_shared <= max_shared_bytes:
            return None
        worst = next((k for k in (probe.get("kernels") or [])
                      if (k.get("shared") or 0) == max_shared), {})
        return {
            "kernel": worst.get("name"),
            "max_shared": max_shared,
            "limit": max_shared_bytes,
            "detail": (
                f"compile-only screen: {worst.get('name') or 'a kernel'} requires {max_shared} "
                f"bytes of shared memory against this device's per-block opt-in limit of "
                f"{max_shared_bytes} ({100 * max_shared / max_shared_bytes:.0f}%), so a launch "
                f"could only raise `out of resource: shared memory`. Refused without launching; "
                f"the figure is the Triton compiler's own metadata.shared."),
        }

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
                collect_kernel_metadata=True,
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
                collect_kernel_metadata=True,
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
             correct_trials: int, perf_trials: int,
             measure_launch_overhead: bool = False) -> dict[str, Any]:
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
                collect_kernel_metadata=True,
                relaxed_elem_tol=self.cfg.relaxed_elem_tol,
                relaxed_pass_frac=self.cfg.relaxed_pass_frac,
                cosine_min=self.cfg.cosine_min,
                fp64_relative_gate=self.cfg.fp64_relative_gate,
                fp64_rel_multiplier=self.cfg.fp64_rel_multiplier,
                fp64_rel_multiplier_lowp=self.cfg.fp64_rel_multiplier_lowp,
            )
            job["num_perf_trials"] = perf_trials
            job["measure_launch_overhead"] = measure_launch_overhead
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
                collect_kernel_metadata=True,
                excessive_speedup_threshold=self.cfg.excessive_speedup,
                measure_launch_overhead=measure_launch_overhead,
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
        # Launch overhead is measured HERE and not in quick_test: this path already runs
        # `perf_trials` (100) timed samples, so ~150 extra model calls is marginal, whereas
        # quick_test runs on every one of a run's hundreds of tuning trials. `measure_overhead`
        # below covers the other place the number is needed -- the per-candidate verdict, which
        # comes from tuning trials and therefore never saw this path.
        return self._run(task, kernel_src_path, backend, tag,
                         self.cfg.correctness_trials, self.cfg.perf_trials,
                         measure_launch_overhead=True)

    def measure_overhead(self, task: TaskSpec, kernel_src_path: Path, tag: str,
                         backend: str = "triton") -> dict[str, Any]:
        """Launch-overhead probe alone, for one already-validated configuration.

        Exists so the bottleneck classifier can have `cpu_issue_ms` without a second 100-sample
        evaluation. `launch_bound` had never fired in any run partly because this number only
        ever came from `full_eval` -- whose sole caller is the final re-eval -- while every
        verdict is computed from tuning trials: `cpu_issue_ms` was None in 848 of 848 trial
        profiles.

        Correctness trials are set to the minimum rather than skipped, because the worker's
        handlers only reach the overhead block on a candidate they have confirmed correct, and
        a probe that ran on a broken kernel would report the cost of raising an exception.
        Exclusive lane: it times.
        """
        if self.cfg.correctness_mode == "dual_witness_relaxed":
            job = make_relaxed_correctness_job(
                str(task.ref_path), str(kernel_src_path),
                num_correct_trials=1,
                backend=backend,
                precision=self.cfg.precision,
                seed=self.seed,
                collect_kernel_metadata=False,
                relaxed_elem_tol=self.cfg.relaxed_elem_tol,
                relaxed_pass_frac=self.cfg.relaxed_pass_frac,
                cosine_min=self.cfg.cosine_min,
                fp64_relative_gate=self.cfg.fp64_relative_gate,
                fp64_rel_multiplier=self.cfg.fp64_rel_multiplier,
                fp64_rel_multiplier_lowp=self.cfg.fp64_rel_multiplier_lowp,
            )
            job["num_perf_trials"] = 0
        else:
            job = make_eval_job(
                str(task.ref_path), str(kernel_src_path),
                measure_performance=False,
                num_correct_trials=1,
                num_perf_trials=0,
                timing_method=self.cfg.timing_method,
                backend=backend,
                precision=self.cfg.precision,
                seed=self.seed,
                build_dir=None,
                collect_kernel_metadata=False,
                excessive_speedup_threshold=self.cfg.excessive_speedup,
            )
        job["measure_launch_overhead"] = True
        result = self.worker.run_job(job, self.cfg.build_timeout_s + self.cfg.eval_timeout_s,
                                     f"{tag}-overhead", lock_mode="exclusive")
        result["phase"] = "launch_overhead"
        return result
