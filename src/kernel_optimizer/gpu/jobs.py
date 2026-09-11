"""GPU job/result dict schemas. Stdlib types only — shared with the WSL worker."""

from __future__ import annotations

from typing import Any

JOB_TYPES = ("baseline", "eval_correctness", "eval_perf", "static_check", "env_probe",
             "eval_correctness_relaxed", "probe_semantics", "probe_noise_floor", "calibrate",
             "task_cost")

FAILURE_KINDS = (
    "compile_error",
    "runtime_error",
    "correctness_mismatch",
    "oom",
    "timeout",
    "worker_crash",
    "static_check_failed",
)


def make_env_probe_job() -> dict[str, Any]:
    return {"job_type": "env_probe"}


def make_calibrate_job() -> dict[str, Any]:
    """Measure this box's ceilings and yardstick workloads (step 2's calibrator).

    Takes no parameters on purpose. Every size inside is derived from the card's own free VRAM
    and L2, so the same job calibrates a 16 GB laptop GPU and a 24 GB datacenter-adjacent one
    without a config knob to get wrong.
    """
    return {"job_type": "calibrate"}


def make_task_cost_job(ref_src_path: str) -> dict[str, Any]:
    """Count the arithmetic and traffic this task requires, from its REFERENCE (step 3).

    On the reference, so the numbers are a property of the TASK and identical for every candidate
    optimizing it -- which is what makes them a usable shared denominator for "% of peak".
    Counting a candidate instead would measure the quantity being optimized.
    """
    return {"job_type": "task_cost", "ref_src_path": ref_src_path}


def make_probe_semantics_job(ref_src_path: str) -> dict[str, Any]:
    """Improvement J: probe the reference model's runtime eval semantics
    (train/eval mode + norm-layer flags) so the agent can match them. Reads the
    live model object's state — not the source text — so it is correct regardless
    of how the reference is written."""
    return {"job_type": "probe_semantics", "ref_src_path": ref_src_path}


def make_static_check_job(kernel_src_path: str, backend: str, precision: str) -> dict[str, Any]:
    return {
        "job_type": "static_check",
        "kernel_src_path": kernel_src_path,
        "backend": backend,
        "precision": precision,
    }


def make_compile_probe_job(ref_src_path: str, kernel_src_path: str, *,
                           backend: str,
                           extra_kernel_src_paths: list[str] | None = None) -> dict[str, Any]:
    """Compile-only feasibility probe: shared bytes without a launch. See run_compile_probe.

    `extra_kernel_src_paths` probes further materialized variants of the same candidate in the
    SAME worker process, which is the only way this screen is affordable during sampling:
    measured on box 2, one probe in its own process costs a median 16.7 s -- almost entirely
    process start plus torch/CUDA/KernelBench import -- against the 18.6 s the wasted trial it
    replaces already cost, so one-per-process saves nothing. Forty-eight variants in one
    process took 11.02 s, a marginal 7 ms each.

    The result then carries a per-path `results` map alongside the primary variant's own
    verdict, so a single-path caller sees an unchanged shape.
    """
    job = {
        "job_type": "compile_probe",
        "ref_src_path": ref_src_path,
        "kernel_src_path": kernel_src_path,
        "backend": backend,
    }
    if extra_kernel_src_paths:
        job["extra_kernel_src_paths"] = list(extra_kernel_src_paths)
    return job


def make_baseline_job(
    ref_src_path: str,
    *,
    num_trials: int,
    timing_method: str,
    precision: str,
    use_torch_compile: bool,
    matmul_precision: str | None = None,
) -> dict[str, Any]:
    return {
        "job_type": "baseline",
        "ref_src_path": ref_src_path,
        "num_trials": num_trials,
        "timing_method": timing_method,
        "precision": precision,
        "use_torch_compile": use_torch_compile,
        "matmul_precision": matmul_precision,
    }


def make_eval_job(
    ref_src_path: str,
    kernel_src_path: str,
    *,
    measure_performance: bool,
    num_correct_trials: int,
    num_perf_trials: int,
    timing_method: str,
    backend: str,
    precision: str,
    seed: int,
    build_dir: str | None,
    collect_kernel_metadata: bool,
    plausibility: dict[str, Any] | None = None,
    measure_launch_overhead: bool = False,
) -> dict[str, Any]:
    job = {
        "job_type": "eval_perf" if measure_performance else "eval_correctness",
        "ref_src_path": ref_src_path,
        "kernel_src_path": kernel_src_path,
        "num_correct_trials": num_correct_trials,
        "num_perf_trials": num_perf_trials,
        "timing_method": timing_method,
        "backend": backend,
        "precision": precision,
        "seed": seed,
        "build_dir": build_dir,
        # Backend-NEUTRAL: the worker picks the Triton JIT reader or the cubin reader from
        # `backend`. It used to be `collect_triton_metadata`, set by callers as
        # `(backend == "triton")`, which made the cubin path for cuda/cutlass/cute
        # unreachable -- see `_wants_kernel_metadata` in worker_main.py.
        "collect_kernel_metadata": collect_kernel_metadata,
        # Off by default. Requested only by full_eval and the baselines: it costs ~150 extra
        # model calls, which is negligible beside their 100 timed samples but would be a real
        # tax on the 20-sample tuning trials, of which a run does hundreds.
        "measure_launch_overhead": measure_launch_overhead,
    }
    # The plausibility bound, when the driver could compute one. MERGED rather than defaulted:
    # a job with no bound must carry no threshold key at all, so the worker reports
    # `plausibility_checked: False` instead of comparing against a constant nobody derived.
    if plausibility:
        job.update(plausibility)
    return job


def make_relaxed_correctness_job(
    ref_src_path: str,
    kernel_src_path: str,
    *,
    num_correct_trials: int,
    backend: str,
    precision: str,
    seed: int,
    collect_kernel_metadata: bool,
    relaxed_elem_tol: float,
    relaxed_pass_frac: float,
    cosine_min: float,
    fp64_relative_gate: bool = False,
    fp64_rel_multiplier: float = 2.0,
    fp64_rel_multiplier_lowp: float = 3.0,
    plausibility: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Improvement A: dual-precision witness relaxed correctness (no timing)."""
    job = {
        "job_type": "eval_correctness_relaxed",
        "ref_src_path": ref_src_path,
        "kernel_src_path": kernel_src_path,
        "num_correct_trials": num_correct_trials,
        "num_perf_trials": 0,
        "backend": backend,
        "precision": precision,
        "seed": seed,
        "collect_kernel_metadata": collect_kernel_metadata,
        "relaxed_elem_tol": relaxed_elem_tol,
        "relaxed_pass_frac": relaxed_pass_frac,
        "cosine_min": cosine_min,
        "fp64_relative_gate": fp64_relative_gate,
        "fp64_rel_multiplier": fp64_rel_multiplier,
        "fp64_rel_multiplier_lowp": fp64_rel_multiplier_lowp,
    }
    if plausibility:
        job.update(plausibility)
    return job


def failure_result(kind: str, detail: str) -> dict[str, Any]:
    return {
        "ok": False,
        "compiled": False,
        "correct": False,
        "failure_kind": kind,
        "log_tail": detail[-4000:],
    }