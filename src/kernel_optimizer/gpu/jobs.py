"""GPU job/result dict schemas. Stdlib types only — shared with the WSL worker."""

from __future__ import annotations

from typing import Any

JOB_TYPES = ("baseline", "eval_correctness", "eval_perf", "static_check", "env_probe",
             "eval_correctness_relaxed", "probe_semantics")

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
    collect_triton_metadata: bool,
    excessive_speedup_threshold: float = 10.0,
) -> dict[str, Any]:
    return {
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
        "collect_triton_metadata": collect_triton_metadata,
        "excessive_speedup_threshold": excessive_speedup_threshold,
    }


def make_relaxed_correctness_job(
    ref_src_path: str,
    kernel_src_path: str,
    *,
    num_correct_trials: int,
    backend: str,
    precision: str,
    seed: int,
    collect_triton_metadata: bool,
    relaxed_elem_tol: float,
    relaxed_pass_frac: float,
    cosine_min: float,
    fp64_relative_gate: bool = False,
    fp64_rel_multiplier: float = 2.0,
    fp64_rel_multiplier_lowp: float = 3.0,
) -> dict[str, Any]:
    """Improvement A: dual-precision witness relaxed correctness (no timing)."""
    return {
        "job_type": "eval_correctness_relaxed",
        "ref_src_path": ref_src_path,
        "kernel_src_path": kernel_src_path,
        "num_correct_trials": num_correct_trials,
        "num_perf_trials": 0,
        "backend": backend,
        "precision": precision,
        "seed": seed,
        "collect_triton_metadata": collect_triton_metadata,
        "relaxed_elem_tol": relaxed_elem_tol,
        "relaxed_pass_frac": relaxed_pass_frac,
        "cosine_min": cosine_min,
        "fp64_relative_gate": fp64_relative_gate,
        "fp64_rel_multiplier": fp64_rel_multiplier,
        "fp64_rel_multiplier_lowp": fp64_rel_multiplier_lowp,
    }


def failure_result(kind: str, detail: str) -> dict[str, Any]:
    return {
        "ok": False,
        "compiled": False,
        "correct": False,
        "failure_kind": kind,
        "log_tail": detail[-4000:],
    }