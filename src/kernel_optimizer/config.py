"""Config loading: YAML -> pydantic AppConfig, with dotted CLI overrides."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

from kernel_optimizer.models.core import DeviceLimits


class RunConfig(BaseModel):
    runs_dir: Path = Path("runs")
    seed: int = 0


class OpencodeConfig(BaseModel):
    server_url: str | None = None  # null => harness launches `opencode serve`
    launch_cwd: Path = Path("D:/Pyhon_projects/opop")
    host: str = "127.0.0.1"
    port: int = 4096
    agent: str = "build"
    request_timeout_s: float = 1200.0
    permission_mode: str = "sandbox_config"  # or "sse_auto_approve"
    startup_timeout_s: float = 60.0


class AgentModuleConfig(BaseModel):
    model: str | None = None  # None => agents.default_model
    max_retries: int = 2
    timeout_s: float = 1200.0
    n_candidates: int = 4  # generator / rewriter / novelty batch size
    # Transport (ReadTimeout / connection) failures retried on a FRESH session. Capped
    # separately from max_retries because a hung endpoint costs a full request_timeout_s
    # per attempt with no diagnostic value, whereas a schema-invalid response costs
    # seconds and the feedback usually fixes it. Default 2 preserves the previous total
    # retry budget exactly (max_retries=2 => 3 attempts), so the win here comes from the
    # fresh session, not from cutting attempts: the L3:43 repair that burned 0.99h timed
    # out twice while queued behind its own aborted turn on one session.
    max_transport_retries: int = 2


class AgentsConfig(BaseModel):
    default_model: str = "openai/gpt-5.6-sol"
    generator: AgentModuleConfig = AgentModuleConfig()
    parameterizer: AgentModuleConfig = AgentModuleConfig()
    analyst: AgentModuleConfig = AgentModuleConfig(max_retries=1)
    rewriter: AgentModuleConfig = AgentModuleConfig(n_candidates=2)
    novelty: AgentModuleConfig = AgentModuleConfig(n_candidates=2)
    repair: AgentModuleConfig = AgentModuleConfig()

    def module(self, name: str) -> AgentModuleConfig:
        cfg: AgentModuleConfig = getattr(self, name)
        if cfg.model is None:
            cfg = cfg.model_copy(update={"model": self.default_model})
        return cfg


class BudgetConfig(BaseModel):
    trials_per_space: int = 40
    rewrite_rounds_per_family: int = 3
    no_improve_rounds: int = 2
    min_improvement_pct: float = 2.0
    max_families_total: int = 3
    max_families_active: int = 2
    max_families_total_hard: int = 6  # absolute cap once dead families stop counting
    max_seed_candidates: int = 4
    repair_attempts: int = 2
    wall_clock_hours: float = 12.0
    # Improvement K: lightweight parameter-space expansion when a knob is at the
    # tried-range boundary and still improving with idle resources. 0 = disabled.
    space_expansions_per_candidate: int = 0
    space_expansion_idle_frac: float = 0.8  # resource must be < this frac of limit
    # Improvement B1: how many upcoming candidates' parameterizer calls to run on a
    # background thread while the current candidate holds the GPU. 0 = disabled
    # (fully synchronous, the pre-B1 behaviour). Only parameterization is prefetched;
    # analyst/rewriter depend on tuning results and stay strictly ordered.
    prefetch_parameterization: int = 1


class EvalConfig(BaseModel):
    correctness_trials: int = 5
    perf_trials: int = 100
    quick_correctness_trials: int = 3
    quick_perf_trials: int = 20
    timing_method: str = "cuda_event"
    precision: str = "fp32"
    atol: float = 1e-2
    rtol: float = 1e-2
    eval_timeout_s: float = 600.0
    build_timeout_s: float = 1200.0
    suspicious_speedup: float = 2.0
    excessive_speedup: float = 10.0
    # Improvement A: correctness judging mode.
    #   strict              -> KernelBench eval_kernel_against_ref (allclose 1e-4 for fp32)
    #   dual_witness_relaxed-> compare against the reference at BOTH tf32 and ieee fp32,
    #                          accept if EITHER matches under a relative-error slack gate.
    correctness_mode: str = "strict"
    relaxed_elem_tol: float = 0.01   # per-element relative error threshold
    relaxed_pass_frac: float = 0.99  # fraction of elements that must be within tol
    cosine_min: float = 0.99985      # second criterion: flattened cosine similarity
    # Second acceptance path, following torch._dynamo.utils.same(): compute an fp64
    # GOLDEN reference and accept when the candidate's RMSE against it is no more than
    # `fp64_rel_multiplier` x the REFERENCE's own RMSE against it. The absolute gate
    # above is unreachable on tasks whose own two-precision spread exceeds it -- all
    # three L3 tasks measure floors of 0.9554/0.9767/0.9778 against a 0.99 requirement --
    # and this measures the floor against TRUTH instead of comparing two imprecise
    # results. Both PyTorch (multiplier 2.0, or 3.0 for fp16/bf16 results) and
    # KernelBench (a 100x looser tolerance for a declared low-precision kernel) give a
    # low-precision candidate more slack than an fp32 one; we applied one tolerance to
    # every candidate. Does NOT replace the existing gate -- it is an additional way to
    # pass, so nothing previously accepted becomes rejected.
    fp64_relative_gate: bool = False
    fp64_rel_multiplier: float = 2.0       # torch uses 2.0 for fp32-class results
    fp64_rel_multiplier_lowp: float = 3.0  # ... and 3.0 for fp16/bf16 (avoids false alarms)


class GpuConcurrencyConfig(BaseModel):
    enabled: bool = True
    max_shared_jobs: int = 2  # correctness/compile/static-check only; timing is exclusive
    vram_budget_frac: float = 0.45
    timing_cooldown_s: float = 2.0


class WslConfig(BaseModel):
    distro: str = "Ubuntu"
    # MUST be on ext4, never under /mnt/* (a 9p mount of a Windows drive). Small-file
    # reads on 9p cost ~100x more than on ext4, and `import torch` reads thousands of
    # them: measured 26.8s vs 2.4s of per-worker startup (3x alternating, including a
    # real triton compile+launch). Every eval is a one-shot process, so with ~1000 jobs
    # per L3 run this alone was 6.7h of the 11.7h wall clock.
    # Built by scripts/setup_wsl_venv.sh (lean: torch/triton/numpy/pydantic/einops).
    venv: str = "~/kernel-opt-venv"
    # Read-only, a handful of files per job -> 9p is fine and keeps it Windows-visible.
    kernelbench_src: str = "/mnt/d/Pyhon_projects/opop/KernelBench/src"
    # Extra site dir for deps missing from the venv (pip --target), optional.
    extra_pythonpath: str = ""
    # Also ext4: the triton cache is read on every compile, same 9p penalty.
    triton_cache_dir: str = "~/.triton-cache-kopt"


class GpuConfig(BaseModel):
    concurrency: GpuConcurrencyConfig = GpuConcurrencyConfig()


class AppConfig(BaseModel):
    run: RunConfig = RunConfig()
    opencode: OpencodeConfig = OpencodeConfig()
    agents: AgentsConfig = AgentsConfig()
    budgets: BudgetConfig = BudgetConfig()
    evaluation: EvalConfig = EvalConfig()
    wsl: WslConfig = WslConfig()
    gpu: GpuConfig = GpuConfig()
    device: DeviceLimits = DeviceLimits()
    kernelbench_root: Path = Path("D:/Pyhon_projects/opop/KernelBench")


def _apply_override(data: dict[str, Any], dotted: str, value: str) -> None:
    keys = dotted.split(".")
    node = data
    for k in keys[:-1]:
        node = node.setdefault(k, {})
    # YAML-parse the value so "8" -> int, "true" -> bool, etc.
    node[keys[-1]] = yaml.safe_load(value)


def load_config(path: Path | None = None, overrides: list[str] | None = None) -> AppConfig:
    data: dict[str, Any] = {}
    if path is not None:
        loaded = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        if loaded:
            data = loaded
    for ov in overrides or []:
        if "=" not in ov:
            raise ValueError(f"override must be key.path=value, got: {ov}")
        key, _, value = ov.partition("=")
        _apply_override(data, key.strip(), value.strip())
    return AppConfig.model_validate(data)
