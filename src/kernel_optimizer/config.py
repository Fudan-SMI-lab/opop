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


class GpuConcurrencyConfig(BaseModel):
    enabled: bool = True
    max_shared_jobs: int = 2  # correctness/compile/static-check only; timing is exclusive
    vram_budget_frac: float = 0.45
    timing_cooldown_s: float = 2.0


class WslConfig(BaseModel):
    distro: str = "Ubuntu"
    # Reuse the proven kernelfoundry WSL venv (torch 2.9+cu129, triton 3.5).
    venv: str = "/mnt/d/Pyhon_projects/opop/kernelfoundry/.venv-wsl"
    kernelbench_src: str = "/mnt/d/Pyhon_projects/opop/KernelBench/src"
    # Extra site dir for deps missing from the venv (pip --target), optional.
    extra_pythonpath: str = ""
    triton_cache_dir: str = "/mnt/d/Pyhon_projects/opop/v2/.triton-cache-wsl"


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
