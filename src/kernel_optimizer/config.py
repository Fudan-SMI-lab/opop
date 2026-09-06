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
    # 30 min. Was 1200s (20 min), which was killing real work: across the five completed L3 runs
    # 8 agent calls died at exactly 1200-1201s with `prompt transport error (ReadTimeout)` --
    # 5 repairs, 2 rewriters, 1 generator, spread over all three tasks. Each kill discards a
    # candidate or a whole rewrite round, so the cost is search progress, not just a retry.
    # Sizing: the slowest SUCCESSFUL call measured is 576s (generator p90), and one glm-5.3
    # generator call on L3:21 took 979s (16m19s) with a single large reasoning turn -- so 20 min
    # left under 4 min of headroom for the reasoning-heavy arm. 1800s clears both by >=1.8x.
    # This is model-agnostic: it is the transport read timeout, not a token or effort setting.
    request_timeout_s: float = 1800.0
    permission_mode: str = "sandbox_config"  # or "sse_auto_approve"
    startup_timeout_s: float = 60.0
    # Merged into every agent sandbox's opencode.json. That file makes the sandbox a
    # project root, which stops opencode's upward config search — so a provider declared
    # only in an ancestor directory cannot be resolved from inside a sandbox. Providers in
    # the user's GLOBAL config still work (it is always loaded), which is why openai has
    # never needed this and a repo-local provider does. Point it at a file with
    # `sandbox_config_path` instead of inlining secrets in the experiment config.
    sandbox_extra_config: dict = Field(default_factory=dict)
    sandbox_config_path: Path | None = None
    # Environment variables for the `opencode serve` process. Some opencode settings have
    # NO config-file route and are read only from the environment; the one we need is
    # OPENCODE_EXPERIMENTAL_OUTPUT_TOKEN_MAX, which sets the per-turn output-token ceiling.
    # The binary computes it as `Math.min(model.limit.output, ENV ?? 32000)`, so the env var
    # is a hard CEILING that config can only lower: raising a model's `limit.output` alone
    # cannot get past the 32000 default (measured, scripts/probe_glm_limit_output.py). A
    # reasoning-heavy model that needs more than 32000 output+reasoning tokens per turn is
    # truncated mid-thought without it.
    server_env: dict[str, str] = Field(default_factory=dict)


class AgentModuleConfig(BaseModel):
    model: str | None = None  # None => agents.default_model
    max_retries: int = 2
    # NOT CONSUMED ANYWHERE (verified: nothing reads `AgentModuleConfig.timeout_s`). The single
    # effective ceiling is OpencodeConfig.request_timeout_s, which sets the httpx client timeout
    # in wiring.py. Kept in step with it so a reader who sets this per-module does not end up
    # with a value that silently contradicts the real one -- but setting it changes nothing.
    timeout_s: float = 1800.0
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
    # 5, was 3. At 3 the round cap fired BEFORE the convergence test it exists to defer to:
    # across 18 L3 runs, 16 families froze on `budget_exhausted` and 3 of them were still
    # gaining >=2% on that final round -- and those 3 gains (47.7%, 44.5%, 11.5%) are the
    # largest single-round improvements in the project. Their trajectories were accelerating
    # (18.6 -> 15.4 -> 8.1), not plateauing. Meanwhile the wall clock was NOT the constraint:
    # both runs froze their best families at ~6.3 h of a 12 h budget, leaving ~45% unused.
    #
    # `no_improve_rounds` + `min_improvement_pct` already stop a family that has stopped
    # improving, so raising this cap hands the stopping decision back to the mechanism designed
    # for it, with `wall_clock_hours` as the hard backstop. Measured cost of one round: 38.9 min
    # median / 46.3 min mean, so +2 rounds is +2.6 h median for the 2 concurrently-active
    # families (max_families_active=2) and +5.2 h in the pessimistic all-4 case.
    #
    # Why not a progress-conditional cap instead: `min_improvement_pct` is RELATIVE, so a 9.43 ms
    # family must find 0.19 ms to qualify while an 18.6 ms family needs 0.37 ms. A conditional
    # cap therefore penalises the fastest families -- exactly the ones most likely to produce the
    # run's best result -- so it would cut off the families we most want to continue.
    rewrite_rounds_per_family: int = 5
    no_improve_rounds: int = 2
    min_improvement_pct: float = 2.0
    # 6, was 3. At 3 Loop D was unreachable: `max_seed_candidates: 4` produces 4 seed
    # families, so the novelty gate's `>= max_families_total` was true before the first
    # check. Across all 19 runs `origin:novelty` is 0 and `module=novelty` agent calls are 0
    # -- one of the paper's four loops had no evidence at all. 6 leaves room for 2 novel
    # families on top of 4 seeds, and matches `max_families_total_hard` so the two caps agree.
    #
    # Verified before raising it (configs/smoke_l1_novelty.yaml, run-l1-19-20260906-183211):
    # the NoveltyGeneratorAgent path works end to end -- prompt, schema, sandbox seeding,
    # similarity gate, registration, parameterization, tuning. Raising this without that
    # smoke would have risked discovering a broken code path hours into an L3 run.
    max_families_total: int = 6
    # 3, was 2. Raised together with enabling Loop D (novelty). `active_families()` ranks
    # families with `rewrite_rounds_used == 0` FIRST -- deliberately, so a branch is never
    # dropped before it has shown its headroom -- which means a newly injected novelty
    # family jumps the queue ahead of the incumbents. At 2 slots it would displace the two
    # families currently holding the best latencies, and those are the ones most likely to
    # produce the run's winner (on run-l3-43-20260906-091019 all four families were still
    # improving >=2% when the budget froze them). A third slot lets a novel family be tried
    # without evicting both leaders in the same round.
    #
    # It does NOT make the run longer, which is worth stating because the opposite is the
    # intuitive guess. Rewrite rounds are SERIAL (`_rewrite_round` iterates
    # `active_families()` and each `_do_rewrite` blocks on the GPU), and the total available
    # is `max_seed_candidates * rewrite_rounds_per_family` = 20 either way. At a 38.8 min
    # median per round a 12 h budget affords ~18 of them, so WALL CLOCK is the binding
    # constraint in both settings -- run-l3-21-20260905-195615 already overshot at 12.82 h
    # with active=2. What this changes is the DISTRIBUTION: more families reach a first
    # round before the clock stops, each getting fewer rounds. That is the intended trade
    # for giving Loop D somewhere to land.
    max_families_active: int = 3
    max_families_total_hard: int = 6  # absolute cap once dead families stop counting
    max_seed_candidates: int = 4
    repair_attempts: int = 2
    wall_clock_hours: float = 12.0
    # Improvement K: lightweight parameter-space expansion when a knob is at the
    # tried-range boundary and still improving with idle resources. 0 = disabled.
    space_expansions_per_candidate: int = 0
    space_expansion_idle_frac: float = 0.8  # resource must be < this frac of limit
    # P4: among the knobs already cleared for widening, prefer those whose boundary value
    # is not already failing. A value added beyond a FAILING edge fails 43% of the time
    # (16/37 across 19 runs) vs 15% (13/84) beyond a healthy one, and a failed trial
    # returns no latency, so those trials buy nothing.
    #
    # This is a PREFERENCE, not a filter, and the difference is measured
    # (scripts/audit_expansion_failure_veto.py): as a filter it would empty the request
    # list on 8 of 177 expansions, and an empty list cancels the expansion outright --
    # losing the fresh tuning budget as well as the widening. Two of those 8 are their
    # run's best candidate (cand-0d0dcd49, and cand-60fdcae9 = the 8.06 ms L3:43 winner).
    # As a preference it avoids 131 of 155 failing-edge aims (every one with a healthy
    # alternative in the same expansion) and leaves the other 24 exactly as today.
    # 1.0 = disabled.
    max_edge_failure_frac: float = 0.30
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
