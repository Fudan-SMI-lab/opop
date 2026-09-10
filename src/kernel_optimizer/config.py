"""Config loading: YAML -> pydantic AppConfig, with dotted CLI overrides."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

from kernel_optimizer.models.core import DeviceLimits


class StrictConfig(BaseModel):
    """Base for every config block: an unknown key is an ERROR, not a silent drop.

    Pydantic's DEFAULT is to ignore unknown keys, and for a config loaded by `load_config` --
    which reads ONE file with no base layer, so an omitted key falls back to a field default --
    that default turns every typo into a silent revert to the default behaviour. The failure is
    not hypothetical and not cosmetic:

      * A `v3:` block is the FIRST thing any experiment sets and no config had ever carried one,
        so the YAML->pydantic path for those switches was entirely unexercised. Measured before
        this change: `v3.diagnosis.modes: vector` (plural) and `expectation_ledgers: true`
        validated cleanly and left `mode="label"` / `expectation_ledger=False`. A treatment arm
        written with either typo would have been a SECOND CONTROL ARM, the two runs would have
        agreed, and the conclusion drawn from 24 h of GPU time would have been "the vector shape
        changes nothing" -- the diagnosis inverted, the same shape as scoring an fp16 kernel
        against a tf32 ceiling and reading 107.8% of peak as "saturated, stop optimising".
      * The same family already cost this project a schema: `ResourceExpectation.expected_pct`
        validated and vanished, so an agent reasoned from a magnitude nothing ever checked. That
        one was closed with `extra="forbid"`; this is the same fix at the config layer.
      * `_apply_override` builds its path with `setdefault`, so `--set v3.diagnosis.modes=vector`
        creates the key rather than failing. Forbidding at the MODEL layer closes the YAML path
        and the CLI path together, instead of validating a key list in two places.

    Verified at the time of the change: all 10 files in `configs/` validate with no dropped key,
    so this rejects only what was already being ignored.
    """

    model_config = ConfigDict(extra="forbid")


class RunConfig(StrictConfig):
    runs_dir: Path = Path("runs")
    seed: int = 0


class OpencodeConfig(StrictConfig):
    server_url: str | None = None  # null => harness launches `opencode serve`
    launch_cwd: Path = Path("D:/Pyhon_projects/opop")
    host: str = "127.0.0.1"
    port: int = 4096
    agent: str = "build"
    # 25 min. Measured over all 997 agent-call attempts on record (1000 minus 3 in-flight),
    # scripts/probe_agent_timeouts.py:
    #
    #   successful calls: n=982, p50 114s, p90 252s, p99 534s, MAX 1167s
    #   calls exceeding 1200s: 0.  Calls exceeding 1800s: 0.
    #   ReadTimeout: 15 calls hung (18 hangs -- three hung twice), 14 of the 15 FINISHED on
    #   a retry, the retries taking 0.4/1.1/2.1/2.2/2.8/3.3/3.9/4.2/4.2/4.3/4.7/5.1/13.4/19.4 min.
    #
    # So a timed-out call is HUNG, not slow: not one of the 15 was real work approaching the
    # limit, and a fresh session completes the same prompt in ~4 min (median). The timeout's
    # only job is to notice a hang, and its value is what each hang costs -- the 18 hangs on
    # record cost 9.00 h at 1800s against 7.50 h at 1500s and 6.00 h at 1200s.
    #
    # This CORRECTS the reasoning that raised it from 1200 to 1800. That change was made on
    # the belief that the 1200s kills were destroying real work ("each kill discards a
    # candidate or a whole rewrite round"); re-measured, 7 of those 8 recovered on retry and
    # only one call was ever truly lost (an L3:21 repair that timed out twice). Raising the
    # ceiling did not rescue work -- it made every hang 50% more expensive.
    #
    # Not lowered to 1200 either: the slowest successful call is 1167s, leaving 33s (2.9%) of
    # headroom, so a call 3% slower than anything yet seen would be killed for being slow.
    # 1500s keeps 333s (29%) of headroom over that maximum while returning half the loss.
    # A hang is still only detectable by its duration; distinguishing "hung" from "thinking"
    # needs a token-level heartbeat on the transport, which is deferred (D-4).
    # Model-agnostic: this is the transport read timeout, not a token or effort setting.
    request_timeout_s: float = 1500.0
    # An in-flight agent call is aborted when the CONTAINER's memory reaches this fraction of
    # its cgroup limit. This deliberately replaced a total wall-clock deadline on a call, which
    # was the wrong instrument: an agent legitimately runs long because it compiles kernels and
    # benchmarks them, and cutting it at a deadline destroys that work. Measured twice -- the
    # 4057s call had already written `rw_1.py` 8 minutes before it returned, and an earlier
    # 1500s ceiling discarded a finished rewrite. Duration is not the failure.
    #
    # The real failure is an agent SUBPROCESS taking the machine down: one agent-written sweep
    # produced a 272,341-line PTX, ptxas reached 111 GiB resident, the container hit its memory
    # cgroup limit (125.5 GB against a 124.5 GB memory.high, 16.0M throttle events) and the
    # orchestrator itself was throttled into D-state -- 3 GB from a hard OOM where the kernel
    # chooses the victim instead of us. That is a resource condition and it is observable, so
    # that is what is bounded. A long call using nothing is left entirely alone.
    #
    # 0.0 disables the check. Ignored where there is no readable cgroup limit (Windows, macOS),
    # since a bound that cannot be measured must not be approximated.
    memory_abort_frac: float = 0.92
    resource_poll_s: float = 20.0
    # G20. How long a call may produce NOTHING before it counts as hung, as a FRACTION of
    # `request_timeout_s` above -- never an absolute interval. Deriving it keeps the two from
    # drifting: raising the transport ceiling automatically raises how long silence is tolerated,
    # and shortening it shortens this too. A hardcoded '20 minutes' would turn vacuous or
    # trigger-happy the moment the ceiling changed.
    #
    # Why it exists: measured across 9 rewriter calls, the median is 18.9 min and 33% ran into
    # the ceiling. Duration alone must never end a call (an agent compiling and benchmarking is
    # working), but SILENCE should -- waiting out the full ceiling on a call that has stopped
    # producing costs a sample that could have been another measurement.
    #
    # 0.0 disables the check. The productivity signal is the agent sandbox's file tree, so this
    # is inert for a call given no directory.
    idle_abort_frac: float = 0.5
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


class AgentModuleConfig(StrictConfig):
    model: str | None = None  # None => agents.default_model
    max_retries: int = 2
    # NOT CONSUMED ANYWHERE (verified: nothing reads `AgentModuleConfig.timeout_s`). The single
    # effective ceiling is OpencodeConfig.request_timeout_s, which sets the httpx client timeout
    # in wiring.py. Kept in step with it so a reader who sets this per-module does not end up
    # with a value that silently contradicts the real one -- but setting it changes nothing.
    timeout_s: float = 1500.0
    n_candidates: int = 4  # generator / rewriter / novelty batch size
    # Transport (ReadTimeout / connection) failures retried on a FRESH session. Capped
    # separately from max_retries because a hung endpoint costs a full request_timeout_s
    # per attempt with no diagnostic value, whereas a schema-invalid response costs
    # seconds and the feedback usually fixes it. Default 2 preserves the previous total
    # retry budget exactly (max_retries=2 => 3 attempts), so the win here comes from the
    # fresh session, not from cutting attempts: the L3:43 repair that burned 0.99h timed
    # out twice while queued behind its own aborted turn on one session.
    max_transport_retries: int = 2


class AgentsConfig(StrictConfig):
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


class BudgetConfig(StrictConfig):
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


class EvalConfig(StrictConfig):
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


class GpuConcurrencyConfig(StrictConfig):
    enabled: bool = True
    max_shared_jobs: int = 2  # correctness/compile/static-check only; timing is exclusive
    vram_budget_frac: float = 0.45
    timing_cooldown_s: float = 2.0


class WslConfig(StrictConfig):
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


class GpuConfig(StrictConfig):
    concurrency: GpuConcurrencyConfig = GpuConcurrencyConfig()
    # P1: refuse configurations whose compiled shared-memory requirement exceeds the device's
    # per-block opt-in limit, before paying for a launch. On L3:43 this class was 180 of 1004
    # trials -- 18% of the budget, 0.93 h. On by default because the screen only ever acts on
    # the compiler's own figure against the device's own limit, and returns "no opinion" for
    # every other outcome. Set false to restore the pre-screen behaviour for a comparison.
    compile_screen_enabled: bool = True


class V3SearchConfig(StrictConfig):
    """S1 / S1b: what the tuner is allowed to change about the SPACE it samples from.

    Both default OFF. Every stage of v3 needs its own control run -- the same task, the same
    budget, the switch flipped -- and a stage that cannot be turned off cannot be measured. That
    is not a style preference: S1 and S2 both change what the tuner and the agent see, so a run
    from before a stage landed is not comparable to one from after it, and the only honest
    comparison is two runs that differ in one switch.
    """

    # S1: express a compile-time-known infeasibility as a SHRUNKEN DOMAIN rather than sampling a
    # dead point and rejecting it. Measured waste it targets: 180 of 1004 trials (18%) on L3:43
    # were configurations that could not run and were knowable as such before any launch.
    #
    # The shrink must be driven by the existing compile-time truth (`_shared_memory_ok`), never by
    # a new hand-written formula. Agent-written shared-memory constraints were measured at a median
    # of 32% of the true limit, and L3:43's was true for all 36 configurations it saw -- it never
    # rejected anything. Truth comes from the compiler, not from an expression.
    declare_infeasible_out_of_space: bool = False

    # S1b: down-weight a categorical VALUE whose failures no partner can explain. The criterion
    # and every constant in it are measured; `tuning/deweight.py` carries the full derivation.
    # In short: evidence pooled on (candidate_id, knob, value) across that candidate's spaces --
    # never across candidates, never across runs -- fires at 7 failures with zero passes, retracts
    # permanently on the first pass, and down-weights to 1/4 rather than removing.
    #
    # Two earlier versions of this criterion were refused by measurement, and both are worth
    # keeping in view because they were each argued to be the safe one:
    #
    #   "retire a value after N total failures" -- at N=6 it retired 13 values on one box of which
    #   5 later succeeded (38.5%), PJ_BC=64 among them, which really passed 102 of 176 times.
    #   Failure is almost always CONDITIONAL while retirement is unconditional.
    #
    #   "unconditional + median-M sample sufficiency + a live control" -- my own replacement, and
    #   prospective replay measured it at 67.5% value-level mis-kill against the blacklist's 2.3%.
    #   median-M is adaptive, so early in a space it is tiny and fires before the evidence exists.
    #
    # What survived: the count floor was doing the work all along, and the pooling SCOPE decides
    # whether the evidence means anything. Pooling across a whole run pools across CANDIDATES,
    # which mixes evidence about different source code -- the root cause is the candidate's own
    # uncompensated `dot` -- and its apparent 249-trial saving rests on 4 rules, whose 0 mis-kills
    # carry a 95% upper bound of 52.7%. Per-candidate saves 206 from 33 rules, bound 8.7%, and is
    # the only scope whose floor is stable under leave-one-out ([7,7,7,7,7] against [7,7,7,6,7] and
    # [5,7,7,7,7]).
    deweight_unconditional_failures: bool = False


class V3DiagnosisConfig(StrictConfig):
    """S2 / S2b / S2d: what the agent is told about resources, and in what shape."""

    # S2: `label` is v2's behaviour -- one `kind` string per candidate, which measured 19 of 20
    # reports on L3:21 carrying the same label. `vector` emits one independent record per
    # dimension, and when several are binding it says so instead of choosing between them.
    #
    # The vector is written to events.jsonl either way, because recording costs nothing and is not
    # what carries risk. This switch decides only what reaches the PROMPT -- which is the thing
    # with an external counter-example (few-shot optimisation exemplars measurably LOWERED fast_1,
    # 10% to 6% on KernelBench L1), so it is the thing that needs a control run.
    mode: Literal["label", "vector"] = "label"

    # S2d: the agent states, per dimension, which way it expects a rewrite to move the resource
    # ("up"/"down"/"unchanged"/"unknown"), and afterwards the harness reconciles that against what
    # was measured AND against whether the movement bought any latency.
    #
    # The direction is never a number: change rates are task-specific and, measured three ways, not
    # derivable in advance -- no closed form for shared memory (0 of 96 exact), the map is not
    # separable (0 of 10 one-step deltas agreed), and even the SIGN is unreliable (13 non-monotone
    # slices, BK 16->32 dropping 58 registers and 32->64 adding 87). So the agent is asked only for
    # a direction, and accuracy comes from being held to account afterwards rather than from us
    # computing the rate for it.
    #
    # An expectation may never enter ranking, allocation, or acceptance. It has exactly two
    # outlets: the ledger, and the next round's prompt.
    expectation_ledger: bool = False

    # S2b: the access-pattern coordinate (instruction-roofline "walls"), derived statically from
    # tile/stride configuration -- no counters, no run. Off until its positive control passes: at
    # least one hand-built kernel must land ON the 1/32 wall, or the derivation is just returning
    # good news. A probe with no failing control has already cost us five negative "results".
    access_pattern_walls: bool = False


class V3Config(StrictConfig):
    """The v3 stages, each behind its own switch, all off by default.

    Grouped rather than flat so that a stage's switches sit together and a YAML that sets one does
    not have to know the others exist. NOTE that `load_config` reads a single file with no base
    layer: an omitted key silently falls back to the default here, so THESE defaults are the
    production behaviour, not `configs/default.yaml`. That has bitten this project before -- an
    omitted `device:` block left every L3 agent being told its GPU was "unknown".
    """

    search: V3SearchConfig = V3SearchConfig()
    diagnosis: V3DiagnosisConfig = V3DiagnosisConfig()


class AppConfig(StrictConfig):
    run: RunConfig = RunConfig()
    opencode: OpencodeConfig = OpencodeConfig()
    agents: AgentsConfig = AgentsConfig()
    budgets: BudgetConfig = BudgetConfig()
    evaluation: EvalConfig = EvalConfig()
    wsl: WslConfig = WslConfig()
    gpu: GpuConfig = GpuConfig()
    v3: V3Config = V3Config()
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
