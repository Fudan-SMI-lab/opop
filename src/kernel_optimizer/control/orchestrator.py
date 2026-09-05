"""Outer-loop orchestrator: baseline -> seeds -> per-candidate pipeline
(parameterize -> tune -> stats -> analyze) -> rewrite rounds (loop C) ->
novelty rounds (loop D) -> final re-eval -> report.

Every step is guarded by a step_key; replay()-known steps are skipped on resume.
"""

from __future__ import annotations

import ast
import csv
import hashlib
import io
import random
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from kernel_optimizer.agents.base import AgentOutcome
from kernel_optimizer.agents.modules import (
    AnalystInputs,
    BottleneckAnalystAgent,
    CandidateGeneratorAgent,
    GeneratorInputs,
    NoveltyGeneratorAgent,
    NoveltyInputs,
    ParameterizerAgent,
    ParameterizerInputs,
    RepairAgent,
    RepairInputs,
    RewriterInputs,
    StructureRewriterAgent,
)
from kernel_optimizer.agents.runtime import AgentCallError
from kernel_optimizer.config import AppConfig
from kernel_optimizer.control.convergence import ConvergencePolicy
from kernel_optimizer.control.families import FamilyManager, NoveltyRejection
from kernel_optimizer.evaluation.benchmark import Benchmarker
from kernel_optimizer.evaluation.correctness import CorrectnessEvaluator, latency_from_result
from kernel_optimizer.evaluation.profilerx import LightProfiler
from kernel_optimizer.models.core import (
    Baseline,
    Candidate,
    Constraint,
    ParameterSpace,
    ParamSet,
    ParamValue,
    TaskSpec,
    TrialRecord,
)
from kernel_optimizer.models.reports import BottleneckReport, TuningStats
from kernel_optimizer.paramspace import materializer
from kernel_optimizer.paramspace.guard import check_config
from kernel_optimizer.paramspace.triton_lint import device_helper_names, jit_kernel_names
from kernel_optimizer.paramspace.validation import (
    SpaceAccepted,
    SpaceValidator,
    error_excerpt,
)
from kernel_optimizer.store.run_store import RunStore
from kernel_optimizer.tuning.stats import TuningStatsAnalyzer
from kernel_optimizer.tuning.tpe import OptunaTPETuner


def _detect_candidate_precision(source: str, params: ParamSet) -> str:
    """Best-effort classification of the arithmetic precision the candidate uses.

    Reports the precision the candidate actually computes in so the report can
    compare it against the same-precision baseline (an ieee candidate vs the ieee
    baseline, a tf32 candidate vs the tf32 baseline). Purely descriptive — never
    changes what runs. Returns one of "tf32", "fp16", "bf16", "ieee_fp32", or
    "unknown".
    """
    # A precision/dtype PARAMS knob is the most reliable signal (the tuner may have
    # selected it), so check the materialized params first. Name-agnostic: match on
    # the VALUE (fp16/bf16/tf32/ieee...), so DOT_PRECISION / COMPUTE_DTYPE / any name
    # is recognized.
    for _key, val in params.values.items():
        if isinstance(val, str):
            v = val.strip().lower()
            if v in ("tf32", "ieee", "fp16", "bf16", "float16", "bfloat16"):
                if v in ("ieee",):
                    return "ieee_fp32"
                if v == "float16":
                    return "fp16"
                if v == "bfloat16":
                    return "bf16"
                return v
    text = source.lower()
    # Explicit tl.dot input_precision literal in the source.
    if 'input_precision="tf32"' in text or "input_precision='tf32'" in text:
        return "tf32"
    if 'input_precision="ieee"' in text or "input_precision='ieee'" in text:
        return "ieee_fp32"
    if "float16" in text or "tl.float16" in text or ".half(" in text or "torch.half" in text:
        return "fp16"
    if "bfloat16" in text or "tl.bfloat16" in text:
        return "bf16"
    if "tl.dot" in text:
        # tl.dot with no explicit precision: Triton defaults to the tf32 path on
        # fp32 inputs for this GPU generation.
        return "tf32"
    return "unknown"


def _honest_verdict(precision: str, speedups: dict[str, float]) -> dict[str, Any]:
    """Judge the candidate against the baseline computed at the SAME precision.

    speedups maps baseline kind -> (baseline_ms / candidate_ms). tf32 baselines
    are recorded with a "_tf32" suffix by measure_baseline in dual-precision mode.
    A tf32 candidate should be compared against torch_compile_tf32, not against the
    slower ieee torch_compile, or the reported win is against a strawman.
    """
    # Which baseline precision is the fair comparison for this candidate?
    is_tensor_core = precision in ("tf32", "fp16", "bf16")
    compile_key = "torch_compile_tf32" if is_tensor_core else "torch_compile"
    eager_key = "eager_tf32" if is_tensor_core else "eager"
    # Fall back to the untagged baseline if the same-precision one was not recorded
    # (e.g. strict mode only measured ieee baselines).
    if compile_key not in speedups and "torch_compile" in speedups:
        compile_key = "torch_compile"
    if eager_key not in speedups and "eager" in speedups:
        eager_key = "eager"
    verdict: dict[str, Any] = {
        "candidate_precision": precision,
        "compared_against": compile_key if compile_key in speedups else eager_key,
    }
    ref = speedups.get(compile_key)
    if ref is None:
        ref = speedups.get(eager_key)
    if ref is not None:
        verdict["same_precision_speedup"] = ref
        verdict["beats_same_precision_baseline"] = ref > 1.0
    return verdict


def boundary_knobs_to_expand(stats: TuningStats, idle_frac: float,
                             space: ParameterSpace | None = None,
                             min_effect_pct: float = 2.0) -> list[dict]:
    """Improvement K: which knobs are at the tried-range boundary AND still improving,
    given that at least one hardware resource has headroom (< idle_frac of its limit).

    Returns a list of {name, direction} for the boundary knobs. Empty if no knob is
    blocked at a boundary, or if resources are already saturated (no headroom to
    exploit — expanding would just OOM/spill, so we defer to a structural rewrite).
    Purely a read over deterministic stats; never runs anything.

    Two filters keep this from firing on non-opportunities:
    - Only ORDERED NUMERIC knobs are expandable: extending a categorical knob (e.g.
      COMPUTE_DTYPE's fp16/bf16/tf32/ieee) is meaningless — there is no "next value"
      beyond its edge.
    - The knob must have a MEANINGFUL EFFECT (>= min_effect_pct). On a flat latency
      surface the "argmin at an edge" test passes on noise for every knob, but a knob
      that changes latency by ~0% is irrelevant, not blocked — expanding its range
      cannot help (observed live on L3:21 cand-dc6526b6: all 6 knobs flagged
      at_boundary with 0.0-0.4% effect).
    - The boundary must be EXTENDABLE. A knob sitting on a hard hardware floor/ceiling
      has no next value to offer, so asking for one wastes a parameterizer call and a
      full re-tune over an unchanged space. NUM_WARPS=1 is the case that fired: it was
      requested in 4 of 5 expansions on L3:48 and expanded zero times, because one warp
      IS the minimum launch allocation — the analyst says so itself
      (`blocked_by: "threads"`, "further decrease is impossible"). Twice it was the only
      requested knob, so the whole expansion was a no-op that still cost 40 trials.

      The wall list must cover every knob shape that HAS a floor, which is where it was
      short: `BLOCK_K=[16,32,64]` was asked to add 8, which is ILLEGAL for a `tl.dot`
      contraction dimension (`K >= 16`). It fails to compile and takes the whole
      expansion down with it -- observed twice, on QKV_BLOCK_K and PV_BLOCK_K, each
      costing a parameterizer call and 40 trials. That floor is a compiler fact of the
      same kind as the warp floor, so it belongs in the same table.
    """
    if stats is None or not stats.param_stats:
        return []
    # Hard limits per direction: a value at or beyond one of these cannot be used,
    # whatever the latency trend says. Keyed by (name SUFFIX, direction) -> the wall.
    #
    # Matched by suffix, not exact name, because agents prefix these freely: the runs so
    # far contain NUM_WARPS, NUM_STAGES, PW_WARPS, APPLY_WARPS, FINISH_WARPS, PW_STAGES,
    # EXPAND_NUM_STAGES, FUSED_NUM_WARPS, SUMMARY_NUM_WARPS, SCAN_NUM_WARPS and
    # OUTPUT_NUM_WARPS, plus BLOCK_K shapes as QKV_BLOCK_K, PV_BLOCK_K, EXPAND_BLOCK_K.
    # Exact matching covered 10 of the 11 historical min-at-1 requests and let
    # `EXPAND_NUM_STAGES` through (L3:21 09-04, cand-82819823); suffix matching covers
    # all 11.
    #
    # BLOCK_K's floor is 16 because `tl.dot` requires the CONTRACTION dim to be >= 16.
    # That rule is ASYMMETRIC -- M and N have no such floor -- so only the K suffix is
    # listed; see agents/prompts/triton_pitfalls.md rule 4.
    HARD_EDGE = {("WARPS", "min"): 1, ("STAGES", "min"): 1, ("BLOCK_K", "min"): 16}
    res = stats.resource_at_best
    # Headroom = some resource is comfortably below its limit. If we can't tell
    # (no profile), be permissive — the guard + validation still gate the result.
    has_headroom = True
    if res is not None:
        fracs = [f for f in (res.regs_frac_of_limit, res.shared_frac_of_limit)
                 if f is not None]
        if fracs:
            has_headroom = any(f < idle_frac for f in fracs)
    if not has_headroom:
        return []

    def _is_numeric_knob(name: str) -> bool:
        if space is None:
            return True  # cannot tell; validation still gates the expansion
        try:
            dom = space.domain(name)
        except KeyError:
            return False
        return dom.kind in ("int", "float")

    def _at_hard_edge(name: str, direction: str) -> bool:
        """True if the knob's offered range already touches an unextendable limit.

        The wall is the last USABLE value, so a range whose edge has reached it has
        nothing further to offer in that direction and the request is a guaranteed
        no-op. A range one step away is still allowed: NUM_WARPS=[2,4,8] may add 1,
        because 1 is legal -- it is merely (measurably) slow, which is the tuner's
        business rather than this filter's.

        Note this is deliberately NOT a "would the next value cross the wall" test.
        I implemented that variant first; checked exhaustively over 6392 candidate
        domains it produces an identical verdict in every case, because a ladder whose
        next step would cross the wall has its edge AT the wall already. The extra
        arithmetic was dead code, so the real fix for the BLOCK_K=8 failures is the
        HARD_EDGE entry above, not a cleverer predicate here.
        """
        wall = next((v for (suffix, d), v in HARD_EDGE.items()
                     if d == direction and name.upper().endswith(suffix)), None)
        if wall is None or space is None:
            return False
        try:
            choices = space.domain(name).choices
        except KeyError:
            return False
        numeric = [c for c in choices if isinstance(c, (int, float))]
        if not numeric:
            return False
        return min(numeric) <= wall if direction == "min" else max(numeric) >= wall

    return [
        {"name": ps.name, "direction": ps.boundary_direction}
        for ps in stats.param_stats
        if ps.at_boundary and ps.boundary_direction in ("min", "max")
        and _is_numeric_knob(ps.name)
        and (ps.effect_pct or 0.0) >= min_effect_pct
        and not _at_hard_edge(ps.name, ps.boundary_direction)
    ]


@dataclass
class Wiring:
    evaluator: CorrectnessEvaluator
    benchmarker: Benchmarker
    profiler: LightProfiler
    validator: SpaceValidator
    stats_analyzer: TuningStatsAnalyzer
    families: FamilyManager
    convergence: ConvergencePolicy
    generator: CandidateGeneratorAgent
    parameterizer: ParameterizerAgent
    analyst: BottleneckAnalystAgent
    rewriter: StructureRewriterAgent
    novelty: NoveltyGeneratorAgent
    repair: RepairAgent


@dataclass
class CandidateRun:
    """Per-candidate pipeline products kept in memory for the current run."""

    candidate: Candidate
    source: str = ""
    space: ParameterSpace | None = None
    trials: list[TrialRecord] = field(default_factory=list)
    stats: TuningStats | None = None
    report: BottleneckReport | None = None
    best_ms: float | None = None


class Orchestrator:
    def __init__(self, deps: Wiring, cfg: AppConfig, store: RunStore, task: TaskSpec):
        self.deps = deps
        self.cfg = cfg
        self.store = store
        self.task = task
        self.t0 = time.monotonic()
        self.baselines: list[Baseline] = []
        self.eval_semantics: dict = {}  # improvement J: reference train/eval semantics
        self.runs: dict[str, CandidateRun] = {}
        self.failed_hypotheses: dict[str, list[dict]] = {}  # family_id -> tried-and-failed
        # Improvement B1: one worker thread that runs the NEXT candidate's
        # parameterization (an LLM call) while the CURRENT one occupies the GPU.
        # Measured on L3:43: agent 2.88h and GPU 7.34h with 0.00h of overlap, purely
        # because the pipeline is synchronous. Parameterization is the only agent step
        # that needs no trial results, so it is the only one safe to run ahead;
        # analyst/rewriter read tuning stats and must stay ordered.
        self._prefetch_pool: ThreadPoolExecutor | None = None
        self._prefetched: dict[str, Future] = {}

    # ------------------------------------------------------------------ helpers

    def _elapsed_hours(self) -> float:
        return (time.monotonic() - self.t0) / 3600.0

    def _step_done(self, key: str) -> None:
        self.store.append("STEP_DONE", {"step_key": key})

    def _register(self, source: str, origin: str, parents: list[str], backend: str,
                  approach: str) -> Candidate | None:
        cand = self.deps.families.register_candidate(source, origin, parents, backend,
                                                     approach)
        if cand is None:
            return None
        cand_dir = self.store.candidate_dir(cand.candidate_id)
        (cand_dir / "source.py").write_text(source, encoding="utf-8")
        self.store.append("CANDIDATE_REGISTERED", {"candidate": cand.model_dump()})
        self.runs[cand.candidate_id] = CandidateRun(candidate=cand, source=source)
        return cand

    # ------------------------------------------------------------------ stages

    def run(self) -> dict[str, Any]:
        try:
            return self._run()
        finally:
            # A non-daemon ThreadPoolExecutor keeps the process alive at exit, so an
            # exception mid-run would otherwise hang the CLI. Do not wait on an
            # in-flight prompt here (it can run for request_timeout_s).
            self._shutdown_prefetch(wait=False)

    def _run(self) -> dict[str, Any]:
        self.store.write_state_snapshot({"phase": "started", "task": self.task.model_dump()})
        self._baseline()
        self._generate_seeds()

        self._pipeline_batch(list(self.runs))

        self._restore_family_control_state()

        # Loop C: rewrite rounds on active families, then Loop D: novelty rounds.
        round_no = 0
        while True:
            verdict = self.deps.convergence.global_verdict(
                list(self.deps.families.families.values()), self._elapsed_hours()
            )
            self.store.append("CONVERGENCE_DECIDED", {"decision": verdict.model_dump()})
            if verdict.verdict == "freeze":
                break
            round_no += 1
            progressed = self._rewrite_round(round_no)
            if not progressed:
                added = self._novelty_round(round_no)
                if not added:
                    # Nothing to rewrite and nothing novel accepted: freeze leftovers.
                    for fam in self.deps.families.families.values():
                        if fam.status == "active":
                            fam.status = "frozen_budget"
                    continue

        # Drain the prefetch thread only once all pipelining is done — Loop C and D
        # pipeline candidates too, so shutting down after the seed loop would disable
        # B1 for the majority of them.
        self._shutdown_prefetch()
        result = self._finalize()
        self.store.append("RUN_FINISHED", {"summary": result})
        return result

    def _baseline(self) -> None:
        key = "baseline"
        state = self.store.replay()
        if key in state.steps_done:
            self.baselines = [Baseline.model_validate(b["baseline"])
                              for b in state.baselines]
            # Restore probed eval semantics (improvement J) from the log on resume.
            for ev in state.events:
                if ev.type == "SEMANTICS_PROBED":
                    self.eval_semantics = ev.payload.get("semantics", {}) or {}
            return
        # Improvement J: probe the reference's runtime eval semantics (train/eval +
        # norm-layer flags) before generation, so agents can match them. Advisory —
        # an empty dict degrades gracefully in the contract doc.
        self.eval_semantics = self.deps.benchmarker.probe_semantics(self.task)
        self.store.append("SEMANTICS_PROBED", {"semantics": self.eval_semantics})
        self.baselines = self.deps.benchmarker.measure_baseline(self.task)
        for b in self.baselines:
            self.store.append("BASELINE_DONE", {"baseline": b.model_dump()})
        self._step_done(key)

    def _generate_seeds(self) -> None:
        key = "generate_seeds"
        state = self.store.replay()
        if key in state.steps_done:
            # Rebuild candidates from the log.
            for cand_payload in state.candidates.values():
                cand = Candidate.model_validate(cand_payload)
                src_path = self.store.candidate_dir(cand.candidate_id) / "source.py"
                source = src_path.read_text(encoding="utf-8")
                self.deps.families.candidates[cand.candidate_id] = cand
                self.deps.families._sources[cand.candidate_id] = source
                fam = self.deps.families.families.get(cand.family_id)
                if fam is None:
                    from kernel_optimizer.models.core import Family

                    self.deps.families.families[cand.family_id] = Family(
                        family_id=cand.family_id, anchor_candidate_id=cand.candidate_id,
                        member_ids=[cand.candidate_id])
                elif cand.candidate_id not in fam.member_ids:
                    fam.member_ids.append(cand.candidate_id)
                self.runs[cand.candidate_id] = CandidateRun(candidate=cand, source=source)
            if self.runs:
                return

        outcome: AgentOutcome = self.deps.generator.invoke(
            GeneratorInputs(
                task=self.task,
                ref_source=Path(self.task.ref_path).read_text(encoding="utf-8"),
                device=self.cfg.device,
                n_candidates=min(self.cfg.agents.generator.n_candidates,
                                 self.cfg.budgets.max_seed_candidates),
                eval_semantics=self.eval_semantics,
            )
        )
        for gen_cand in outcome.output.candidates[: self.cfg.budgets.max_seed_candidates]:
            source = outcome.sandbox.read_output(gen_cand.file)
            self._register(source, "seed", [], gen_cand.backend, gen_cand.approach_summary)
        if not self.runs:
            raise RuntimeError("generator produced no registrable candidates")
        self._step_done(key)

    # ----------------------------------------------------- per-candidate pipeline

    def _candidate_pipeline(self, cand_id: str) -> None:
        crun = self.runs[cand_id]
        key = f"pipeline:{cand_id}"
        state = self.store.replay()
        if key in state.steps_done:
            self._restore_pipeline(crun, state)
            return

        # Partial resume: a space already published for this candidate means
        # parameterization (incl. witnesses) succeeded before the interruption.
        existing_space = next(
            (ParameterSpace.model_validate(s) for s in state.spaces.values()
             if s["candidate_id"] == cand_id),
            None,
        )
        measured_cache: dict[str, TrialRecord] = {}
        if existing_space is not None:
            crun.space = existing_space
            src_path = self.store.candidate_dir(cand_id) / "source.py"
            crun.source = src_path.read_text(encoding="utf-8")
            for t in state.trials.get(existing_space.space_id, []):
                record = TrialRecord.model_validate(t)
                measured_cache[record.params.key()] = record
            anchors = tuple(r.params for r in measured_cache.values()
                            if r.status == "complete")[:2]
            if not anchors:
                anchors = (ParamSet(values=materializer.extract_defaults(crun.source)),)
        else:
            accepted = self._parameterize_with_repair(crun)
            if accepted is None:
                crun.candidate.status = "dropped"
                self._step_done(key)
                return
            crun.space = accepted.space
            self.store.append("SPACE_PUBLISHED", {"space": accepted.space.model_dump()})
            anchors = tuple(w.params for w in accepted.witnesses)
            for witness in accepted.witnesses:
                if witness.latency_mean_ms is not None:
                    record = TrialRecord(
                        trial_id=f"wit-{witness.params.key()}",
                        candidate_id=cand_id, space_id=accepted.space.space_id,
                        params=witness.params, status="complete",
                        latency_ms=latency_from_result(witness.worker_result),
                        profile=self.deps.profiler.extract(witness.worker_result),
                    )
                    measured_cache[witness.params.key()] = record

        self._tune(crun, anchors, measured_cache)
        self._stats_and_analysis(crun)
        self._maybe_expand_space(crun)
        crun.candidate.status = "tuned"
        self._step_done(key)

    def _restore_pipeline(self, crun: CandidateRun, state) -> None:
        for space_payload in state.spaces.values():
            if space_payload["candidate_id"] == crun.candidate.candidate_id:
                crun.space = ParameterSpace.model_validate(space_payload)
                for t in state.trials.get(crun.space.space_id, []):
                    crun.trials.append(TrialRecord.model_validate(t))
        complete = [t for t in crun.trials if t.status == "complete" and t.latency_ms]
        if complete:
            best = min(complete, key=lambda t: t.latency_ms.mean)
            crun.best_ms = best.latency_ms.mean
            self.deps.families.update_best(
                crun.candidate.family_id, crun.candidate.candidate_id,
                best.params, best.latency_ms.mean,
            )
        if crun.space is not None and crun.trials:
            crun.stats = self.deps.stats_analyzer.analyze(crun.space, crun.trials)
        # Restore the bottleneck report — an in-memory-only pipeline product. Without
        # this, a resumed run leaves crun.report is None, so _rewrite_round silently
        # burns rewrite rounds without ever calling the rewriter -> spurious
        # budget_exhausted freeze and Loop C never runs (regression: level3:43 run).
        for ev in reversed(state.events):
            if (ev.type == "BOTTLENECK_REPORTED"
                    and ev.payload.get("candidate_id") == crun.candidate.candidate_id):
                crun.report = BottleneckReport.model_validate(ev.payload["report"])
                break

    def _shutdown_prefetch(self, wait: bool = True) -> None:
        """Drain the prefetch thread. Outcomes not claimed by a candidate are simply
        dropped: the agent call already happened and is logged, so nothing is lost
        beyond the tokens it cost.

        wait=False is for the error path — an in-flight prompt can take up to
        request_timeout_s (1200s), and blocking on it would stall the CLI's exit.
        """
        for fut in self._prefetched.values():
            fut.cancel()
        self._prefetched.clear()
        if self._prefetch_pool is not None:
            self._prefetch_pool.shutdown(wait=wait, cancel_futures=True)
            self._prefetch_pool = None

    def _parameterize_agent_call(self, source: str, feedback: str,
                                 cand_id: str | None = None) -> AgentOutcome:
        """The pure-LLM half of parameterization: no GPU, no shared mutable state.

        Split out so it can be run ahead of time on the prefetch thread (improvement
        B1). Sandboxes are per-call (uuid4 call_id) and RunStore.append is locked, so
        this is safe off the main thread.
        """
        return self.deps.parameterizer.invoke(
            ParameterizerInputs(
                task=self.task, candidate_source=source,
                device=self.cfg.device, prior_feedback=feedback,
                candidate_id=cand_id,
            )
        )

    def _prefetch_parameterization(self, cand_id: str) -> None:
        """Kick off the next candidate's parameterizer call in the background."""
        if self.cfg.budgets.prefetch_parameterization <= 0:
            return
        if cand_id in self._prefetched or cand_id not in self.runs:
            return
        crun = self.runs[cand_id]
        if not crun.source or crun.candidate.status == "dropped":
            return
        # On resume, a candidate whose pipeline already completed is skipped by
        # _candidate_pipeline, so prefetching for it would spend a real agent call
        # (and its tokens) on a result nothing can claim.
        if f"pipeline:{cand_id}" in self.store.replay().steps_done:
            return
        if self._prefetch_pool is None:
            self._prefetch_pool = ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="prefetch")
        self._prefetched[cand_id] = self._prefetch_pool.submit(
            self._parameterize_agent_call, crun.source, "", cand_id)

    def _take_prefetched(self, cand_id: str, source: str) -> AgentOutcome | None:
        """Claim a prefetched parameterization if it is still valid for this source."""
        fut = self._prefetched.pop(cand_id, None)
        if fut is None:
            return None
        try:
            outcome = fut.result()
        except AgentCallError:
            return None          # fall through to a fresh, synchronous attempt
        except Exception:        # noqa: BLE001 — a prefetch must never break the run
            return None
        # The prefetch was issued against the source as it stood then. If a repair has
        # since rewritten it, the result is stale and must be discarded.
        return outcome if source == self.runs[cand_id].source else None

    def _parameterize_with_repair(self, crun: CandidateRun) -> SpaceAccepted | None:
        """Loop A: parameterize; on failure feed typed errors to repair, retry."""
        cand = crun.candidate
        source = crun.source
        feedback = ""
        repair_history: list[dict] = []
        for attempt in range(self.cfg.budgets.repair_attempts + 1):
            outcome = None
            if attempt == 0 and not feedback:
                outcome = self._take_prefetched(cand.candidate_id, source)
            if outcome is None:
                try:
                    outcome = self._parameterize_agent_call(source, feedback,
                                                           cand.candidate_id)
                except AgentCallError as exc:
                    self.store.append("SPACE_REJECTED",
                                      {"candidate_id": cand.candidate_id,
                                       "reason": "agent_error", "detail": str(exc)[:500]})
                    return None

            param_source = outcome.sandbox.read_output(outcome.output.file)
            work_dir = self.store.candidate_dir(cand.candidate_id)
            verdict = self.deps.validator.validate_and_publish(
                cand, param_source, outcome.output, self.task, work_dir,
                version=attempt + 1,
            )
            if isinstance(verdict, SpaceAccepted):
                crun.source = param_source
                (work_dir / "source.py").write_text(param_source, encoding="utf-8")
                # Update stored source (structure may legitimately be reorganized).
                self.deps.families._sources[cand.candidate_id] = param_source
                for witness in verdict.witnesses:
                    self.store.append("QUICKTEST_DONE", {
                        "candidate_id": cand.candidate_id,
                        "params": witness.params.model_dump(),
                        "latency_mean_ms": witness.latency_mean_ms,
                    })
                return verdict

            self.store.append("SPACE_REJECTED", {
                "candidate_id": cand.candidate_id, "attempt": attempt,
                # error_excerpt keeps the TAIL, where the verdict lives. A flat [:800]
                # cut the rich mismatch message (1149 chars) mid-word and dropped the
                # reference's own noise-floor line -- the part that says whether the
                # candidate is actually wrong or merely inside the task's own spread.
                # The AGENT already receives the untruncated verdict.detail; this is the
                # permanent record, which every later analysis reads instead of the run.
                "reason": verdict.reason, "detail": error_excerpt(verdict.detail, 2000),
            })
            if attempt >= self.cfg.budgets.repair_attempts:
                return None

            # Witness execution failures go to the repair agent; other rejections
            # go back to the parameterizer as feedback.
            if verdict.reason.startswith("witness_"):
                # Close out the PREVIOUS repair with the failure it actually produced.
                # verdict.detail here is the outcome of the last repaired source, so it
                # belongs to that attempt -- attaching it at append time instead labelled
                # each diagnosis with the failure it was RESPONDING to, telling the agent
                # its TF32 hypothesis "still failed with" the crash that came before it.
                if repair_history and repair_history[-1].get("failure_detail") is None:
                    # 900, not 600: the witness detail now leads with a ~200-char
                    # "[minimal witness config {...}] the DEFAULT config passed" prefix
                    # (docs/finding-minimal-witness-forces-fp16.md), and at 600 that prefix
                    # would displace the metrics the agent needs -- trading one missing
                    # piece of context for another.
                    repair_history[-1]["failure_detail"] = verdict.detail[:900]
                try:
                    repaired = self.deps.repair.invoke(
                        RepairInputs(
                            task=self.task, broken_source=param_source,
                            failure_kind=verdict.reason, failure_detail=verdict.detail,
                            device=self.cfg.device,
                            eval_semantics=self.eval_semantics,
                            ref_source=Path(self.task.ref_path).read_text(encoding="utf-8"),
                            candidate_id=cand.candidate_id,
                            prior_attempts=[h for h in repair_history
                                            if h.get("failure_detail")] or None,
                        )
                    )
                    source = repaired.sandbox.read_output(repaired.output.file)
                    # Record what was tried; its failure_detail is filled in on the next
                    # iteration, once we know what this fix actually did.
                    repair_history.append({
                        "diagnosis": repaired.output.diagnosis[:600],
                        "failure_detail": None,
                    })
                    self.store.append("REPAIR_PRODUCED", {
                        "candidate_id": cand.candidate_id,
                        "diagnosis": repaired.output.diagnosis[:500],
                        # The agent's own account of what it EDITED, as distinct from
                        # what it believes went wrong. Without it the event log records
                        # only the reasoning, so a post-hoc reader cannot tell a repair
                        # that rewrote the arithmetic from one that changed a single
                        # dtype -- the two have very different implications when the
                        # diagnosis turns out to describe task noise rather than a bug.
                        "change_summary": repaired.output.change_summary[:500],
                        "source_sha": hashlib.sha256(source.encode()).hexdigest(),
                        "prior_rejected": sum(1 for h in repair_history
                                              if h.get("failure_detail")),
                    })
                    feedback = ""
                except AgentCallError as exc:
                    feedback = f"repair also failed: {str(exc)[:300]}"
            else:
                feedback = f"{verdict.reason}: {error_excerpt(verdict.detail, 1200)}"
        return None

    def _tune(self, crun: CandidateRun, anchors: tuple[ParamSet, ...],
              measured_cache: dict[str, TrialRecord]) -> None:
        """Loop B: TPE over the published space; correctness-first per trial.

        measured_cache: params-key -> already-measured record (witnesses or
        pre-interruption trials); reused instead of re-running on the GPU.
        """
        cand = crun.candidate
        space = crun.space
        conc = self.cfg.gpu.concurrency
        tuner = OptunaTPETuner(
            space,
            guard_ok=lambda p: check_config(space, p, self.cfg.device) is None,
            budget=self.cfg.budgets.trials_per_space,
            seed=self.cfg.run.seed,
            anchors=anchors,
            constant_liar=conc.enabled,
        )
        cand_dir = self.store.candidate_dir(cand.candidate_id)
        trials_dir = cand_dir / "trials"
        trials_dir.mkdir(exist_ok=True)

        while True:
            asked = tuner.ask()
            if asked is None:
                break
            trial_id, params = asked
            cached = measured_cache.get(params.key())
            if cached is not None:
                record = cached.model_copy(update={"trial_id": trial_id})
                # Reused measurements must still be journalled. replay() rebuilds
                # measured_cache purely from TRIAL_DONE, and the report and lineage read
                # the same events, so a silently-reused record is invisible to both: a
                # resume would re-run it on the GPU, and the expansion anchor carrying
                # the pre-expansion optimum (the fix for L3:43 cand-0c3b5820's
                # 20.0 -> 22.6ms regression) would not appear in the trial log at all.
                # Marked so it is never mistaken for a fresh measurement.
                record = record.model_copy(update={"space_id": space.space_id})
                self.store.append("TRIAL_DONE", {"trial": record.model_dump(),
                                                 "reused_measurement": True})
            else:
                record = self._run_trial(crun, space, trial_id, params, trials_dir)
                self.store.append("TRIAL_DONE", {"trial": record.model_dump()})
            tuner.tell(trial_id, record)
            crun.trials.append(record)

        best = tuner.best()
        if best is not None:
            # crun.best_ms tracks the candidate's best over ALL its spaces, so a
            # re-tune (improvement K's expansion) that lands worse must not erase a
            # better earlier result. FamilyManager.update_best is already monotonic.
            if crun.best_ms is None or best.latency_ms.mean < crun.best_ms:
                crun.best_ms = best.latency_ms.mean
            improved = self.deps.families.update_best(
                cand.family_id, cand.candidate_id, best.params, best.latency_ms.mean
            )
            self.store.append("TUNING_DONE", {
                "candidate_id": cand.candidate_id, "space_id": space.space_id,
                "best_ms": best.latency_ms.mean, "improved_family": improved,
                "snapshot": tuner.snapshot(),
            })
        else:
            self.store.append("TUNING_DONE", {
                "candidate_id": cand.candidate_id, "space_id": space.space_id,
                "best_ms": None, "snapshot": tuner.snapshot(),
            })

    def _run_trial(self, crun: CandidateRun, space: ParameterSpace, trial_id: str,
                   params: ParamSet, trials_dir: Path) -> TrialRecord:
        cand = crun.candidate
        try:
            mat_src = materializer.materialize(crun.source, params)
        except materializer.MaterializeError as exc:
            return TrialRecord(
                trial_id=trial_id, candidate_id=cand.candidate_id,
                space_id=space.space_id, params=params, status="fail",
                failure_kind="materialize_error", failure_detail=str(exc)[:500],
            )
        path = trials_dir / f"{trial_id}.py"
        path.write_text(mat_src, encoding="utf-8")
        result = self.deps.evaluator.quick_test(
            self.task, path, tag=f"{cand.candidate_id}-{trial_id}",
            backend=cand.backend,
        )

        lat = latency_from_result(result)
        if not result.get("ok") or lat is None:
            return TrialRecord(
                trial_id=trial_id, candidate_id=cand.candidate_id,
                space_id=space.space_id, params=params, status="fail",
                failure_kind=result.get("failure_kind") or "runtime_error",
                failure_detail=error_excerpt(str(result.get("log_tail", "")), 800),
            )
        return TrialRecord(
            trial_id=trial_id, candidate_id=cand.candidate_id, space_id=space.space_id,
            params=params, status="complete", latency_ms=lat,
            profile=self.deps.profiler.extract(result),
        )

    def _stats_and_analysis(self, crun: CandidateRun) -> None:
        if crun.space is None or not crun.trials:
            return
        crun.stats = self.deps.stats_analyzer.analyze(crun.space, crun.trials)
        self.store.append("STATS_DONE", {"stats": crun.stats.model_dump()})
        unlaunched = self._unlaunched_kernels(crun)
        if unlaunched:
            # Improvement M: a kernel defined but never launched across the WHOLE tuning
            # budget is dead code, and the budget measured something other than the
            # advertised optimization. Deterministic (compares defined @triton.jit names
            # against profile.kernel_names), so it is journalled as fact rather than left
            # to the analyst -- which on L3:21 cand-c0b3b7cd proposed the same
            # inference-BN fusion again after 31 trials had all timed the fallback.
            self.store.append("KERNELS_NEVER_LAUNCHED", {
                "candidate_id": crun.candidate.candidate_id,
                "space_id": crun.space.space_id,
                "never_launched": sorted(unlaunched),
                "n_trials_measured": len(crun.trials),
            })
        if crun.best_ms is None:
            return  # nothing correct; analysis would have no signal
        try:
            outcome = self.deps.analyst.invoke(
                AnalystInputs(
                    task=self.task, candidate_source=crun.source, stats=crun.stats,
                    trials_csv=self._trials_csv(crun), device=self.cfg.device,
                    candidate_id=crun.candidate.candidate_id,
                    eval_semantics=self.eval_semantics,
                    never_launched_kernels=sorted(unlaunched),
                )
            )
            crun.report = outcome.output
            self.store.append("BOTTLENECK_REPORTED",
                              {"candidate_id": crun.candidate.candidate_id,
                               "report": crun.report.model_dump()})
        except AgentCallError as exc:
            self.store.append("AGENT_CALL_FAILED",
                              {"module": "analyst", "final": True,
                               "candidate_id": crun.candidate.candidate_id,
                               "error": str(exc)[:500]})

    def _maybe_expand_space(self, crun: CandidateRun) -> None:
        """Improvement K: if a knob hit the tried-range boundary while still improving
        AND resources have headroom, ask the parameterizer to extend ONLY those knobs'
        choices (structure unchanged), revalidate, and re-tune once. Bounded by
        budgets.space_expansions_per_candidate; disabled when that is 0. A lightweight
        alternative to a full structural rewrite for the 'blocked by search range'
        case. Guarded by validation + guard, so a bad expansion is safely rejected."""
        budget = self.cfg.budgets.space_expansions_per_candidate
        if budget <= 0 or crun.stats is None or crun.space is None:
            return
        cand = crun.candidate
        for _ in range(budget):
            knobs = boundary_knobs_to_expand(
                crun.stats, self.cfg.budgets.space_expansion_idle_frac, crun.space,
                min_effect_pct=self.cfg.budgets.min_improvement_pct)
            if not knobs:
                return
            directive = self._expand_directive_text(crun, knobs)
            prior_constraints = tuple(
                (c.expr, c.rationale) for c in crun.space.constraints)
            # A rejected expansion is often a fixable formatting problem (e.g. an
            # unsupported constraint form), not a dead end — retry once with the
            # validator's reason as feedback before giving up.
            verdict = None
            param_source = ""
            outcome = None
            feedback = ""
            for expand_attempt in range(2):
                try:
                    outcome = self.deps.parameterizer.invoke(
                        ParameterizerInputs(
                            task=self.task, candidate_source=crun.source,
                            device=self.cfg.device, expand_directive=directive,
                            prior_feedback=feedback,
                            candidate_id=cand.candidate_id,
                            prior_constraints=prior_constraints,
                        )
                    )
                except AgentCallError as exc:
                    self.store.append("SPACE_EXPANSION_REJECTED",
                                      {"candidate_id": cand.candidate_id,
                                       "reason": "agent_error", "detail": str(exc)[:500]})
                    return
                param_source = outcome.sandbox.read_output(outcome.output.file)
                work_dir = self.store.candidate_dir(cand.candidate_id)
                verdict = self.deps.validator.validate_and_publish(
                    cand, param_source, outcome.output, self.task, work_dir,
                    version=crun.space.version + 1,
                )
                if isinstance(verdict, SpaceAccepted):
                    break
                self.store.append("SPACE_EXPANSION_REJECTED", {
                    "candidate_id": cand.candidate_id, "attempt": expand_attempt,
                    "reason": verdict.reason,
                    "detail": error_excerpt(verdict.detail, 2000)})
                feedback = (f"the expanded space was rejected ({verdict.reason}): "
                            f"{error_excerpt(verdict.detail, 1200)}")
            if not isinstance(verdict, SpaceAccepted):
                return
            # A "valid" expansion that added no choices is a no-op: accepting it costs a
            # full re-tune (40 trials) over the space just tuned, and its only possible
            # outcome is to rediscover the same optimum. Observed twice on L3:48
            # (cand-f4a2ce82 sp-cbb94366 and cand-3bcc57ce sp-a848207a were byte-identical
            # to their predecessors). The knob-side hard-edge filter prevents most of
            # these, but the agent can also decline to widen a knob for its own reasons,
            # so check the delivered space rather than trusting the request.
            prev_choices = {d.name: tuple(d.choices) for d in crun.space.domains}
            new_choices = {d.name: tuple(d.choices) for d in verdict.space.domains}
            if new_choices == prev_choices:
                self.store.append("SPACE_EXPANSION_REJECTED", {
                    "candidate_id": cand.candidate_id,
                    "reason": "no_new_choices",
                    "detail": f"expansion of {[k['name'] for k in knobs]} returned an "
                              f"identical domain set; skipping the re-tune"})
                return
            work_dir = self.store.candidate_dir(cand.candidate_id)
            restored = self._restore_dropped_constraints(crun.space, verdict.space)
            if restored:
                self.store.append("EXPANSION_CONSTRAINTS_RESTORED", {
                    "candidate_id": cand.candidate_id,
                    "space_id": verdict.space.space_id,
                    "restored": [c.expr for c in restored]})
            # Accept the expanded space and re-tune once over it.
            prev_best = crun.best_ms
            crun.source = param_source
            (work_dir / "source.py").write_text(param_source, encoding="utf-8")
            self.deps.families._sources[cand.candidate_id] = param_source
            crun.space = verdict.space
            self.store.append("SPACE_PUBLISHED", {"space": verdict.space.model_dump()})
            self.store.append("SPACE_EXPANDED", {
                "candidate_id": cand.candidate_id, "knobs": knobs,
                "prev_best_ms": prev_best})
            anchors = tuple(w.params for w in verdict.witnesses)
            measured_cache: dict[str, TrialRecord] = {}
            for witness in verdict.witnesses:
                if witness.latency_mean_ms is not None:
                    measured_cache[witness.params.key()] = TrialRecord(
                        trial_id=f"wit-{witness.params.key()}",
                        candidate_id=cand.candidate_id, space_id=verdict.space.space_id,
                        params=witness.params, status="complete",
                        latency_ms=latency_from_result(witness.worker_result),
                        profile=self.deps.profiler.extract(witness.worker_result),
                    )
            # An expansion only ADDS choices, so the pre-expansion optimum is still a
            # legal config — but the re-tune starts a FRESH TPE study whose only
            # anchors are the two witnesses. Without carrying the old optimum over,
            # 40 fresh trials can simply fail to rediscover it and the candidate goes
            # BACKWARDS (live on L3:43 cand-0c3b5820: 20.0 -> 22.6 ms). Seed it as an
            # anchor and reuse its measurement so it costs no GPU time.
            prev_trials = list(crun.trials)
            prior_best = min(
                (t for t in prev_trials if t.status == "complete" and t.latency_ms),
                key=lambda t: t.latency_ms.mean, default=None)
            if prior_best is not None and check_config(
                    verdict.space, prior_best.params, self.cfg.device) is None:
                key = prior_best.params.key()
                if key not in measured_cache:
                    measured_cache[key] = prior_best.model_copy(
                        update={"space_id": verdict.space.space_id})
                if prior_best.params not in anchors:
                    anchors = (prior_best.params, *anchors)
            self._tune(crun, anchors, measured_cache)
            self._stats_and_analysis(crun)
            # Stop if the expansion did not meaningfully help (avoid chasing a flat edge).
            if (prev_best is not None and crun.best_ms is not None
                    and crun.best_ms > prev_best * (1 - self.cfg.budgets.min_improvement_pct / 100.0)):
                return

    def _restore_dropped_constraints(
        self, old: ParameterSpace, new: ParameterSpace
    ) -> list[Constraint]:
        """Improvement K, part 2: re-add prior constraints the expansion dropped, but
        ONLY where they do not strangle the expansion's own purpose.

        An expansion is meant to be a pure relaxation of the domain: it adds choices and
        keeps everything else. The agent has to re-declare the whole space to do that, and
        the constraints are not in the file it reads, so it drops them. Measured over all
        30 expansions on record: the replacement space admitted configurations its
        predecessor had excluded in 21 of them, 15.7% of the shared sub-grid. Those
        newly-admitted configurations paid nothing — 0 of 21 candidates found a better
        latency there, 18 were strictly worse or failed outright — and they failed at
        48.3% against 26.0% for the region both spaces allowed. The prompt now shows the
        prior constraints, which is the real fix; this is the backstop for when it is
        ignored.

        The test for each dropped constraint is empirical rather than trusting either
        side: re-add it only if every NEWLY ADDED choice still has at least one feasible
        configuration under the restored set. That keeps a stale constraint from vetoing
        the expansion it was requested for (the case on record: a rewritten body made an
        N tile of 8 legal, and the old `BLOCK_N % 16 == 0` would have forbidden the only
        value being added), while restoring the resource bounds, which do not interact
        with the new choices that way.

        Mutates `new.constraints` in place and returns what it restored.
        """
        have = {c.expr for c in new.constraints}
        dropped = [c for c in old.constraints if c.expr not in have]
        if not dropped:
            return []
        old_choices = {d.name: set(d.choices) for d in old.domains}
        added: list[tuple[str, ParamValue]] = [
            (d.name, v) for d in new.domains
            for v in d.choices if v not in old_choices.get(d.name, set())
        ]
        restored: list[Constraint] = []
        for cons in dropped:
            trial = ParameterSpace(
                space_id=new.space_id, candidate_id=new.candidate_id,
                source_sha=new.source_sha, version=new.version,
                domains=new.domains,
                constraints=[*new.constraints, cons, *restored],
            )
            if all(self._choice_is_reachable(trial, name, value)
                   for name, value in added):
                restored.append(cons)
        new.constraints.extend(restored)
        return restored

    def _choice_is_reachable(self, space: ParameterSpace, name: str,
                             value: ParamValue, samples: int = 2000) -> bool:
        """Does at least one config pinning `name=value` satisfy `space`'s constraints?

        Random search rather than exhaustive: a 16-knob space has far too many
        combinations to enumerate, and a false "unreachable" here only means a
        constraint is not restored, which is the pre-existing behaviour.
        """
        rnd = random.Random(f"{space.space_id}:{name}:{value}")
        choices = {d.name: list(d.choices) for d in space.domains}
        if value not in choices.get(name, []):
            return True
        for _ in range(samples):
            params = ParamSet(values={
                n: (value if n == name else rnd.choice(vals))
                for n, vals in choices.items()})
            if check_config(space, params, self.cfg.device) is None:
                return True
        return False

    def _expand_directive_text(self, crun: CandidateRun, knobs: list[dict]) -> str:
        lines = ["The following knobs hit the edge of their offered range and latency "
                 "was still improving toward that edge (resources had headroom):"]
        stat_by_name = {ps.name: ps for ps in (crun.stats.param_stats if crun.stats else [])}
        for k in knobs:
            ps = stat_by_name.get(k["name"])
            cur = crun.space.domain(k["name"]).choices if crun.space else []
            lines.append(
                f"- `{k['name']}`: extend toward {k['direction']} "
                f"(current choices {list(cur)}; best sat at the {k['direction']} edge)"
            )
        # Spell out the full knob inventory so the agent re-declares every one of them
        # (a partial space.params list is rejected as key_mismatch).
        if crun.space is not None:
            lines.append("")
            lines.append("Every knob below must appear in your `space.params` response "
                         "(repeat the unexpanded ones verbatim):")
            for d in crun.space.domains:
                mark = " <- EXPAND" if any(k["name"] == d.name for k in knobs) else ""
                lines.append(f"  - {d.name} ({d.kind}): {list(d.choices)}{mark}")
        return "\n".join(lines)

    def _trials_csv(self, crun: CandidateRun) -> str:
        buf = io.StringIO()
        names = crun.space.param_names() if crun.space else []
        writer = csv.writer(buf)
        writer.writerow(["trial_id", *names, "status", "failure_kind", "latency_mean_ms",
                         "latency_std_ms", "n_regs", "n_spills", "shared_bytes",
                         "kernels_launched"])
        for t in crun.trials:
            writer.writerow([
                t.trial_id,
                *[t.params.values.get(n) for n in names],
                t.status, t.failure_kind or "",
                t.latency_ms.mean if t.latency_ms else "",
                t.latency_ms.std if t.latency_ms else "",
                t.profile.n_regs if t.profile else "",
                t.profile.n_spills if t.profile else "",
                t.profile.shared_bytes if t.profile else "",
                # Improvement M: which kernels this trial ACTUALLY launched. Without
                # this the analyst cannot tell a measured optimization from a measured
                # fallback, and will keep proposing a fusion that is already dead code.
                " ".join(t.profile.kernel_names) if t.profile and t.profile.kernel_names
                else "",
            ])
        return buf.getvalue()

    def _unlaunched_kernels(self, crun: CandidateRun) -> set[str]:
        """`@triton.jit` kernels defined in the candidate but launched by NO trial.

        Deterministic dead-code evidence: if a name never appears in any trial's
        `profile.kernel_names` across the whole budget, the tuning measured something
        other than that kernel.

        Two exclusions keep this factual rather than merely suggestive:
        - Returns an empty set when NO trial carries kernel names (a CUDA backend, or
          profiling unavailable) so absence of data never reads as absence of launches.
        - Skips `@triton.jit` DEVICE HELPERS — functions called by name from inside
          another jit body (`scores = _qk_scores(...)`, no `[grid]`). Triton inlines
          those into their caller, so they never appear in `kernel_names` even though
          they run on every trial. L3:43 `cand-d257924a` is the case: `_qk_scores` is
          inlined into `_softmax_stats` and `_attention_from_stats`, both of which
          launched on all 76 trials.
        """
        launched: set[str] = set()
        any_names = False
        for t in crun.trials:
            names = t.profile.kernel_names if t.profile else None
            if names:
                any_names = True
                launched |= set(names)
        if not any_names:
            return set()
        try:
            tree = ast.parse(crun.source)
        except SyntaxError:
            return set()
        defined = jit_kernel_names(tree)
        return defined - launched - device_helper_names(tree, defined)

    # ------------------------------------------------------------- loop C: rewrite

    def _restore_family_control_state(self) -> None:
        """Rebuild memory-only Family control fields from the event log before Loop C.

        `best_history` and `rewrite_rounds_used` live only on the in-memory Family
        object; a resumed run would otherwise start Loop C with an empty history and
        zero rounds-used, forgetting how much of the rewrite budget each family
        already spent. Both are derived from the FAMILY_ROUND_RECORDED stream (one
        event per completed rewrite round), the single persisted source of truth.

        `failed_hypotheses` is restored here too, from HYPOTHESES_FAILED: the rewriter
        reads it to avoid re-proposing a change already shown not to help, so losing it
        on resume spends rewrite rounds re-testing known dead ends.

        This is also where `best_history` is SEEDED with the seed-phase result, which
        fixes an off-by-one with two consequences. `ConvergencePolicy.family_verdict`
        checks the round budget BEFORE testing convergence, and the convergence test
        needs `no_improve_rounds + 1` history entries; with the shipped config (3 rounds,
        no_improve_rounds 2) the budget freeze fires at round 4 while the history only
        reaches 3 entries at that same moment, so `stop_kind="converged"` was
        arithmetically unreachable -- confirmed over all 48 recorded families, whose
        history lengths are 0, 1 or 3 and never more, every one `frozen_budget`.

        The second consequence is larger. `_improvement_pct` needs two entries, so a
        family's FIRST rewrite round -- the biggest gains on record, 22.5%/12.9%/15.0%/
        23.9% on run-l3-43-20260905-091705 -- scored a 0.0% slope at the moment round 2
        was allocated, making `active_families`'s slope rule blind exactly when it
        matters most. Seeding gives every family a round-0 datum, so its first round's
        gain counts toward its own ranking.

        Seeded through an EVENT rather than from the live `family.best`, because on a
        resume the live best is the best over ALL spaces including rewrites, which is
        not the seed-phase value. FAMILY_SEEDED is written once per family and replayed
        thereafter.
        """
        state = self.store.replay()
        rounds: dict[str, list[dict]] = {}
        seeded: dict[str, float] = {}
        # Rebuilt from scratch rather than extended: this runs once before Loop C and
        # HYPOTHESES_FAILED is only emitted inside Loop C, so in a fresh run there is
        # nothing to double-count -- but assigning instead of extending keeps that true
        # even if the call site ever moves.
        restored: dict[str, list[dict]] = {}
        for ev in state.events:
            if ev.type == "FAMILY_ROUND_RECORDED":
                rounds.setdefault(ev.payload["family_id"], []).append(ev.payload)
            elif ev.type == "HYPOTHESES_FAILED":
                restored.setdefault(
                    ev.payload["family_id"], []).extend(ev.payload["hypotheses"])
            elif ev.type == "FAMILY_SEEDED":
                seeded[ev.payload["family_id"]] = ev.payload["best_ms"]
        for family_id, hyps in restored.items():
            self.failed_hypotheses[family_id] = hyps
        # Seed round 0 for every family that has a correct candidate. A family with no
        # best has nothing to seed and cannot be rewritten anyway.
        for family_id, family in self.deps.families.families.items():
            if family_id in seeded or family.best is None:
                continue
            seed_ms = family.best.latency_ms
            self.store.append("FAMILY_SEEDED", {"family_id": family_id,
                                                "best_ms": seed_ms})
            seeded[family_id] = seed_ms
        for family_id, family in self.deps.families.families.items():
            evs = rounds.get(family_id, [])
            head = [seeded[family_id]] if family_id in seeded else []
            family.best_history = head + [e["best_ms"] for e in evs]
            # rewrite_rounds_used counts ROUNDS RUN, so the seed datum must not count.
            family.rewrite_rounds_used = len(evs)

    def _rewrite_round(self, round_no: int) -> bool:
        """One rewrite round across active families. Returns True if any family
        actually attempted a rewrite this round."""
        progressed = False
        for family in self.deps.families.active_families():
            verdict = self.deps.convergence.family_verdict(family)
            self.store.append("CONVERGENCE_DECIDED", {"decision": verdict.model_dump(),
                                                      "family_id": family.family_id})
            if verdict.verdict == "freeze":
                family.status = ("frozen_converged" if verdict.stop_kind == "converged"
                                 else "frozen_budget")
                continue
            if family.best is None:
                family.status = "frozen_budget"  # nothing correct in this family
                continue

            source_crun = self.runs.get(family.best.candidate_id)
            if source_crun is None or source_crun.report is None:
                family.rewrite_rounds_used += 1
                continue

            progressed = True
            key = f"rewrite:{family.family_id}:{family.rewrite_rounds_used}"
            state = self.store.replay()
            if key in state.steps_done:
                family.rewrite_rounds_used += 1
                continue

            best_before = family.best.latency_ms
            self._do_rewrite(family.family_id, source_crun)
            family.rewrite_rounds_used += 1
            best_after = (self.deps.families.families[family.family_id].best.latency_ms
                          if self.deps.families.families[family.family_id].best else
                          best_before)
            self.deps.families.record_round(family.family_id, best_after)
            # Persist the round's incumbent so best_history survives resume — it is
            # otherwise memory-only, leaving `best history: []` after a restart and
            # making the `converged` stop_kind unreachable on resumed runs.
            self.store.append("FAMILY_ROUND_RECORDED", {
                "family_id": family.family_id, "best_ms": best_after,
                "round": round_no})
            if best_after >= best_before and source_crun.report is not None:
                tried = [{"id": hyp.id, "change": hyp.change, "round": round_no}
                         for hyp in source_crun.report.hypotheses]
                self.failed_hypotheses.setdefault(family.family_id, []).extend(tried)
                # Journalled for the same reason as best_history: this is memory-only
                # state that a resume would silently lose, and the rewriter uses it to
                # avoid re-proposing changes already shown not to help. Losing it wastes
                # rewrite rounds, the scarcest budget in the loop.
                if tried:
                    self.store.append("HYPOTHESES_FAILED", {
                        "family_id": family.family_id, "round": round_no,
                        "hypotheses": tried})
            self._step_done(key)
        return progressed

    def _do_rewrite(self, family_id: str, parent_crun: CandidateRun) -> None:
        parent = parent_crun.candidate
        family = self.deps.families.families[family_id]
        try:
            best_source = materializer.materialize(parent_crun.source, family.best.params)
        except materializer.MaterializeError:
            best_source = parent_crun.source
        try:
            outcome = self.deps.rewriter.invoke(
                RewriterInputs(
                    task=self.task, best_source=best_source, report=parent_crun.report,
                    failed_hypotheses=self.failed_hypotheses.get(family_id, []),
                    device=self.cfg.device,
                    n_candidates=self.cfg.agents.rewriter.n_candidates,
                    eval_semantics=self.eval_semantics,
                )
            )
        except AgentCallError as exc:
            self.store.append("AGENT_CALL_FAILED",
                              {"module": "rewriter", "final": True,
                               "family_id": family_id, "error": str(exc)[:500]})
            return
        registered: list[str] = []
        for rw in outcome.output.candidates:
            source = outcome.sandbox.read_output(rw.file)
            cand = self._register(source, "rewrite", [parent.candidate_id],
                                  parent.backend, rw.change_summary)
            if cand is None:
                self.store.append("NOVELTY_REJECTED", {
                    "origin": "rewrite", "reason": "duplicate_signature",
                    "family_id": family_id})
                continue
            self.store.append("REWRITE_PRODUCED", {
                "candidate_id": cand.candidate_id, "family_id": family_id,
                "hypothesis_id": rw.hypothesis_id,
                "change_summary": rw.change_summary[:500]})
            registered.append(cand.candidate_id)
        # B1 also applies here: rewrite/novelty candidates are 10 of the 14 candidates
        # in a typical L3 run, so prefetching only in the seed loop covered under a
        # third of the parameterizer calls. The rewriter hands back every candidate
        # before any is pipelined, so the next one's parameterization can be issued
        # while the current one holds the GPU.
        self._pipeline_batch(registered)

    def _pipeline_batch(self, cand_ids: list[str]) -> None:
        """Run the per-candidate pipeline over a batch, prefetching the next one's
        parameterizer call while the current one occupies the GPU."""
        for i, cand_id in enumerate(cand_ids):
            for nxt in cand_ids[i + 1:][: self.cfg.budgets.prefetch_parameterization]:
                self._prefetch_parameterization(nxt)
            self._candidate_pipeline(cand_id)

    # ------------------------------------------------------------- loop D: novelty

    def _novelty_round(self, round_no: int) -> bool:
        """Generate distinctly-different new family seeds. Returns True if any
        novel candidate was accepted and pipelined."""
        if len(self.deps.families.families) >= self.cfg.budgets.max_families_total:
            return False
        key = f"novelty:{round_no}"
        state = self.store.replay()
        if key in state.steps_done:
            return False

        summaries = []
        for fam in self.deps.families.families.values():
            anchor = self.deps.families.candidates[fam.anchor_candidate_id]
            summaries.append({
                "family_id": fam.family_id,
                "approach_summary": anchor.approach_summary,
                "best_ms": fam.best.latency_ms if fam.best else None,
                "status": fam.status,
                "anchor_source": self.deps.families.source_of(fam.anchor_candidate_id),
            })
        try:
            outcome = self.deps.novelty.invoke(
                NoveltyInputs(
                    task=self.task,
                    ref_source=Path(self.task.ref_path).read_text(encoding="utf-8"),
                    family_summaries=summaries, device=self.cfg.device,
                    n_candidates=self.cfg.agents.novelty.n_candidates,
                    eval_semantics=self.eval_semantics,
                )
            )
        except AgentCallError as exc:
            self.store.append("AGENT_CALL_FAILED",
                              {"module": "novelty", "final": True,
                               "error": str(exc)[:500]})
            self._step_done(key)
            return False

        accepted_any = False
        accepted_ids: list[str] = []
        for nc in outcome.output.candidates:
            source = outcome.sandbox.read_output(nc.file)
            result = self.deps.families.accept_novel_seed(
                source, nc.backend, nc.approach_summary, nc.difference_claim
            )
            if isinstance(result, NoveltyRejection):
                self.store.append("NOVELTY_REJECTED", {
                    "reason": result.reason, "detail": result.detail[:500]})
                continue
            cand_dir = self.store.candidate_dir(result.candidate_id)
            (cand_dir / "source.py").write_text(source, encoding="utf-8")
            self.store.append("CANDIDATE_REGISTERED", {"candidate": result.model_dump()})
            self.store.append("NOVELTY_PRODUCED", {
                "candidate_id": result.candidate_id,
                "difference_claim": nc.difference_claim[:500]})
            self.runs[result.candidate_id] = CandidateRun(candidate=result, source=source)
            accepted_any = True
            accepted_ids.append(result.candidate_id)
        self._pipeline_batch(accepted_ids)
        self._step_done(key)
        return accepted_any

    # ------------------------------------------------------------------- finalize

    def _finalize(self) -> dict[str, Any]:
        best_family = None
        for fam in self.deps.families.families.values():
            if fam.best is None:
                continue
            if best_family is None or fam.best.latency_ms < best_family.best.latency_ms:
                best_family = fam

        result: dict[str, Any] = {
            "task": self.task.model_dump(),
            "baselines": [b.model_dump() for b in self.baselines],
            "families": self.deps.families.lineage_tree(),
            "elapsed_hours": round(self._elapsed_hours(), 3),
        }
        if best_family is None:
            result["best"] = None
            return result

        crun = self.runs[best_family.best.candidate_id]
        final_src = materializer.materialize(crun.source, best_family.best.params)
        final_path = self.store.run_dir / "report" / "best_kernel.py"
        final_path.write_text(final_src, encoding="utf-8")
        reeval = self.deps.benchmarker.final_reeval(self.task, final_path,
                                                    backend=crun.candidate.backend)
        lat = latency_from_result(reeval)
        precision = _detect_candidate_precision(final_src, best_family.best.params)
        result["best"] = {
            "candidate_id": best_family.best.candidate_id,
            "family_id": best_family.family_id,
            "params": best_family.best.params.model_dump(),
            "tuned_ms": best_family.best.latency_ms,
            "final_reeval_ok": bool(reeval.get("ok")),
            "final_reeval_ms": lat.mean if lat else None,
            "excessive_speedup_flag": bool(reeval.get("excessive_speedup")),
            "precision": precision,
        }
        # Speedup vs every recorded baseline (4-way when dual-precision baselines
        # were measured: eager, eager_tf32, torch_compile, torch_compile_tf32).
        if lat and lat.mean > 0:
            speedups: dict[str, float] = {}
            for b in self.baselines:
                if b.latency_ms.mean > 0:
                    speedups[b.kind] = round(b.latency_ms.mean / lat.mean, 4)
            result["best"]["speedups"] = speedups
            # Back-compat scalar fields (vs the ieee/untagged baselines).
            if "eager" in speedups:
                result["best"]["speedup_vs_eager"] = speedups["eager"]
            if "torch_compile" in speedups:
                result["best"]["speedup_vs_compile"] = speedups["torch_compile"]
            # Honest same-precision verdict: compare the candidate against the
            # baseline computed at the SAME precision the candidate actually uses.
            # A tf32 candidate must be judged against tf32 torch.compile, not the
            # slower ieee baseline, or the speedup is measured against a strawman.
            result["best"]["honest_verdict"] = _honest_verdict(precision, speedups)
        return result

