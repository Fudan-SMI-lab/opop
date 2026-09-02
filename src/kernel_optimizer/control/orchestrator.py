"""Outer-loop orchestrator: baseline -> seeds -> per-candidate pipeline
(parameterize -> tune -> stats -> analyze) -> rewrite rounds (loop C) ->
novelty rounds (loop D) -> final re-eval -> report.

Every step is guarded by a step_key; replay()-known steps are skipped on resume.
"""

from __future__ import annotations

import csv
import io
import time
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
    ParameterSpace,
    ParamSet,
    TaskSpec,
    TrialRecord,
)
from kernel_optimizer.models.reports import BottleneckReport, TuningStats
from kernel_optimizer.paramspace import materializer
from kernel_optimizer.paramspace.guard import check_config
from kernel_optimizer.paramspace.validation import SpaceAccepted, SpaceValidator
from kernel_optimizer.store.run_store import RunStore
from kernel_optimizer.tuning.stats import TuningStatsAnalyzer
from kernel_optimizer.tuning.tpe import OptunaTPETuner


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
        self.runs: dict[str, CandidateRun] = {}
        self.failed_hypotheses: dict[str, list[dict]] = {}  # family_id -> tried-and-failed

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
        self.store.write_state_snapshot({"phase": "started", "task": self.task.model_dump()})
        self._baseline()
        self._generate_seeds()

        for cand_id in list(self.runs):
            self._candidate_pipeline(cand_id)

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

        result = self._finalize()
        self.store.append("RUN_FINISHED", {"summary": result})
        return result

    def _baseline(self) -> None:
        key = "baseline"
        state = self.store.replay()
        if key in state.steps_done:
            self.baselines = [Baseline.model_validate(b["baseline"])
                              for b in state.baselines]
            return
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

    def _parameterize_with_repair(self, crun: CandidateRun) -> SpaceAccepted | None:
        """Loop A: parameterize; on failure feed typed errors to repair, retry."""
        cand = crun.candidate
        source = crun.source
        feedback = ""
        for attempt in range(self.cfg.budgets.repair_attempts + 1):
            try:
                outcome = self.deps.parameterizer.invoke(
                    ParameterizerInputs(
                        task=self.task, candidate_source=source,
                        device=self.cfg.device, prior_feedback=feedback,
                    )
                )
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
                "reason": verdict.reason, "detail": verdict.detail[:800],
            })
            if attempt >= self.cfg.budgets.repair_attempts:
                return None

            # Witness execution failures go to the repair agent; other rejections
            # go back to the parameterizer as feedback.
            if verdict.reason.startswith("witness_"):
                try:
                    repaired = self.deps.repair.invoke(
                        RepairInputs(
                            task=self.task, broken_source=param_source,
                            failure_kind=verdict.reason, failure_detail=verdict.detail,
                            device=self.cfg.device,
                        )
                    )
                    source = repaired.sandbox.read_output(repaired.output.file)
                    self.store.append("REPAIR_PRODUCED", {
                        "candidate_id": cand.candidate_id,
                        "diagnosis": repaired.output.diagnosis[:500],
                    })
                    feedback = ""
                except AgentCallError as exc:
                    feedback = f"repair also failed: {str(exc)[:300]}"
            else:
                feedback = f"{verdict.reason}: {verdict.detail[:500]}"
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
            else:
                record = self._run_trial(crun, space, trial_id, params, trials_dir)
                self.store.append("TRIAL_DONE", {"trial": record.model_dump()})
            tuner.tell(trial_id, record)
            crun.trials.append(record)

        best = tuner.best()
        if best is not None:
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
                failure_detail=str(result.get("log_tail", ""))[:500],
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
        if crun.best_ms is None:
            return  # nothing correct; analysis would have no signal
        try:
            outcome = self.deps.analyst.invoke(
                AnalystInputs(
                    task=self.task, candidate_source=crun.source, stats=crun.stats,
                    trials_csv=self._trials_csv(crun), device=self.cfg.device,
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

    def _trials_csv(self, crun: CandidateRun) -> str:
        buf = io.StringIO()
        names = crun.space.param_names() if crun.space else []
        writer = csv.writer(buf)
        writer.writerow(["trial_id", *names, "status", "failure_kind", "latency_mean_ms",
                         "latency_std_ms", "n_regs", "n_spills", "shared_bytes"])
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
            ])
        return buf.getvalue()

    # ------------------------------------------------------------- loop C: rewrite

    def _restore_family_control_state(self) -> None:
        """Rebuild memory-only Family control fields from the event log before Loop C.

        `best_history` and `rewrite_rounds_used` live only on the in-memory Family
        object; a resumed run would otherwise start Loop C with an empty history and
        zero rounds-used, forgetting how much of the rewrite budget each family
        already spent. Both are derived from the FAMILY_ROUND_RECORDED stream (one
        event per completed rewrite round), the single persisted source of truth.
        """
        state = self.store.replay()
        rounds: dict[str, list[dict]] = {}
        for ev in state.events:
            if ev.type == "FAMILY_ROUND_RECORDED":
                rounds.setdefault(ev.payload["family_id"], []).append(ev.payload)
        for family_id, evs in rounds.items():
            family = self.deps.families.families.get(family_id)
            if family is None:
                continue
            family.best_history = [e["best_ms"] for e in evs]
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
                for hyp in source_crun.report.hypotheses:
                    self.failed_hypotheses.setdefault(family.family_id, []).append(
                        {"id": hyp.id, "change": hyp.change, "round": round_no}
                    )
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
                )
            )
        except AgentCallError as exc:
            self.store.append("AGENT_CALL_FAILED",
                              {"module": "rewriter", "final": True,
                               "family_id": family_id, "error": str(exc)[:500]})
            return
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
            self._candidate_pipeline(cand.candidate_id)

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
                )
            )
        except AgentCallError as exc:
            self.store.append("AGENT_CALL_FAILED",
                              {"module": "novelty", "final": True,
                               "error": str(exc)[:500]})
            self._step_done(key)
            return False

        accepted_any = False
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
            self._candidate_pipeline(result.candidate_id)
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
        result["best"] = {
            "candidate_id": best_family.best.candidate_id,
            "family_id": best_family.family_id,
            "params": best_family.best.params.model_dump(),
            "tuned_ms": best_family.best.latency_ms,
            "final_reeval_ok": bool(reeval.get("ok")),
            "final_reeval_ms": lat.mean if lat else None,
            "excessive_speedup_flag": bool(reeval.get("excessive_speedup")),
        }
        eager = next((b for b in self.baselines if b.kind == "eager"), None)
        compiled = next((b for b in self.baselines if b.kind == "torch_compile"), None)
        if lat:
            if eager and eager.latency_ms.mean > 0:
                result["best"]["speedup_vs_eager"] = round(eager.latency_ms.mean / lat.mean, 4)
            if compiled and compiled.latency_ms.mean > 0:
                result["best"]["speedup_vs_compile"] = round(
                    compiled.latency_ms.mean / lat.mean, 4)
        return result
