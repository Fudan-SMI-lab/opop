"""Parameter-space acceptance: structural checks + two GPU witness quick tests."""

from __future__ import annotations

import itertools
import random
import uuid
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from kernel_optimizer.config import EvalConfig
from kernel_optimizer.evaluation.correctness import CorrectnessEvaluator, latency_from_result
from kernel_optimizer.models.core import (
    Candidate,
    Constraint,
    DeviceLimits,
    ParamDomain,
    ParameterSpace,
    ParamSet,
    TaskSpec,
)
from kernel_optimizer.models.reports import ParameterizationResult
from kernel_optimizer.paramspace import materializer
from kernel_optimizer.paramspace.guard import ConstraintError, check_config, eval_constraint


class SpaceRejection(BaseModel):
    reason: str
    detail: str


def error_excerpt(log_tail: str, limit: int = 2000) -> str:
    """Keep the END of a traceback, not the beginning.

    The worker captures the last 4000 chars of the traceback; the actual exception
    type and message are on its LAST line, while the first ~500 chars are the
    harness's own call frames. Truncating from the front (`log_tail[:500]`) therefore
    handed the repair agent a diagnosis-free prefix — observed live on L3:43, where
    three seeds were rejected with a detail that ended mid-frame at
    `torch/nn/modules/module.py ... return self._call_impl(*args, **kwargs)` and never
    named the error. Prefer the tail; keep a short head for context when both fit.
    """
    text = (log_tail or "").strip()
    if len(text) <= limit:
        return text
    head = limit // 4
    return f"{text[:head]}\n...[{len(text) - limit} chars elided]...\n{text[-(limit - head):]}"


class WitnessResult(BaseModel):
    params: ParamSet
    latency_mean_ms: float | None = None
    worker_result: dict[str, Any]


class SpaceAccepted(BaseModel):
    space: ParameterSpace
    witnesses: list[WitnessResult]


class SpaceValidator:
    def __init__(self, correctness: CorrectnessEvaluator, device: DeviceLimits,
                 eval_cfg: EvalConfig, min_feasible_frac: float = 0.25,
                 feasibility_samples: int = 512, seed: int = 0,
                 max_witness_retries: int = 2):
        self.correctness = correctness
        self.device = device
        self.eval_cfg = eval_cfg
        self.min_feasible_frac = min_feasible_frac
        self.feasibility_samples = feasibility_samples
        self.seed = seed
        # Each retry is a real GPU quick test (~30-60s on L3), so the walk is bounded.
        # 2 is enough to step past a single out-of-range corner such as fp16 on a task
        # whose outputs exceed 65504, which is the case this exists for.
        self.max_witness_retries = max_witness_retries

    def validate_and_publish(
        self,
        candidate: Candidate,
        source: str,
        proposal: ParameterizationResult,
        task: TaskSpec,
        work_dir: Path,
        version: int = 1,
    ) -> SpaceAccepted | SpaceRejection:
        # 1. structural checks -------------------------------------------------
        try:
            defaults = materializer.extract_defaults(source)
        except materializer.MaterializeError as exc:
            return SpaceRejection(reason=f"materialize:{exc.kind}", detail=exc.detail)

        domains = [
            ParamDomain(name=p.name, kind=p.kind, choices=list(p.choices),
                        description=p.description)
            for p in proposal.space.params
        ]
        constraints = [Constraint(expr=c.expr, rationale=c.rationale)
                       for c in proposal.space.constraints]
        declared = {d.name for d in domains}
        if declared != set(defaults):
            return SpaceRejection(
                reason="key_mismatch",
                detail=f"PARAMS keys {sorted(defaults)} != declared {sorted(declared)}",
            )
        for d in domains:
            if len(d.choices) < 2:
                return SpaceRejection(reason="degenerate_domain",
                                      detail=f"{d.name} has <2 choices")
            if len(set(map(repr, d.choices))) != len(d.choices):
                return SpaceRejection(reason="duplicate_choices", detail=d.name)
            if defaults[d.name] not in d.choices:
                return SpaceRejection(
                    reason="default_not_in_choices",
                    detail=f"{d.name} default {defaults[d.name]!r} not in {d.choices}",
                )

        space = ParameterSpace(
            space_id=f"sp-{uuid.uuid4().hex[:8]}",
            candidate_id=candidate.candidate_id,
            source_sha=candidate.source_sha,
            version=version,
            domains=domains,
            constraints=constraints,
        )

        # 2. constraint satisfiability over a sampled choice grid ---------------
        frac = self._feasible_fraction(space)
        if isinstance(frac, SpaceRejection):
            return frac
        if frac < self.min_feasible_frac:
            return SpaceRejection(
                reason="infeasible_space",
                detail=f"only {frac:.0%} of sampled grid satisfies constraints "
                       f"(need >= {self.min_feasible_frac:.0%})",
            )

        # 3. two GPU witnesses ---------------------------------------------------
        default_params = ParamSet(values=defaults)
        minimal_params = ParamSet(values={d.name: d.choices[0] for d in domains})
        if check_config(space, minimal_params, self.device) is not None:
            minimal_params = self._first_feasible(space, exclude=default_params)
            if minimal_params is None:
                return SpaceRejection(reason="no_feasible_witness",
                                      detail="cannot find a second feasible config")

        witness_sources: dict[str, str] = {}
        witnesses: list[WitnessResult] = []
        for label, params in (("default", default_params), ("minimal", minimal_params)):
            try:
                mat_src = materializer.materialize(source, params)
            except materializer.MaterializeError as exc:
                return SpaceRejection(reason=f"materialize:{exc.kind}", detail=exc.detail)
            witness_sources[label] = mat_src

        if witness_sources["default"] == witness_sources["minimal"]:
            return SpaceRejection(
                reason="inert_space",
                detail="default and minimal configs materialize to identical source",
            )

        for label, params in (("default", default_params), ("minimal", minimal_params)):
            path = work_dir / f"witness_{label}.py"
            path.write_text(witness_sources[label], encoding="utf-8")
            result = self.correctness.quick_test(
                task, path, tag=f"{candidate.candidate_id}-wit-{label}",
                backend=candidate.backend,
            )
            if not result.get("ok"):
                # The SECOND witness only has to prove the space is not inert -- that some
                # config other than the default actually runs. It does not have to be the
                # CHEAPEST one, and insisting on that was rejecting whole spaces for a
                # reason no repair could fix.
                #
                # `minimal_params` is choices[0] of every knob, and candidate_contract.md
                # asks for choices ordered cheap->expensive with "fp16" first on the
                # precision knob. On level3/48, whose outputs reach 1e22 against fp16's
                # 65504 ceiling, that corner overflows by construction: 7 of 7 candidates
                # declaring a COMPUTE_DTYPE knob were rejected here while 7 of 7 without one
                # were published, and repair had already fixed the real defect at attempt 1
                # in four of those seven before burning its remaining budget on an
                # impossible config. See docs/finding-minimal-witness-forces-fp16.md.
                #
                # So retry with the next feasible config before giving up. Only a candidate
                # that fails at EVERY alternative is rejected -- the anti-inertness
                # guarantee is preserved, since acceptance still requires two distinct
                # sources both passing a real GPU correctness test.
                if label == "minimal":
                    alt = self._next_witness(
                        space, exclude=(default_params, minimal_params),
                        source=source, work_dir=work_dir, task=task, candidate=candidate,
                        witness_sources_default=witness_sources["default"])
                    if alt is not None:
                        witnesses.append(alt)
                        break
                passed_note = ("the DEFAULT config passed; only this one failed. "
                               if label == "minimal" else "")
                return SpaceRejection(
                    reason=f"witness_{label}_failed",
                    # Which witness failed is the difference between "your kernel is broken"
                    # and "the cheapest corner of your own space is out of range", and the
                    # label was previously dropped on the floor -- it reached the event's
                    # `reason` but never the repair prompt, so every diagnosis in these
                    # chains read as though the kernel were globally wrong.
                    detail=f"[{label} witness config {params.values}] {passed_note}"
                           f"{result.get('failure_kind')}: "
                           f"{error_excerpt(result.get('log_tail', ''))}",
                )
            lat = latency_from_result(result)
            witnesses.append(
                WitnessResult(params=params,
                              latency_mean_ms=lat.mean if lat else None,
                              worker_result=result)
            )
        return SpaceAccepted(space=space, witnesses=witnesses)

    def _next_witness(self, space: ParameterSpace, exclude: tuple[ParamSet, ...],
                      source: str, work_dir: Path, task: TaskSpec,
                      candidate: Candidate, witness_sources_default: str) -> WitnessResult | None:
        """Find a second witness that is neither the default nor an already-failed config.

        Bounded: tries at most `max_witness_retries` alternatives, because each one is a
        real GPU quick test and an exhaustive walk of the grid would cost more than the
        tuning it is gating.
        """
        tried = [p.values for p in exclude]
        attempted = 0
        for combo in itertools.product(*[d.choices for d in space.domains]):
            if attempted >= self.max_witness_retries:
                return None
            params = ParamSet(values=dict(zip(space.param_names(), combo)))
            if params.values in tried:
                continue
            if check_config(space, params, self.device) is not None:
                continue
            try:
                mat_src = materializer.materialize(source, params)
            except materializer.MaterializeError:
                continue
            if mat_src == witness_sources_default:
                continue  # inert against the default; no evidence of a live knob
            attempted += 1
            path = work_dir / f"witness_alt{attempted}.py"
            path.write_text(mat_src, encoding="utf-8")
            result = self.correctness.quick_test(
                task, path, tag=f"{candidate.candidate_id}-wit-alt{attempted}",
                backend=candidate.backend,
            )
            if result.get("ok"):
                lat = latency_from_result(result)
                return WitnessResult(params=params,
                                     latency_mean_ms=lat.mean if lat else None,
                                     worker_result=result)
        return None

    # -- helpers ---------------------------------------------------------------

    def _feasible_fraction(self, space: ParameterSpace) -> float | SpaceRejection:
        env_base = self.device.as_env()
        grids = [d.choices for d in space.domains]
        total = 1
        for g in grids:
            total *= len(g)
        rng = random.Random(self.seed)
        if total <= self.feasibility_samples:
            samples = list(itertools.product(*grids))
        else:
            samples = [tuple(rng.choice(g) for g in grids)
                       for _ in range(self.feasibility_samples)]
        names = space.param_names()
        ok = 0
        for combo in samples:
            env = {**env_base, **dict(zip(names, combo))}
            try:
                if all(eval_constraint(c.expr, env) for c in space.constraints):
                    ok += 1
            except ConstraintError as exc:
                return SpaceRejection(reason="constraint_invalid", detail=str(exc))
        return ok / len(samples) if samples else 0.0

    def _first_feasible(self, space: ParameterSpace,
                        exclude: ParamSet) -> ParamSet | None:
        grids = [d.choices for d in space.domains]
        names = space.param_names()
        for combo in itertools.product(*grids):
            params = ParamSet(values=dict(zip(names, combo)))
            if params.values == exclude.values:
                continue
            if check_config(space, params, self.device) is None:
                return params
        return None
