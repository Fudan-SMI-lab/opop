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
                 feasibility_samples: int = 512, seed: int = 0):
        self.correctness = correctness
        self.device = device
        self.eval_cfg = eval_cfg
        self.min_feasible_frac = min_feasible_frac
        self.feasibility_samples = feasibility_samples
        self.seed = seed

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
                return SpaceRejection(
                    reason=f"witness_{label}_failed",
                    detail=f"{result.get('failure_kind')}: {result.get('log_tail', '')[:500]}",
                )
            lat = latency_from_result(result)
            witnesses.append(
                WitnessResult(params=params,
                              latency_mean_ms=lat.mean if lat else None,
                              worker_result=result)
            )
        return SpaceAccepted(space=space, witnesses=witnesses)

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
