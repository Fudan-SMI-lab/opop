"""Optuna grouped-TPE tuner with ask/tell, guard filtering, and anchors."""

from __future__ import annotations

import uuid
from collections.abc import Callable

import optuna
from optuna.samplers import TPESampler
from optuna.trial import TrialState

from kernel_optimizer.models.core import ParameterSpace, ParamSet, TrialRecord


class OptunaTPETuner:
    """Bayesian (TPE) search over a choice-grid ParameterSpace.

    Guard-rejected draws are told as PRUNED and re-asked (bounded), so the
    sampler learns to avoid infeasible regions without costing GPU time.
    """

    def __init__(
        self,
        space: ParameterSpace,
        guard_ok: Callable[[ParamSet], bool],
        budget: int,
        seed: int = 0,
        anchors: tuple[ParamSet, ...] = (),
        constant_liar: bool = False,
        max_guard_rejects_per_ask: int = 64,
    ):
        self.space = space
        self.guard_ok = guard_ok
        self.budget = budget
        self.max_guard_rejects = max_guard_rejects_per_ask
        sampler = TPESampler(
            seed=seed,
            multivariate=True,
            group=True,
            n_startup_trials=10,
            constant_liar=constant_liar,
        )
        self.study = optuna.create_study(direction="minimize", sampler=sampler)
        for anchor in anchors:
            self.study.enqueue_trial(dict(anchor.values), skip_if_exists=True)
        self._pending: dict[str, optuna.trial.Trial] = {}
        self._asked = 0
        self._best_record: TrialRecord | None = None
        self._seen: set[str] = set()

    def ask(self) -> tuple[str, ParamSet] | None:
        if self._asked >= self.budget:
            return None
        rejects = 0
        while rejects < self.max_guard_rejects:
            trial = self.study.ask()
            values: dict = {}
            for domain in self.space.domains:
                values[domain.name] = trial.suggest_categorical(
                    domain.name, list(domain.choices)
                )
            params = ParamSet(values=values)
            key = params.key()
            if key in self._seen:
                # Duplicate draw: prune and move on (counts toward reject budget).
                self.study.tell(trial, state=TrialState.PRUNED)
                rejects += 1
                continue
            if not self.guard_ok(params):
                self.study.tell(trial, state=TrialState.PRUNED)
                rejects += 1
                continue
            trial_id = f"tr-{uuid.uuid4().hex[:8]}"
            self._pending[trial_id] = trial
            self._seen.add(key)
            self._asked += 1
            return trial_id, params
        return None  # space is effectively exhausted for the sampler

    def tell(self, trial_id: str, record: TrialRecord) -> None:
        trial = self._pending.pop(trial_id, None)
        if trial is None:
            raise KeyError(f"unknown or already-told trial {trial_id}")
        if record.status == "complete" and record.latency_ms is not None:
            # `robust_ms` (median, falling back to the mean) rather than `.mean`. A trial is
            # timed with `quick_perf_trials` samples -- 20 by default -- and at that count a
            # few 300-700 us scheduling stalls drag the mean 35-136% above the kernel's real
            # cost. Measured with scripts/probe_robust_objective.py against a 2000-sample
            # ground truth: the mean's CV at n=20 is 24-37% versus the median's 3-8%, and on
            # a pair of configurations whose true costs differ by 7.6% the mean identifies
            # the faster one 64.8% of the time -- barely above chance -- while the median
            # manages 93.2%. Since TPE decides where to sample next from these comparisons,
            # a near-coin-flip objective wastes the trial budget exploring noise.
            #
            # Live example this fixes (run-l2-37-20260907-010645): a space expansion was
            # credited with 32.60 -> 30.70 us, a 5.8% "gain" that cleared
            # min_improvement_pct 2.0 and earned the family another rewrite round -- while
            # the difference was 1.90 us against a combined standard error of 17.85 us, and
            # the supposedly-better point was SLOWER by min. 40 trials spent on noise.
            self.study.tell(trial, record.latency_ms.robust_ms)
            if (
                self._best_record is None
                or record.latency_ms.robust_ms < self._best_record.latency_ms.robust_ms
            ):
                self._best_record = record
        else:
            self.study.tell(trial, state=TrialState.FAIL)

    def best(self) -> TrialRecord | None:
        return self._best_record

    def snapshot(self) -> dict:
        return {
            "asked": self._asked,
            "budget": self.budget,
            "pending": len(self._pending),
            "incumbent_ms": (
                self._best_record.latency_ms.robust_ms if self._best_record else None
            ),
        }
