"""Optuna grouped-TPE tuner with ask/tell, guard filtering, and anchors."""

from __future__ import annotations

import uuid
from collections.abc import Callable

import optuna
from optuna.samplers import TPESampler
from optuna.trial import TrialState

from kernel_optimizer.models.core import ParameterSpace, ParamSet, TrialRecord
from kernel_optimizer.tuning import ordered_domains


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
        deweight_reject: Callable[[ParamSet], bool] | None = None,
        ordered_categoricals: bool = False,
    ):
        self.space = space
        self.guard_ok = guard_ok
        self.budget = budget
        self.max_guard_rejects = max_guard_rejects_per_ask
        # S1b. None = off, which is the pre-v3 behaviour. A callable returns True to refuse a
        # drawn configuration probabilistically because it holds a value whose failures no
        # partner explains; see tuning/deweight.py for the criterion and its measurements.
        self.deweight_reject = deweight_reject
        # S8 / item 3.1. False = off and LITERALLY the old path: `distance_funcs` is not even
        # called, and `categorical_distance_func=None` is what TPESampler already received.
        #
        # It has to be a switch rather than unconditional because it changes what the sampler
        # draws, so a run with it on is not comparable with the finished ones. Measured before being
        # written (scripts/probes/is_categorical_distance_func_functional.py): the argument is
        # deprecated in optuna 4.9.0 but still functional, its CONTENT is read (a scrambled rung
        # order loses 34.1 of 120 near-optimum draws, 12/12 seeds), and against today's sampler it
        # wins +4.5 near-best draws and 12.1% mean cost on a synthetic 8-rung ladder, 12/12 seeds.
        #
        # `or None`: an empty dict and None are the same sampler, but passing {} would emit optuna's
        # deprecation FutureWarning for a call that asked for nothing. The distinction between
        # "asked for, nothing qualified" and "not asked for" is kept in the LOG instead, where it
        # belongs -- the orchestrator journals `ordered_domains.snapshot()` from the same predicate
        # on the same domains. Not stored on the tuner: every stand-in for a tuner would then have to
        # remember the attribute, and one that forgot would journal a silent None.
        self.ordered_categoricals = ordered_categoricals
        distance = (ordered_domains.distance_funcs(list(space.domains)) or None
                    if ordered_categoricals else None)
        sampler = TPESampler(
            seed=seed,
            multivariate=True,
            group=True,
            n_startup_trials=10,
            constant_liar=constant_liar,
            categorical_distance_func=distance,
        )
        self.study = optuna.create_study(direction="minimize", sampler=sampler)
        for anchor in anchors:
            self.study.enqueue_trial(dict(anchor.values), skip_if_exists=True)
        self._pending: dict[str, optuna.trial.Trial] = {}
        self._asked = 0
        # Trials that have returned a result. S7's recompute cadence runs on this rather than on
        # `_asked` (see `n_told`), and it is counted here rather than derived as `_asked - pending`
        # because that difference is also 0 for a trial told twice -- which raises, but only after the
        # cadence has already been computed from a wrong number.
        self._told = 0
        self._best_record: TrialRecord | None = None
        self._seen: set[str] = set()
        # v4.1 §7: params-keys whose NEXT draws must be measured fresh — enqueued by the
        # conditional scanner for scan endpoints (anchors re-measured, old winners out of
        # the baseline). A COUNTER, not a set: a C4 block measures the SAME endpoint config
        # twice (discovery + validation slots), so two fresh passes of one key must both
        # survive the dedup branch. Each pass is handed to the caller with fresh=True via
        # `is_fresh`, so the measured-cache reuse path is bypassed for exactly those draws.
        # Without this, `skip_if_exists` + `_seen` + the orchestrator's measured_cache
        # would either drop the re-measurement or fake it with a cache copy (the fake A/A
        # zero the v3 review's E1 blocker names).
        self._fresh_intent: dict[str, int] = {}
        # trial_id -> the draw consumed a fresh intent. Queried by the orchestrator to
        # bypass its measured cache; a separate map (not a wider ask() tuple) so every
        # existing caller of ask() keeps its two-element contract.
        self._fresh_asked: set[str] = set()

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
            fresh = self._fresh_intent.get(key, 0) > 0
            if key in self._seen and not fresh:
                # Duplicate draw: prune and move on (counts toward reject budget).
                self.study.tell(trial, state=TrialState.PRUNED)
                rejects += 1
                continue
            if not self.guard_ok(params):
                self.study.tell(trial, state=TrialState.PRUNED)
                rejects += 1
                continue
            if self.deweight_reject is not None and self.deweight_reject(params):
                # S1b. Told PRUNED and re-asked, exactly like a guard rejection, and counted
                # against the SAME bounded reject budget -- so a space where many values have
                # fired degrades into "ask fewer times", never into a spin. Crucially this costs
                # no GPU: the refusal happens before the trial is run.
                #
                # PRUNED rather than FAIL for the same reason the guard path uses it: Optuna keeps
                # pruned trials in the TPE model and drops failed ones, so pruning lets the
                # sampler see that this region was visited and passed over. It is not a claim that
                # the point is infeasible -- the point may well be drawn and measured on a later
                # ask, which is the whole difference between down-weighting and removal.
                self.study.tell(trial, state=TrialState.PRUNED)
                rejects += 1
                continue
            if fresh:
                # One fresh pass consumed per intent unit; a C4's duplicated endpoint
                # carries TWO units, so both survive dedup and a later ordinary draw of
                # the same key is deduped as before.
                remaining = self._fresh_intent[key] - 1
                if remaining > 0:
                    self._fresh_intent[key] = remaining
                else:
                    del self._fresh_intent[key]
            trial_id = f"tr-{uuid.uuid4().hex[:8]}"
            if fresh:
                self._fresh_asked.add(trial_id)
            self._pending[trial_id] = trial
            self._seen.add(key)
            self._asked += 1
            return trial_id, params
        return None  # space is effectively exhausted for the sampler

    def is_fresh(self, trial_id: str) -> bool:
        """v4.1 §7: did this draw consume a fresh-measurement intent? The orchestrator must
        bypass its measured cache for such a draw — a cache copy with a swapped trial_id
        would fake the scanner's A/A anchor as a zero difference."""
        return trial_id in self._fresh_asked

    # ---------------------------------------------------------------- S7 / item 3.2

    def enqueue(self, params: ParamSet) -> str | None:
        """S7: put one point at the head of the queue. None on success, else the refusal reason.

        The point is consumed by the ordinary `ask()`, so it is subject to every rule a drawn point is
        subject to and it counts against the SAME `budget`. That is what makes "only adds candidate
        points, does not change `trials_per_space`" true rather than asserted: S7 changes which points
        the budget is spent on and never how many there are.

        The two refusals happen HERE rather than being discovered later, because Optuna's queue has no
        way to report either one:

        * `already_drawn` -- an enqueued duplicate would come back from `ask()`, be told PRUNED by the
          dedup branch, and consume one of the bounded `max_guard_rejects` re-asks. The caller would see
          a successful enqueue and a trial that never appeared.
        * `guard_rejected` -- the same, and it is the more important of the two: a queued point that the
          space's own constraints forbid must NOT be forced through. S7 proposes; the guard still
          decides, exactly as for a drawn point.

        A value outside the domain's declared choices cannot be refused here because Optuna does not
        raise on one -- it warns and samples that knob normally, which would silently turn a suggestion
        into an ordinary draw. `slope_guide._toward_wall` therefore only ever proposes declared choices;
        this is the reason it reads the domain instead of extrapolating a neighbour.
        """
        key = params.key()
        if key in self._seen:
            return "already_drawn"
        if not self.guard_ok(params):
            return "guard_rejected"
        self.study.enqueue_trial(dict(params.values), skip_if_exists=True)
        return None

    def enqueue_fresh(self, params: ParamSet) -> str | None:
        """v4.1 §7: enqueue a point that MUST be measured fresh even if already drawn.

        The scanner's anchors are deliberate re-measurements (old winners leave the
        baseline; a live A/A), so `already_drawn` is not a refusal here — the queue entry
        skips Optuna's own dedup (`skip_if_exists=False`) and the key gains one fresh unit
        so `ask()` lets it through its dedup branch and reports it via `is_fresh`, which
        bypasses the measured cache for that draw. Guard rules still apply: the scanner
        proposes, the guard decides — identical to `enqueue`.
        """
        if not self.guard_ok(params):
            return "guard_rejected"
        key = params.key()
        self._fresh_intent[key] = self._fresh_intent.get(key, 0) + 1
        self.study.enqueue_trial(dict(params.values), skip_if_exists=False)
        return None

    @property
    def n_told(self) -> int:
        """Trials that have RETURNED a result -- the clock S7's recompute cadence runs on.

        Not `_asked`: a wall is derived from measured latencies, and with `constant_liar` there can be
        asked-but-untold trials at any moment, so a cadence counted on asks would recompute against a
        stats table that has not moved.
        """
        return self._told

    def drawn_keys(self) -> set[str]:
        """A copy of the parameter keys already drawn, for S7's dedup before it proposes.

        A copy rather than the set itself: a caller that mutated it would change what `ask()` treats as
        a duplicate, and the resulting extra PRUNED draws would look like the sampler exhausting the
        space.
        """
        return set(self._seen)

    def tell(self, trial_id: str, record: TrialRecord) -> None:
        trial = self._pending.pop(trial_id, None)
        if trial is None:
            raise KeyError(f"unknown or already-told trial {trial_id}")
        self._told += 1
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
            # P1: a config refused for a HARD, config-determined reason is reported PRUNED, not
            # FAIL. Optuna excludes FAIL from the TPE model but keeps PRUNED in it -- measured:
            # 12 trials reported FAIL leave 1 trial visible to the sampler, the same 12 reported
            # PRUNED leave 13. So reporting these as FAIL threw the information away, which is
            # why L3:43's per-candidate shared-memory failure rate (21-33%, 180 of 1004 trials)
            # never decayed over a run: TPE kept proposing a region it was never told about.
            #
            # Only for reasons that are a property of the CONFIGURATION and would recur
            # identically -- an over-limit shared-memory requirement, a guard rejection, a
            # materialize error. A `runtime_error` or `correctness_mismatch` stays FAIL: those
            # can be non-deterministic or a defect in the candidate rather than in the point,
            # and teaching the sampler to avoid that region would be teaching it noise.
            hard = record.failure_kind in ("infeasible_shared_memory", "guard_rejected",
                                           "materialize_error")
            self.study.tell(trial, state=TrialState.PRUNED if hard else TrialState.FAIL)

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
