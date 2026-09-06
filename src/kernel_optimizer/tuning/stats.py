"""Deterministic tuning-record analysis: boundary-blocked params, failure clusters."""

from __future__ import annotations

import statistics
from collections import Counter, defaultdict

from kernel_optimizer.models.core import DeviceLimits, ParameterSpace, TrialRecord
from kernel_optimizer.models.reports import (
    FailureCluster,
    ParamStat,
    ResourceSnapshot,
    TuningStats,
)


class TuningStatsAnalyzer:
    def __init__(self, device: DeviceLimits):
        self.device = device

    def analyze(self, space: ParameterSpace, trials: list[TrialRecord]) -> TuningStats:
        complete = [t for t in trials if t.status == "complete" and t.latency_ms]
        failed = [t for t in trials if t.status == "fail"]
        # robust_ms throughout: these curves are what the analyst reads to decide which
        # knob is worth pushing, and a per-choice best driven by one scheduling stall
        # points it at the wrong knob. Same statistic the tuner optimizes.
        best = min(complete, key=lambda t: t.latency_ms.robust_ms) if complete else None

        param_stats = [
            self._param_stat(domain.name, domain.choices, complete, failed)
            for domain in space.domains
        ]
        return TuningStats(
            candidate_id=space.candidate_id,
            space_id=space.space_id,
            n_complete=len(complete),
            n_fail=len(failed),
            best=best,
            param_stats=param_stats,
            resource_at_best=self._resource_snapshot(best),
            failure_clusters=self._failure_clusters(space, trials),
        )

    def _param_stat(self, name: str, choices: list, complete: list[TrialRecord],
                    failed: list[TrialRecord]) -> ParamStat:
        """Per-knob view of one tuning space.

        `latency_by_value` is a MEDIAN per choice, deliberately: latency on a laptop GPU is
        noisy (100-sample baselines show std up to 1.78 ms), and a per-choice minimum rewards
        whichever configuration caught the quietest moment on the card. That is the right
        statistic for the analyst to read as a trend.

        It is the wrong statistic for deciding which way to EXTEND a range, because the
        objective is a minimum: `best_ms` is `min(trials)`, families are compared on their
        best, and the delivered result is one configuration. So `at_boundary` /
        `boundary_direction` are anchored to `best_trial_value` -- the value the fastest trial
        actually used -- while `best_value` keeps reporting the median's pick.

        Measured cost of not doing this (`scripts/audit_boundary_direction_vs_best_trial.py`,
        1126 knobs): the two picks disagree on 44.7%, and on 16.3% the median flagged an edge
        pointing AWAY from the value the winning trial used. Expansions born from those flags
        found a better configuration 1 of 38 times (2.6%) against 12 of 55 (21.8%) for
        well-directed ones, Fisher two-sided p=0.013. Not a relabelling of the known
        min-vs-max yield gap: within `direction=max` alone it is 0/25 against 12/53.

        Concretely, `cand-47371017` (a 9.78 ms run best) reported `GEMM_STAGES best_value=5,
        at_boundary=max` while its winning trial ran `GEMM_STAGES=1`; an expansion would have
        added STAGES=6 and left the winning edge unexplored.

        The change is mostly SUBTRACTIVE, which is the conservative direction. The monotone
        tail test still reads the median curve: the anchor picks which edge to consider, and
        the trend must still agree the curve heads there. Replayed over the same 1126 knobs,
        18.4% of flags are withdrawn (winner on an edge the median slopes away from -- one
        lucky trial against a contrary trend, not worth an expansion) and 3.1% newly raised.
        Expansion is not starved: 78 of the 96 spaces that previously had at least one request
        still have one, with requests per space falling 2.2 -> 1.5 and all now aimed at the
        winner's edge.
        """
        lat_by_value: dict[str, float] = {}
        grouped: dict[str, list[float]] = defaultdict(list)
        for t in complete:
            grouped[repr(t.params.values.get(name))].append(t.latency_ms.robust_ms)
        for key, vals in grouped.items():
            lat_by_value[key] = statistics.median(vals)

        fail_by_value: dict[str, float] = {}
        for choice in choices:
            key = repr(choice)
            n_ok = len(grouped.get(key, []))
            n_bad = sum(1 for t in failed if repr(t.params.values.get(name)) == key)
            total = n_ok + n_bad
            if total:
                fail_by_value[key] = n_bad / total

        best_value = choices[0]
        best_trial_value = None
        at_boundary = False
        boundary_direction = None
        effect_pct = 0.0
        measured = [(c, lat_by_value[repr(c)]) for c in choices if repr(c) in lat_by_value]
        if measured:
            best_value = min(measured, key=lambda kv: kv[1])[0]
            lats = [kv[1] for kv in measured]
            lo, hi = min(lats), max(lats)
            effect_pct = 0.0 if lo <= 0 else (hi - lo) / lo * 100.0

            # The objective's own winner: the value used by the single fastest trial.
            fastest = min(complete, key=lambda t: t.latency_ms.robust_ms)
            candidate_value = fastest.params.values.get(name)
            # Only usable as an anchor if it is one of this domain's measured choices.
            measured_choices = [c for c, _ in measured]
            if candidate_value in measured_choices:
                best_trial_value = candidate_value

            # Boundary = edge of the MEASURED range; choices beyond that either
            # don't exist or consistently fail (the paper's "blocked" case).
            # Anchored on the fastest trial's value (see docstring), falling back to the
            # median's pick when the trial's value is not among the measured choices.
            anchor = best_trial_value if best_trial_value is not None else best_value
            best_m_idx = measured_choices.index(anchor)
            at_min_edge = best_m_idx == 0
            at_max_edge = best_m_idx == len(measured_choices) - 1
            if at_min_edge or at_max_edge:
                full_idx = choices.index(anchor)
                beyond = choices[:full_idx] if at_min_edge else choices[full_idx + 1:]
                beyond_blocked = all(
                    fail_by_value.get(repr(c), 0.0) >= 1.0 or repr(c) not in lat_by_value
                    for c in beyond
                )
                ordered = [lat for _, lat in measured]
                tail = ordered[:3] if at_min_edge else ordered[-3:]
                monotone = (
                    len(tail) >= 2
                    and (all(a <= b for a, b in zip(tail, tail[1:])) if at_min_edge
                         else all(a >= b for a, b in zip(tail, tail[1:])))
                )
                if monotone and (not beyond or beyond_blocked):
                    at_boundary = True
                    boundary_direction = "min" if at_min_edge else "max"

        return ParamStat(
            name=name,
            best_value=best_value,
            best_trial_value=best_trial_value,
            at_boundary=at_boundary,
            boundary_direction=boundary_direction,
            effect_pct=round(effect_pct, 2),
            latency_by_value=lat_by_value,
            failure_rate_by_value=fail_by_value,
        )

    def _resource_snapshot(self, best: TrialRecord | None) -> ResourceSnapshot | None:
        if best is None or best.profile is None:
            return None
        p = best.profile
        return ResourceSnapshot(
            n_regs=p.n_regs,
            regs_frac_of_limit=(
                p.n_regs / self.device.max_regs_per_thread if p.n_regs else None
            ),
            shared_bytes=p.shared_bytes,
            shared_frac_of_limit=(
                p.shared_bytes / self.device.max_shared_bytes_optin
                if p.shared_bytes
                else None
            ),
            n_spills=p.n_spills,
        )

    def _failure_clusters(self, space: ParameterSpace,
                          trials: list[TrialRecord]) -> list[FailureCluster]:
        if not trials:
            return []
        overall_fail = sum(1 for t in trials if t.status == "fail") / len(trials)
        if overall_fail == 0:
            return []
        clusters: list[FailureCluster] = []
        for domain in space.domains:
            for choice in domain.choices:
                key = repr(choice)
                subset = [t for t in trials if repr(t.params.values.get(domain.name)) == key]
                if len(subset) < 2:
                    continue
                fails = [t for t in subset if t.status == "fail"]
                rate = len(fails) / len(subset)
                if rate >= max(2 * overall_fail, 0.5) and rate > overall_fail:
                    kinds = Counter(t.failure_kind for t in fails if t.failure_kind)
                    clusters.append(
                        FailureCluster(
                            param=domain.name,
                            value=key,
                            failure_rate=round(rate, 3),
                            dominant_kind=kinds.most_common(1)[0][0] if kinds else None,
                        )
                    )
        return clusters
