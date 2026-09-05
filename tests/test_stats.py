"""TuningStatsAnalyzer tests: boundary detection, failure clusters, resources."""

from kernel_optimizer.models.core import (
    DeviceLimits,
    LatencyStats,
    ParamDomain,
    ParameterSpace,
    ParamSet,
    ProfileRecord,
    TrialRecord,
)
from kernel_optimizer.tuning.stats import TuningStatsAnalyzer

DEVICE = DeviceLimits()

SPACE = ParameterSpace(
    space_id="sp", candidate_id="c", source_sha="x",
    domains=[
        ParamDomain(name="TILE", kind="int", choices=[64, 128, 256, 512]),
        ParamDomain(name="WARPS", kind="int", choices=[2, 4, 8]),
    ],
)


def trial(tid, tile, warps, ms=None, kind=None, regs=None, shared=None):
    if ms is not None:
        return TrialRecord(
            trial_id=tid, candidate_id="c", space_id="sp",
            params=ParamSet(values={"TILE": tile, "WARPS": warps}),
            status="complete",
            latency_ms=LatencyStats(mean=ms, std=0.01, min=ms, max=ms, n_samples=20),
            profile=ProfileRecord(n_regs=regs, shared_bytes=shared, n_spills=0),
        )
    return TrialRecord(
        trial_id=tid, candidate_id="c", space_id="sp",
        params=ParamSet(values={"TILE": tile, "WARPS": warps}),
        status="fail", failure_kind=kind or "oom",
    )


def test_boundary_detection_monotone_toward_max():
    # Latency strictly improves with TILE; 512 always fails -> best measured at 256 edge?
    trials = [
        trial("t1", 64, 4, ms=10.0),
        trial("t2", 128, 4, ms=8.0),
        trial("t3", 256, 4, ms=6.0),
        trial("t4", 512, 4, kind="oom"),
        trial("t5", 512, 8, kind="oom"),
    ]
    stats = TuningStatsAnalyzer(DEVICE).analyze(SPACE, trials)
    tile_stat = next(s for s in stats.param_stats if s.name == "TILE")
    # 512 has no complete measurement; the measured optimum 256 is the max of the
    # measured range and the trend is monotone downward (toward max).
    assert tile_stat.best_value == 256
    assert tile_stat.at_boundary and tile_stat.boundary_direction == "max"
    assert tile_stat.failure_rate_by_value.get("512") == 1.0


def test_no_boundary_when_interior_best():
    trials = [
        trial("t1", 64, 4, ms=8.0),
        trial("t2", 128, 4, ms=5.0),
        trial("t3", 256, 4, ms=7.0),
        trial("t4", 512, 4, ms=9.0),
    ]
    stats = TuningStatsAnalyzer(DEVICE).analyze(SPACE, trials)
    tile_stat = next(s for s in stats.param_stats if s.name == "TILE")
    assert tile_stat.best_value == 128
    assert not tile_stat.at_boundary


def test_failure_clusters():
    trials = [
        trial("t1", 64, 2, ms=10.0),
        trial("t2", 128, 2, ms=9.0),
        trial("t3", 512, 2, kind="oom"),
        trial("t4", 512, 4, kind="oom"),
        trial("t5", 512, 8, kind="oom"),
        trial("t6", 256, 4, ms=8.0),
    ]
    stats = TuningStatsAnalyzer(DEVICE).analyze(SPACE, trials)
    clusters = {(c.param, c.value): c for c in stats.failure_clusters}
    assert ("TILE", "512") in clusters
    assert clusters[("TILE", "512")].dominant_kind == "oom"


def test_resource_snapshot_fractions():
    trials = [trial("t1", 128, 4, ms=5.0, regs=200, shared=90000)]
    stats = TuningStatsAnalyzer(DEVICE).analyze(SPACE, trials)
    snap = stats.resource_at_best
    assert snap is not None
    assert snap.n_regs == 200
    assert 0.7 < snap.regs_frac_of_limit < 0.8
    assert snap.shared_bytes == 90000
    assert snap.shared_frac_of_limit is not None and snap.shared_frac_of_limit > 0.8


def test_empty_trials():
    stats = TuningStatsAnalyzer(DEVICE).analyze(SPACE, [])
    assert stats.n_complete == 0 and stats.best is None
    assert stats.failure_clusters == []


def test_boundary_follows_the_fastest_trial_not_the_median():
    """The knob's flagged edge must be the one the winning configuration sits on.

    `latency_by_value` is a per-choice MEDIAN (right for a noisy laptop GPU), but the
    objective is a minimum -- `best_ms` is `min(trials)`. When the two disagree, the median
    can flag an edge on the OPPOSITE side from the value that actually won, and improvement K
    then spends its expansion extending away from it.

    Shaped after the real case: `cand-47371017` (a 9.78 ms run best) reported
    `GEMM_STAGES best_value=5, at_boundary=max` with its winning trial at `GEMM_STAGES=1`.
    Here the median's argmin is TILE=512 (the max edge) while the single fastest trial ran
    TILE=64, so the pre-change rule would have pointed `max` and the fix must point `min`.
    """
    trials = [
        trial("w", 64, 4, ms=6.0),     # the fastest trial in the space
        trial("a", 64, 8, ms=10.0),    # ... but 64's MEDIAN (8.0) is not the smallest
        trial("b", 128, 4, ms=9.0),
        trial("c", 256, 4, ms=10.0),
        trial("d", 512, 4, ms=7.5),    # the median's argmin sits on the opposite edge
    ]
    tile = next(s for s in TuningStatsAnalyzer(DEVICE).analyze(SPACE, trials).param_stats
                if s.name == "TILE")
    assert tile.best_value == 512, "the median table itself is unchanged"
    assert tile.best_trial_value == 64, "the winning trial's value is reported alongside"
    assert tile.at_boundary and tile.boundary_direction == "min", \
        "the flagged edge must be the winner's edge, not the median's"


def test_a_winner_the_median_trend_contradicts_is_withdrawn_not_flipped():
    """When anchor and trend disagree, the flag is WITHDRAWN rather than re-aimed.

    The monotone-tail test still reads the median curve, on purpose: the anchor decides which
    edge to consider, and the trend must still agree the curve heads there. A winner on an
    edge the median curve slopes away from is the ambiguous case -- one lucky trial against a
    contrary trend -- and spending a scarce expansion on it is not justified.

    This is why replaying 1126 real knobs is mostly subtractive (18.4% of flags withdrawn,
    3.1% newly raised) rather than a wholesale re-aiming, and it is the behaviour that keeps
    the change conservative: a withdrawn flag costs an expansion opportunity, a wrongly-aimed
    one wastes the expansion itself (measured yield 2.6% vs 21.8%).
    """
    trials = [
        trial("w", 64, 4, ms=1.0),     # fastest trial, on the min edge
        trial("a", 64, 8, ms=30.0),    # but 64's median (15.5) is the WORST
        trial("b", 128, 4, ms=9.0),
        trial("c", 256, 4, ms=8.0),
        trial("d", 512, 4, ms=7.0),    # ... and the curve slopes toward max throughout
    ]
    tile = next(s for s in TuningStatsAnalyzer(DEVICE).analyze(SPACE, trials).param_stats
                if s.name == "TILE")
    assert tile.best_trial_value == 64
    assert not tile.at_boundary and tile.boundary_direction is None


def test_median_and_winner_agreeing_leaves_the_verdict_untouched():
    """The common case must be byte-identical to the pre-change behaviour.

    Replayed over all 1126 real knobs on disk the new rule leaves 77.6% unchanged; this
    pins the shape of that majority so the anchor swap cannot quietly alter it.
    """
    trials = [
        trial("t1", 64, 4, ms=10.0),
        trial("t2", 128, 4, ms=8.0),
        trial("t3", 256, 4, ms=6.0),
        trial("t4", 512, 4, kind="oom"),
    ]
    tile = next(s for s in TuningStatsAnalyzer(DEVICE).analyze(SPACE, trials).param_stats
                if s.name == "TILE")
    assert tile.best_value == tile.best_trial_value == 256
    assert tile.at_boundary and tile.boundary_direction == "max"


def test_boundary_falls_back_to_the_median_when_the_winner_is_unmeasurable():
    """A knob with no completed trial must not crash or invent an anchor.

    `best_trial_value` is None when nothing completed, and the boundary rule then has no
    winner to anchor on -- it must fall back to the median's pick rather than raise.
    """
    stats = TuningStatsAnalyzer(DEVICE).analyze(
        SPACE, [trial("f1", 64, 4, kind="oom"), trial("f2", 128, 4, kind="oom")])
    tile = next(s for s in stats.param_stats if s.name == "TILE")
    assert tile.best_trial_value is None
    assert not tile.at_boundary  # nothing measured => nothing to flag
    assert stats.best is None
