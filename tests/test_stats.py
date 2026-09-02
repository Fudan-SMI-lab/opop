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
