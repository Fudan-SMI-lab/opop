"""TPE tuner tests (no GPU): ask/tell loop over a synthetic objective."""

from kernel_optimizer.models.core import (
    Constraint,
    DeviceLimits,
    LatencyStats,
    ParamDomain,
    ParameterSpace,
    ParamSet,
    TrialRecord,
)
from kernel_optimizer.paramspace.guard import check_config
from kernel_optimizer.tuning.tpe import OptunaTPETuner

SPACE = ParameterSpace(
    space_id="sp", candidate_id="c", source_sha="x",
    domains=[
        ParamDomain(name="A", kind="int", choices=[1, 2, 4, 8]),
        ParamDomain(name="B", kind="int", choices=[16, 32, 64]),
    ],
    constraints=[Constraint(expr="A * B <= 256")],
)
DEVICE = DeviceLimits()


def synthetic_latency(params: ParamSet) -> float:
    # Optimum at A=4, B=64 (within the constraint A*B<=256).
    a, b = params.values["A"], params.values["B"]
    return abs(a - 4) * 2 + abs(b - 64) / 16 + 1.0


def make_record(trial_id: str, params: ParamSet, ms: float) -> TrialRecord:
    return TrialRecord(
        trial_id=trial_id, candidate_id="c", space_id="sp", params=params,
        status="complete",
        latency_ms=LatencyStats(mean=ms, std=0.0, min=ms, max=ms, n_samples=5),
    )


def guard_ok(p: ParamSet) -> bool:
    return check_config(SPACE, p, DEVICE) is None


def test_budget_respected_and_dedup():
    tuner = OptunaTPETuner(SPACE, guard_ok, budget=10, seed=0)
    seen = set()
    count = 0
    while True:
        asked = tuner.ask()
        if asked is None:
            break
        trial_id, params = asked
        assert guard_ok(params), "guard-rejected params leaked through ask()"
        key = params.key()
        assert key not in seen, "duplicate config asked"
        seen.add(key)
        tuner.tell(trial_id, make_record(trial_id, params, synthetic_latency(params)))
        count += 1
    assert count <= 10
    assert tuner.best() is not None


def test_anchor_enqueued_first():
    anchor = ParamSet(values={"A": 2, "B": 32})
    tuner = OptunaTPETuner(SPACE, guard_ok, budget=5, seed=0, anchors=(anchor,))
    trial_id, params = tuner.ask()
    assert params.values == anchor.values
    tuner.tell(trial_id, make_record(trial_id, params, synthetic_latency(params)))


def test_fail_trials_do_not_become_best():
    tuner = OptunaTPETuner(SPACE, guard_ok, budget=6, seed=1)
    while True:
        asked = tuner.ask()
        if asked is None:
            break
        trial_id, params = asked
        record = TrialRecord(
            trial_id=trial_id, candidate_id="c", space_id="sp", params=params,
            status="fail", failure_kind="oom",
        )
        tuner.tell(trial_id, record)
    assert tuner.best() is None


def test_finds_good_region():
    tuner = OptunaTPETuner(SPACE, guard_ok, budget=12, seed=0)
    while True:
        asked = tuner.ask()
        if asked is None:
            break
        trial_id, params = asked
        tuner.tell(trial_id, make_record(trial_id, params, synthetic_latency(params)))
    best = tuner.best()
    assert best is not None
    # 12 trials over a 12-config feasible grid should reach the optimum region.
    assert best.latency_ms.mean <= 3.5


def test_snapshot():
    tuner = OptunaTPETuner(SPACE, guard_ok, budget=3, seed=0)
    trial_id, params = tuner.ask()
    snap = tuner.snapshot()
    assert snap["asked"] == 1 and snap["pending"] == 1 and snap["budget"] == 3
