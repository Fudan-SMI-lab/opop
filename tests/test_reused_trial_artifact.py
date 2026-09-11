"""A reused measurement must leave the same artefact a fresh trial leaves.

Found by measurement. Three of box 2's fourteen `BOTTLENECK_CLASSIFIED` events carried
`cpu_issue_ms: None` with no `LAUNCH_OVERHEAD_FAILED` beside them -- so the classification ran without
the launch-overhead probe and nothing said why. The cause is in `_tune`: when a TPE ask lands on an
already-measured param set, the record is copied with a NEW `trial_id` and `_run_trial` is skipped
entirely -- and `_run_trial` is the only writer of `trials/<trial_id>.py`. So the file for that id never
existed.

`_measure_best_overhead` resolves the winning trial id back to source. When the winner is a reused
record it finds nothing and returns SILENTLY, which makes the whole `launch_bound` branch unreachable
for that candidate with nothing in the log to distinguish it from a kernel that simply is not launch
bound. Measured across five runs: **128 of 128 reused records had no file**, costing 3 of 11
classifications on the control arm and 3 of 14 on the treatment arm -- both arms at similar rates, so
not an arm-parity defect, but a fifth of every run's launch diagnostics.

These tests drive the REAL `_tune` loop and the REAL `_measure_best_overhead`, substituting only the
evaluator, tuner and store. A test asserting on the source text of the reuse branch would pass on an
orchestrator that writes the file to the wrong place, and the pre-existing
`test_reused_measurement_is_journalled_with_flag` is exactly that shape -- the recorded
`source-text-assertions-can-encode-the-bug` failure. So the assertions here are: does the file exist,
and does the probe reach the winner.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("optuna")

from kernel_optimizer.config import AppConfig  # noqa: E402
from kernel_optimizer.models.core import (  # noqa: E402
    Candidate,
    LatencyStats,
    ParamDomain,
    ParameterSpace,
    ParamSet,
    TaskSpec,
    TrialRecord,
)

_SOURCE = '''
PARAMS = {"BLOCK_M": 32}

import torch


class ModelNew(torch.nn.Module):
    def forward(self, x):
        return x * PARAMS["BLOCK_M"]
'''


class _Store:
    def __init__(self, run_dir: Path) -> None:
        self.events: list[tuple[str, dict]] = []
        self.run_dir = run_dir

    def append(self, type_: str, payload: dict) -> None:
        self.events.append((type_, payload))

    def of(self, type_: str) -> list[dict]:
        return [p for t, p in self.events if t == type_]

    def candidate_dir(self, candidate_id: str) -> Path:
        d = self.run_dir / "candidates" / candidate_id
        d.mkdir(parents=True, exist_ok=True)
        return d

    def replay(self):
        return type("S", (), {"steps_done": set(), "trials": {}})()


def _lat(ms: float) -> LatencyStats:
    return LatencyStats(mean=ms, median=ms, std=0.0, min=ms, max=ms, n_samples=20,
                        samples=[ms] * 20)


def _orch(store: _Store, tmp_path: Path):
    """A real Orchestrator carrying only the collaborators these paths touch."""
    from kernel_optimizer.control import orchestrator as orch_mod

    cfg = AppConfig()
    cfg.budgets.trials_per_space = 4
    cfg.gpu.compile_screen_enabled = False
    o = object.__new__(orch_mod.Orchestrator)
    o.cfg = cfg
    o.store = store
    o.deweight_ledger = None
    o.calibration = None
    # `TaskSpec` has NO `task_id` and REQUIRES `ref_src_sha` -- copied from the model, not guessed.
    o.task = TaskSpec(level=3, problem_id=43, name="43_MinGPTCausalAttention",
                      ref_path=tmp_path / "ref.py", ref_src_sha="0" * 64)
    return o


def _crun(source: str = _SOURCE):
    from kernel_optimizer.control.orchestrator import CandidateRun

    cand = Candidate(candidate_id="cand-1", family_id="fam-1", origin="seed", backend="triton",
                     source_sha="0" * 64, structural_signature="s", approach_summary="a")
    space = ParameterSpace(
        space_id="sp-1", candidate_id="cand-1", version=1, source_sha="0" * 64,
        domains=[ParamDomain(name="BLOCK_M", kind="int", choices=[32, 64])],
        constraints=[],
    )
    crun = object.__new__(CandidateRun)
    crun.candidate = cand
    crun.space = space
    crun.source = source
    crun.trials = []
    crun.best_ms = None
    crun.overhead = None
    return crun


# --------------------------------------------------------------------------------------------------
# The defect: a reused record's trial_id names a file that was never written
# --------------------------------------------------------------------------------------------------


def test_a_reused_measurement_writes_the_same_py_a_fresh_trial_writes(tmp_path, monkeypatch):
    """Drives the real reuse branch. The record is copied with a new `trial_id`, so unless the branch
    writes the file itself, `trials/<new_id>.py` does not exist -- which is what 128 of 128 reused
    records across five runs looked like."""
    from kernel_optimizer.control import orchestrator as orch_mod

    store = _Store(tmp_path / "run")
    o = _orch(store, tmp_path)
    crun = _crun()
    # `_tune` derives trials_dir itself as `store.candidate_dir(cid)/"trials"` -- read from the real
    # signature `(crun, anchors, measured_cache)` rather than passed in, so the test cannot decide
    # where the production code writes.
    trials_dir = store.candidate_dir("cand-1") / "trials"

    params = ParamSet(values={"BLOCK_M": 32})
    cached = TrialRecord(trial_id="tr-original", candidate_id="cand-1", space_id="sp-0",
                         params=params, status="complete", latency_ms=_lat(3.0))

    # A tuner that asks for the cached param set once and then stops, so the ONLY path exercised is
    # the reuse branch. Substituting the tuner (not reimplementing the branch) keeps the production
    # code in charge of what it writes.
    asks = [("tr-reused", params)]

    class _Tuner:
        def ask(self):
            return asks.pop(0) if asks else None

        def tell(self, trial_id, record):  # noqa: ANN001, ARG002
            pass

        def best(self):
            return cached

        def snapshot(self):
            return {}

    monkeypatch.setattr(orch_mod, "OptunaTPETuner", lambda **kw: _Tuner())
    o.deps = type("D", (), {"evaluator": None, "profiler": None,
                            "families": type("F", (), {
                                "update_best": staticmethod(lambda *a, **k: False)})()})()
    o._tune(crun, (), {params.key(): cached})

    reused = store.of("TRIAL_DONE")
    assert reused and reused[0].get("reused_measurement") is True, \
        "the reuse branch must be the one that ran"
    new_id = reused[0]["trial"]["trial_id"]
    assert new_id == "tr-reused", new_id
    written = trials_dir / ("%s.py" % new_id)
    assert written.exists(), (
        "a reused measurement left no source for its own trial_id -- everything that resolves a "
        "trial id back to source then finds nothing")
    # The SAME bytes a fresh trial would write, not a reconstruction: `_run_trial` writes
    # `materialize(crun.source, params)`.
    from kernel_optimizer.paramspace import materializer
    assert written.read_text(encoding="utf-8") == materializer.materialize(_SOURCE, params)


def test_a_failed_artifact_write_is_journalled_not_swallowed(tmp_path, monkeypatch):
    """A missing artefact is exactly what went unnoticed for five runs, so the write failing must be an
    event rather than a silent pass. It must also not end the run: the measurement is already good."""
    from kernel_optimizer.control import orchestrator as orch_mod

    store = _Store(tmp_path / "run")
    o = _orch(store, tmp_path)
    # A source with no PARAMS literal cannot be materialized, so the write raises inside the branch.
    crun = _crun(source="import torch\n\n\nclass ModelNew(torch.nn.Module):\n    pass\n")
    trials_dir = tmp_path / "trials"
    trials_dir.mkdir()

    params = ParamSet(values={"BLOCK_M": 32})
    cached = TrialRecord(trial_id="tr-original", candidate_id="cand-1", space_id="sp-0",
                         params=params, status="complete", latency_ms=_lat(3.0))
    asks = [("tr-reused", params)]

    class _Tuner:
        def ask(self):
            return asks.pop(0) if asks else None

        def tell(self, trial_id, record):  # noqa: ANN001, ARG002
            pass

        def best(self):
            return cached

        def snapshot(self):
            return {}

    monkeypatch.setattr(orch_mod, "OptunaTPETuner", lambda **kw: _Tuner())
    o.deps = type("D", (), {"evaluator": None, "profiler": None,
                            "families": type("F", (), {
                                "update_best": staticmethod(lambda *a, **k: False)})()})()
    o._tune(crun, (), {params.key(): cached})

    assert store.of("TRIAL_ARTIFACT_FAILED"), \
        "the write failed and nothing recorded it -- the same silence that hid the original defect"
    assert store.of("TRIAL_DONE"), "the measurement is still good; the run must continue"


# --------------------------------------------------------------------------------------------------
# The consequence: the silent return that hid it
# --------------------------------------------------------------------------------------------------


def test_no_source_for_the_winning_trial_is_recorded_not_silent(tmp_path):
    """`_measure_best_overhead` returned silently when the winner had no file, so an UNMEASURED
    verdict was indistinguishable in the log from a measured negative one. That is the difference
    between "this kernel is not launch bound" and "we never looked"."""
    store = _Store(tmp_path / "run")
    o = _orch(store, tmp_path)
    crun = _crun()
    # A complete winner whose `.py` is deliberately absent -- the production state of every candidate
    # whose best trial was a reused measurement.
    crun.trials = [TrialRecord(trial_id="tr-missing", candidate_id="cand-1", space_id="sp-1",
                               params=ParamSet(values={"BLOCK_M": 32}), status="complete",
                               latency_ms=_lat(3.0))]
    o.deps = type("D", (), {"evaluator": None})()

    o._measure_best_overhead(crun)

    failed = store.of("LAUNCH_OVERHEAD_FAILED")
    assert failed, "a missing source must be recorded, or `launch_bound` silently never fires"
    assert failed[0]["trial_id"] == "tr-missing"
    assert "launch_bound" in failed[0]["error"], \
        "the message must say what became unreachable, not merely that a file was absent"
    assert not store.of("LAUNCH_OVERHEAD_MEASURED")


def test_a_winner_with_no_complete_trial_is_still_silent(tmp_path):
    """The other direction. Not every early return should be an event: a candidate with nothing
    correct has no winner to measure, and that is normal, not a defect. Firing here would put a
    FAILED event on every candidate whose trials all failed."""
    store = _Store(tmp_path / "run")
    o = _orch(store, tmp_path)
    crun = _crun()
    crun.trials = [TrialRecord(trial_id="tr-bad", candidate_id="cand-1", space_id="sp-1",
                               params=ParamSet(values={"BLOCK_M": 32}), status="fail",
                               failure_kind="correctness_mismatch")]
    o.deps = type("D", (), {"evaluator": None})()

    o._measure_best_overhead(crun)

    assert not store.of("LAUNCH_OVERHEAD_FAILED"), \
        "no complete trial is not a missing artefact; flagging it would make the event meaningless"
    assert not store.of("LAUNCH_OVERHEAD_MEASURED")
