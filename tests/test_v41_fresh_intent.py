"""v4.1 §7 wiring: fresh-measurement intent through tpe.py, and the bridge's isolation.

THE test this file exists for: `test_fresh_anchor_is_really_remeasured` — the v3 review's
E1 implementation blocker said the shipping dedup + measured-cache would either drop a
scan anchor's re-measurement or fake it with a cache copy (a fake A/A zero). Its positive
control (`test_positive_control_ordinary_duplicate_is_deduped`) proves the assertion bites:
the SAME flow without fresh intent is deduped, so a broken fresh path cannot pass both.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from kernel_optimizer.config import AppConfig, load_config
from kernel_optimizer.models.core import (
    LatencyStats,
    ParamDomain,
    ParameterSpace,
    ParamSet,
    TrialRecord,
)
from kernel_optimizer.tuning.tpe import OptunaTPETuner


def _space() -> ParameterSpace:
    return ParameterSpace(
        space_id="sp-fresh", candidate_id="cand-fresh", version=1, source_sha="beef",
        domains=[ParamDomain(name="A", kind="int", choices=[1, 2, 3, 4]),
                 ParamDomain(name="B", kind="int", choices=[10, 20])],
        constraints=[])


def _tuner(budget: int = 20) -> OptunaTPETuner:
    return OptunaTPETuner(_space(), guard_ok=lambda p: True, budget=budget, seed=7)


def _record(trial_id: str, params: ParamSet, ms: float) -> TrialRecord:
    return TrialRecord(
        trial_id=trial_id, candidate_id="cand-fresh", space_id="sp-fresh",
        params=params, status="complete",
        latency_ms=LatencyStats(mean=ms, median=ms, std=0.0, min=ms, max=ms,
                                n_samples=20))


class TestFreshIntent:
    def test_fresh_anchor_is_really_remeasured(self):
        """A key that was already drawn and told comes back from ask() again when (and
        only when) it carries fresh intent — and is flagged by is_fresh."""
        t = _tuner()
        # draw + tell the point once, the ordinary way
        first = t.ask()
        assert first is not None
        tid, params = first
        t.tell(tid, _record(tid, params, 1.0))
        # now enqueue the SAME params fresh
        assert t.enqueue_fresh(params) is None
        nxt = t.ask()
        assert nxt is not None
        tid2, params2 = nxt
        assert params2.key() == params.key()      # the same configuration again
        assert t.is_fresh(tid2) is True           # flagged: measured cache must be bypassed
        assert tid2 != tid
        t.tell(tid2, _record(tid2, params2, 1.1))

    def test_positive_control_ordinary_duplicate_is_deduped(self):
        """WITHOUT fresh intent the same enqueue comes back as a different point (the
        dedup branch prunes the duplicate) — proves the fresh path above is load-bearing."""
        t = _tuner()
        first = t.ask()
        tid, params = first
        t.tell(tid, _record(tid, params, 1.0))
        assert t.enqueue(params) == "already_drawn"     # the legacy path refuses outright
        nxt = t.ask()
        if nxt is not None:
            _, params2 = nxt
            assert params2.key() != params.key()

    def test_c4_double_endpoint_survives_dedup_twice(self):
        """A C4 measures the same endpoint config twice (discovery + validation): two
        fresh units, two draws of the same key, both flagged."""
        t = _tuner()
        params = ParamSet(values={"A": 3, "B": 20})
        assert t.enqueue_fresh(params) is None
        assert t.enqueue_fresh(params) is None
        seen_fresh = 0
        for _ in range(2):
            got = t.ask()
            assert got is not None
            tid, p = got
            if p.key() == params.key():
                assert t.is_fresh(tid)
                seen_fresh += 1
            t.tell(tid, _record(tid, p, 2.0))
        assert seen_fresh == 2

    def test_fresh_intent_is_consumed_not_permanent(self):
        """After the fresh draws are spent, the key dedupes as before — fresh intent is
        a bounded pass, not a permanent hole in the dedup."""
        t = _tuner()
        params = ParamSet(values={"A": 1, "B": 10})
        assert t.enqueue_fresh(params) is None
        got = t.ask()
        tid, p = got
        assert p.key() == params.key() and t.is_fresh(tid)
        t.tell(tid, _record(tid, p, 1.0))
        assert t.enqueue(params) == "already_drawn"

    def test_guard_still_decides_fresh_points(self):
        t = OptunaTPETuner(_space(), guard_ok=lambda p: p.values.get("A") != 4,
                           budget=20, seed=7)
        assert t.enqueue_fresh(ParamSet(values={"A": 4, "B": 10})) == "guard_rejected"


class TestConfigExclusivity:
    def test_active_plus_slope_guide_is_refused(self, tmp_path: Path):
        cfg = tmp_path / "bad.yaml"
        cfg.write_text(
            "v4:\n  conditional_scan:\n    mode: active\n"
            "v3:\n  slope_guide:\n    enabled: true\n", encoding="utf-8")
        with pytest.raises(ValueError, match="mutually exclusive"):
            load_config(cfg)

    def test_observe_plus_slope_guide_is_allowed(self, tmp_path: Path):
        cfg = tmp_path / "ok.yaml"
        cfg.write_text(
            "v4:\n  conditional_scan:\n    mode: observe\n"
            "v3:\n  slope_guide:\n    enabled: true\n", encoding="utf-8")
        loaded = load_config(cfg)
        assert loaded.v4.conditional_scan.mode == "observe"

    def test_default_mode_is_off(self):
        assert AppConfig().v4.conditional_scan.mode == "off"
