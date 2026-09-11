"""The batch prescreen gets its OWN deadline, and it must be shorter than `build_timeout_s`.

WHY THIS IS SAFE, which is the whole justification and the only reason this is not a search-space
change. `prescreen_batch` caches **only answers** (the recorded
`caching-a-probe-failure-makes-a-transient-permanent` fix). So a prescreen that times out leaves every
key ABSENT, `cached_shared_verdict` returns `None`, `_shared_memory_ok` returns True for an unscreened
configuration, and the configuration receives a real trial with the FULL `build_timeout_s`. Not one
configuration is removed from the search. Contrast `build_timeout_s` itself, which a real trial's
compile uses: shortening THAT would discard a legitimate slow-compiling candidate, which is why the
user's standing instruction (`never-narrow-the-search-space-to-control-cost`) forbids it.

Measured, on the runs this comes from:
  * box 3 `run-l3-48-20260911-052647`: two batches of 40 sat at 1200.5 s and 1201.0 s -- exactly
    `build_timeout_s` -- and answered NOTHING. 0.67 h of a 12 h budget for zero verdicts.
  * an answering batch on the same box: 490 s for 40 configs, 17 infeasible.
  * the design cost: 48 configurations in 11.02 s, a marginal ~7 ms each.
  * D5 on L3:21: 76-260 s for 40, because the marginal cost is per KERNEL and an L3:21 variant
    compiles several.

Hence `base + per_config * n`: 30 + 3*40 = 150 s at the production batch size, 5x the slowest useful
observation's per-config rate and an eighth of the dead 1200 s tail.

Every test here drives the REAL `_prescreen_timeout_s` and the REAL `prescreen_batch`, substituting
only the worker. A test that recomputed the formula itself would pass on an evaluator that still
passed `build_timeout_s` to `run_job` -- the recorded `a-test-that-copies-the-loop-does-not-test-it`
failure -- so the assertions are on **what timeout the worker was actually handed**.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from kernel_optimizer.config import AppConfig
from kernel_optimizer.evaluation.correctness import CorrectnessEvaluator, prescreen_timeout_s
from kernel_optimizer.models.core import TaskSpec

_SRC = "PARAMS = {'BLOCK_M': 32}\n"


class _Worker:
    """Records the timeout every job was given, and answers whatever `verdict` says."""

    def __init__(self, verdict: dict[str, Any] | None = None) -> None:
        self.timeouts: list[float] = []
        self.tags: list[str] = []
        self.verdict = verdict

    def run_job(self, job, timeout, tag, lock_mode=None):  # noqa: ANN001, ARG002
        self.timeouts.append(float(timeout))
        self.tags.append(tag)
        if self.verdict is None:
            raise TimeoutError("prescreen deadline reached")
        return self.verdict


def _ev(worker: _Worker, **over: float) -> CorrectnessEvaluator:
    cfg = AppConfig()
    for k, v in over.items():
        setattr(cfg.evaluation, k, v)
    return CorrectnessEvaluator(worker, cfg.evaluation, cfg.gpu.concurrency)


def _task() -> TaskSpec:
    return TaskSpec(level=3, problem_id=48, name="48_Mamba2ReturnY",
                    ref_path=Path("ref.py"), ref_src_sha="0" * 64)


def _paths(tmp_path: Path, n: int) -> list[Path]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    out = []
    for i in range(n):
        p = tmp_path / f"s{i:03d}.py"
        p.write_text(_SRC.replace("32", str(32 + i)), encoding="utf-8")
        out.append(p)
    return out


def test_the_prescreen_is_given_its_own_short_deadline_not_build_timeout_s(tmp_path):
    """The defect this fixes: `run_job` was handed `build_timeout_s` (1200 s), so a runaway ptxas
    burned the full real-trial budget for zero verdicts."""
    w = _Worker(verdict={"ok": True, "results": {}})
    ev = _ev(w, build_timeout_s=1200.0, prescreen_base_timeout_s=30.0,
             prescreen_per_config_timeout_s=3.0)

    ev.prescreen_batch(_task(), _paths(tmp_path, 40), tag="t", backend="triton")

    assert len(w.timeouts) == 1, w.timeouts
    assert w.timeouts[0] == pytest.approx(150.0), (
        "30 + 3*40 = 150 s, not build_timeout_s: %r" % w.timeouts)
    assert w.timeouts[0] < 1200.0


def test_the_deadline_scales_with_the_batch_size(tmp_path):
    """`base + per_config * n`, because the marginal cost is per kernel and a multi-kernel candidate
    compiles several per variant (D5: 76-260 s for 40 on L3:21). A single constant would be either
    too tight for a big batch or pointless for a small one."""
    small, big = _Worker(verdict={"ok": True, "results": {}}), _Worker(verdict={"ok": True, "results": {}})
    _ev(small, prescreen_base_timeout_s=30.0, prescreen_per_config_timeout_s=3.0).prescreen_batch(
        _task(), _paths(tmp_path / "small", 4), tag="t", backend="triton")
    _ev(big, prescreen_base_timeout_s=30.0, prescreen_per_config_timeout_s=3.0).prescreen_batch(
        _task(), _paths(tmp_path / "big", 40), tag="t", backend="triton")

    assert small.timeouts[0] == pytest.approx(42.0), small.timeouts   # 30 + 3*4
    assert big.timeouts[0] == pytest.approx(150.0), big.timeouts      # 30 + 3*40
    assert big.timeouts[0] > small.timeouts[0]


def test_the_prescreen_deadline_can_never_exceed_build_timeout_s(tmp_path):
    """The clamp. A mis-set config must not make the SCREEN the longer deadline -- that would restore
    exactly the tail this fixes, and worse, silently."""
    w = _Worker(verdict={"ok": True, "results": {}})
    ev = _ev(w, build_timeout_s=60.0, prescreen_base_timeout_s=30.0,
             prescreen_per_config_timeout_s=3.0)

    ev.prescreen_batch(_task(), _paths(tmp_path, 40), tag="t", backend="triton")

    assert w.timeouts[0] == pytest.approx(60.0), (
        "150 s must be clamped to build_timeout_s=60: %r" % w.timeouts)


def test_a_timed_out_prescreen_caches_NOTHING_so_no_configuration_is_excluded(tmp_path):
    """The safety argument, executed rather than asserted in prose. This is what makes a short
    prescreen deadline a cost cap instead of a search-space restriction."""
    w = _Worker(verdict=None)                      # the worker raises, as on a real timeout
    ev = _ev(w, prescreen_base_timeout_s=30.0, prescreen_per_config_timeout_s=3.0)
    paths = _paths(tmp_path, 5)

    ev.prescreen_batch(_task(), paths, tag="t", backend="triton")   # must not raise

    for p in paths:
        assert ev.cached_shared_verdict(p.read_text(encoding="utf-8"), "triton", 101376) is None, (
            "an unanswered configuration must read UNKNOWN, so the guard lets it through to a real "
            "trial: %s" % p.name)


def test_a_shorter_deadline_does_not_touch_a_real_trials_compile_budget(tmp_path):
    """`build_timeout_s` stays what it was. If this ever regresses, a legitimate candidate whose
    ptxas needs ten minutes is discarded -- the failure mode the user's instruction forbids."""
    w = _Worker(verdict={"ok": True, "results": {}})
    cfg = AppConfig()
    cfg.evaluation.build_timeout_s = 1200.0
    ev = CorrectnessEvaluator(w, cfg.evaluation, cfg.gpu.concurrency)

    ev.prescreen_batch(_task(), _paths(tmp_path, 40), tag="t", backend="triton")

    assert ev.cfg.build_timeout_s == 1200.0, "the real-trial compile budget must be untouched"
    assert w.timeouts[0] < ev.cfg.build_timeout_s


def test_the_event_reports_the_deadline_it_was_actually_judged_against(tmp_path, monkeypatch):
    """`timed_out` compared `elapsed` against `build_timeout_s`. With the screen on its own budget
    that test can never fire: a screen cut off at 150 s would report `timed_out: false`, which is the
    silent direction. The orchestrator must read the prescreen's own limit.

    Drives the real `_prescreen_space` so the field comes from production, not from this test.
    """
    pytest.importorskip("optuna")   # the orchestrator imports the tuner; A800-only, as elsewhere
    from kernel_optimizer.control import orchestrator as orch_mod
    from kernel_optimizer.control.orchestrator import CandidateRun
    from kernel_optimizer.models.core import Candidate, ParamDomain, ParameterSpace

    class _Store:
        def __init__(self) -> None:
            self.events: list[tuple[str, dict]] = []

        def append(self, t: str, p: dict) -> None:
            self.events.append((t, p))

    class _Ev:
        """A stub with only the two methods this path calls. It deliberately does NOT carry a
        timeout helper: the orchestrator must reach the shared `prescreen_timeout_s`, so a stub
        evaluator cannot silently supply a different number."""

        def prescreen_batch(self, task, paths, tag, backend) -> None:  # noqa: ANN001, ARG002
            return None

        def cached_shared_verdict(self, src, backend, cap):  # noqa: ANN001, ARG002
            return None                                     # nothing answered: a timeout's shape

    cfg = AppConfig()
    cfg.evaluation.build_timeout_s = 1200.0
    cfg.budgets.trials_per_space = 16
    cfg.gpu.compile_screen_enabled = True

    calls = {"n": 0}
    monkeypatch.setattr(orch_mod.time, "time",
                        lambda: 1000.0 if (calls.__setitem__("n", calls["n"] + 1) or calls["n"] == 1)
                        else 1000.0 + 140.0)

    ev = _Ev()
    store = _Store()
    o = object.__new__(orch_mod.Orchestrator)
    o.cfg = cfg
    o.store = store
    o.deps = type("D", (), {"evaluator": ev})()
    o.task = _task()

    cand = Candidate(candidate_id="cand-1", family_id="fam-1", origin="seed", backend="triton",
                     source_sha="0" * 64, structural_signature="s", approach_summary="a")
    space = ParameterSpace(space_id="sp-1", candidate_id="cand-1", version=1, source_sha="0" * 64,
                           domains=[ParamDomain(name="BLOCK_M", kind="int", choices=[32, 64])],
                           constraints=[])
    crun = object.__new__(CandidateRun)
    crun.candidate, crun.space = cand, space
    crun.source = ("PARAMS = {\"BLOCK_M\": 32}\n\nimport torch\n\n\nclass ModelNew(torch.nn.Module):\n"
                   "    def forward(self, x):\n        return x * PARAMS[\"BLOCK_M\"]\n")
    d = tmp_path / "trials"
    d.mkdir(exist_ok=True)
    o._prescreen_space(crun, d)

    got = [p for t, p in store.events if t == "SPACE_PRESCREENED"]
    assert len(got) == 1, store.events
    p = got[0]
    assert p["timeout_s"] == pytest.approx(36.0), (
        "2 configs => 30 + 3*2 = 36 s, and the event must say so rather than 1200: %r" % p)
    assert p["answered"] == 0
    assert p["timed_out"] is True, (
        "140 s elapsed against a 36 s prescreen deadline with no answers IS a timeout; comparing "
        "against build_timeout_s=1200 would report false: %r" % p)

def test_a_config_missing_the_prescreen_fields_fails_LOUDLY_not_as_a_probe_failure(tmp_path):
    """The defect the first version of this change actually had.

    `prescreen_batch` wraps `run_job` in `except Exception` -- correctly, because a screen must never
    be a verdict. The first version computed the deadline INSIDE that block, so a config object
    without the two new fields raised `AttributeError`, was swallowed into `{"ok": False}`, and read
    as a PROBE failure. A mis-wired config would then have been byte-indistinguishable from a
    timing-out `ptxas` on every batch, forever, with no event saying so.

    Found by `test_a_failed_probe_is_never_cached_as_a_screen_verdict`, whose stub config carries
    only the fields the screen genuinely needs -- and which therefore failed the moment the screen
    started needing two more. That is a guard earning its keep rather than a nuisance, so the fix
    moved the computation out of the try and this test pins it there.

    A config error is a programming error and must reach the operator. A probe failure is a fact
    about one attempt and must be swallowed. The two must not share an exit.
    """
    class _Partial:
        """Everything `prescreen_batch` needs EXCEPT the prescreen deadline fields."""
        precision = "fp32"
        build_timeout_s = 60.0
        eval_timeout_s = 60.0
        correctness_mode = "dual_witness_relaxed"

    w = _Worker(verdict={"ok": True, "results": {}})
    ev = CorrectnessEvaluator.__new__(CorrectnessEvaluator)
    ev.worker, ev.cfg, ev.seed = w, _Partial(), 0
    ev._screen_cache = {}
    ev._static_cache = {}

    with pytest.raises(AttributeError):
        ev.prescreen_batch(_task(), _paths(tmp_path, 3), tag="t", backend="triton")

    assert w.timeouts == [], "the worker must not have been called with a bogus deadline"
    assert not ev._screen_cache, "and nothing may be cached from a config error"


def test_the_shared_helper_is_what_both_call_sites_use():
    """`prescreen_timeout_s` is module-level so the worker's deadline and the event's `timeout_s`
    cannot drift. A per-class copy on either side would let them disagree silently -- and the
    disagreement is invisible, because both numbers look plausible on their own."""
    cfg = AppConfig().evaluation
    cfg.build_timeout_s = 1200.0
    cfg.prescreen_base_timeout_s = 30.0
    cfg.prescreen_per_config_timeout_s = 3.0

    assert prescreen_timeout_s(cfg, 0) == pytest.approx(30.0)
    assert prescreen_timeout_s(cfg, 40) == pytest.approx(150.0)
    assert prescreen_timeout_s(cfg, -5) == pytest.approx(30.0), "a negative count must not shorten it"
    cfg.build_timeout_s = 60.0
    assert prescreen_timeout_s(cfg, 40) == pytest.approx(60.0), "the clamp"
