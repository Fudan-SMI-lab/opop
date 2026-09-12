"""D9. A single-configuration compile screen gets its OWN deadline, not a real trial's.

THE DEFECT. `compile_screen` handed `run_job` the full `build_timeout_s` (1200 s) while its
sibling `prescreen_batch`, 88 lines away in the same file, used `prescreen_timeout_s`. That
sibling was created by `59d5a71` along with 30 lines arguing that a screen's deadline must be
separate from a real trial's compile budget -- and the argument was never carried across. So the
generalizable statement of the defect is **a fix applied to one caller leaves its siblings**, and
the guard for it here is a grep test over the whole module, not an assertion about one function.

WHAT IT COST, measured on the three finished runs (1243 single-configuration screens):

    p50 11.4-13.6 s · p90 26.5-28.5 s · p99 38.7-105.0 s   vs a 1200 s deadline = 11x its own p99

A hung screen is followed by a real trial paying its own 1800 s (`build_timeout_s +
eval_timeout_s`), which is where box 1's four 50.1-minute gaps between consecutive `TRIAL_DONE`
came from: 1200 + 1800 = 3000 s. Counted as time past a 120 s cap -- because a screen that
ANSWERS at 1202 s costs the same 20 minutes as one that hangs -- box 1 loses ~2.1 h of a 12 h
budget (17-20%), box 2 loses 0.02 h and box 3 0.01 h.

WHY 120 s AND NOT 33 s. `prescreen_timeout_s(cfg, 1)` = 33 s, which is only 4.5-6.5 s above the
measured p90: the batch budget scales per configuration and is not calibrated for n=1. Hence
`max(batch_at_1, floor)`. 120 s still answers 99.0 / 99.6 / 99.7% of the screens that would have
answered.

WHY THIS CANNOT DISCARD A CANDIDATE, which is the whole safety argument and is EXECUTED below
rather than asserted in prose: a screen caches only ANSWERS, so a timeout leaves the key absent,
`cached_shared_verdict` returns `None` (three-valued on purpose), and the configuration gets a
full real trial with the untouched `build_timeout_s`.

Every test drives the REAL `compile_screen` and asserts on **what deadline the worker was handed**.
A test that recomputed the formula would pass on an evaluator still passing `build_timeout_s` --
the recorded `a-test-that-copies-the-loop-does-not-test-it` failure.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest

from kernel_optimizer.config import AppConfig
from kernel_optimizer.evaluation import correctness as corr_mod
from kernel_optimizer.evaluation.correctness import (
    CorrectnessEvaluator,
    prescreen_timeout_s,
    screen_timeout_s,
)
from kernel_optimizer.models.core import TaskSpec

_SRC = "PARAMS = {'BLOCK_M': 32}\n"


class _Worker:
    def __init__(self, verdict: dict[str, Any] | None = None) -> None:
        self.timeouts: list[float] = []
        self.tags: list[str] = []
        self.verdict = verdict

    def run_job(self, job, timeout, tag, lock_mode=None):  # noqa: ANN001, ARG002
        self.timeouts.append(float(timeout))
        self.tags.append(tag)
        if self.verdict is None:
            raise TimeoutError("screen deadline reached")
        return self.verdict


def _ev(worker: _Worker, **over: float) -> CorrectnessEvaluator:
    cfg = AppConfig()
    for k, v in over.items():
        setattr(cfg.evaluation, k, v)
    return CorrectnessEvaluator(worker, cfg.evaluation, cfg.gpu.concurrency)


def _task() -> TaskSpec:
    return TaskSpec(level=3, problem_id=43, name="43_MinGPTCausalAttention",
                    ref_path=Path("ref.py"), ref_src_sha="0" * 64)


def _src(tmp_path: Path, name: str = "s.py") -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    p = tmp_path / name
    p.write_text(_SRC, encoding="utf-8")
    return p


def test_the_single_screen_is_given_its_own_deadline_not_build_timeout_s(tmp_path):
    """The defect itself, driven through the real `compile_screen`. This must FAIL on the
    unfixed code, which passed `self.cfg.build_timeout_s`."""
    w = _Worker(verdict={"ok": True, "max_shared": 1024, "kernels": []})
    ev = _ev(w, build_timeout_s=1200.0, screen_floor_timeout_s=120.0,
             prescreen_base_timeout_s=30.0, prescreen_per_config_timeout_s=3.0)

    ev.compile_screen(_task(), _src(tmp_path), tag="t", backend="triton",
                      max_shared_bytes=101376)

    assert len(w.timeouts) == 1, w.timeouts
    assert w.timeouts[0] == pytest.approx(120.0), (
        "the screen's own deadline, not build_timeout_s=1200: %r" % w.timeouts)
    assert w.timeouts[0] < 1200.0


def test_no_COMPILE_ONLY_probe_anywhere_in_the_module_passes_build_timeout_s():
    """THE GENERALIZABLE GUARD, and the reason this file exists rather than a one-line change.

    `59d5a71` fixed `prescreen_batch` and left `compile_screen`. Asserting only on
    `compile_screen` would let the same class of defect reappear at a third screen. So the guard
    scans the module structurally: for every function that builds a `make_compile_probe_job`,
    require that its `run_job` deadline is a screen deadline and not `build_timeout_s`.

    WHY THE DISCRIMINATOR IS THE JOB TYPE AND NOT THE TAG. The first version of this test matched
    on the string "screen" in the call, and it immediately flagged two calls in `screen()` -- which
    turned out to be correct code, not a second instance of the defect. `screen()` is the
    CORRECTNESS screen: it launches the kernel and runs `quick_correctness_trials`, so it genuinely
    needs `build_timeout_s + eval_timeout_s`, exactly like `quick_test` and `full_eval`. Shortening
    its deadline would discard legitimately slow-compiling candidates, which
    `never-narrow-the-search-space-to-control-cost` forbids.

    So D9's population is precisely the COMPILE-ONLY probes -- the ones that never launch anything
    and whose entire output is `metadata.shared`. That is the distinction the tag failed to draw,
    and drawing it is what makes this guard usable rather than a permanent false positive.
    """
    src = Path(corr_mod.__file__).read_text(encoding="utf-8")
    # Split into function bodies at module and method indentation.
    bodies = re.split(r"\n    def ", src)
    probes = [b for b in bodies if "make_compile_probe_job(" in b]
    assert len(probes) >= 2, (
        "expected at least prescreen_batch and compile_screen to build compile-only probes; "
        "found %d -- the scan is broken, not the code" % len(probes))

    offenders = []
    for body in probes:
        name = body.split("(", 1)[0].strip()
        for call in re.findall(r"self\.worker\.run_job\((.*?)\)\s*$", body, re.S | re.M):
            flat = " ".join(call.split())
            if "build_timeout_s" in flat and "screen_timeout_s" not in flat:
                offenders.append(f"{name}: {flat[:160]}")
    assert not offenders, (
        "a COMPILE-ONLY probe must never be given a real trial's compile budget:\n"
        + "\n".join(offenders))


def test_the_correctness_screen_KEEPS_the_full_budget():
    """The other side of the same distinction, pinned so a later tidy-up does not "fix" it.

    `screen()` runs correctness trials on the GPU. Its deadline must stay
    `build_timeout_s + eval_timeout_s`: it is a real launch, and D9 says nothing about it.
    """
    src = Path(corr_mod.__file__).read_text(encoding="utf-8")
    body = src.split("\n    def screen(", 1)[1].split("\n    def ", 1)[0]
    calls = re.findall(r"self\.worker\.run_job\((.*?)\)\s*$", body, re.S | re.M)
    assert len(calls) == 2, calls   # the shared-lane call and its exclusive OOM retry
    for call in calls:
        flat = " ".join(call.split())
        assert "build_timeout_s + self.cfg.eval_timeout_s" in flat, (
            "the correctness screen launches the kernel and must keep the full budget: %r" % flat)
        assert "screen_timeout_s" not in flat, (
            "and it must NOT be given the compile-only probe's short deadline: %r" % flat)


def test_the_floor_is_load_bearing_because_the_batch_formula_at_n_1_is_too_tight():
    """33 s vs a measured p90 of 26.5-28.5 s: the batch budget is not calibrated for one config,
    so without the floor the single-config case would start refusing screens about to answer.

    Asserted as an INEQUALITY against the measured p90 rather than as `== 120.0`, so the test
    states why the number is what it is instead of restating the constant.
    """
    cfg = AppConfig().evaluation
    cfg.build_timeout_s = 1200.0
    cfg.prescreen_base_timeout_s = 30.0
    cfg.prescreen_per_config_timeout_s = 3.0

    batch_at_one = prescreen_timeout_s(cfg, 1)
    assert batch_at_one == pytest.approx(33.0), batch_at_one

    measured_p90_max = 28.5     # the slowest of the three boxes' p90
    measured_p99_max = 105.0    # the slowest of the three boxes' p99
    assert batch_at_one - measured_p90_max < 6.5, (
        "the batch formula at n=1 sits within 6.5 s of the measured p90, which is why it cannot "
        "be the deadline on its own")
    assert screen_timeout_s(cfg) >= measured_p99_max, (
        "the screen deadline must cover the slowest measured p99 (105.0 s), or it starts "
        "refusing screens that would have answered: %r" % screen_timeout_s(cfg))


def test_the_screen_deadline_is_always_below_build_timeout_s():
    """An invariant, checked over a sweep rather than at the shipped values, because the
    dangerous direction is a mis-set config making the SCREEN the longer deadline -- which would
    silently restore the tail this fixes."""
    cfg = AppConfig().evaluation
    for build in (60.0, 120.0, 300.0, 1200.0, 3600.0):
        for floor in (0.0, 33.0, 120.0, 600.0, 5000.0):
            cfg.build_timeout_s = build
            cfg.screen_floor_timeout_s = floor
            got = screen_timeout_s(cfg)
            assert got > 0.0, (build, floor, got)
            if floor <= build:
                assert got <= build, (
                    "the screen deadline must not exceed a real trial's compile budget: "
                    "build=%r floor=%r got=%r" % (build, floor, got))


def test_a_timed_out_screen_caches_NOTHING_so_no_configuration_is_excluded(tmp_path):
    """The safety argument, executed. This is what makes a shorter screen deadline a cost cap
    rather than a search-space restriction: the configuration still gets its full real trial."""
    w = _Worker(verdict=None)       # raises, as on a real timeout
    ev = _ev(w, screen_floor_timeout_s=120.0)
    p = _src(tmp_path)

    got = ev.compile_screen(_task(), p, tag="t", backend="triton", max_shared_bytes=101376)

    assert got is None, "a screen that could not answer must never refuse a configuration"
    assert ev.cached_shared_verdict(p.read_text(encoding="utf-8"), "triton", 101376) is None, (
        "an unanswered configuration must read UNKNOWN so the guard lets it through")
    assert not ev._screen_cache, "a failure is a fact about one attempt and must not be cached"


def test_a_real_trials_compile_budget_is_untouched(tmp_path):
    """`build_timeout_s` stays what it was, and `quick_test`/`full_eval` keep receiving
    `build_timeout_s + eval_timeout_s`. A regression here discards legitimately slow candidates."""
    w = _Worker(verdict={"ok": True, "max_shared": 1024, "kernels": []})
    ev = _ev(w, build_timeout_s=1200.0, eval_timeout_s=600.0)

    ev.compile_screen(_task(), _src(tmp_path), tag="t", backend="triton",
                      max_shared_bytes=101376)

    assert ev.cfg.build_timeout_s == 1200.0
    assert w.timeouts[0] < ev.cfg.build_timeout_s
    src = Path(corr_mod.__file__).read_text(encoding="utf-8")
    assert src.count("self.cfg.build_timeout_s + self.cfg.eval_timeout_s") >= 4, (
        "the four real-trial exits must still pass the full deadline as ONE budget")


# --------------------------------------------------------------------- revert checks
# Two variants of the UNFIXED code. Each must be caught, or this change has no guard.


def test_a_config_missing_the_new_field_fails_LOUDLY_not_as_a_probe_failure(tmp_path):
    """THE DEFECT THIS CHANGE ACTUALLY HAD, in its first version.

    `compile_screen` wraps `run_job` in `except Exception` -- correctly, because a screen must never
    be a verdict. The first version computed the new deadline INSIDE that block, so a config object
    without `screen_floor_timeout_s` raised `AttributeError`, was swallowed into `{"ok": False}`,
    and read as a PROBE failure. A mis-wired config would then be byte-indistinguishable from a
    timing-out `ptxas` on every screen, forever, with no event saying so.

    This is the SECOND time this exact mistake has been made in this file: `59d5a71` made it for
    `prescreen_batch`, and `test_a_failed_probe_is_never_cached_as_a_screen_verdict` plus
    `test_the_compile_screen_only_refuses_on_the_compilers_own_number` caught it again here --
    stub configs that carry only the fields the screen genuinely needs, which is what makes them
    fail the moment the screen starts needing one more. That is a guard earning its keep.

    A config error is a programming error and must reach the operator. A probe failure is a fact
    about one attempt and must be swallowed. The two must not share an exit.
    """
    class _Partial:
        """Everything `compile_screen` needs EXCEPT the new deadline field."""
        precision = "fp32"
        build_timeout_s = 60.0
        eval_timeout_s = 60.0
        correctness_mode = "dual_witness_relaxed"
        prescreen_base_timeout_s = 30.0
        prescreen_per_config_timeout_s = 3.0

    w = _Worker(verdict={"ok": True, "max_shared": 1024, "kernels": []})
    ev = CorrectnessEvaluator.__new__(CorrectnessEvaluator)
    ev.worker, ev.cfg, ev.seed = w, _Partial(), 0
    ev._screen_cache = {}
    ev._static_cache = {}

    with pytest.raises(AttributeError):
        ev.compile_screen(_task(), _src(tmp_path), tag="t", backend="triton",
                          max_shared_bytes=101376)

    assert w.timeouts == [], "the worker must not have been called with a bogus deadline"
    assert not ev._screen_cache, "and nothing may be cached from a config error"


def test_the_deadline_is_computed_outside_the_swallowing_try():
    """Structural, because the behavioural test above can be satisfied by accident. The
    computation must be a statement of its own, before the `try` -- exactly as `prescreen_batch`
    does it after the same lesson."""
    src = Path(corr_mod.__file__).read_text(encoding="utf-8")
    body = src.split("\n    def compile_screen(", 1)[1].split("\n    def ", 1)[0]
    before_try, _, after_try = body.partition("\n            try:")
    assert "screen_timeout_s(self.cfg)" in before_try, (
        "the deadline must be computed BEFORE the try, or an AttributeError from a mis-wired "
        "config is reported as a probe failure")
    assert "screen_timeout_s(" not in after_try, (
        "and it must not ALSO be computed inside it")


def test_revert_restoring_build_timeout_s_to_the_screen_is_caught(tmp_path, monkeypatch):
    """Variant 1: put `build_timeout_s` back. Simulated by making `screen_timeout_s` return it,
    which is exactly what the old line did."""
    w = _Worker(verdict={"ok": True, "max_shared": 1024, "kernels": []})
    ev = _ev(w, build_timeout_s=1200.0)
    monkeypatch.setattr(corr_mod, "screen_timeout_s", lambda cfg: float(cfg.build_timeout_s))

    ev.compile_screen(_task(), _src(tmp_path), tag="t", backend="triton",
                      max_shared_bytes=101376)

    assert w.timeouts[0] == pytest.approx(1200.0), (
        "the revert must actually reproduce the defect, or this control proves nothing")
    # And the assertion the real test makes would now fail, which is the point:
    with pytest.raises(AssertionError):
        assert w.timeouts[0] == pytest.approx(120.0)


def test_revert_setting_the_floor_to_zero_is_caught():
    """Variant 2: keep the new call site but zero the floor. The deadline collapses to 33 s,
    below the measured p90 -- so this must be visible, not silently accepted."""
    cfg = AppConfig().evaluation
    cfg.build_timeout_s = 1200.0
    cfg.prescreen_base_timeout_s = 30.0
    cfg.prescreen_per_config_timeout_s = 3.0
    cfg.screen_floor_timeout_s = 0.0

    got = screen_timeout_s(cfg)
    assert got == pytest.approx(33.0), got
    assert got < 105.0, "the revert must reproduce the too-tight deadline"
    with pytest.raises(AssertionError):
        assert got >= 105.0, "which is what the shipped configuration guarantees"


def test_the_new_config_field_is_accepted_and_a_typo_is_rejected():
    """`EvalConfig` extends `StrictConfig` (`extra="forbid"`), so a mistyped key must raise
    rather than be silently ignored -- the failure mode that let `suspicious_speedup` sit in every
    config file with no consumer."""
    from pydantic import ValidationError

    from kernel_optimizer.config import EvalConfig

    assert EvalConfig(screen_floor_timeout_s=90.0).screen_floor_timeout_s == 90.0
    assert EvalConfig().screen_floor_timeout_s == pytest.approx(120.0)
    with pytest.raises(ValidationError):
        EvalConfig(screen_floor_timeout=90.0)   # missing the _s suffix
