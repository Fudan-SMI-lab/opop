"""Synthetic positive controls for the P2' reader's marginal arithmetic.

WHY THESE EXIST, and why a test on real logs would not have caught the bug. The reader's
marginal was wrong for three months in a way that live data could never expose: it returned
`(n-f)/n` where the conditioned side computes `1 - n/f`. Both are "a percentage", both look
plausible, both produce numbers in the right range, and on real logs the two disagree only
in ways that look like measurement noise -- the defect showed up as the MARGINAL BEING BAD,
which was exactly the conclusion the probe was written to support. A wrong ruler that
flatters your hypothesis is invisible from the output alone.

So the control is a case where the CORRECT ANSWER IS KNOWN BY CONSTRUCTION. Feed the reader
trials whose medians are exactly F=10, N=8. Then any estimator of "the relative change from
F to N" must return 1 - 8/10 = +0.20, the same value and sign the scanner would compute for
g_d. If the reader returns -0.25 instead, it is using the retracted form -- and the test
says so with the exact number, rather than a run failing somewhere downstream.

This is the discipline the memory calls "a probe needs a positive control": a test that only
checks the probe does not crash confirms nothing about whether its arithmetic means what its
docstring claims. Each case below pins one property:

  identical_endpoints   F == N must give exactly 0.0, not a small float artefact
  known_positive        F=10, N=8  -> +0.20 (N faster). The counterexample from review.
  known_negative        F=8, N=10  -> -0.25 (N slower). Asymmetric on purpose: the ratio
                        form is NOT antisymmetric, so a "just negate it" fix fails here.
  legacy_sign_flag      the retracted form is still reachable for regenerating old tables,
                        and must give the OLD number -- so the two are never confused
  agrees_with_scanner   the same endpoints through the production scanner formula and
                        through the reader must agree to floating-point tolerance
  thin_bucket           one trial per side is a "median" of one: decline, do not answer
  cross_candidate       trials from another candidate must not enter the pool
"""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import pytest

_PROBE = Path(__file__).resolve().parents[1] / "scripts" / "probes" / "v41_p2prime.py"


def _load():
    spec = importlib.util.spec_from_file_location("v41_p2prime", _PROBE)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


p2 = _load()


def trial(axis: str, value, ms: float, *, candidate: str = "cand-a",
          trial_id: str = "tr-x") -> dict:
    """One completed TRIAL_DONE record, in the shape the reader consumes.

    `latency_ms` uses the key `median` with no `_ms` suffix -- the recorded trap: writing
    `median_ms` makes every trial read as None and the probe prints a confident
    "no comparable trials" instead of failing.
    """
    return {
        "candidate_id": candidate,
        "trial_id": trial_id,
        "status": "complete",
        "params": {"values": {axis: value}},
        "latency_ms": {"median": ms, "mean": ms},
    }


def pool(f_ms: list[float], n_ms: list[float], *, axis: str = "BK",
         f_val=32, n_val=64, candidate: str = "cand-a") -> list[dict]:
    out = []
    for i, ms in enumerate(f_ms):
        out.append(trial(axis, f_val, ms, candidate=candidate, trial_id=f"tr-f{i}"))
    for i, ms in enumerate(n_ms):
        out.append(trial(axis, n_val, ms, candidate=candidate, trial_id=f"tr-n{i}"))
    return out


def test_identical_endpoints_are_exactly_zero():
    slope, skip = p2._marginal_slope(pool([10.0, 10.0], [10.0, 10.0]), "BK",
                                     repr(32), repr(64))
    assert skip is None
    assert slope == pytest.approx(0.0, abs=1e-12), (
        "F == N must be exactly no change; a nonzero value here means the two sides are "
        "not being divided by the same denominator")


def test_known_positive_is_plus_point_two_not_minus_point_two_five():
    """The external review's counterexample, as an executable assertion.

    F=10, N=8: the near-wall end is 20% faster, so g = 1 - 8/10 = +0.20 -- the same
    convention and sign as scanner.py's g_d. The retracted form gave (8-10)/8 = -0.25,
    which manufactured an apparent |g_m - y| of 0.45 against a holdout of +0.20.
    """
    slope, skip = p2._marginal_slope(pool([10.0, 10.0], [8.0, 8.0]), "BK",
                                     repr(32), repr(64))
    assert skip is None
    assert slope == pytest.approx(0.20, abs=1e-12)
    assert slope > 0, "positive must mean the N (near-wall) end is FASTER"
    # The specific wrong answer, named so a regression is unmistakable.
    assert slope != pytest.approx(-0.25, abs=1e-9)


def test_known_negative_is_asymmetric_so_negation_is_not_a_fix():
    """F=8, N=10 gives 1 - 10/8 = -0.25, NOT -0.20.

    The ratio form is deliberately not antisymmetric: swapping the ends does not flip the
    sign of the same magnitude. That is why `gates.reversed_interval` uses -U/(1-U) rather
    than negating, and why a future "fix" that just multiplies by -1 must fail a test.
    """
    slope, skip = p2._marginal_slope(pool([8.0, 8.0], [10.0, 10.0]), "BK",
                                     repr(32), repr(64))
    assert skip is None
    assert slope == pytest.approx(-0.25, abs=1e-12)
    # Naive negation of the positive case would have produced -0.20 here.
    assert slope != pytest.approx(-0.20, abs=1e-9)
    # And the exact reversal algebra from gates.py must reproduce it from +0.20.
    g_forward = 0.20
    assert -g_forward / (1.0 - g_forward) == pytest.approx(-0.25, abs=1e-12)


def test_legacy_sign_flag_reproduces_the_retracted_number():
    """The retracted form stays reachable, and gives the retracted answer.

    Historical tables must be regenerable -- overwriting them silently is what makes an
    audit impossible -- but the old and new values must never be interchangeable.
    """
    old, skip = p2._marginal_slope(pool([10.0, 10.0], [8.0, 8.0]), "BK",
                                   repr(32), repr(64), legacy_sign=True)
    assert skip is None
    assert old == pytest.approx(-0.25, abs=1e-12)
    new, _ = p2._marginal_slope(pool([10.0, 10.0], [8.0, 8.0]), "BK", repr(32), repr(64))
    assert (old > 0) != (new > 0), "the two forms must disagree in sign on this case"


def test_reader_agrees_with_the_production_scanner_formula():
    """Same endpoints, two code paths: the probe and the live scanner must agree.

    scanner.py computes g = 1 - exp(log N - log F) on FRESH replicate means. Reproduced
    here in one line rather than imported, because importing it would test that the two
    call the same function -- not that the reader's own arithmetic matches the definition.
    """
    f, n = 10.0, 8.0
    scanner_g = 1.0 - math.exp(math.log(n) - math.log(f))
    reader_g, skip = p2._marginal_slope(pool([f, f], [n, n]), "BK", repr(32), repr(64))
    assert skip is None
    assert reader_g == pytest.approx(scanner_g, abs=1e-12)


def test_a_median_of_one_trial_declines_rather_than_answering():
    slope, skip = p2._marginal_slope(pool([10.0], [8.0]), "BK", repr(32), repr(64))
    assert slope is None
    assert skip == "thin_bucket_at_cutoff", (
        "one trial per side is a single noisy sample wearing a robust name; answering "
        "would let the conditioned side win against a coin flip")


def test_unsampled_endpoint_declines_and_is_not_an_error():
    slope, skip = p2._marginal_slope(pool([10.0, 10.0], [8.0, 8.0]), "BK",
                                     repr(32), repr(128))   # 128 never measured
    assert slope is None
    assert skip == "values_unsampled_at_cutoff"


def test_a_single_measured_value_has_no_slope():
    only_f = [trial("BK", 32, 10.0, trial_id="tr-a"), trial("BK", 32, 10.2, trial_id="tr-b")]
    slope, skip = p2._marginal_slope(only_f, "BK", repr(32), repr(64))
    assert slope is None
    assert skip == "no_slope_at_cutoff"


def test_nonpositive_latency_never_reaches_the_division():
    bad = pool([10.0, 10.0], [0.0, -1.0])
    slope, skip = p2._marginal_slope(bad, "BK", repr(32), repr(64))
    # The zero/negative trials are filtered before bucketing, so the N side ends up empty
    # and the estimator declines -- it must never divide by or take a log of them.
    assert slope is None
    assert skip in ("values_unsampled_at_cutoff", "no_slope_at_cutoff", "nonpositive_median")


def test_repr_keyed_buckets_do_not_collapse_true_and_one():
    """JSON collapses True/1, so both sides are compared through repr().

    A bool knob and an int knob sharing a bucket would average unrelated configurations.
    """
    mixed = [trial("FLAG", True, 10.0, trial_id="tr-t0"),
             trial("FLAG", True, 10.0, trial_id="tr-t1"),
             trial("FLAG", 1, 8.0, trial_id="tr-i0"),
             trial("FLAG", 1, 8.0, trial_id="tr-i1")]
    slope, skip = p2._marginal_slope(mixed, "FLAG", repr(True), repr(1))
    assert skip is None
    # repr(True) == 'True' and repr(1) == '1' are different buckets, so a slope exists.
    assert slope == pytest.approx(0.20, abs=1e-12)


def test_cross_candidate_trials_must_not_enter_a_candidate_scoped_pool():
    """Defect 2, as a unit assertion on the pool the caller builds.

    On live logs 90-98% of the late-contrast 'marginal' came from other candidates'
    implementations of the same knob NAME. The estimator itself cannot detect this -- the
    guarantee has to come from the caller passing a candidate-scoped history -- so this
    test pins the consequence: the same axis with a foreign candidate's much slower trials
    changes the answer, which is why the scope must be enforced upstream.
    """
    own = pool([10.0, 10.0], [8.0, 8.0], candidate="cand-a")
    foreign = pool([40.0, 40.0], [40.0, 40.0], candidate="cand-b")
    clean, _ = p2._marginal_slope(own, "BK", repr(32), repr(64))
    polluted, _ = p2._marginal_slope(own + foreign, "BK", repr(32), repr(64))
    assert clean == pytest.approx(0.20, abs=1e-12)
    assert polluted != pytest.approx(clean, abs=1e-3), (
        "if this ever stops differing, the pooling defect has become undetectable and "
        "the candidate scoping in analyse() is no longer load-bearing")
