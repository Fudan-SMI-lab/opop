"""The resource->latency conversion RATE: what one unit of a resource actually bought.

This is step 1 of the third core design point, and the reason it is only step 1 is measured rather
than cautious. Over 5213 complete trials on three boxes:

  * `n_spills` is the ONLY dimension with a strong, SIGN-STABLE latency relationship -- median r2
    0.573 / 0.733 / 0.800 with the slope sign agreeing in 95 of 99 usable series. n_regs, occupancy
    and shared_bytes sit under r2 0.10 and flip sign in most series, reproducing the recorded
    `resource-map-is-not-separable` finding.
  * It is a RATE, not a threshold: the spill COUNT explains +0.661 / +0.518 / +0.407 of variance
    beyond a 0/1 "does it spill" indicator, and with the zeros removed entirely the fit holds at
    r2 0.745 / 0.594 / 0.762 with 93 of 95 slopes still positive.
  * It is NOT transferable: across candidates in one run the slope spans up to 15.4x, and
    normalising by the candidate's own clean latency leaves p90/p10 at 6.2x.

So the tests below assert three things in equal measure: the rate is computed where the evidence
supports it, it is WITHHELD everywhere else, and it is journalled without being allowed to change
anything -- a run carrying it must stay comparable with one that does not.
"""

from __future__ import annotations

from kernel_optimizer.evaluation.conversion_rate import (
    MIN_DISTINCT,
    MIN_R2,
    MIN_TRIALS,
    RATE_DIMENSIONS,
    conversion_rates,
    rates_payload,
)
from kernel_optimizer.models.core import LatencyStats, ParamSet, TrialRecord


def _trial(spills, ms, *, status="complete", regs=100, n=1):
    """A real `TrialRecord`, built through the real models.

    Shapes were guessed wrong once in this project (`BestRecord(params={...})` against a pydantic
    model wanting `ParamSet`), and a fixture invented to match the reader proves nothing -- so
    every field here goes through validation.
    """
    from kernel_optimizer.models.core import ProfileRecord

    prof = None
    if spills is not None or regs is not None:
        prof = ProfileRecord(n_spills=spills, n_regs=regs)
    return TrialRecord(
        trial_id="t%d" % n, candidate_id="cand-x", space_id="sp-1",
        params=ParamSet(values={"BLOCK": 64}),
        status=status,
        latency_ms=LatencyStats(mean=ms, std=0.01, min=ms - 0.01, max=ms + 0.01,
                                median=ms, n_samples=20) if ms is not None else None,
        profile=prof,
    )


def _clean_series(slope=0.02, base=1.0, values=(0, 2, 4, 8, 12, 16, 20, 24, 30, 36)):
    """A series with a genuine linear spill cost and negligible noise."""
    return [_trial(v, base + slope * v, n=i) for i, v in enumerate(values)]


def test_a_real_spill_rate_is_estimated_in_the_resources_own_unit():
    rates = conversion_rates(_clean_series(slope=0.02, base=1.0))
    assert len(rates) == 1, [r.dimension for r in rates]
    r = rates[0]
    assert r.dimension == "n_spills"
    assert abs(r.slope_ms_per_unit - 0.02) < 1e-6, r.slope_ms_per_unit
    assert r.r2 > 0.99
    assert r.unit == "local-memory slot"
    assert (r.lo, r.hi) == (0, 36)
    assert r.regime == "mixed", "zeros and non-zeros are both present"
    # And as a percentage of the candidate's own latency at the lowest spill count: 0.02/1.0.
    assert abs(r.slope_pct_of_base - 2.0) < 0.01, r.slope_pct_of_base


def test_only_n_spills_is_rated_because_only_it_survived_the_measurement():
    """The list is deliberately one name long. Adding a second without re-running the probes would
    put a slope on a dimension whose sign flips between candidates -- n_regs was 13+/30- on box 1
    and 15+/36- on box 2 -- and a confidently-wrong rate is worse than no rate, because a consumer
    treats it as a prior."""
    assert RATE_DIMENSIONS == ("n_spills",), RATE_DIMENSIONS
    # A series where registers vary perfectly with latency and spills do not vary at all must yield
    # NO rate: the strong relationship is on a dimension the evidence does not support.
    trials = [_trial(4, 1.0 + 0.01 * rg, regs=rg, n=i)
              for i, rg in enumerate((60, 80, 100, 120, 140, 160, 180, 200, 220, 240))]
    assert conversion_rates(trials) == []


def test_a_measured_zero_is_kept_because_it_is_the_informative_end_of_the_series():
    """`is not None`, never truthiness. The same falsy drop one layer up recorded a measured 0 as
    'not measured' on 11 of 128 dimension records on box 2, and it fires on exactly the healthy
    candidates. Here it would silently delete the clean end of every series."""
    rates = conversion_rates(_clean_series(values=(0, 0, 0, 5, 10, 15, 20, 25, 30, 35)))
    assert rates and rates[0].lo == 0, (
        "the zero-spill trials were dropped, so the series no longer spans the regime boundary")
    assert rates[0].regime == "mixed"


def test_too_few_trials_yields_no_rate():
    assert conversion_rates(_clean_series(values=(0, 4, 8, 12, 16))) == [], (
        "5 trials is below the %d-trial gate" % MIN_TRIALS)


def test_two_distinct_values_yield_no_rate_however_clean_they_look():
    """Two points define a line through themselves and report r2 = 1.0 -- the most confident-looking
    output this module could produce and the least informative."""
    trials = [_trial(0 if i % 2 else 20, 1.0 if i % 2 else 1.4, n=i) for i in range(12)]
    assert conversion_rates(trials) == [], "only %d distinct values are required" % MIN_DISTINCT


def test_a_series_with_no_relationship_yields_no_rate():
    """The r2 gate. Latency here is unrelated to spills, which is what n_regs and occupancy look
    like across the corpus (median r2 0.03-0.08)."""
    lat = [1.0, 1.4, 0.9, 1.5, 1.1, 0.8, 1.6, 1.0, 1.3, 0.95]
    trials = [_trial(v, lat[i], n=i) for i, v in enumerate((0, 2, 4, 8, 12, 16, 20, 24, 30, 36))]
    rates = conversion_rates(trials)
    assert rates == [], "a relationship below r2=%.1f must not be reported" % MIN_R2


def test_a_slope_that_reverses_when_the_zeros_are_removed_is_withheld():
    """The threshold-vs-rate check, applied per candidate rather than taken on faith.

    This is the 2-of-51 case on box 2. If the whole-series slope is positive only because the
    non-spilling trials happen to be fast, while inside the spilling regime MORE spills are FASTER,
    then the number describes a jump between two regimes and not a rate inside either. Withholding
    it is the honest answer -- and note that spilling really was faster in 9/44 and 17/51 mixed
    series, so this is not a hypothetical shape.

    THE FIXTURE HAS TO CLEAR THE r2 GATE OR IT ASSERTS NOTHING. A first version was rejected at
    r2=0.363 by `MIN_R2` before the reversal check ever ran, so the revert variant that deleted the
    reversal check did not change its verdict -- caught by
    `scripts/revert_check_conversion_rate.py`, and exactly the `probe-needs-a-positive-control`
    failure. This shape puts the zero group far enough below the spilling group that the overall
    upward trend is strong (r2 > 0.4) while the within-regime trend is clearly downward.
    """
    trials = [_trial(0, 1.00, n=0), _trial(0, 1.02, n=1), _trial(0, 0.99, n=2)]
    # Within the spilling regime, latency FALLS as spills rise -- but every spilling trial is far
    # above every clean one, so the whole-series fit still rises steeply.
    for i, (v, ms) in enumerate(((10, 4.00), (14, 3.90), (18, 3.80), (22, 3.70),
                                 (26, 3.60), (30, 3.50), (34, 3.40))):
        trials.append(_trial(v, ms, n=3 + i))

    # First, the non-vacuity check: the whole-series fit MUST clear the r2 gate, or the assertion
    # below would be satisfied by the wrong gate.
    xs = [t.profile.n_spills for t in trials]
    ys = [t.latency_ms.robust_ms for t in trials]
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    icept = my - slope * mx
    r2 = 1.0 - (sum((y - (icept + slope * x)) ** 2 for x, y in zip(xs, ys))
                / sum((y - my) ** 2 for y in ys))
    assert slope > 0 and r2 >= MIN_R2, (
        "the fixture must reach the reversal check: slope=%.5f r2=%.4f (gate %.1f)"
        % (slope, r2, MIN_R2))

    rates = conversion_rates(trials)
    assert rates == [], (
        "a sign reversal between the whole series and the spilling regime must withhold the rate: "
        "%r" % [(r.slope_ms_per_unit, r.r2) for r in rates])


def test_a_consistent_negative_slope_is_reported_rather_than_suppressed():
    """The positive control for the test above -- otherwise it could pass by withholding always.

    A candidate where more spills really are consistently faster gets its (negative) rate, because
    the measurements found that case: spilling was faster in 9 of 44 mixed series on box 1. This
    module reports what was measured, not what the folklore says.
    """
    rates = conversion_rates([_trial(v, 2.0 - 0.01 * v, n=i)
                              for i, v in enumerate((4, 8, 12, 16, 20, 24, 28, 32, 36, 40))])
    assert len(rates) == 1
    assert rates[0].slope_ms_per_unit < 0
    assert rates[0].regime == "always_positive", "no zero-spill trial is present here"


def test_an_always_spilling_series_is_labelled_so_it_is_not_compared_with_a_mixed_one():
    """Box 3's single-round run had 3 of 5 series never leaving the spilling regime. A rate from
    such a series answers a different question from one spanning the boundary, and only this label
    lets a reader tell them apart."""
    rates = conversion_rates(_clean_series(values=(4, 8, 12, 16, 20, 24, 28, 32, 36, 40)))
    assert rates and rates[0].regime == "always_positive"


def test_a_never_spilling_series_yields_no_rate_at_all():
    """Every trial at zero means the resource never varied -- there is nothing to fit, and the
    distinct-value gate is what refuses it."""
    assert conversion_rates([_trial(0, 1.0 + 0.001 * i, n=i) for i in range(12)]) == []


def test_incomplete_and_unmeasured_trials_are_ignored_not_coerced():
    """A failed trial is not a data point, and an unmeasured profile has no spill count. Neither may
    be read as one -- coercing a missing value to 0 would make every candidate look like it
    eliminated all spilling.

    THE FAILED TRIAL HERE CARRIES A LATENCY, deliberately. `TrialRecord` permits
    `status="fail"` with `latency_ms` set (models/core.py: the field is `LatencyStats | None` and
    status is an independent Literal), so a trial that was timed and then failed a later check is a
    legal record. A first version of this test gave the failed trial `ms=None`, which meant the
    latency check caught it and the STATUS check was never exercised -- the revert variant that
    deleted the status guard passed, caught by `scripts/revert_check_conversion_rate.py`. Relying on
    "a failed trial happens to have no latency" couples this to a coincidence of today's producers;
    `status` is the authoritative field.
    """
    trials = _clean_series()
    trials.append(_trial(200, 99.0, status="fail", n=98))     # timed, then failed: must be ignored
    trials.append(_trial(None, 5.0, n=99))                    # complete but spills unmeasured
    rates = conversion_rates(trials)
    assert rates and rates[0].hi == 36, (
        "a failed or unmeasured trial entered the fit: hi=%r" % rates[0].hi)
    assert rates[0].n == 10, "the fit used %d points instead of the 10 complete ones" % rates[0].n


def test_the_latency_used_is_the_median_the_run_ranked_by():
    """Correlating against the mean would relate the resource to a number no decision used. The
    mean's measured ranking accuracy on this hardware is 64.8% against the median's 93.2%.

    Built so the two statistics disagree in the SIGN of the relationship: the median rises with
    spills while the mean falls, so a reader of the wrong field gets the opposite answer.
    """
    trials = []
    for i, v in enumerate((0, 4, 8, 12, 16, 20, 24, 28, 32, 36)):
        median_ms = 1.0 + 0.02 * v
        mean_ms = 3.0 - 0.02 * v
        trials.append(TrialRecord(
            trial_id="t%d" % i, candidate_id="cand-x", space_id="sp-1",
            params=ParamSet(values={"BLOCK": 64}), status="complete",
            latency_ms=LatencyStats(mean=mean_ms, std=0.5, min=0.5, max=6.0,
                                    median=median_ms, n_samples=20),
            profile=__import__("kernel_optimizer.models.core", fromlist=["ProfileRecord"])
            .ProfileRecord(n_spills=v, n_regs=100),
        ))
    rates = conversion_rates(trials)
    assert rates and rates[0].slope_ms_per_unit > 0, (
        "the fit used the mean, whose slope here is negative: %r"
        % (rates[0].slope_ms_per_unit if rates else None))


def test_the_prompt_line_carries_its_confidence_and_its_non_transferability():
    """A bare "0.021 ms per slot" reads as a hardware fact. It is not one: across candidates in a
    single run the slope spans up to 15.4x, so a line without that warning invites exactly the
    extrapolation the data forbids. And `r2`/`n` must be present, or the number joins the class of
    agent-facing figures that looked precise and were vacuous in 36 of 36 cases.
    """
    line = conversion_rates(_clean_series())[0].for_prompt()
    assert "n_spills" in line
    assert "r2=" in line and "trials" in line, line
    assert "THIS CANDIDATE" in line, "the line must scope the claim to the candidate"
    assert "10x" in line and "not a property of the hardware" in line.lower(), (
        "the line must say the slope does not transfer: %s" % line)


def test_the_payload_records_a_finding_even_when_nothing_cleared_the_gates():
    """An absent event reads exactly like a mechanism that never ran -- the shape that left
    `launch_bound` unreachable across 848 trials without anyone noticing."""
    payload = rates_payload([_trial(0, 1.0, n=i) for i in range(12)])
    assert payload["rates"] == []
    assert payload["n_trials_seen"] == 12, (
        "the payload must say how many trials were considered, or 'no candidate had enough data' "
        "cannot be told from 'the estimator was never called'")
    assert payload["gates"]["min_r2"] == MIN_R2
    assert payload["dimensions_considered"] == list(RATE_DIMENSIONS)


def test_the_payload_survives_a_pydantic_round_trip():
    """It goes into events.jsonl, so it must serialise. A `ConversionRate` leaking into the payload
    as an object would raise at append time -- inside the orchestrator's blanket except, where it
    would be recorded as a failure of the estimator rather than of the serialisation."""
    import json

    payload = rates_payload(_clean_series())
    assert payload["rates"], "this fixture must produce a rate or the test asserts nothing"
    round_tripped = json.loads(json.dumps(payload))
    assert round_tripped["rates"][0]["dimension"] == "n_spills"


def test_the_rate_is_frozen_so_a_consumer_cannot_edit_the_number_or_its_confidence():
    """The slope and its r2 must travel together and unmodified: a caller that could raise the
    slope while leaving r2 describing the old fit would produce a prompt line whose stated
    confidence belongs to a different number."""
    import pydantic
    import pytest

    r = conversion_rates(_clean_series())[0]
    with pytest.raises(pydantic.ValidationError):
        r.slope_ms_per_unit = 99.0


# --- wiring: journalled, and INERT ------------------------------------------------------------


def test_the_rates_are_journalled_per_candidate_by_the_real_run():
    """Emitted from `_stats_and_analysis`, driving the real method rather than reimplementing it.

    A test that recomputed the payload in its own body would pass on any orchestrator at all --
    the defect that let two of this project's own tests assert nothing.
    """
    import tempfile
    from pathlib import Path

    from kernel_optimizer.control.orchestrator import CandidateRun, Orchestrator
    from kernel_optimizer.models.core import Candidate, ParameterSpace, ParamDomain
    from kernel_optimizer.store.run_store import RunStore

    with tempfile.TemporaryDirectory() as td:
        store = RunStore.create(Path(td) / "runs", "run-cr", {"task": "level3:43"})

        orch = Orchestrator.__new__(Orchestrator)
        orch.store = store

        class Stats:
            def analyze(self, space, trials):
                from kernel_optimizer.models.reports import TuningStats

                return TuningStats(space_id=space.space_id, candidate_id=crun.candidate.candidate_id,
                                   n_trials=len(trials), n_complete=len(trials), n_fail=0)

        class Deps:
            pass

        deps = Deps()
        deps.stats_analyzer = Stats()
        orch.deps = deps
        # `_unlaunched_kernels` and the analyst path are not what this test is about; the method
        # returns early on `best_ms is None`, AFTER the emission being asserted.
        orch._unlaunched_kernels = lambda crun: []

        crun = CandidateRun(candidate=Candidate(
            candidate_id="cand-x", family_id="fam-1", origin="seed", backend="triton",
            source_sha="0" * 64, structural_signature="1" * 64))
        crun.space = ParameterSpace(
            space_id="sp-1", candidate_id="cand-x", version=1, source_sha="0" * 64,
            domains=[ParamDomain(name="BLOCK", kind="int", choices=[32, 64])])
        crun.trials = _clean_series()
        crun.best_ms = None          # stop before the analyst

        orch._stats_and_analysis(crun)

        evs = [e for e in store.replay().events if e.type == "CONVERSION_RATES"]
        assert len(evs) == 1, [e.type for e in store.replay().events]
        payload = evs[0].payload
        assert payload["candidate_id"] == "cand-x"
        assert payload["space_id"] == "sp-1"
        assert payload["rates"] and payload["rates"][0]["dimension"] == "n_spills"
        assert abs(payload["rates"][0]["slope_ms_per_unit"] - 0.02) < 1e-6


def test_nothing_in_the_codebase_reads_the_rate_yet():
    """The comparability guarantee, asserted rather than promised.

    Step 1 journals the number and MUST NOT consume it: a run that ranked, allocated or prompted
    differently because of it would not be comparable with the three completed runs or with either
    E1 arm, and the whole point of landing this now is that E1/E3 can accumulate the corpus while
    staying comparable. The same boundary J2d-8 enforces for expectations.
    """
    from pathlib import Path

    root = Path(__file__).resolve().parents[1] / "src"
    offenders = []
    for path in root.rglob("*.py"):
        if path.name in ("conversion_rate.py",):
            continue
        text = path.read_text(encoding="utf-8")
        for needle in ("conversion_rates", "rates_payload", "ConversionRate",
                       "slope_ms_per_unit", "for_prompt()"):
            if needle not in text:
                continue
            # The orchestrator's journalling call is the ONE permitted reference.
            if path.name == "orchestrator.py" and needle in ("rates_payload",):
                continue
            if needle == "for_prompt()":
                # `for_prompt` is a common method name in this codebase (the resource vector has
                # one); only a call on a ConversionRate would matter, and that requires importing
                # this module.
                if "conversion_rate" not in text:
                    continue
            offenders.append("%s: %s" % (path.relative_to(root), needle))
    assert not offenders, (
        "the conversion rate is being consumed somewhere, so a run carrying it is no longer "
        "comparable with one that does not: %s" % offenders)


def test_an_estimator_failure_cannot_end_a_run():
    """A diagnostic must never be fatal, and its failure must be visible as its own event rather
    than silently swallowed -- the shape recorded in
    `a-support-field-must-not-damage-what-it-describes`, where a swallowed AttributeError also
    truncated the main artefact it was describing.
    """
    import tempfile
    from pathlib import Path

    from kernel_optimizer.control.orchestrator import CandidateRun, Orchestrator
    from kernel_optimizer.models.core import Candidate, ParameterSpace, ParamDomain
    from kernel_optimizer.store.run_store import RunStore

    with tempfile.TemporaryDirectory() as td:
        store = RunStore.create(Path(td) / "runs", "run-cr2", {"task": "level3:43"})
        orch = Orchestrator.__new__(Orchestrator)
        orch.store = store

        class Stats:
            def analyze(self, space, trials):
                from kernel_optimizer.models.reports import TuningStats

                return TuningStats(space_id=space.space_id, candidate_id=crun.candidate.candidate_id,
                                   n_trials=len(trials), n_complete=len(trials), n_fail=0)

        class Deps:
            pass

        deps = Deps()
        deps.stats_analyzer = Stats()
        orch.deps = deps
        orch._unlaunched_kernels = lambda crun: []

        crun = CandidateRun(candidate=Candidate(
            candidate_id="cand-y", family_id="fam-1", origin="seed", backend="triton",
            source_sha="0" * 64, structural_signature="2" * 64))
        crun.space = ParameterSpace(
            space_id="sp-2", candidate_id="cand-y", version=1, source_sha="0" * 64,
            domains=[ParamDomain(name="BLOCK", kind="int", choices=[32, 64])])

        class Exploding(list):
            def __iter__(self):
                raise RuntimeError("boom")

        crun.trials = Exploding(_clean_series())
        crun.best_ms = None

        # Must not raise.
        orch._stats_and_analysis(crun)
        kinds = [e.type for e in store.replay().events]
        assert "CONVERSION_RATES_FAILED" in kinds, kinds
        assert "STATS_DONE" in kinds, "the main artefact must still have been written first"
