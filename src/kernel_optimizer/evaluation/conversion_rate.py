"""How much latency did one unit of a resource actually buy? (the third core design point)

WHAT THIS ADDS TO `conversion.py`, AND WHY IT IS A SEPARATE MODULE. That one answers "did the
resource change convert into speed" QUALITATIVELY -- improved / no_conversion / regressed / flat --
and it deliberately computes no number, because pressure deltas have no common unit ACROSS
dimensions (shared_bytes/101376 swings by tens of thousands whenever a tile changes while
n_regs/255 moves by a few: measured median swings 0.162 vs 0.012, a 13.5x difference that comes
entirely from the normalising denominator).

That argument is correct and it is re-confirmed by the measurements below. But it only rules out a
CROSS-dimension composite. Inside ONE dimension the unit is fixed, so d(latency)/d(resource) is
dimensionally meaningful -- and it is estimable from a candidate's own tuning trials, which already
sweep the parameters that move it.

WHY ONLY `n_spills`. Measured over 5213 complete trials on three boxes
(scripts/probe_conversion_efficiency.py), per (candidate, dimension) series with n>=8 trials and
>=3 distinct values of the resource:

    dimension              box1 r2   sign      box2 r2   sign      box3 r2
    n_spills                 0.733   43+/0-      0.573   49+/2-      0.800
    shared_bytes             0.059   36+/8-      0.046   39+/12-     0.307
    n_regs                   0.083   13+/30-     0.045   15+/36-     0.053
    occupancy                0.049   3+/29-      0.029   11+/40-     0.223
    threads_launched         0.008   4+/5-       0.015   1+/8-       0.028
    peak_alloc_bytes         (2 usable series)   (1)                 (0)
    candidate_aten_bytes     (2)                 (1)                 (0)
    candidate_aten_ops       (1)                 (0)                 (0)

`n_spills` is the only dimension whose relationship is both strong and SIGN-STABLE -- 95 of 99
usable series agree. n_regs and occupancy flip sign in most series, which reproduces the recorded
`resource-map-is-not-separable` finding ("even the SIGN is unreliable") rather than contradicting
it. The last three dimensions are per-CANDIDATE constants, so they do not vary inside a candidate
and can never have a slope at all; they stay with the qualitative verdict.

AND IT IS A RATE, NOT A THRESHOLD -- the check that had to pass before any of this was worth
shipping (scripts/probe_spill_threshold_vs_rate.py). A high r2 would be unremarkable if series only
held `0` and `some large number`, because then the fit merely says "a spilling kernel is slower",
which is common knowledge and not an efficiency. Three-way split on the mixed series:

                                          box1     box2     box3
    r2 with spills as a COUNT             0.733    0.573    0.845
    r2 with spills as a 0/1 THRESHOLD     0.072    0.055    0.438
    the count explains beyond threshold   +0.661   +0.518   +0.407
    r2 WITH THE ZEROS REMOVED             0.745    0.594    0.762
    slope sign in that regime             41+/0-   47+/2-   5+/0-

The positive-only fit does not collapse, so more spill slots really do cost more time inside the
spilling regime. Note also that spilling was FASTER in 9/44 and 17/51 mixed series (latency ratios
range 0.493x..4.869x), which is exactly why "eliminate spills" is not an unconditional rule and
why a measured local slope beats a slogan.

WHAT THIS IS NOT, and the boundary is load-bearing: the slope is NOT transferable. Across
candidates within ONE run it spans up to 15.4x (run-l3-48-20260907-202457: 0.00571..0.08817
ms/slot), and normalising by the candidate's own clean latency does not fix it (pooled p90/p10
still 6.2x, up to 18.2x inside one run). So it describes the local terrain around THIS candidate at
THESE parameters, and any consumer that extrapolates it to a new structure is wrong by up to an
order of magnitude. It explains what just happened; it does not predict the next rewrite.
"""

from __future__ import annotations

import math
from typing import Any

from pydantic import BaseModel, ConfigDict

# The one dimension the measurements support. A list of one, deliberately: adding a second name
# here without re-running the probes would put a slope on a dimension whose sign flips between
# candidates, and a confidently-wrong rate is worse than none because a consumer treats it as a
# prior. See the module docstring for the per-dimension figures.
RATE_DIMENSIONS = ("n_spills",)

# Minimum trials in a series. Below this an OLS slope is a curiosity: box 3's single-round run has
# 5 series at n>=8 and 3 of those never leave the spilling regime.
MIN_TRIALS = 8

# Minimum distinct values of the resource. Two points define a line through themselves and report
# r2 = 1.0, which would be the most confident-looking output this module could produce and the
# least informative.
MIN_DISTINCT = 3

# Minimum r2 before a slope is reported at all. An EVIDENCE gate, not a tuned parameter: the
# measured medians are 0.573 (box 2), 0.733 (box 1) and 0.800 (box 3), so a real spill relationship
# clears it comfortably while the other dimensions' 0.03-0.08 would not come close if one were ever
# added to RATE_DIMENSIONS.
MIN_R2 = 0.4


class ConversionRate(BaseModel):
    """d(latency)/d(resource) for ONE candidate and ONE dimension, in that resource's own unit."""

    model_config = ConfigDict(frozen=True)

    dimension: str
    unit: str
    n: int
    n_distinct: int
    lo: float
    hi: float
    # ms per unit of the resource.
    slope_ms_per_unit: float
    r2: float
    # The slope as a percentage of the candidate's median latency at the resource's LOWEST observed
    # value, so two candidates' rates can be read side by side. Included for legibility only -- it
    # is NOT more transferable than the raw slope (pooled p90/p10 6.2x either way), and the caller
    # must not treat it as a constant.
    slope_pct_of_base: float | None = None
    # "mixed" (both zero and non-zero present) | "always_positive" | "always_zero".
    # A rate from an `always_positive` series is still a rate, but a reader comparing it against a
    # `mixed` one is comparing two different questions.
    regime: str = "mixed"
    # Set when the sign inside the non-zero regime disagrees with the sign over the whole series --
    # the 2-of-51 case on box 2. The slope is then withheld, and this says why.
    note: str = ""

    def for_prompt(self) -> str:
        """One line for the agent, carrying its own confidence and its own limits.

        `r2` and `n` are present because a bare "0.021 ms per slot" reads as a hardware fact; this
        project has already measured agent-authored resource constraints that looked precise and
        were vacuously true in 36 of 36 cases. The non-transferability clause is present because
        the slope varies up to 15.4x between candidates in one run, so an agent extrapolating it to
        a new structure would be reasoning from a number that does not apply there.
        """
        base = ""
        if self.slope_pct_of_base is not None:
            base = " (%.2f%% of this candidate's latency at %s=%g)" % (
                self.slope_pct_of_base, self.dimension, self.lo)
        return (
            "%s -> latency: MEASURED ON THIS CANDIDATE, each %s costs about %.4f ms%s "
            "[r2=%.2f over %d trials, %d distinct values %g..%g]. This is the local slope for "
            "this structure at these parameters, NOT a property of the hardware or the task: "
            "across candidates in one run it varies by more than 10x, so do not carry it into a "
            "new structure."
            % (self.dimension, self.unit, self.slope_ms_per_unit, base, self.r2, self.n,
               self.n_distinct, self.lo, self.hi))


def _ols(xs: list[float], ys: list[float]) -> tuple[float, float] | None:
    """(slope, r2), or None when x does not vary or y is constant."""
    n = len(xs)
    if n < 3:
        return None
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    ss_tot = sum((y - my) ** 2 for y in ys)
    if sxx <= 0 or ss_tot <= 0:
        return None
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    slope = sxy / sxx
    intercept = my - slope * mx
    ss_res = sum((y - (intercept + slope * x)) ** 2 for x, y in zip(xs, ys))
    return slope, 1.0 - ss_res / ss_tot


def _median(vs: list[float]) -> float:
    s = sorted(vs)
    return s[len(s) // 2] if s else float("nan")


def _series(trials: Any, dimension: str) -> list[tuple[float, float]]:
    """(resource, latency) for every complete trial that measured both.

    Latency is `robust_ms` -- the median when present -- because that is the statistic the run
    RANKED by. Correlating against a mean would relate the resource to a number no decision used,
    and the mean's measured ranking accuracy on this hardware is 64.8% against the median's 93.2%.
    """
    out: list[tuple[float, float]] = []
    for t in trials or []:
        if getattr(t, "status", None) != "complete":
            continue
        lat = getattr(t, "latency_ms", None)
        ms = getattr(lat, "robust_ms", None) if lat is not None else None
        if not isinstance(ms, (int, float)) or ms <= 0:
            continue
        prof = getattr(t, "profile", None)
        if prof is None:
            continue
        v = getattr(prof, dimension, None)
        # `is not None`, never truthiness: a measured ZERO is the most informative value this
        # dimension has, and dropping it here would remove exactly the clean end of the series.
        # The same falsy drop one layer up cost 11 of 128 dimension records on box 2.
        if v is None or isinstance(v, bool) or not isinstance(v, (int, float)):
            continue
        out.append((float(v), float(ms)))
    return out


def conversion_rates(trials: Any) -> list[ConversionRate]:
    """Estimate the local resource->latency slope for one candidate, from its own trials.

    Returns an empty list when nothing clears the evidence gates, which is the common case early in
    a run and on any candidate whose spill count never varies. An empty list is not a failure and
    must not be logged as one.
    """
    rates: list[ConversionRate] = []
    for dimension in RATE_DIMENSIONS:
        pairs = _series(trials, dimension)
        if len(pairs) < MIN_TRIALS:
            continue
        xs = [p[0] for p in pairs]
        ys = [p[1] for p in pairs]
        distinct = sorted(set(xs))
        if len(distinct) < MIN_DISTINCT:
            continue

        n_zero = sum(1 for x in xs if x == 0)
        regime = ("always_zero" if n_zero == len(xs)
                  else "always_positive" if n_zero == 0 else "mixed")

        fit = _ols(xs, ys)
        if fit is None:
            continue
        slope, r2 = fit
        if r2 < MIN_R2 or not math.isfinite(slope) or not math.isfinite(r2):
            continue

        # The threshold-vs-rate check, applied per candidate rather than taken on faith from the
        # corpus. If removing the zeros reverses the sign, the whole-series slope is describing the
        # jump between two regimes and not a rate within either, so the number is withheld and the
        # reason recorded. Measured as the 2-of-51 case on box 2.
        note = ""
        if regime == "mixed":
            pos = [(x, y) for x, y in pairs if x > 0]
            if len({p[0] for p in pos}) >= 2 and len(pos) >= 3:
                pos_fit = _ols([p[0] for p in pos], [p[1] for p in pos])
                if pos_fit is not None and (pos_fit[0] > 0) != (slope > 0):
                    note = ("withheld: the slope reverses sign when the non-spilling trials are "
                            "removed (%.5f over all trials vs %.5f within the spilling regime), "
                            "so this is a difference between two regimes rather than a rate "
                            "inside either" % (slope, pos_fit[0]))
        if note:
            continue

        base_ms = _median([y for x, y in pairs if x == min(distinct)])
        pct = (100.0 * slope / base_ms) if base_ms and base_ms > 0 else None
        rates.append(ConversionRate(
            dimension=dimension,
            unit="local-memory slot" if dimension == "n_spills" else "unit",
            n=len(pairs), n_distinct=len(distinct),
            lo=min(distinct), hi=max(distinct),
            slope_ms_per_unit=round(slope, 6),
            r2=round(r2, 4),
            slope_pct_of_base=round(pct, 4) if pct is not None else None,
            regime=regime,
        ))
    return rates


def rates_payload(trials: Any) -> dict:
    """What to journal. Always returns a dict, so "measured nothing" is recorded as a fact.

    An absent event reads exactly like a mechanism that never ran -- the shape that left
    `launch_bound` unreachable across 848 trials without anyone noticing. `n_considered` is carried
    so a reader can tell "no candidate had enough trials" from "the estimator was never called".
    """
    rates = conversion_rates(trials)
    return {
        "rates": [r.model_dump() for r in rates],
        "dimensions_considered": list(RATE_DIMENSIONS),
        "gates": {"min_trials": MIN_TRIALS, "min_distinct": MIN_DISTINCT, "min_r2": MIN_R2},
        "n_trials_seen": sum(1 for t in (trials or [])
                             if getattr(t, "status", None) == "complete"),
    }
