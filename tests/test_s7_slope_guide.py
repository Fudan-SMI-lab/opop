"""S7 / item 3.2: slope-guided sampling.

WHAT THESE TESTS ARE BUILT AROUND, and it is the measurement that argues AGAINST the mechanism as much
as for it (`scripts/probes/s7_feasibility_replay.py`: prefixes of each candidate's real trial ORDER, the
shipping finders on each prefix, per-choice medians rebuilt FROM THE PREFIX so the replay cannot see what
the sampler could not):

  FOR      a wall is available by the halfway point on 25 of 34 candidates (74%), median earliest prefix
           0.38 of the sequence -- there IS budget left to act on it.
  AGAINST  the knob walled at that early prefix is still walled at the end on only 18 of 34 (53%).
           12 of 34 walls VANISH by 100%, and 4 of 34 move to a DIFFERENT knob.

Both from the SOFT criterion. The hard one is unanswerable on this host: 0 of 151 local candidates carry
a single `infeasible_shared_memory` record, because every locally backed-up run predates the
shared-memory screen and refused parameter sets are `find_walls`' entire input.

So the properties under test are the ones that make a 53%-reliable signal safe to act on -- ADVISORY
(nothing is forbidden, nothing locked in), RECOMPUTED (derived from scratch each time, so a wall that
stops holding stops being pushed), CAPPED (bounded dose per recompute) -- plus the parity properties that
let the 12h pair be read at all: the budget is untouched, the space is untouched, and the guard still
decides.
"""

from __future__ import annotations

import ast
import io
from pathlib import Path

from kernel_optimizer.models.core import (
    LatencyStats,
    ParamDomain,
    ParameterSpace,
    ParamSet,
    ProfileRecord,
    TrialRecord,
)
from kernel_optimizer.models.reports import ParamStat, TuningStats
from kernel_optimizer.tuning import slope_guide as sg

SRC = Path("src/kernel_optimizer")


# ---------------------------------------------------------------------------------------------
# fixtures, shaped like the emitter's records rather than like the reader
# ---------------------------------------------------------------------------------------------


def _space(knob: str = "BLOCK_M", choices: list | None = None,
           extra: dict[str, list] | None = None) -> ParameterSpace:
    domains = [ParamDomain(name=knob, kind="int", choices=choices or [16, 32, 64, 128])]
    for name, ch in (extra or {}).items():
        domains.append(ParamDomain(name=name, kind="int", choices=ch))
    return ParameterSpace(space_id="sp-x", candidate_id="cand-x", source_sha="0" * 8,
                          domains=domains)


def _trial(values: dict, ms: float | None, *, status: str = "complete",
           failure: str | None = None, spills: int | None = 0) -> TrialRecord:
    """A TrialRecord with the emitter's own field spellings.

    `LatencyStats` requires mean/std/min/max/n_samples and exposes `robust_ms` as a PROPERTY that is
    never serialized -- which is why the module reproduces median-else-mean instead of looking the name
    up, and why a fixture that invented a `robust_ms` field would prove nothing.
    """
    lat = None
    if ms is not None:
        lat = LatencyStats(mean=ms, std=0.05, min=ms, max=ms, n_samples=20, median=ms)
    return TrialRecord(
        trial_id=f"t-{sorted(values.items())}-{ms}", candidate_id="cand-x", space_id="sp-x",
        params=ParamSet(values=values), status=status, failure_kind=failure, latency_ms=lat,
        profile=ProfileRecord(n_regs=200, n_spills=spills, shared_bytes=32768, num_warps=4,
                              num_stages=2, compile_s=0.4))


def _stats(space: ParameterSpace, trials: list[TrialRecord]) -> TuningStats:
    """The REAL analyzer on the REAL space, not a hand-built stats object.

    The wall criteria read `ParamStat.latency_by_value`, which is the harness's own per-choice median
    table; a hand-made table could be shaped to make a wall appear and would then be testing the
    fixture. `DeviceLimits()` defaults are the 4090's (shared optin 101376), asserted elsewhere.
    """
    from kernel_optimizer.models.core import DeviceLimits
    from kernel_optimizer.tuning.stats import TuningStatsAnalyzer

    return TuningStatsAnalyzer(DeviceLimits()).analyze(space, trials)


def _hard_wall_case() -> tuple[ParameterSpace, list[TrialRecord]]:
    """A truncated knob: 16/32/64 measured and improving, 128 REFUSED by the compiler.

    Note 64 is deliberately NOT measured, so there is an undrawn choice below the refusal for the guide
    to propose. `find_walls` needs >= 3 measured values, so the extra knob carries a second value to keep
    the ladder legal without giving BLOCK_M a fourth.
    """
    space = _space(choices=[16, 32, 64, 128])
    trials = [
        _trial({"BLOCK_M": 16}, 4.0),
        _trial({"BLOCK_M": 24}, 3.5),
        _trial({"BLOCK_M": 32}, 3.0),
        _trial({"BLOCK_M": 128}, None, status="fail", failure="infeasible_shared_memory"),
    ]
    space = _space(choices=[16, 24, 32, 64, 128])
    return space, trials


# ---------------------------------------------------------------------------------------------
# (1) the cadence: recompute on trials TOLD, never on trials asked
# ---------------------------------------------------------------------------------------------


def test_the_cadence_counts_finished_trials():
    g = sg.SlopeGuide(space=_space(), recompute_every=10)
    assert [n for n in range(1, 31) if g.due(n)] == [10, 20, 30]


def test_zero_told_is_never_due():
    """Otherwise the first recompute would run against an empty stats table and find nothing, burning
    a recompute to journal a zero."""
    assert sg.SlopeGuide(space=_space(), recompute_every=10).due(0) is False


def test_a_non_positive_cadence_disables_the_recompute_rather_than_dividing_by_zero():
    for every in (0, -1):
        g = sg.SlopeGuide(space=_space(), recompute_every=every)
        assert not any(g.due(n) for n in range(1, 50))


def test_the_cadence_reads_the_tuners_told_count_and_not_its_asked_count():
    """The distinction is real, not theoretical: with `constant_liar` the orchestrator keeps
    asked-but-untold trials in flight, so a cadence on asks would recompute against a stats table that
    has not moved -- and `n_told` is what the tuner exposes for exactly this.
    """
    text = io.open(SRC / "control" / "orchestrator.py", encoding="utf-8").read()
    tree = ast.parse(text)
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == "_tune")
    body = ast.get_source_segment(text, fn) or ""
    call = next(line for line in body.splitlines() if "guide.due(" in line)
    assert "n_told" in call and "_asked" not in call


def test_told_counts_only_returned_trials_and_survives_a_double_tell():
    """`n_told` must not be derivable as `asked - pending`: that difference is also correct for a trial
    told twice, which raises -- but only AFTER the cadence has been computed from a wrong number.
    """
    from kernel_optimizer.tuning.tpe import OptunaTPETuner

    space = _space()
    t = OptunaTPETuner(space, guard_ok=lambda p: True, budget=4, seed=0)
    assert t.n_told == 0
    tid, params = t.ask()
    assert t.n_told == 0, "asking is not measuring"
    t.tell(tid, _trial(dict(params.values), 2.0))
    assert t.n_told == 1
    try:
        t.tell(tid, _trial(dict(params.values), 2.0))
    except KeyError:
        pass
    else:
        raise AssertionError("telling the same trial twice must raise")
    assert t.n_told == 1, "a refused tell must not advance the cadence"


# ---------------------------------------------------------------------------------------------
# (2) what gets proposed: an UNDRAWN declared choice, toward the wall, from the incumbent
# ---------------------------------------------------------------------------------------------


def test_a_hard_wall_proposes_an_undrawn_choice_below_the_refused_value():
    space, trials = _hard_wall_case()
    g = sg.SlopeGuide(space=space, max_enqueued_per_recompute=2)
    out = g.suggest(_stats(space, trials), trials, set())
    assert len(out) == 1
    s = out[0]
    assert s.knob == "BLOCK_M"
    assert s.source == "hard_wall"
    assert s.knob_value == 64, \
        "64 is the largest declared choice below the refused 128 that has not been drawn"
    assert s.tail_gain_pct > 0


def test_no_value_at_or_beyond_the_refused_one_is_proposed():
    """The compiler has already refused 128. A value AT or BEYOND it would return
    `infeasible_shared_memory`, so the mechanism would spend a trial to be told what it already knew --
    and would manufacture its own extra evidence that the wall exists.

    The earlier version of this test used a space whose top choice WAS the refused value, so the bound
    was never exercised: dropping it changed nothing and the revert-check said NOT CAUGHT. This space
    declares 256 above the refusal, which is undrawn and would be picked first without the bound.
    """
    space = _space(choices=[16, 24, 32, 64, 128, 256])
    trials = [_trial({"BLOCK_M": 16}, 4.0), _trial({"BLOCK_M": 24}, 3.5),
              _trial({"BLOCK_M": 32}, 3.0),
              _trial({"BLOCK_M": 128}, None, status="fail",
                     failure="infeasible_shared_memory")]
    out = sg.SlopeGuide(space=space).suggest(_stats(space, trials), trials, set())
    assert [s.knob_value for s in out] == [64], \
        "64 is the last undrawn choice BEFORE the refusal; 128 and 256 are past it"


def test_every_declared_choice_below_the_refusal_being_drawn_yields_nothing():
    """The other half of the bound: when nothing undrawn is left on the walled side, the knob is skipped
    rather than pushed to a value past the refusal.
    """
    space = _space(choices=[16, 24, 32, 128])
    trials = [_trial({"BLOCK_M": 16}, 4.0), _trial({"BLOCK_M": 24}, 3.5),
              _trial({"BLOCK_M": 32}, 3.0),
              _trial({"BLOCK_M": 128}, None, status="fail",
                     failure="infeasible_shared_memory")]
    g = sg.SlopeGuide(space=space)
    assert g.suggest(_stats(space, trials), trials, set()) == []
    assert g.n_skipped_no_unmeasured_value == 1
    assert g.snapshot()["n_suggested"] == 0


def test_an_already_drawn_value_is_not_proposed_again():
    """A drawn value adds nothing to `latency_by_value` -- the per-choice median table both criteria
    read -- so re-proposing it cannot widen the coverage that makes a wall visible. Failed draws count
    as drawn: the compiler has already answered.
    """
    space = _space(choices=[16, 24, 32, 48, 64, 128])
    trials = [_trial({"BLOCK_M": 16}, 4.0), _trial({"BLOCK_M": 24}, 3.5),
              _trial({"BLOCK_M": 32}, 3.0),
              _trial({"BLOCK_M": 64}, None, status="fail", failure="runtime_error"),
              _trial({"BLOCK_M": 128}, None, status="fail", failure="infeasible_shared_memory")]
    g = sg.SlopeGuide(space=space)
    out = g.suggest(_stats(space, trials), trials, set())
    assert [s.knob_value for s in out] == [48], \
        "64 was drawn (and failed), so the next undrawn value toward the wall is 48"


def test_only_declared_choices_are_ever_proposed():
    """Optuna does NOT raise on a queued value outside the distribution -- it warns and samples that
    knob normally, which would silently turn a suggestion into an ordinary draw. So the proposal has to
    come from the domain's own list, never from arithmetic on the measured values.
    """
    space = _space(choices=[16, 24, 32, 64, 128])
    _, trials = _hard_wall_case()
    g = sg.SlopeGuide(space=space)
    for s in g.suggest(_stats(space, trials), trials, set()):
        assert s.knob_value in space.domain(s.knob).choices
        for knob, value in s.values.items():
            assert value in space.domain(knob).choices


def test_every_other_knob_stays_at_the_incumbents_value():
    """One knob varied from the measured optimum. From the optimum 6 of 6 hard walls attribute to a
    single knob; from the space's default corner only 1 of 6 -- a suggestion built on a slow point would
    probe a place where every other knob is small and the walled one is not binding.
    """
    space = _space(choices=[16, 24, 32, 64, 128], extra={"NUM_WARPS": [4, 8]})
    trials = [
        _trial({"BLOCK_M": 16, "NUM_WARPS": 4}, 4.0),
        _trial({"BLOCK_M": 24, "NUM_WARPS": 8}, 3.5),
        _trial({"BLOCK_M": 32, "NUM_WARPS": 8}, 3.0),      # the incumbent
        _trial({"BLOCK_M": 128, "NUM_WARPS": 8}, None, status="fail",
               failure="infeasible_shared_memory"),
    ]
    g = sg.SlopeGuide(space=space)
    out = g.suggest(_stats(space, trials), trials, set())
    assert len(out) == 1
    assert out[0].values == {"BLOCK_M": 64, "NUM_WARPS": 8}


def test_the_incumbent_is_the_robust_objective_and_never_the_luckiest_sample():
    """`robust_ms` (median else mean) is this project's measured objective: 93.2% rank-correctness at
    n=20 against the mean's 64.8%, while `min` reports the luckiest sample (biased +9.8% to +156%). A
    second rule here could name a different winner than the one the rest of the framework selects.

    The trap this fixture had to avoid: the "lucky" trial must sit OUTSIDE the last three measured
    values, or its inflated median breaks the monotone tail and the wall is filtered out before the
    incumbent is ever chosen -- which would make the test pass for the wrong reason and then fail as
    soon as anything changed.
    """
    space = _space(choices=[8, 16, 24, 32, 64, 128], extra={"NUM_WARPS": [4, 8]})
    # BLOCK_M=8 holds the lowest single SAMPLE (min 0.5) and the highest MEDIAN (9.0), beside
    # NUM_WARPS=4. Every other trial runs NUM_WARPS=8. So a `min`-based incumbent would carry
    # NUM_WARPS=4 into the suggestion and a robust one carries 8.
    lucky = TrialRecord(
        trial_id="t-lucky", candidate_id="cand-x", space_id="sp-x",
        params=ParamSet(values={"BLOCK_M": 8, "NUM_WARPS": 4}), status="complete",
        latency_ms=LatencyStats(mean=9.0, std=3.0, min=0.5, max=12.0, n_samples=20, median=9.0),
        profile=ProfileRecord(n_regs=200, n_spills=0, shared_bytes=32768, num_warps=4,
                              num_stages=2, compile_s=0.4))
    trials = [
        lucky,
        _trial({"BLOCK_M": 16, "NUM_WARPS": 8}, 4.0),
        _trial({"BLOCK_M": 24, "NUM_WARPS": 8}, 3.5),
        _trial({"BLOCK_M": 32, "NUM_WARPS": 8}, 3.0),
        _trial({"BLOCK_M": 128, "NUM_WARPS": 8}, None, status="fail",
               failure="infeasible_shared_memory"),
    ]
    g = sg.SlopeGuide(space=space)
    out = g.suggest(_stats(space, trials), trials, set())
    assert out, "the wall must survive the slope filter, or this tests nothing"
    assert out[0].values["NUM_WARPS"] == 8, \
        "the incumbent must be chosen on robust_ms, not on the luckiest single sample"


def test_a_point_already_drawn_is_not_proposed_even_when_the_value_is_undrawn():
    """The knob's value can be undrawn while the whole POINT has been asked -- the value appeared
    beside a different partner. Enqueueing it would consume one of the tuner's bounded re-asks and
    produce no trial, while the log recorded a successful suggestion.
    """
    space = _space(choices=[16, 24, 32, 64, 128])
    _, trials = _hard_wall_case()
    already = ParamSet(values={"BLOCK_M": 64}).key()
    g = sg.SlopeGuide(space=space)
    assert g.suggest(_stats(space, trials), trials, {already}) == []
    assert g.n_skipped_already_proposed == 1


# ---------------------------------------------------------------------------------------------
# (3) the dose, and the slope filter that decides a wall is worth acting on
# ---------------------------------------------------------------------------------------------


def _two_walls() -> tuple[ParameterSpace, list[TrialRecord]]:
    """Two knobs, both truncated at 128, with deliberately DIFFERENT slopes.

    A full 3x3 factorial rather than a diagonal, and that is not cosmetic: on a diagonal each knob value
    appears exactly once, so a single trial's latency lands in both knobs' median curves and one
    off-trend point destroys monotonicity for both. My first attempt did exactly that and produced ZERO
    walls -- the ordering assertion then passed on an empty list, which is
    `probe-needs-a-positive-control` in its purest form.

    Latency is `10 - 0.1*i - 3*j` over (SHALLOW index i, STEEP index j). Marginalising gives
    SHALLOW: 7.0 / 6.9 / 6.8 (gain 2.9%) and STEEP: 9.9 / 6.9 / 3.9 (gain 60.6%) -- both monotone toward
    the wall, an order of magnitude apart, so the sort has something to get wrong.
    """
    vals = [16, 24, 32]
    space = ParameterSpace(
        space_id="sp-x", candidate_id="cand-x", source_sha="0" * 8,
        domains=[ParamDomain(name=n, kind="int", choices=[16, 24, 32, 64, 128])
                 for n in ("SHALLOW", "STEEP")])
    trials = [_trial({"SHALLOW": a, "STEEP": b}, 10.0 - 0.1 * i - 3.0 * j)
              for i, a in enumerate(vals) for j, b in enumerate(vals)]
    trials.append(_trial({"SHALLOW": 128, "STEEP": 128}, None, status="fail",
                         failure="infeasible_shared_memory"))
    return space, trials


def test_the_cap_bounds_the_dose():
    """12 of 34 early walls VANISH by the end of tuning, so an uncapped mechanism could spend a large
    share of a 40-trial budget on knobs the final measurement does not support.
    """
    space, trials = _two_walls()
    stats = _stats(space, trials)
    assert len(sg.SlopeGuide(space=space, max_enqueued_per_recompute=2)
               .suggest(stats, trials, set())) == 2, \
        "both walls must be findable, or the cap test proves nothing"
    assert len(sg.SlopeGuide(space=space, max_enqueued_per_recompute=1)
               .suggest(stats, trials, set())) == 1
    assert sg.SlopeGuide(space=space, max_enqueued_per_recompute=0
                         ).suggest(stats, trials, set()) == []


def test_a_flat_or_worsening_wall_is_not_acted_on():
    """The shipping slope filter: a wall real but with latency FLAT or WORSENING toward it is worthless
    -- lifting it buys nothing (measured: drops 3 of 6 walls, worst at -54.8%). Spending a trial toward
    it is the sampling analogue of spending a rewrite on a non-problem.
    """
    space = _space(choices=[16, 24, 32, 64, 128])
    trials = [_trial({"BLOCK_M": 16}, 3.0), _trial({"BLOCK_M": 24}, 3.5),
              _trial({"BLOCK_M": 32}, 4.0),          # worsening toward the wall
              _trial({"BLOCK_M": 128}, None, status="fail",
                     failure="infeasible_shared_memory")]
    g = sg.SlopeGuide(space=space)
    assert g.suggest(_stats(space, trials), trials, set()) == []
    assert g.n_skipped_no_wall == 1


def test_the_slope_filter_is_the_shipping_one_and_not_a_second_copy():
    """`select_for_probing` is what decides a wall is worth acting on, and the report and the prompt
    already agree with it. A private re-implementation of `monotone and tail_gain_pct > 0` here would be
    free to drift from that -- the trap recorded as `a-fix-applied-to-one-caller-leaves-its-siblings`.
    """
    text = io.open(SRC / "tuning" / "slope_guide.py", encoding="utf-8").read()
    tree = ast.parse(text)
    calls = {ast.unparse(n.func) for n in ast.walk(tree) if isinstance(n, ast.Call)}
    assert any("select_for_probing" in c for c in calls), \
        "the shipping filter must be CALLED, not paraphrased"
    # An AST walk, not a text search: the substring `tail_gain_pct` appears legitimately (the field is
    # read to sort and to report). What must not exist is a COMPARISON in this module that re-decides
    # worthiness from it.
    for node in ast.walk(tree):
        if isinstance(node, ast.Compare):
            src = ast.unparse(node)
            assert "tail_gain_pct" not in src, \
                f"worthiness must come from select_for_probing, not from {src!r}"


def test_walls_are_ordered_steepest_first():
    """The mechanism spends a bounded number of trials, so which wall gets them is the whole allocation
    decision. Two walls with a 20x slope difference, and the steep one has to come first.
    """
    space, trials = _two_walls()
    out = sg.SlopeGuide(space=space, max_enqueued_per_recompute=2).suggest(
        _stats(space, trials), trials, set())
    assert [s.knob for s in out] == ["STEEP", "SHALLOW"]
    gains = [s.tail_gain_pct for s in out]
    assert gains == sorted(gains, reverse=True)
    assert gains[0] > 5 * gains[1], \
        "the fixture must make the two slopes clearly different, or the sort is untested"


def test_one_row_per_knob_even_when_both_criteria_name_it():
    """Two criteria naming the same knob is AGREEMENT, not two independent reasons to spend two trials
    on it. Here both name the same SIDE, so the count is the only observable -- the opposite-sides case
    is `test_the_same_knob_is_pushed_once_even_when_the_two_criteria_disagree_on_the_side`.
    """
    space = _space(choices=[16, 24, 32, 64, 128])
    trials = [
        _trial({"BLOCK_M": 16}, 4.0, spills=0),
        _trial({"BLOCK_M": 24}, 3.5, spills=0),
        _trial({"BLOCK_M": 32}, 3.0, spills=40),           # the incumbent spills => soft applicable
        _trial({"BLOCK_M": 128}, None, status="fail", failure="infeasible_shared_memory"),
    ]
    stats = _stats(space, trials)
    # Both criteria really do fire on this knob -- otherwise the dedup is untested.
    from kernel_optimizer.evaluation import soft_wall, wall_attribution
    refused = [{"BLOCK_M": 128}]
    assert [w.param for w in wall_attribution.find_walls(stats, refused)] == ["BLOCK_M"]
    assert [w.param for w in soft_wall.find_soft_walls(stats, trials).walls] == ["BLOCK_M"]

    g = sg.SlopeGuide(space=space, max_enqueued_per_recompute=4, use_soft_wall=True)
    out = g.suggest(stats, trials, set())
    assert [s.knob for s in out] == ["BLOCK_M"], "the same knob must not be pushed twice"


def test_a_known_refusal_bounds_the_target_even_when_the_soft_slope_wins_the_dedup():
    """The defect this pins was live until target resolution was moved AFTER the dedup.

    A knob can carry both walls, and the row that survives dedup is whichever criterion reported the
    steeper slope. The soft criterion has no refusal to bound it, so when IT wins, a
    resolve-then-dedup order would push the knob to its extreme declared value -- which for this knob is
    exactly the value the compiler already refused. The mechanism would then have manufactured its own
    `infeasible_shared_memory` trial: a trial spent to be told what it already knew, and one that makes
    the wall look more real in the log than the measurement supports.

    HOW THE TWO SLOPES COME APART, because they usually do not: both read a three-value tail, so on a
    knob where every trial has both measurements they are EQUAL and the stable sort keeps the hard row.
    They differ when the two criteria see different value sets -- here `BLOCK_M=24` has no `n_spills`
    reading (the profile field is absent, which happens for real: `ProfileRecord` fields are optional and
    a CUDA-backend trial has no Triton metadata at all), so it enters the hard wall's latency table and
    not the soft wall's spill table. Hard's tail is then 24/32/64 (9.1%) and soft's is 16/32/64 (85.0%).
    """
    space = _space(choices=[16, 24, 32, 64, 96, 128])
    trials = [
        _trial({"BLOCK_M": 16}, 20.0, spills=0),
        _trial({"BLOCK_M": 24}, 3.3, spills=None),         # no spill reading -> soft cannot see it
        _trial({"BLOCK_M": 32}, 3.2, spills=20),
        _trial({"BLOCK_M": 64}, 3.0, spills=40),
        _trial({"BLOCK_M": 128}, None, status="fail", failure="infeasible_shared_memory"),
    ]
    stats = _stats(space, trials)
    g = sg.SlopeGuide(space=space, max_enqueued_per_recompute=2, use_soft_wall=True)
    out = g.suggest(stats, trials, set())
    assert len(out) == 1
    assert out[0].source == "soft_wall", \
        "this fixture is only meaningful if the SOFT row is the one that survives dedup"
    assert out[0].knob_value == 96, \
        "the compiler refused 128; the surviving row must respect that whatever criterion named it"


def _low_hard_and_high_soft() -> tuple[ParameterSpace, list[TrialRecord]]:
    """ONE knob carrying a LOW hard wall and a HIGH soft wall at the same time.

    A U-shaped latency curve does it: 16 -> 128 reads 1.0 / 2.0 / 3.0 / 0.5, so the low tail
    (64/32/16) is monotone improving toward the refused 8 (hard wall, side "low", gain 66.7%), and the
    top of the spill curve is also improving (soft wall, side "high", gain 75.0%). Both criteria fire on
    BLOCK_M with different sides, and the soft one has the steeper slope -- which is what makes this the
    fixture for the cross-side bound AND for the dedup.

    Physically ordinary rather than contrived: 128 is the incumbent, and a knob that is bad in the middle
    of its range and good at both ends is what a tile size does when one end fits in cache and the other
    amortises launch overhead.

    `12` is declared and never drawn ON PURPOSE. Without an undrawn choice on the LOW side too, the hard
    row resolves to None, only one row survives, and dropping the dedup changes nothing observable -- the
    revert-check said NOT CAUGHT for exactly that reason. With 12 present, no dedup means two trials on
    one knob pointing in opposite directions (256 and 12), which is the behaviour worth guarding.
    """
    space = _space(choices=[8, 12, 16, 32, 64, 128, 256])
    trials = [
        _trial({"BLOCK_M": 16}, 1.0, spills=0),
        _trial({"BLOCK_M": 32}, 2.0, spills=10),
        _trial({"BLOCK_M": 64}, 3.0, spills=20),
        _trial({"BLOCK_M": 128}, 0.5, spills=40),      # the incumbent
        _trial({"BLOCK_M": 8}, None, status="fail", failure="infeasible_shared_memory"),
    ]
    return space, trials


def test_a_low_side_refusal_does_not_bound_a_high_side_proposal():
    """The refusal bound applies only when the SIDES agree. A knob truncated at its bottom puts no
    ceiling on a high-side proposal, and applying the bound across sides would exclude values the
    compiler never objected to -- narrowing the space, the one thing this mechanism must never do.

    On this fixture the surviving row is the HIGH soft wall (75.0% against the low hard wall's 66.7%),
    while the refusal is at 8 on the LOW side. Bounding across sides would drop every choice at or below
    8 -- which is none of them -- so the observable damage is the reverse: a `side == "low"` bound applied
    to a high-side scan keeps only values BELOW 8 and the knob would be skipped entirely.
    """
    space, trials = _low_hard_and_high_soft()
    g = sg.SlopeGuide(space=space, use_soft_wall=True)
    out = g.suggest(_stats(space, trials), trials, set())
    assert len(out) == 1, "a low-side refusal must not cause the high-side proposal to be skipped"
    assert out[0].source == "soft_wall" and out[0].knob_value == 256


def test_the_same_knob_is_pushed_once_even_when_the_two_criteria_disagree_on_the_side():
    """Two criteria naming the same knob is AGREEMENT that the knob matters, not two independent reasons
    to spend two trials on it -- and least of all a reason to spend one trial going UP and another going
    DOWN, which is exactly what dropping the dedup produces on this fixture (256 from the soft row, 12
    from the hard row).

    The previous version of this test used a knob where both criteria named the same SIDE and resolved to
    the same target, so the two rows collapsed to one value and dropping the dedup changed nothing
    observable -- the revert-check said NOT CAUGHT. Here the rows point in opposite directions and reach
    different declared choices, so the COUNT is the behaviour.
    """
    space, trials = _low_hard_and_high_soft()
    g = sg.SlopeGuide(space=space, max_enqueued_per_recompute=4, use_soft_wall=True)
    out = g.suggest(_stats(space, trials), trials, set())
    assert [s.knob for s in out] == ["BLOCK_M"], \
        "one knob, one suggestion -- not one per criterion, and not one per direction"
    assert g.snapshot()["knobs_pushed"] == {"BLOCK_M": 1}
    # The row that survives is the steeper one, and the fixture must really offer both.
    assert out[0].source == "soft_wall" and out[0].knob_value == 256
    rows = g._walls(_stats(space, trials), trials, {"BLOCK_M": 128})
    assert len(rows) == 1, "the dedup happens inside _walls, so this is where the count is decided"


def test_a_proposal_is_a_step_from_the_incumbent_toward_the_wall():
    """"Toward the wall" is defined relative to the OPTIMUM. Without that anchor it degenerates into
    "any undrawn value on that side of the range", which can point the opposite way.

    Both directions of the failure, and it was live until the incumbent was resolved before the walls:

      LOW wall, undrawn value only ABOVE the incumbent.  The refusal is at 8 and every choice between it
      and the optimum (16/32/64/128) has been drawn, so the correct answer is to SKIP the knob -- there is
      nothing left to learn on the walled side. Unanchored, the ascending scan runs past the incumbent to
      256: the knob's TOP value, a step directly AWAY from the wall, recorded in the log as a step toward
      it.

      HIGH wall, undrawn value BELOW the incumbent.  16/32/64 are drawn with 64 the optimum and the spill
      curve rising, so the wall is above -- but 8 and 24 are undrawn, and an unanchored descending scan
      picks 24, i.e. it proposes moving DOWN the knob in order to explore its upper end. Anchored, nothing
      above 64 exists in that space and the knob is correctly skipped.

    Asserted by comparing the anchored call against the unanchored one, so the test states what the anchor
    CHANGES rather than only that today's output looks sensible. Note the space here deliberately omits the
    `12` that `_low_hard_and_high_soft` declares: with an undrawn choice available on the walled side, BOTH
    scans find it and the anchor makes no difference -- which is a property of that fixture, not of the
    anchor, and testing it there would have proved nothing.
    """
    # (a) LOW wall, nothing undrawn between the refusal and the optimum.
    space = _space(choices=[8, 16, 32, 64, 128, 256])
    trials = [
        _trial({"BLOCK_M": 16}, 1.0, spills=0),
        _trial({"BLOCK_M": 32}, 2.0, spills=10),
        _trial({"BLOCK_M": 64}, 3.0, spills=20),
        _trial({"BLOCK_M": 128}, 0.5, spills=40),
        _trial({"BLOCK_M": 8}, None, status="fail", failure="infeasible_shared_memory"),
    ]
    stats = _stats(space, trials)
    g = sg.SlopeGuide(space=space)
    assert g._walls(stats, trials, {"BLOCK_M": 128}) == []
    assert [r[1] for r in g._walls(stats, trials, {})] == [256], \
        "without the anchor the low-side wall proposes the knob's TOP value"
    assert g.suggest(stats, trials, set()) == [], \
        "the shipping path must take the anchored answer"

    # (b) HIGH wall: the anchor stops a proposal below the optimum.
    space2 = _space(choices=[8, 16, 24, 32, 64])
    trials2 = [_trial({"BLOCK_M": 16}, 4.0, spills=0), _trial({"BLOCK_M": 32}, 3.5, spills=10),
               _trial({"BLOCK_M": 64}, 3.0, spills=40)]
    stats2 = _stats(space2, trials2)
    g2 = sg.SlopeGuide(space=space2, use_soft_wall=True)
    assert g2._walls(stats2, trials2, {"BLOCK_M": 64}) == []
    assert [r[1] for r in g2._walls(stats2, trials2, {})] == [24], \
        "without the anchor the high-side wall proposes a value BELOW the optimum"


def test_the_anchor_still_finds_an_undrawn_value_on_the_walled_side():
    """The positive control for the previous test, and it is necessary: an anchor implemented as "return
    None for a low-side wall" would pass every assertion there. On the fixture that DOES declare an undrawn
    choice between the refusal and the optimum, the anchored answer is that choice.
    """
    space, trials = _low_hard_and_high_soft()
    g = sg.SlopeGuide(space=space)
    assert [r[1] for r in g._walls(_stats(space, trials), trials, {"BLOCK_M": 128})] == [12], \
        "12 is undrawn and lies between the refused 8 and the optimum 128"


# ---------------------------------------------------------------------------------------------
# (4) the soft criterion is opt-in, and its own switch is not enough
# ---------------------------------------------------------------------------------------------


def test_the_soft_criterion_is_off_by_default():
    """A spill wall has NO independent probe confirmation. It must not steer the sampler unless asked
    for explicitly -- and `use_soft_wall` alone is not enough (see the orchestrator test below).
    """
    assert sg.SlopeGuide(space=_space()).use_soft_wall is False


def test_a_spill_wall_can_steer_when_it_is_turned_on():
    """The positive control for the switch: without this, `use_soft_wall=False` passing every test would
    be indistinguishable from a soft path that never works at all.

    NOTE what the target is, and it differs from the hard case by design. A hard wall has a REFUSED value
    that bounds the proposal from above (proposing it or beyond would buy the same refusal back). A soft
    wall has no refusal at all -- nothing was refused, the range is merely clipped -- so the furthest
    undrawn value on the spilling side is the legitimate target, extreme end included. That is the
    difference between "the compiler says no" and "the compiler says yes but slowly".
    """
    space = _space(choices=[16, 24, 32, 64, 128])
    trials = [_trial({"BLOCK_M": 16}, 4.0, spills=0),
              _trial({"BLOCK_M": 24}, 3.5, spills=8),
              _trial({"BLOCK_M": 32}, 3.0, spills=40)]     # incumbent spills, curve rises, lat improves
    stats = _stats(space, trials)
    assert sg.SlopeGuide(space=space).suggest(stats, trials, set()) == []
    on = sg.SlopeGuide(space=space, use_soft_wall=True).suggest(stats, trials, set())
    assert len(on) == 1 and on[0].source == "soft_wall" and on[0].knob == "BLOCK_M"
    assert on[0].knob_value == 128, \
        "a soft wall refuses nothing, so the furthest undrawn value on the spilling side is legal"


def test_a_hard_and_a_soft_wall_on_the_same_knob_target_different_values():
    """The asymmetry above, asserted directly so it cannot be "fixed" into uniformity by accident: the
    hard wall stops BELOW the refusal, the soft wall does not, because only one of them has a refusal.
    """
    space = _space(choices=[16, 24, 32, 64, 128])
    common = [_trial({"BLOCK_M": 16}, 4.0, spills=0), _trial({"BLOCK_M": 24}, 3.5, spills=8),
              _trial({"BLOCK_M": 32}, 3.0, spills=40)]
    soft = sg.SlopeGuide(space=space, use_soft_wall=True).suggest(
        _stats(space, common), common, set())
    hard_trials = common + [_trial({"BLOCK_M": 128}, None, status="fail",
                                   failure="infeasible_shared_memory")]
    hard = sg.SlopeGuide(space=space).suggest(_stats(space, hard_trials), hard_trials, set())
    assert soft[0].knob_value == 128 and hard[0].knob_value == 64


def test_the_source_of_every_suggestion_is_recorded():
    """Hard and soft walls are pooled by slope here because both only pick a DIRECTION -- but they carry
    different evidence (one has a compiler confirmation, one does not), so the log must keep them
    separable for a later reading.
    """
    space = _space(choices=[16, 24, 32, 64, 128])
    trials = [_trial({"BLOCK_M": 16}, 4.0, spills=0), _trial({"BLOCK_M": 24}, 3.5, spills=8),
              _trial({"BLOCK_M": 32}, 3.0, spills=40)]
    g = sg.SlopeGuide(space=space, use_soft_wall=True)
    out = g.suggest(_stats(space, trials), trials, set())
    assert out[0].payload()["source"] == "soft_wall"
    assert g.snapshot()["sources"] == {"soft_wall": 1}


# ---------------------------------------------------------------------------------------------
# (5) parity: the budget, the space, and the guard are untouched
# ---------------------------------------------------------------------------------------------


def test_an_enqueued_point_counts_against_the_same_budget():
    """"Only adds candidate points; does not change `trials_per_space`" has to be TRUE, not asserted:
    the enqueued point is consumed by the ordinary `ask()` and consumes one unit of the same budget.
    """
    from kernel_optimizer.tuning.tpe import OptunaTPETuner

    space = _space(choices=[16, 24, 32, 64, 128])
    t = OptunaTPETuner(space, guard_ok=lambda p: True, budget=3, seed=0)
    assert t.enqueue(ParamSet(values={"BLOCK_M": 64})) is None
    drawn = []
    while True:
        got = t.ask()
        if got is None:
            break
        tid, params = got
        drawn.append(params.values["BLOCK_M"])
        t.tell(tid, _trial(dict(params.values), 2.0))
    assert len(drawn) == 3, "the budget must be exactly what it was without S7"
    assert 64 in drawn, "the enqueued point must actually be drawn"


def test_the_enqueued_point_is_drawn_next():
    """Otherwise the guidance could arrive dozens of trials late, which is the very defect S7 exists to
    fix -- and its absence would be invisible in a log that only says "enqueued".
    """
    from kernel_optimizer.tuning.tpe import OptunaTPETuner

    space = _space(choices=[16, 24, 32, 64, 128])
    t = OptunaTPETuner(space, guard_ok=lambda p: True, budget=6, seed=0)
    tid, params = t.ask()
    t.tell(tid, _trial(dict(params.values), 2.0))
    assert t.enqueue(ParamSet(values={"BLOCK_M": 64})) is None
    _, nxt = t.ask()
    assert nxt.values["BLOCK_M"] == 64


def test_the_guard_still_refuses_an_enqueued_point():
    """S7 proposes; the space's own constraints still decide. A queued point forced past the guard would
    be S7 overriding the candidate's declared feasibility -- and it would then be measured and possibly
    counted as the winner.
    """
    from kernel_optimizer.tuning.tpe import OptunaTPETuner

    space = _space(choices=[16, 24, 32, 64, 128])
    t = OptunaTPETuner(space, guard_ok=lambda p: p.values["BLOCK_M"] != 64, budget=4, seed=0)
    assert t.enqueue(ParamSet(values={"BLOCK_M": 64})) == "guard_rejected"


def test_enqueueing_an_already_drawn_point_is_refused_rather_than_silently_wasted():
    """It would come back from `ask()`, be told PRUNED by the dedup branch, and consume one of the
    bounded `max_guard_rejects` re-asks -- while the caller saw a successful enqueue and a trial that
    never appeared.
    """
    from kernel_optimizer.tuning.tpe import OptunaTPETuner

    space = _space(choices=[16, 24, 32, 64, 128])
    t = OptunaTPETuner(space, guard_ok=lambda p: True, budget=4, seed=0)
    tid, params = t.ask()
    t.tell(tid, _trial(dict(params.values), 2.0))
    assert t.enqueue(params) == "already_drawn"


def test_the_guide_never_shrinks_a_domain_or_adds_a_constraint():
    """The one property that makes this a sampling hint rather than a space change. Asserted on the
    OBJECT after a real suggest, because "only adds points" is the claim the whole comparability
    argument rests on.
    """
    space = _space(choices=[16, 24, 32, 64, 128], extra={"NUM_WARPS": [4, 8]})
    before = space.model_dump_json()
    _, trials = _hard_wall_case()
    trials = [t.model_copy(update={"params": ParamSet(
        values={**t.params.values, "NUM_WARPS": 8})}) for t in trials]
    sg.SlopeGuide(space=space, use_soft_wall=True).suggest(_stats(space, trials), trials, set())
    assert space.model_dump_json() == before
    assert [len(d.choices) for d in space.domains] == [5, 2]
    assert space.constraints == []


def test_drawn_keys_is_a_copy():
    """A caller that mutated the tuner's dedup set would change what `ask()` treats as a duplicate, and
    the extra PRUNED draws would look like the sampler exhausting the space.
    """
    from kernel_optimizer.tuning.tpe import OptunaTPETuner

    t = OptunaTPETuner(_space(), guard_ok=lambda p: True, budget=4, seed=0)
    tid, params = t.ask()
    keys = t.drawn_keys()
    keys.add("not-a-real-key")
    assert "not-a-real-key" not in t.drawn_keys()
    assert params.key() in t.drawn_keys()


# ---------------------------------------------------------------------------------------------
# (6) robustness: the guide must not be able to end a candidate, and it must read replayed trials
# ---------------------------------------------------------------------------------------------


def test_replayed_dict_trials_read_identically():
    """`robust_ms` is a @property and never serialized: a reader that looked the name up in a replayed
    trial would treat every trial as untimed and silently suggest nothing -- a plausible constant, which
    is the failure mode that hides (`a-constant-reading-is-a-broken-probe`).
    """
    space, trials = _hard_wall_case()
    stats = _stats(space, trials)
    as_dicts = [t.model_dump() for t in trials]
    a = sg.SlopeGuide(space=space).suggest(stats, trials, set())
    b = sg.SlopeGuide(space=space).suggest(stats, as_dicts, set())
    assert [(s.knob, s.knob_value, s.source) for s in a] == \
           [(s.knob, s.knob_value, s.source) for s in b]
    assert a and "robust_ms" not in str(as_dicts[0].get("latency_ms")), \
        "the fixture must be a real serialization, where robust_ms is absent"


def test_an_incumbent_that_is_not_a_point_of_this_space_is_skipped():
    """Reachable after an expansion re-tune: `crun.trials` accumulates across a candidate's spaces, so
    the incumbent can predate the space being tuned. Optuna does not raise on either defect -- a missing
    knob is sampled, and an undeclared value is warned-and-sampled -- so both would silently degrade a
    suggestion into an ordinary draw while the log recorded a suggestion.
    """
    space = _space(choices=[16, 24, 32, 64, 128], extra={"NUM_WARPS": [4, 8]})
    _, trials = _hard_wall_case()          # these trials have no NUM_WARPS at all
    g = sg.SlopeGuide(space=space)
    assert g.suggest(_stats(space, trials), trials, set()) == []
    assert g.n_skipped_incomplete_incumbent == 1

    # ...and the value-not-declared half of the same guard.
    space2 = _space(choices=[16, 24, 32, 64, 128], extra={"NUM_WARPS": [4, 8]})
    trials2 = [t.model_copy(update={"params": ParamSet(
        values={**t.params.values, "NUM_WARPS": 16})}) for t in trials]      # 16 not in [4, 8]
    g2 = sg.SlopeGuide(space=space2)
    assert g2.suggest(_stats(space2, trials2), trials2, set()) == []
    assert g2.n_skipped_incomplete_incumbent == 1


def test_no_completed_trial_yields_nothing_rather_than_raising():
    space = _space(choices=[16, 24, 32, 64, 128])
    trials = [_trial({"BLOCK_M": 128}, None, status="fail",
                     failure="infeasible_shared_memory")]
    assert sg.SlopeGuide(space=space).suggest(_stats(space, trials), trials, set()) == []


def test_the_guide_is_pure_and_touches_no_gpu():
    """The whole mechanism is affordable only because `analyze` and both wall finders are pure. A worker
    call inside the tuning loop every 10 trials would be a cost, not a hint.
    """
    text = io.open(SRC / "tuning" / "slope_guide.py", encoding="utf-8").read()
    for banned in ("worker", "run_job", "subprocess", "quick_test", "torch", "open("):
        assert banned not in text, f"{banned!r} must not appear in a pure recompute"


# ---------------------------------------------------------------------------------------------
# (7) the wiring: three separate switches, off by default, journalled
# ---------------------------------------------------------------------------------------------


def test_all_three_switches_exist_separately_and_the_mechanism_is_off():
    """NOT one boolean, and the reason is P4: if the mechanism fires and the run does not improve,
    `enabled` / `recompute_every` / `max_enqueued_per_recompute` are what separate "the timing was wrong"
    from "the dose was wrong" from "the signal is not useful". A single flag makes that negative result
    uninterpretable.
    """
    from kernel_optimizer.config import AppConfig

    cfg = AppConfig().v3.slope_guide
    assert cfg.enabled is False
    assert cfg.recompute_every == 10
    assert cfg.max_enqueued_per_recompute == 2
    assert cfg.use_soft_wall is False


def test_the_off_path_creates_no_guide_at_all():
    """Off must be the LITERAL old path: no guide object, so no recompute, no enqueue and no event.
    A guide constructed and then not consulted would still change the tuner's construction.
    """
    from kernel_optimizer.config import AppConfig
    from kernel_optimizer.control import orchestrator as orch_mod

    o = object.__new__(orch_mod.Orchestrator)
    o.cfg = AppConfig()
    assert o._make_slope_guide(_space()) is None


def test_the_switches_reach_the_guide():
    """The positive control for the previous test: without it, a `_make_slope_guide` that returned None
    unconditionally would pass every off-path assertion.
    """
    from kernel_optimizer.config import AppConfig
    from kernel_optimizer.control import orchestrator as orch_mod

    cfg = AppConfig()
    cfg.v3.slope_guide.enabled = True
    cfg.v3.slope_guide.recompute_every = 7
    cfg.v3.slope_guide.max_enqueued_per_recompute = 3
    o = object.__new__(orch_mod.Orchestrator)
    o.cfg = cfg
    g = o._make_slope_guide(_space())
    assert g is not None
    assert (g.recompute_every, g.max_enqueued_per_recompute) == (7, 3)


def test_the_guide_cannot_end_a_candidate():
    """Same rule as the wall attribution and `_reconcile_round`: a sampling HINT that raised would
    present a bookkeeping defect as a candidate defect, mid-tuning, with trials already spent.
    """
    from kernel_optimizer.config import AppConfig
    from kernel_optimizer.control import orchestrator as orch_mod

    class _Store:
        def __init__(self):
            self.events = []

        def append(self, t, p):
            self.events.append((t, p))

    class _Boom:
        def analyze(self, space, trials):
            raise RuntimeError("analyzer exploded")

    cfg = AppConfig()
    cfg.v3.slope_guide.enabled = True
    o = object.__new__(orch_mod.Orchestrator)
    o.cfg = cfg
    o.store = _Store()
    o.deps = type("D", (), {"stats_analyzer": _Boom()})()
    space, trials = _hard_wall_case()
    crun = object.__new__(orch_mod.CandidateRun)
    crun.candidate = type("C", (), {"candidate_id": "cand-x"})()
    crun.trials = trials
    guide = o._make_slope_guide(space)
    o._slope_guide_step(crun, space, object(), guide)      # must not raise
    assert [t for t, _ in o.store.events] == ["SLOPE_GUIDE_FAILED"]
    assert "analyzer exploded" in o.store.events[0][1]["error"]


def test_every_step_is_journalled_with_what_fired_and_what_was_refused():
    """"the mechanism fired" and "the mechanism fired and the sampler drew the point" are different
    claims, and only the log can separate them afterwards. Driven through the REAL
    `_slope_guide_step` with a real store, because a source-order assertion cannot tell whether the
    append is conditional -- that exact defect survived a source-order guard in item 2 and was caught
    only by the revert-check.
    """
    from kernel_optimizer.config import AppConfig
    from kernel_optimizer.control import orchestrator as orch_mod
    from kernel_optimizer.models.core import DeviceLimits
    from kernel_optimizer.tuning.stats import TuningStatsAnalyzer
    from kernel_optimizer.tuning.tpe import OptunaTPETuner

    class _Store:
        def __init__(self):
            self.events = []

        def append(self, t, p):
            self.events.append((t, p))

    def _run(guard_ok):
        cfg = AppConfig()
        cfg.v3.slope_guide.enabled = True
        o = object.__new__(orch_mod.Orchestrator)
        o.cfg = cfg
        o.store = _Store()
        o.deps = type("D", (), {"stats_analyzer": TuningStatsAnalyzer(DeviceLimits())})()
        space, trials = _hard_wall_case()
        crun = object.__new__(orch_mod.CandidateRun)
        crun.candidate = type("C", (), {"candidate_id": "cand-x"})()
        crun.trials = trials
        tuner = OptunaTPETuner(space, guard_ok=guard_ok, budget=10, seed=0)
        o._slope_guide_step(crun, space, tuner, o._make_slope_guide(space))
        return o.store.events

    # (a) accepted
    events = _run(lambda p: True)
    assert [t for t, _ in events] == ["SLOPE_GUIDE_STEP"]
    p = events[0][1]
    assert len(p["enqueued"]) == 1 and p["refused"] == []
    assert p["enqueued"][0]["knob"] == "BLOCK_M" and p["enqueued"][0]["knob_value"] == 64
    assert p["n_suggested"] == 1 and p["n_recomputes"] == 1

    # (b) refused by the guard -- still journalled, and distinguishable from (a)
    events = _run(lambda pp: pp.values.get("BLOCK_M") != 64)
    p2 = events[0][1]
    assert p2["enqueued"] == [] and len(p2["refused"]) == 1
    assert p2["refused"][0]["refused"] == "guard_rejected"
    assert p2["n_suggested"] == 1, \
        "the guide suggested it; the tuner refused it -- both facts have to be in the log"


def test_the_acceptance_measurement_is_recorded_next_to_the_predictions():
    """The 10.5% firing rate relocates this module, and the relocation must travel with the predictions.

    Not a style check. The measurement says 95.1% of recomputes decline because NO WALL EXISTS -- so S7
    does not fix C2's coverage problem, it inherits it. Without those numbers beside P2/P3, a P2/P3 failure
    after the 12h pair gets attributed to S7's dose or timing, which is precisely the misattribution the
    three separate switches exist to prevent.

    Asserted on the NUMBERS and the artifacts, not on the prose that explains them: a guard on wording is
    the `source-text-assertions-can-encode-the-bug` trap, and it would fail on a rephrasing while passing
    on a rewrite that dropped the point.
    """
    for path in (SRC / "tuning" / "slope_guide.py", SRC / "config.py"):
        text = io.open(path, encoding="utf-8").read()
        assert "10.5%" in text and "95.1%" in text, f"{path.name} must carry the firing rate"
        assert "P4" in text, f"{path.name} must carry the predictions the rate is read against"
        assert "s7_acceptance" in text or "s7-acceptance" in text, \
            f"{path.name} must name the probe the numbers come from, so they can be re-derived"
    assert (Path("docs") / "analysis-s7-acceptance-firing-rate.md").exists()
    for name in ("s7-acceptance-soft.txt", "s7-acceptance-hard.txt"):
        assert (Path("docs") / name).exists(), \
            f"the raw probe output {name} must be committed beside its analysis"


def test_the_hard_criterions_absence_on_this_corpus_is_recorded_as_absence_of_input():
    """0 of 152 candidates carry a shared-memory refusal, so the hard arm's zero is the absence of
    `find_walls`' only INPUT -- not evidence that the mechanism never fires. A zero recorded without that
    distinction is exactly `a-constant-reading-is-a-broken-probe`: it is plausible, and it is wrong.
    """
    text = io.open(Path("docs") / "s7-acceptance-hard.txt", encoding="utf-8").read()
    assert "UNANSWERABLE ON THIS CORPUS" in text
    assert "absence of INPUT" in text
    assert "candidates where it EVER fired:  0/" in text, \
        "the hard arm really did produce a zero here; the point is that it is labelled"


def test_the_snapshot_separates_the_ways_the_mechanism_can_decline():

    """`n_suggested: 0` alone is unreadable. A P4 reading needs to tell "the signal is not useful" from
    "the mechanism never got to fire", and the three skip counters are what distinguish no wall to act
    on / no undrawn value left / the point was already asked.
    """
    g = sg.SlopeGuide(space=_space())
    snap = g.snapshot()
    for key in ("recompute_every", "max_enqueued_per_recompute", "use_soft_wall", "n_recomputes",
                "n_suggested", "n_skipped_no_wall", "n_skipped_no_unmeasured_value",
                "n_skipped_already_proposed", "n_skipped_incomplete_incumbent",
                "knobs_pushed", "sources"):
        assert key in snap, key


def test_tuning_done_carries_the_snapshot_and_is_none_when_off():
    """None when off (a run reads exactly as before) versus a DICT with `n_suggested: 0` when on and
    nothing fired -- different states, and only the second says the mechanism ran. The same distinction
    S1b and S8 already keep.
    """
    text = io.open(SRC / "control" / "orchestrator.py", encoding="utf-8").read()
    tree = ast.parse(text)
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == "_tune")
    body = ast.get_source_segment(text, fn) or ""
    assert body.count('"slope_guide": slope') == 2, \
        "both TUNING_DONE branches must carry it, or a space with no best silently drops it"
    assert "guide.snapshot() if guide is not None else None" in body
