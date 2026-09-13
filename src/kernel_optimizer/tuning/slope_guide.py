"""S7: slope-guided sampling -- move the wall search INSIDE the tuning loop and enqueue candidates.

WHAT IS STRUCTURALLY DIFFERENT FROM 2e. 2e runs AFTER tuning ends, so its slope reaches only the
prompt. Measured consequence: on the three-arm run every one of the 80 trials was spent before the
wall was found, and only 25% of families ever received wall text. S7 recomputes the same pure
functions (`TuningStatsAnalyzer.analyze` + the wall finders -- none of which touches a GPU) every
`recompute_every` finished trials and uses `study.enqueue_trial` to push candidate points onto the
queue.

WHAT IT DOES AND DOES NOT DO TO THE SPACE. It only ADDS points. No value is removed, no domain is
narrowed, no constraint is added, and `trials_per_space` is unchanged -- an enqueued trial is consumed
by the same `ask()` and counts against the same `_asked` budget as any drawn one, so the budget is
identical and every configuration that was drawable stays drawable. What it DOES change is WHICH
points get drawn inside that budget, and that is why it needs a paired control arm while item 2 does
not.

THE MEASURED CASE FOR IT, AND THE MEASURED CASE AGAINST, BOTH FROM THE SAME REPLAY
(`scripts/probes/s7_feasibility_replay.py`: prefixes of each candidate's real trial ORDER, the shipping
finders run on each prefix, per-choice medians rebuilt FROM THE PREFIX so the replay cannot see what
the sampler could not):

  FOR      a wall is available by the halfway point on 25 of 34 candidates (74%), median earliest
           prefix 0.38 of the sequence -- so there IS budget left to act on it.
  AGAINST  the knob walled at that early prefix is still walled at the end on only 18 of 34 (53%).
           12 of 34 walls VANISH by 100% and 4 of 34 move to a DIFFERENT knob.

Both figures are from the SOFT criterion. The hard criterion is unanswerable on this host: 0 of 151
local candidates carry a single `infeasible_shared_memory` record, because every locally backed-up run
predates the shared-memory screen, and refused parameter sets are `find_walls`' entire input. So the
earlier "a probe-worthy wall by the halfway point with 40 trials left" figure stands UNVERIFIED here
and has to be re-measured on a run that carries refusals -- which the 12h pair will.

The 53% is the number this module is designed around, and it is why the guidance is ADVISORY,
RECOMPUTED and CAPPED rather than a commitment:

  * advisory   -- an enqueued point is a suggestion. Nothing is forbidden and nothing is locked in, so
                  a wrong guess costs at most the trials it enqueued.
  * recomputed -- the whole wall set is re-derived from scratch every `recompute_every` trials. A knob
                  that stops being walled stops being pushed, with no retraction bookkeeping to get
                  wrong.
  * capped     -- `max_enqueued_per_recompute` bounds the dose. With 12 of 34 early walls destined to
                  vanish, an uncapped mechanism could spend a large share of a 40-trial budget on knobs
                  the final measurement does not support.

WHY VANISHING IS EXPECTED RATHER THAN A DEFECT, and why it still has to be bounded. A hard wall is a
refused value OUTSIDE the measured range, and the range grows as sampling widens, so it catches up with
the refused value. A soft wall needs a monotone spill curve over at least three measured values, and an
early prefix holds fewer values, so its curve is more easily monotone by accident. Both mechanisms make
an early signal genuinely weaker than a late one. The mechanism cannot fix that -- only avoid betting
the budget on it.

THREE SWITCHES, NOT ONE BOOLEAN, and the reason is P4 below: if the mechanism fires and the run does not
improve, `enabled` / `recompute_every` / `max_enqueued_per_recompute` are what separate "the timing was
wrong" from "the dose was wrong" from "the signal is not useful". A single flag would make that negative
result uninterpretable. Wired exactly like S1b (`tuning/deweight.py`): a `None` collaborator means off
and takes the literal old path, and the effect is journalled so it is readable from the log rather than
argued from the code.

THE PREDICTIONS, DECLARED BEFORE THE RUN because a result nobody predicted can be read either way:

  P1  `RESOURCE_WALL_ATTRIBUTED`'s first appearance moves EARLIER, normalised to tuning progress.
  P2  probe-worthy walls per candidate INCREASE.
  P3  the share of families that receive wall text rises above 25%.
  P4  if P1-P3 hold and latency does NOT improve, then "knowing about the wall earlier" is not the
      bottleneck, and C2's problem is the conversion rate of the REWRITE itself. That is a result, not
      a failure, and being able to state it is why this module exists.

WHAT THE SHIPPING CODE ACTUALLY DID ON THE REAL CORPUS, and it changes how P2/P3 must be read
(`scripts/probes/s7_acceptance.py`, this module replayed over 19 L3 runs / 152 candidates / 790
recomputes, per-choice medians rebuilt from each prefix, soft criterion since the hard one has no input
here -- full analysis in `docs/analysis-s7-acceptance-firing-rate.md`):

  it fires on          16 of 152 candidates (10.5%), enqueueing 48 points in total
  it fires EARLY        median first firing at 0.25 of the candidate's own sequence -- on an 80-trial
                        candidate that leaves 60 trials, so P1's precondition holds
  on real slopes        median tail gain of the acted-on knob +34.8% (range +0.5% .. +66.6%)
  and it declines       751 of 790 recomputes (95.1%) because there is NO ACTIONABLE WALL AT ALL

That last line is the important one and it relocates this module. S7's premise was "the wall arrives too
late for the sampler". On this corpus the wall does not arrive late (0.25 is early); most of the time it
does not exist. So S7 does NOT fix C2's coverage problem -- it INHERITS it, which is the same conclusion
2e's 25% family coverage and item 5's "raising coverage means changing a criterion" already reached from
two other directions. P2 and P3 are therefore the predictions most likely to fail, and a failure there
means the criterion's applicability is the binding constraint, NOT this mechanism's timing or dose. That
reading has to be written down before the run, or it gets attributed to the dose afterwards.

THE OPEN PREMISE, stated because it is unresolved and this module cannot resolve it: that slope is a
good allocation prior at all. Against remaining tuning gain the five readings of boundary saturation
gave Spearman -0.11..+0.24, below the 0.43/0.52 incumbents; and slope versus coverage is rho = -0.431
with 14 of the 24 under-covered parameters already walled -- i.e. the naive "sample what has not been
sampled yet" finds most walls without any slope at all. S7 may therefore spend budget on
high-slope-but-low-absolute-gain dimensions and come out SLOWER. The 12h pair is the test, and P4 is how
a negative reads.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from kernel_optimizer.evaluation import soft_wall as soft_wall_mod
from kernel_optimizer.evaluation import wall_attribution
from kernel_optimizer.models.core import ParameterSpace, ParamValue
from kernel_optimizer.models.reports import TuningStats

# Cadence and dose. Defaults, not constants: they are separate switches precisely so a "mechanism
# fired, no gain" outcome can be attributed to timing or to dose rather than to the signal.
DEFAULT_RECOMPUTE_EVERY = 10
DEFAULT_MAX_ENQUEUED = 2


@dataclass
class Suggestion:
    """One point to enqueue, with why it was chosen -- for the log, never for a decision."""

    values: dict[str, ParamValue]
    knob: str
    knob_value: ParamValue
    source: str            # "hard_wall" | "soft_wall"
    tail_gain_pct: float

    def payload(self) -> dict[str, Any]:
        return {
            "knob": self.knob,
            "knob_value": self.knob_value,
            "source": self.source,
            "tail_gain_pct": round(self.tail_gain_pct, 2),
        }


@dataclass
class SlopeGuide:
    """Recomputes walls mid-tuning and proposes UNMEASURED values of the walled knob.

    One instance per tuning pass. Holds no evidence across candidates: a wall is a property of one
    candidate's space at one point in its own sequence, and this project has already measured what
    cross-candidate pooling does to a per-candidate rule (S1b's cross-run variant bought its larger
    `saved` with a 95% mis-kill bound of 52.7% against per-candidate's 8.7%).

    `use_soft_wall` is NOT defaulted to True even though the soft criterion is the only one the local
    corpus can exercise. Item 2's spill wall has its own switch, and letting it steer the sampler while
    that switch is off would put an unconfirmed signal (no independent probe exists for a spill wall)
    into the tuning loop through a door nobody opened -- and would make the two mechanisms'
    contributions inseparable in the one run that has both.
    """

    space: ParameterSpace
    recompute_every: int = DEFAULT_RECOMPUTE_EVERY
    max_enqueued_per_recompute: int = DEFAULT_MAX_ENQUEUED
    use_soft_wall: bool = False
    # Counters, journalled so the effect is readable from the log rather than argued.
    n_recomputes: int = 0
    n_suggested: int = 0
    n_skipped_no_wall: int = 0
    n_skipped_no_unmeasured_value: int = 0
    n_skipped_already_proposed: int = 0
    n_skipped_incomplete_incumbent: int = 0
    knobs_pushed: dict[str, int] = field(default_factory=dict)
    sources: dict[str, int] = field(default_factory=dict)

    def due(self, n_told: int) -> bool:
        """Whether a recompute is due after `n_told` FINISHED trials.

        Gated on trials told, not trials asked: a wall is derived from measured latencies, and asking is
        not measuring. With `constant_liar` there can be asked-but-untold trials at any moment, so
        counting asks would recompute against a stats table that has not moved.
        """
        if self.recompute_every <= 0:
            return False
        return n_told > 0 and n_told % self.recompute_every == 0

    def suggest(self, stats: TuningStats, trials: list[Any],
                measured_keys: set[str]) -> list[Suggestion]:
        """Points to enqueue now. Pure: no GPU, no I/O, and the space is not mutated.

        `measured_keys` is the set of parameter keys the tuner has already drawn (`ParamSet.key()`), so
        a suggestion can never be a point that has already been asked -- enqueueing one would spend a
        trial re-measuring something known, the exact opposite of the intent.

        Both wall criteria are pooled by SLOPE rather than kept apart by kind. The hard wall carries an
        independent compiler confirmation and the soft wall does not, but here they only pick a
        DIRECTION to sample: a suggestion is never shown to an agent as a fact, so the confirmation gap
        that governs `for_prompt` does not apply. `source` keeps them separable in the log.
        """
        self.n_recomputes += 1
        theta = self._incumbent(trials)
        if theta is None:
            return []
        base = self._base_in_space(theta)
        if base is None:
            self.n_skipped_incomplete_incumbent += 1
            return []
        # The incumbent is resolved BEFORE the walls because a proposal is defined relative to it: it is
        # "the walled knob takes one step from the optimum toward the wall". Without that anchor,
        # "toward the wall" degenerates into "any undrawn value", which can point the opposite way.
        walls = self._walls(stats, trials, base)
        if not walls:
            self.n_skipped_no_wall += 1
            return []

        cap = self.max_enqueued_per_recompute
        if cap <= 0:
            return []
        out: list[Suggestion] = []
        proposed = set(measured_keys)
        for knob, target, gain, source in walls:
            if len(out) >= cap:
                break
            values = dict(base)
            values[knob] = target
            key = _key_of(values)
            if key in proposed:
                self.n_skipped_already_proposed += 1
                continue
            proposed.add(key)
            out.append(Suggestion(values=values, knob=knob, knob_value=target, source=source,
                                  tail_gain_pct=gain))
            self.knobs_pushed[knob] = self.knobs_pushed.get(knob, 0) + 1
            self.sources[source] = self.sources.get(source, 0) + 1
        self.n_suggested += len(out)
        return out

    # ------------------------------------------------------------------ internals

    def _base_in_space(self, theta: dict[str, ParamValue]) -> dict[str, ParamValue] | None:
        """`theta` restricted to this space's knobs, or None if it is not a point OF this space.

        Both refusals are reachable, and both come from the same place -- `crun.trials` accumulates
        across a candidate's spaces, so after an expansion re-tune the incumbent may predate the space
        being tuned:

        * a knob of this space MISSING from theta. Optuna would sample the missing knob itself, which is
          legal but silently stops the suggestion being "one knob varied from the optimum", and its key
          could no longer be compared against the tuner's dedup set.
        * a value theta holds that this space does NOT declare. Optuna does not raise on an enqueued
          value outside the distribution -- it warns and samples that knob normally -- so this would
          silently degrade into an ordinary draw while the log recorded a suggestion. An expansion only
          ADDS choices so it cannot cause this, but a re-published space is not required to be a
          superset, and nothing downstream would notice.
        """
        base: dict[str, ParamValue] = {}
        for d in self.space.domains:
            if d.name not in theta:
                return None
            value = theta[d.name]
            if value not in d.choices:
                return None
            base[d.name] = value
        return base

    def _walls(self, stats: TuningStats, trials: list[Any],
               base: dict[str, ParamValue]) -> list[tuple[str, ParamValue, float, str]]:
        """`(knob, unmeasured value, tail gain, source)` per walled knob, steepest slope first.

        TARGETS ARE RESOLVED AFTER THE DEDUP, NOT BEFORE, and that ordering is load-bearing. Both
        criteria can name the same knob, and only the hard one knows a value the compiler REFUSED.
        Resolving first and deduping second means the surviving row carries whichever target its own
        criterion chose -- so a knob whose soft slope happens to exceed its hard slope would be pushed
        toward the extreme end, which for that knob can be the refused value itself. Dedup first, then
        resolve with the knob's refusal bound applied whatever won, and the two criteria cannot disagree
        about what is launchable.
        """
        rows: list[tuple[str, str, float, str]] = []      # (knob, side, gain, source)
        drawn = _drawn_values(trials)
        # Refusal bounds keyed by knob, from the HARD walls only. A refused value INSIDE the measured
        # range is deliberately NOT a bound: the tuner reached both sides of it, so the value is
        # perfectly usable beside other partners and excluding it would narrow the space.
        bound: dict[str, tuple[str, float]] = {}

        refused_sets = [_params_of(t) for t in trials
                        if _failure_of(t) == "infeasible_shared_memory" and _params_of(t)]
        if refused_sets:
            walls = wall_attribution.find_walls(stats, refused_sets)
            # The SHIPPING slope filter, called rather than re-implemented. `select_for_probing` is
            # what decides a wall is worth acting on (monotone tail, gain > 0 -- measured to drop 3 of
            # 6 walls, the worst at -54.8%), and a second copy of that predicate here would be free to
            # drift from the one the report and the prompt agree on. `-1` = no cap, because the cap
            # that applies here is `max_enqueued_per_recompute`, not the probe budget.
            worth, _ = wall_attribution.select_for_probing(walls, -1)
            for w in worth:
                bound[w.param] = (w.side, float(w.refused_value))
                rows.append((w.param, w.side, w.tail_gain_pct, "hard_wall"))

        if self.use_soft_wall:
            scan = soft_wall_mod.find_soft_walls(stats, trials)
            for sw in scan.walls:
                # Always the high side: the criterion requires spilling to RISE over ascending values,
                # so the bound end is the top one by construction.
                rows.append((sw.param, "high", sw.tail_gain_pct, "soft_wall"))

        # Steepest first, then one row per knob: two criteria naming the same knob is agreement, not
        # two independent reasons to spend two trials on it.
        rows.sort(key=lambda r: r[2], reverse=True)
        out: list[tuple[str, ParamValue, float, str]] = []
        seen: set[str] = set()
        for knob, side, gain, source in rows:
            if knob in seen:
                continue
            seen.add(knob)
            refused = None
            if knob in bound and bound[knob][0] == side:
                # Only when the sides AGREE. A knob truncated on the low side puts no ceiling on a
                # high-side proposal, and applying the bound across sides would exclude values the
                # compiler never objected to.
                refused = bound[knob][1]
            nxt = self._toward_wall(knob, side, drawn.get(knob, set()), refused,
                                    base.get(knob))
            if nxt is None:
                self.n_skipped_no_unmeasured_value += 1
                continue
            out.append((knob, nxt, gain, source))
        return out

    def _toward_wall(self, knob: str, side: str, drawn: set[str], refused_value: float | None,
                     incumbent_value: ParamValue | None) -> ParamValue | None:
        """The furthest undrawn choice that lies TOWARD the wall FROM the incumbent, or None.

        Four properties, each with a reason:

        * FROM THE DECLARED CHOICES. The space's own choice list is the only set of values the
          materializer and the guard accept, and Optuna does not raise on a queued value outside the
          distribution -- it warns and samples that knob normally, silently turning a suggestion into an
          ordinary draw. An extrapolated neighbour could be a value the candidate cannot be written with.
        * TOWARD THE WALL FROM THE INCUMBENT. Without this anchor "toward the wall" degenerates into "any
          undrawn value": on a knob whose wall is on the LOW side with the low neighbours already drawn,
          the scan would run past the incumbent and propose the knob's TOP value -- a step directly away
          from the wall, presented in the log as a step toward it.
        * NOT DRAWN BEFORE. A drawn value contributes nothing new to `latency_by_value` (the per-choice
          median table both criteria read), so pushing it again cannot widen the coverage that makes a
          wall visible. Drawn-and-FAILED counts as drawn: the compiler has already answered for it.
        * BELOW A KNOWN REFUSAL. For a hard wall the compiler has said `refused_value` cannot launch.
          Proposing it, or anything past it, would spend a trial to be told that again -- so the furthest
          useful value is the last one strictly before it. What that buys is not a new range but a
          sharper reading of the existing one: `tail_gain_pct` comes from the last three MEASURED values,
          so filling a hole near the wall is what makes the slope (and hence "is this wall worth
          freeing") readable at all.

        Returns None when nothing on that side qualifies -- then there is nothing to suggest and the knob
        is skipped rather than re-proposed.
        """
        domain = next((d for d in self.space.domains if d.name == knob), None)
        if domain is None:
            return None
        here = wall_attribution._as_num(incumbent_value) if incumbent_value is not None else None
        nums: list[tuple[float, ParamValue]] = []
        for c in domain.choices:
            n = wall_attribution._as_num(c)
            if n is None:
                continue
            if refused_value is not None:
                if side == "high" and n >= refused_value:
                    continue
                if side == "low" and n <= refused_value:
                    continue
            if here is not None:
                if side == "high" and n <= here:
                    continue
                if side == "low" and n >= here:
                    continue
            nums.append((n, c))
        if not nums:
            return None
        nums.sort(key=lambda p: p[0], reverse=(side == "high"))
        for _, choice in nums:
            if str(choice) not in drawn:
                return choice
        return None

    @staticmethod
    def _incumbent(trials: list[Any]) -> dict[str, ParamValue] | None:
        """The best measured configuration -- the base every suggestion varies ONE knob from.

        The same origin the hard wall's ablation uses, for the same measured reason: from the optimum
        6 of 6 walls attribute to a single knob, from the space's default corner only 1 of 6. A
        suggestion built on a slow configuration would probe a point where every other knob is small
        and the walled knob is not the binding one.

        `robust_ms` (median else mean), never `min`: it is this project's measured objective (93.2%
        rank-correctness at n=20 against the mean's 64.8%) and the whole framework already selects on
        it, so a second rule here could name a different winner than the one the agent is told about.
        """
        best: tuple[float, dict[str, ParamValue]] | None = None
        for t in trials:
            if _status_of(t) != "complete":
                continue
            ms = _robust_of(t)
            params = _params_of(t)
            if ms is None or not params:
                continue
            if best is None or ms < best[0]:
                best = (ms, dict(params))
        return best[1] if best else None

    def snapshot(self) -> dict[str, Any]:
        """What fired, and against how much evidence.

        `n_suggested` alone is unreadable: 2 suggestions out of 2 recomputes and out of 20 differ by an
        order of magnitude in how much of the budget the mechanism claimed. The skip counters separate
        the three ways it can decline -- no wall to act on, no undrawn value left on the walled side,
        and the point already asked -- which is what a P4 reading needs to tell "the signal is not
        useful" from "the mechanism never got to fire".

        THE COUNTERS ARE NOT A PARTITION and must not be summed. `n_skipped_no_wall` and
        `n_skipped_incomplete_incumbent` count RECOMPUTES; `n_skipped_no_unmeasured_value` and
        `n_skipped_already_proposed` count KNOBS, of which one recompute can decline several. A recompute
        whose only walled knob had nothing undrawn increments both kinds, because the row is dropped while
        resolving targets and the resulting empty list is then reported as "no wall". Measured on the real
        corpus: 751 + 194 against 790 recomputes.
        """
        return {
            "recompute_every": self.recompute_every,
            "max_enqueued_per_recompute": self.max_enqueued_per_recompute,
            "use_soft_wall": self.use_soft_wall,
            "n_recomputes": self.n_recomputes,
            "n_suggested": self.n_suggested,
            "n_skipped_no_wall": self.n_skipped_no_wall,
            "n_skipped_no_unmeasured_value": self.n_skipped_no_unmeasured_value,
            "n_skipped_already_proposed": self.n_skipped_already_proposed,
            "n_skipped_incomplete_incumbent": self.n_skipped_incomplete_incumbent,
            "knobs_pushed": dict(sorted(self.knobs_pushed.items())),
            "sources": dict(sorted(self.sources.items())),
        }


# --- readers that work on a TrialRecord and on a replayed dict alike ----------------------------
# Same three-line pattern as `soft_wall`, and for the same reason: `robust_ms` is a @property and is
# never serialized, so a reader that looked the name up in a replayed trial would treat every trial as
# untimed and silently suggest nothing -- a plausible constant, which is the failure mode that hides.


def _status_of(t: Any) -> str | None:
    s = getattr(t, "status", None)
    if s is None and isinstance(t, dict):
        s = t.get("status")
    return str(s) if s is not None else None


def _failure_of(t: Any) -> str | None:
    f = getattr(t, "failure_kind", None)
    if f is None and isinstance(t, dict):
        f = t.get("failure_kind")
    return str(f) if f is not None else None


def _params_of(t: Any) -> dict[str, ParamValue]:
    p = getattr(t, "params", None)
    if p is None and isinstance(t, dict):
        p = t.get("params")
    if p is None:
        return {}
    if isinstance(p, dict):
        return dict(p.get("values") or {})
    return dict(getattr(p, "values", {}) or {})


def _robust_of(t: Any) -> float | None:
    lat = getattr(t, "latency_ms", None)
    if lat is None and isinstance(t, dict):
        lat = t.get("latency_ms")
    if lat is None:
        return None
    if isinstance(lat, dict):
        for key in ("median", "mean"):
            v = lat.get(key)
            if isinstance(v, (int, float)) and not isinstance(v, bool) and v > 0:
                return float(v)
        return None
    v = getattr(lat, "robust_ms", None)
    return float(v) if isinstance(v, (int, float)) and v > 0 else None


def _drawn_values(trials: list[Any]) -> dict[str, set[str]]:
    """`knob -> {stringified values that have been DRAWN}`, failed draws included.

    Stringified because a choice list may mix int, float and str and the same value can arrive as
    either after a JSON round-trip. `str` rather than `repr` only because BOTH sides of the only
    comparison that uses this go through `str` (`_toward_wall` stringifies the declared choice); the
    two must not be mixed, since `repr` quotes a string and `str` does not.
    """
    out: dict[str, set[str]] = {}
    for t in trials:
        for knob, value in _params_of(t).items():
            out.setdefault(str(knob), set()).add(str(value))
    return out


def _key_of(values: dict[str, ParamValue]) -> str:
    """The key `ParamSet.key()` produces, so a suggestion can be compared against the tuner's own
    dedup set without this module having to know how that key is built."""
    from kernel_optimizer.models.core import ParamSet

    return ParamSet(values=values).key()
