"""S1b: down-weight a categorical value whose failures no partner can explain.

WHAT THIS IS FOR. `correctness_mismatch` is the largest failure class measured -- 537 of 2559
trials over five L3 runs, 20.98%, 4.7 h at a 26.82 s median, which is MORE expensive than a
completed trial (24.49 s) because it pays for the correctness pass before being refused. Part of
that class is unconditional: a value that fails beside every partner because the CANDIDATE put low
precision in an uncompensated `dot`, or gated the whole algorithm on `PREC` so anything else falls
into an unstabilised branch. `DOT_PRECISION=tf32` measured 0 passes in 164 trials in one run, and
its sampling share did not decay across the four quarters of that run.

WHY THE TUNER CANNOT LEARN THIS ON ITS OWN. `OptunaTPETuner.tell` reports
`correctness_mismatch` as `TrialState.FAIL`, and Optuna EXCLUDES failed trials from the TPE model
(measured: 12 trials told FAIL leave 1 visible to the sampler; the same 12 told PRUNED leave 13).
So those 164 failures are, to the sampler, as if they never happened. Reporting them PRUNED
instead is not the fix and is refused where that decision is made: a `correctness_mismatch` can be
non-deterministic, or a defect in the candidate rather than a property of the point, so teaching
the sampler to avoid the whole region would be teaching it noise. This module is the criterion
that decides WHICH of those failures are a property of the point.

THE RULE, AND EVERY PART OF IT IS A MEASUREMENT RATHER THAN A PREFERENCE.

  scope     (candidate_id, knob, value), pooled across that candidate's spaces, never across
            candidates and never across runs.
  fire      >= `floor` failures accumulated with ZERO passes.
  retract   the first pass retracts permanently.
  action    down-weight to 1/`strength`, never removal.

Scope is the part I got wrong first and it is worth stating why. The first version pooled across
the whole run, which in the measured corpus means across CANDIDATES: at floor 8, three of the four
rules mixed evidence from more than one candidate and the `DOT_PRECISION=tf32` rule mixed ten.
But the root cause above is a property of the candidate's own source, so pooling ten candidates
only works because most of them happen to make the same mistake -- that is the corpus's property,
not the mechanism's, and a candidate that DID compensate its tf32 would be declared dead by nine
other candidates' history. Three independent criteria then agreed on per-candidate:

  scope           saved  mis-kills  95% upper bound   threshold under leave-one-out
  per-space          63          0   <=8.9%  (n=32)   [7,7,7,6,7]  unstable
  per-candidate     206          0   <=8.7%  (n=33)   [7,7,7,7,7]  STABLE
  per-run (cross)   249          0   <=52.7% (n=4)    [5,7,7,7,7]  unstable

The bound matters as much as the zero: 0 mis-kills out of 4 rules cannot rule out a true rate of
half, so the cross-candidate variant's larger `saved` is bought with an order of magnitude less
evidence. And the threshold column is the same test that refused this mechanism's own ancestor
("N=12 happens to be safe" was picked in hindsight) -- only per-candidate picks the same floor
from every four-run subset.

WHY 7. Measured over 2776 values: a value that later passes never first accumulated more than 6
failures (median 0; only 2 of 2776 reached 5). So 7 is the longest observed failure prefix plus
one, and the mis-judgement probability against threshold is 78.5% at 1, 26.7% at 3, 13.5% at 4,
2.3% at 6, and 0% at 7. IT LEADS BY EXACTLY ONE STEP: a value that fails 7 times and then passes
produces a wrong call. That is a real fragility of this floor and not something the zero hides.

WHY DOWN-WEIGHT AND WHY 1/4. Removal cannot be corrected; down-weighting can. But the strength
has a floor of its own: a fired value appears a median 6 more times, so at 1/8 it is expected to
be drawn a median 0.8 times more -- under one, which makes "retains a non-zero probability" empty
inside a 40-trial budget and degrades the action to removal. At 1/4 the median expectation is 1.5.

WHAT THIS DOES NOT TOUCH. Only the unconditional part of `correctness_mismatch`. The rest of that
class includes candidates that are correct but refused by the gate -- all three L3 tasks' noise
floors sit below `relaxed_pass_frac`, re-verified on the A800 -- and those must be rescued by
fixing the gate, not avoided by the sampler. This module must never be widened to cover them.
"""

from __future__ import annotations

import random
from collections import defaultdict

from kernel_optimizer.models.core import TrialRecord

# The failure class this acts on. Deliberately ONE kind: `runtime_error` can be a transient or a
# whole-candidate defect, and `infeasible_shared_memory` is already handled at compile time by the
# screen (measured: 113 of 113 such trials in five runs were caught by it before any launch).
_POOL_KIND = "correctness_mismatch"

# Leave-one-out over the five measured runs picks this floor from every four-run subset, under
# per-candidate pooling. It is not a constant chosen by hand; see the module docstring.
DEFAULT_FLOOR = 7

# 1/4. Measured: at 1/8 a fired value's expected further draws fall to a median 0.8, under one.
DEFAULT_STRENGTH = 4


class DeweightLedger:
    """Per-run evidence on (candidate, knob, value), and the resulting sampling weights.

    One instance per run. Evidence never crosses a run boundary: the plan's own position is that
    conclusions do not carry across environments, and cross-run leave-one-out measured a 23.83%
    mis-kill rate for exactly this rule shape.
    """

    def __init__(self, floor: int = DEFAULT_FLOOR, strength: int = DEFAULT_STRENGTH,
                 seed: int = 0) -> None:
        if floor < 1:
            raise ValueError(f"floor must be >= 1, got {floor}")
        if strength < 1:
            raise ValueError(f"strength must be >= 1, got {strength}")
        self.floor = floor
        self.strength = strength
        self._fails: dict[tuple[str, str, str], int] = defaultdict(int)
        self._passes: dict[tuple[str, str, str], int] = defaultdict(int)
        self._rng = random.Random(seed)
        # Journalled so the effect is measurable from the event log rather than argued.
        self.fired: set[tuple[str, str, str]] = set()
        self.retracted: set[tuple[str, str, str]] = set()

    @staticmethod
    def _keys(record: TrialRecord) -> list[tuple[str, str, str]]:
        return [(record.candidate_id, knob, str(value))
                for knob, value in record.params.values.items()]

    def observe(self, record: TrialRecord) -> None:
        """Fold one finished trial into the evidence. Order-independent by construction.

        Only `complete` and `correctness_mismatch` move the counters. Everything else -- a
        materialize error, a screened-infeasible config, a runtime error -- says nothing about
        whether this VALUE is correct, and counting it would attribute an unrelated failure to it.
        """
        if record.status == "complete":
            for key in self._keys(record):
                self._passes[key] += 1
                if key in self.fired:
                    # Retraction is permanent, and permanence is structural rather than a flag:
                    # `_fire_ready` requires zero passes, and a pass count never decreases, so a
                    # retracted rule cannot re-fire however many further failures arrive.
                    self.fired.discard(key)
                    self.retracted.add(key)
        elif record.failure_kind == _POOL_KIND:
            for key in self._keys(record):
                self._fails[key] += 1
                if self._fire_ready(key):
                    self.fired.add(key)

    def _fire_ready(self, key: tuple[str, str, str]) -> bool:
        return self._passes[key] == 0 and self._fails[key] >= self.floor

    def is_deweighted(self, candidate_id: str, knob: str, value: object) -> bool:
        return (candidate_id, knob, str(value)) in self.fired

    def accept_probability(self, candidate_id: str, values: dict) -> float:
        """1.0 for an ordinary draw, else 1/strength.

        Deliberately NOT compounded per deweighted value. A configuration holding two fired values
        would be suppressed to 1/strength**2 under compounding, which is a strength this analysis
        never measured -- the replay treated a fired value as binary and says nothing about k. The
        uncompounded form is the weaker suppression, i.e. the one that risks less over-interception,
        and over-interception is the risk this mechanism was interrogated for.
        """
        for knob, value in values.items():
            if self.is_deweighted(candidate_id, knob, value):
                return 1.0 / self.strength
        return 1.0

    def should_reject(self, candidate_id: str, values: dict) -> bool:
        """Sample the decision for one drawn configuration."""
        p = self.accept_probability(candidate_id, values)
        return p < 1.0 and self._rng.random() >= p

    def snapshot(self) -> dict:
        """What fired, what retracted, and on how much evidence -- for the event log.

        `n_fired` alone would be unreadable as a result: 0 mis-kills out of 4 rules and out of 33
        rules are an order of magnitude apart in evidence, so the rule count is the denominator any
        later claim about this mechanism has to be divided by.
        """
        return {
            "floor": self.floor,
            "strength": self.strength,
            "n_fired": len(self.fired),
            "n_retracted": len(self.retracted),
            "fired": sorted(f"{c}|{k}={v}" for c, k, v in self.fired),
            "retracted": sorted(f"{c}|{k}={v}" for c, k, v in self.retracted),
        }
