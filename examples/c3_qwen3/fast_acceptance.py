"""How a lane decides a candidate is meaningfully better, and which candidate it keeps.

The original protocol accepted the first candidate whose repeated measurement merely beat the
baseline by any amount (`gain > 0`) and stopped the lane there. Two consequences the user asked to
fix, both observed on the up_proj campaigns of 2026-09-22:

  * A "win" inside the measurement noise. The accepted ttft candidate improved 0.11% while the
    full-model repeat spread of that very site was 0.372% -- so the acceptance was indistinguishable
    from measuring the baseline twice and keeping the luckier read.
  * Stopping at the first such win rather than searching for the best. The ttft scores were still
    trending down (286.0 -> 282.6 -> 282.1 -> 281.9) when the lane quit.

So acceptance now requires the gain to clear a multiple of the goal's own measured noise floor, and
a lane no longer stops at the first acceptance: it keeps drawing, remembers the best accepted
candidate, and ends only when the window closes or a patience number of draws pass with no new best.
Correctness is untouched -- `prove_candidate`, the logits/NLL floor and the device proof all still
gate every candidate before this code ever sees it. This only decides, among candidates that are
already proven correct, which counts as a real speed win and which one to publish.
"""

from dataclasses import dataclass

from kernel_optimizer.tuning.objective import Objective
from .fast_records import CandidateResult

# The gain must exceed this multiple of the goal's measured score spread. k=1 means "larger than the
# distance from the baseline's median to its own extreme repeat" -- the user's chosen setting. It is
# a speed-acceptance threshold only; it never touches a correctness or equivalence limit.
NOISE_MULTIPLE = 1.0
# When a goal cannot estimate its own spread within one evaluation (a single-sample goal such as
# multi, or an older receipt without the field), fall back to this fractional floor rather than to
# zero -- accepting a sub-this gain would be the very "win inside the noise" this module removes.
# 0.4% is just above the largest full-model repeat spread measured across the goals (0.372%).
FALLBACK_SPREAD_FRAC = 0.004
# Consecutive draws without a new best that end a lane early. The window is still the hard bound;
# this only stops a lane that has plateaued from burning budget it will not use.
DEFAULT_PATIENCE = 4


def score_spread_frac(evaluation) -> float:
    """The goal's measured within-evaluation spread as a fraction of its score, or the fallback."""
    if evaluation is None:
        return FALLBACK_SPREAD_FRAC
    frac = evaluation.metrics.get("score_spread_frac")
    if not isinstance(frac, (int, float)) or isinstance(frac, bool) or frac < 0 or frac != frac:
        return FALLBACK_SPREAD_FRAC
    return max(float(frac), 0.0)


def meaningfully_beats(objective: Objective, baseline: float, candidate: float,
                       spread_frac: float, *, multiple: float = NOISE_MULTIPLE) -> bool:
    """True when `candidate` beats `baseline` by more than `multiple` times the noise band.

    The band is `spread_frac` of the baseline magnitude, so the comparison is in the score's own
    units and direction (via `objective.gain`, which is positive for improvement either way).
    """
    gain = objective.gain(baseline, candidate)
    return gain > multiple * spread_frac * abs(baseline)


@dataclass(frozen=True, slots=True)
class Incumbent:
    """The best accepted candidate a lane has seen, and its confirmed gain over the baseline."""
    candidate: CandidateResult
    score: float
    gain: float
