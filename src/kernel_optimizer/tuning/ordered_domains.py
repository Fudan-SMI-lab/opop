"""S8 / item 3.1: tell the sampler which knobs have an ORDER, and how far apart their values are.

WHAT IS WRONG WITHOUT IT. `OptunaTPETuner.ask` calls `suggest_categorical` for EVERY knob, so TPE
sees `BLOCK_N in {16, 32, 64, 128}` as four unrelated labels: learning that 64 is good changes the
probability of drawing 128 by nothing at all. Measured consequence on the real corpus: the highest
value of a knob that has a wall was sampled at a mean normalised position of 0.52 -- the tuner RAN
INTO the wall rather than walking toward it.

THE PREMISE IS MEASURED, WHICH IS WHAT SEPARATES THIS FROM THE OTHER CANDIDATE CHANGES. Over the
real trials, mean |latency difference between ADJACENT values| divided by mean |difference between
DISTANT values| has p50 = 0.735 < 1. Neighbouring settings really do behave more alike, so the order
carries information a label-only model throws away.

WHICH KNOBS QUALIFY IS READ OFF THE VALUES, NEVER OFF THE `kind` LABEL. The space is authored by an
agent; `kind` is its declaration. `guard.py:113` does enforce the declared type, but a reader that
depends on that enforcement inherits any hole in it, and the question here is not "what did the agent
call this knob" but "do these values lie on a line". Quantified over 9 events.jsonl files and 999
domain declarations:

    kind="int"                                858   216 distinct (name, value-set), 3-6 values each
    kind="str"                                141   only 5 names, all precision/mode switches
                                                    (`COMPUTE_DTYPE` etc.) -- genuinely unordered
    kind="float"                                0   absent from the corpus
    `str` knob whose values are all numeric      0   the trap I expected did NOT occur
    `int` knob that is really an on/off switch    0   "int => ordered" holds in this corpus
    `int` knob with fewer than 3 values          5   nothing for a distance to smooth

So the predicate admits 858/999 = 86% of declarations, 211 of 216 distinct knobs.

THE DISTANCE IS ON THE LADDER'S RUNGS, NOT ON THE RAW VALUES. |2048 - 1024| = 1024 while
|32 - 16| = 16, so a raw arithmetic distance would tell the sampler that the top of a geometric
ladder is sparse and the bottom dense -- an artifact of the encoding, not of the kernel. Every tile,
warp and stage ladder in the corpus is geometric or arithmetic in its own right, so the distance is
taken between INDEX POSITIONS in the sorted choice list, which is what "one step along this knob"
means either way.

WHAT WAS MEASURED BEFORE WRITING ANY OF THIS
(`scripts/probes/is_categorical_distance_func_functional.py`, optuna 4.9.0, 8-rung ladder, +-16%
noise matching this project's own per-trial sigma, 120 trials x 12 seeds):

  * the argument is DEPRECATED in 4.9.0 and scheduled for removal in 5.0.0. It still works: the
    deprecation shim passes the value through. That mattered enough to measure, because an argument
    accepted and silently ignored would have produced a "treatment" arm byte-identical to the
    control -- a failure that does not announce itself.
  * the content is READ: the same kernel with a SCRAMBLED rung order loses 34.1 of 120 near-optimum
    draws, 12 of 12 seeds. This is the control that matters. A CONSTANT distance function -- every
    pair equidistant, i.e. what plain TPE assumes -- does NOT reproduce plain (-22/600 near-best),
    because plain builds a count-based categorical distribution while distance mode builds a kernel
    over distances. So "ordered vs plain" alone could not have told "the order helped" from "the
    estimator changed".
  * against today's sampler it wins on that landscape: +4.5 near-best draws and 12.1% lower mean
    cost, 12 of 12 seeds. That is a SYNTHETIC monotone landscape and licenses implementing this, not
    claiming a speedup: the real spaces have 6-12 interacting knobs.

WHY THIS IS SWITCHED AND OFF BY DEFAULT. It changes what the sampler draws, so a run with it on is
not comparable with the finished runs. Same rule as every other v3 stage.
"""

from __future__ import annotations

from typing import Any

from kernel_optimizer.models.core import ParamDomain

# Two values give one gap, and a "distance" over a single gap cannot distinguish a smooth axis from
# two isolated labels -- both say "these two differ". Three is the smallest number of values whose
# distances carry shape. Matches `wall_attribution._MIN_RAN_VALUES` and for the same reason.
MIN_ORDERED_CHOICES = 3


def _as_num(value: Any) -> float | None:
    """Numeric value of one choice, or None when the domain is not an ordered axis.

    Deliberately the same rule as `wall_attribution._as_num`, including the `bool` exclusion:
    `float(True)` is 1.0, which would make an on/off switch look like an ordered axis and hand the
    sampler a gradient over two unrelated code paths. Reusing an already-verified predicate rather
    than writing a second one, because two implementations of "is this ordered" are free to disagree
    and only one of them would be the one the wall attribution used.
    """
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def ordered_choices(domain: ParamDomain) -> list[float] | None:
    """The domain's choices as numbers when it is an ordered axis, else None.

    Reads the CHOICES, not `domain.kind`. Rejects, in order: a domain holding any bool; a domain any
    of whose choices is non-numeric (this also catches `kind="int"` mislabelling and `kind="str"`
    holding "128"); a domain with fewer than `MIN_ORDERED_CHOICES` values; and a domain whose numeric
    choices contain a duplicate, since two identical positions make the distance between two
    DIFFERENT labels zero and would tell the sampler they are interchangeable.

    Reads the domain and never writes it: a test walks this module's AST and fails on any assignment
    to `choices`/`domains` or any mutating list call, because the standing constraint is that nothing
    here narrows a space. (An earlier version of that test scanned the source TEXT and fired on this
    very paragraph -- `source-text-assertions-can-encode-the-bug`.)
    """
    values = list(domain.choices or ())
    if len(values) < MIN_ORDERED_CHOICES:
        return None
    nums: list[float] = []
    for c in values:
        n = _as_num(c)
        if n is None:
            return None
        nums.append(n)
    if len(set(nums)) != len(nums):
        return None
    return nums


def rung_distance_for(domain: ParamDomain):
    """A distance function over one domain's choices, or None when the domain has no order.

    The distance is the gap between INDEX POSITIONS in the sorted choice list -- the ladder's rungs
    -- not between the raw values. On [16, 32, 64, 128] that makes 64->128 one step, the same as
    16->32; a raw difference would have called it eight steps and told the sampler the top of the
    ladder is a distant, sparsely-explored region when it is one knob click away.

    Closes over a value->rung map built once, so the returned callable is O(1) and holds no reference
    to the domain object.
    """
    nums = ordered_choices(domain)
    if nums is None:
        return None
    rung = {value: i for i, value in enumerate(sorted(nums))}

    def distance(a: Any, b: Any) -> float:
        """Non-negative, symmetric, zero iff same rung -- Optuna requires non-negativity.

        Falls back to the maximum span for a value that is not in the map. That cannot happen for a
        value the sampler drew from this domain, but `suggest_categorical` is also replayed with
        values from an enqueued anchor, and returning 0.0 there would silently declare an unknown
        value identical to everything.
        """
        ra, rb = rung.get(_as_num(a)), rung.get(_as_num(b))
        if ra is None or rb is None:
            return float(len(rung))
        return float(abs(ra - rb))

    return distance


def distance_funcs(domains: list[ParamDomain]) -> dict[str, Any]:
    """`{knob: distance}` for every ordered domain -- the shape TPESampler takes.

    Unordered domains are OMITTED rather than given a constant distance. Optuna applies its default
    (all choices equidistant) to any knob absent from the dict, which is the correct model for a
    precision switch; supplying a constant function instead would move that knob into the
    distance-kernel path for no reason, and the probe measured that path to differ from the default
    even when the distances are all equal.

    An empty dict is returned rather than None when NO domain qualifies, and the caller turns that
    into None -- see `OptunaTPETuner`. Distinguishing "asked for, nothing qualified" from "not asked
    for" is what makes a null result in the event log readable.
    """
    out: dict[str, Any] = {}
    for domain in domains:
        fn = rung_distance_for(domain)
        if fn is not None:
            out[domain.name] = fn
    return out


def snapshot(domains: list[ParamDomain]) -> dict:
    """What the switch did, for the event log -- never for a decision.

    `n_ordered` alone would be unreadable: 5 of 5 and 5 of 12 are the same numerator with opposite
    meanings, and the omitted names are exactly what a later reader needs to check the predicate
    against the space it ran on.
    """
    funcs = distance_funcs(domains)
    return {
        "n_domains": len(domains),
        "n_ordered": len(funcs),
        "ordered": sorted(funcs),
        "unordered": sorted(d.name for d in domains if d.name not in funcs),
    }
