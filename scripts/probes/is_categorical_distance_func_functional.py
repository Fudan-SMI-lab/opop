"""Item 3.1 pre-check: does `categorical_distance_func` still DO anything in the Optuna we run?

WHY THIS HAS TO BE MEASURED BEFORE THE CHANGE IS WRITTEN. `categorical_distance_func` is
DEPRECATED in optuna 4.9.0 -- the exact version pinned in pyproject.toml -- and scheduled for
removal in 5.0.0. A deprecated argument can be one of three things, and they call for three
different decisions:

  (a) still fully functional, warning only        -> implementable now, with a pin and an exit plan
  (b) accepted and silently ignored               -> implementing it would produce a run that LOOKS
                                                      like a treatment arm and is byte-identical to
                                                      the control. That is the worst outcome of the
                                                      three and it would not announce itself.
  (c) rejected                                    -> not implementable

Reading the shim says (a): `_warn_if_deprecated_argument` returns the value when it is not None. But
"the code path exists" is not "the sampling changed" -- this project has already recorded
`an-unreachable-branch-is-not-a-safeguard` (a branch executed 0 times in 6894 trials) and
`a-variant-that-changes-no-behaviour-is-not-a-variant`. So the effect is measured on drawn samples.

WHAT IS MEASURED. A study over one knob whose choices are a geometric ladder, with an objective that
is monotone in the knob so the ordering carries real information. Several samplers on identical
seeds, and the comparison that matters is NOT against plain TPE.

THE CONTROL, AND WHY THE OBVIOUS ONE IS WRONG. My first control was a CONSTANT distance function --
every pair equidistant, which is what plain TPE assumes -- expecting it to reproduce the plain
sampler. It does not: measured, it moved draws near the best by -22 out of 600 while the ordered
function moved them by -3. So supplying ANY distance function changes the estimator (plain TPE builds
a count-based categorical distribution; distance mode builds a kernel over the distances), and a
comparison against plain cannot separate "the ordering helped" from "the kernel changed".

The control that does separate them is a PERMUTED ladder: the same distance function, the same
kernel, the same code path, with the rung order SCRAMBLED so the distances carry no information about
the objective. Ordered vs permuted isolates the ordering. Constant is kept as a third arm because it
prices the kernel change on its own, and plain is kept because it is what today's runs did.

A permuted arm that matches the ordered arm means the ordering bought nothing -- which is the outcome
that would sink item 3.1 regardless of whether the argument "works".
"""
from __future__ import annotations

import random
import statistics
import sys
import warnings
from collections import Counter
from pathlib import Path

import optuna
from optuna.samplers import TPESampler

optuna.logging.set_verbosity(optuna.logging.WARNING)

# Eight rungs rather than five: with five, plain TPE already puts 83 of 120 draws on the best choice,
# so every arm saturates and the comparison measures nothing. Eight is the width of the widest knob
# in the measured corpus after K-expansion.
LADDER = [16, 32, 64, 128, 256, 512, 1024, 2048]
BEST = 2048


def objective_value(x: int) -> float:
    """Monotone in the RUNG, plus a little noise.

    Monotone because the question is whether the sampler can exploit order -- an objective with no
    order to exploit would confound "cannot use order" with "there was none". Noisy because a
    noiseless objective lets any sampler lock on immediately, and this project's own trials carry a
    16% per-trial standard deviation; a probe on a noiseless landscape would answer a question the
    framework never asks.
    """
    rungs = abs(LADDER.index(x) - LADDER.index(BEST))
    return rungs + 1.0


def run(distance, n_trials: int, seed: int) -> tuple[Counter, list[float]]:
    """`distance` is a per-knob CALLABLE or None; the sampler takes a {knob: callable} DICT.

    The dict shape is the contract (`dict[str, Callable[[choice, choice], float]]`), and passing a
    bare callable raises `TypeError: argument of type 'function' is not iterable` from deep inside
    the Parzen estimator -- which is the good case. The quiet version of that mistake would be an
    argument silently ignored.
    """
    rng = random.Random(seed * 7919 + 13)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sampler = TPESampler(seed=seed, multivariate=True, group=True, n_startup_trials=10,
                             categorical_distance_func=({"BLOCK": distance} if distance
                                                        else None))
        study = optuna.create_study(direction="minimize", sampler=sampler)
        got: Counter = Counter()
        values: list[float] = []
        for _ in range(n_trials):
            t = study.ask()
            x = t.suggest_categorical("BLOCK", LADDER)
            got[x] += 1
            # The noise is applied to what the SAMPLER is told, not to the bookkeeping: `got` and
            # `values` record the true rung cost, so the arms are scored on the same landscape while
            # each still sees a noisy view of it.
            true_v = objective_value(x)
            values.append(true_v)
            study.tell(t, true_v * (1.0 + 0.16 * (rng.random() * 2 - 1)))
    return got, values


def ladder_distance(a, b) -> float:
    """Distance on the ladder's own RUNGS, not on the raw values.

    |2048 - 1024| = 1024 while |32 - 16| = 16, so raw arithmetic distance would tell the sampler that
    the top of a geometric ladder is sparse and the bottom dense -- an artifact of the encoding, not
    of the kernel. Rung distance (equivalently, distance in log space) is what "one step along this
    knob" means for a tile size, and every ladder in the measured corpus is geometric.
    """
    return abs(LADDER.index(a) - LADDER.index(b))


_PERMUTED = {v: i for i, v in enumerate([256, 16, 1024, 64, 2048, 32, 512, 128])}


def permuted_distance(a, b) -> float:
    """THE CONTROL. Same kernel, same code path, ordering scrambled so it carries no information.

    A fixed permutation rather than a per-seed random one, so every seed's arms differ only in the
    sampler's own seed. It is not the identity on any adjacent pair of the real ladder, which is what
    makes it uninformative rather than a mild relabelling.
    """
    return abs(_PERMUTED[a] - _PERMUTED[b])


def constant_distance(a, b) -> float:
    """Prices the KERNEL CHANGE alone: every distinct pair equidistant, which is what plain TPE
    assumes -- yet measured NOT to reproduce plain, which is why it is not the control."""
    return 0.0 if a == b else 1.0


def near_best(c: Counter) -> int:
    i = LADDER.index(BEST)
    return sum(n for x, n in c.items() if abs(LADDER.index(x) - i) <= 1)


def main() -> int:
    out: list[str] = []

    def say(s: str = "") -> None:
        out.append(s)

    n, seeds = 120, tuple(range(12))
    arms = (("plain", None), ("ordered", ladder_distance),
            ("permuted-CTL", permuted_distance), ("constant", constant_distance))
    say("=" * 96)
    say("ITEM 3.1 PRE-CHECK -- is `categorical_distance_func` functional, and does ORDER help?")
    say("=" * 96)
    say(f"optuna {optuna.__version__}   (deprecated in 4.9.0, removal scheduled for 5.0.0)")
    say(f"ladder {LADDER}")
    say(f"objective monotone toward {BEST}, +-16% multiplicative noise (this project's own")
    say(f"per-trial sigma), {n} trials x {len(seeds)} seeds")
    say()

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        TPESampler(seed=0, categorical_distance_func={"BLOCK": ladder_distance})
        kinds = {x.category.__name__ for x in w}
    say(f"construction warnings: {sorted(kinds)}")
    say("  (a FutureWarning is expected and is NOT evidence the argument is ignored)")
    say()

    res: dict[str, list[tuple[Counter, list[float]]]] = {a: [] for a, _ in arms}
    for seed in seeds:
        for label, d in arms:
            res[label].append(run(d, n, seed))

    say("-" * 96)
    say(f"{'arm':<14s} " + " ".join(f"{v:>5d}" for v in LADDER)
        + f" {'near-best':>10s} {'mean cost':>10s}")
    say("-" * 96)
    for label, _ in arms:
        pooled: Counter = Counter()
        for c, _v in res[label]:
            pooled.update(c)
        nb = [near_best(c) for c, _v in res[label]]
        mc = [statistics.mean(v) for _c, v in res[label]]
        say(f"{label:<14s} " + " ".join(f"{pooled.get(v, 0) // len(seeds):>5d}" for v in LADDER)
            + f" {statistics.mean(nb):>10.1f} {statistics.mean(mc):>10.3f}")
    say("(counts are per-seed averages; near-best = draws within one rung of the optimum;")
    say(" mean cost = average TRUE rung cost over the run, lower is better)")
    say()

    say("=" * 96)
    say("QUESTION 1 -- is the argument's CONTENT read at all? ordered vs PERMUTED")
    say("(same kernel, same code path, rung order scrambled so distances carry no information)")
    o_nb = [near_best(c) for c, _ in res["ordered"]]
    p_nb = [near_best(c) for c, _ in res["permuted-CTL"]]
    o_mc = [statistics.mean(v) for _, v in res["ordered"]]
    p_mc = [statistics.mean(v) for _, v in res["permuted-CTL"]]
    wins = sum(1 for a, b in zip(o_nb, p_nb) if a > b)
    ties = sum(1 for a, b in zip(o_nb, p_nb) if a == b)
    say(f"  near-best: ordered {statistics.mean(o_nb):.1f} vs permuted "
        f"{statistics.mean(p_nb):.1f}   (ordered wins {wins}/{len(seeds)} seeds, ties {ties})")
    say(f"  mean cost: ordered {statistics.mean(o_mc):.3f} vs permuted "
        f"{statistics.mean(p_mc):.3f}")
    identical = sum(1 for (a, _), (b, _) in zip(res["ordered"], res["permuted-CTL"]) if a == b)
    say(f"  histograms identical between the two: {identical}/{len(seeds)} seeds")
    say()

    # This is the number that prices the CHANGE. The permuted arm is a HARMFUL control -- a wrong
    # order actively misleads the kernel -- so ordered-vs-permuted answers "is the content read",
    # NOT "what does shipping this buy". What we would ship replaces `plain`, so plain is the
    # baseline for the benefit even though it is the wrong baseline for the mechanism.
    say("=" * 96)
    say("QUESTION 2 -- what would shipping it BUY? ordered vs PLAIN (what runs today)")
    say("(the permuted arm is a HARMFUL control: a wrong order misleads the kernel, so the")
    say(" ordered-vs-permuted gap prices the mechanism, not the benefit)")
    pl_nb = [near_best(c) for c, _ in res["plain"]]
    pl_mc = [statistics.mean(v) for _, v in res["plain"]]
    w2 = sum(1 for a, b in zip(o_nb, pl_nb) if a > b)
    t2 = sum(1 for a, b in zip(o_nb, pl_nb) if a == b)
    say(f"  near-best: ordered {statistics.mean(o_nb):.1f} vs plain "
        f"{statistics.mean(pl_nb):.1f}   ({statistics.mean(o_nb) - statistics.mean(pl_nb):+.1f}, "
        f"ordered wins {w2}/{len(seeds)} seeds, ties {t2})")
    say(f"  mean cost: ordered {statistics.mean(o_mc):.3f} vs plain "
        f"{statistics.mean(pl_mc):.3f}   "
        f"({100.0 * (statistics.mean(pl_mc) - statistics.mean(o_mc)) / statistics.mean(pl_mc):+.1f}%"
        f" better)")
    say()
    if identical == len(seeds):
        say("VERDICT: INERT. Scrambling the order changed nothing, so the argument is accepted and")
        say("its CONTENT ignored. Implementing 3.1 through it would give a treatment arm")
        say("byte-identical to the control -- the failure mode that does not announce itself.")
    elif statistics.mean(o_nb) > statistics.mean(p_nb):
        say("VERDICT: FUNCTIONAL -- the content IS read. Scrambling the order costs "
            f"{statistics.mean(o_nb) - statistics.mean(p_nb):.1f} near-best draws, so the argument")
        say("is not a no-op in this optuna.")
        gain_nb = statistics.mean(o_nb) - statistics.mean(pl_nb)
        gain_mc = statistics.mean(pl_mc) - statistics.mean(o_mc)
        if gain_nb > 0 and gain_mc > 0:
            say(f"AND IT BEATS TODAY'S SAMPLER on this landscape: {gain_nb:+.1f} near-best draws, "
                f"mean cost {statistics.mean(pl_mc):.3f} -> {statistics.mean(o_mc):.3f}.")
            say("Note what that is and is not: a SYNTHETIC monotone landscape. It licenses")
            say("implementing 3.1, not claiming a speedup -- the real spaces are 6-12 knobs with")
            say("interactions, and the ONLY corpus evidence for the premise is the measured")
            say("neighbour/distant latency-gap ratio p50 0.735.")
        else:
            say(f"BUT IT DOES NOT BEAT TODAY'S SAMPLER: near-best {gain_nb:+.1f}, mean cost "
                f"{gain_mc:+.3f}. The mechanism works and buys nothing here, which has to be")
            say("resolved before it ships -- an arm that changes sampling without improving it")
            say("costs comparability for no return.")
    else:
        say("VERDICT: FUNCTIONAL BUT THE ORDER DOES NOT HELP HERE. The argument changes sampling,")
        say("but a SCRAMBLED order does as well or better -- so on this landscape the effect is the")
        say("kernel change, not the ordering. Item 3.1's premise (adjacent values are more alike,")
        say("measured p50 0.735) is about the KERNEL's assumption; this says the assumption did not")
        say("convert into better sampling, and that has to be resolved before it ships.")
    say()
    say("The `constant` arm prices the estimator change on its own: plain TPE builds a count-based")
    say("categorical distribution, distance mode builds a kernel over distances, so 'no distance")
    say("function' and 'a distance function that says everything is equidistant' are NOT the same")
    say("sampler. That is why plain is not the control.")
    say()
    say("STILL TO DECIDE even if functional: the argument is removed in optuna 5.0.0, so shipping")
    say("it needs an upper pin and a recorded exit. `pyproject.toml` says `optuna~=4.9`, which")
    say("admits 4.x only -- so 5.0.0 cannot arrive silently through a resolve.")

    text = "\n".join(out) + "\n"
    if len(sys.argv) > 1:
        Path(sys.argv[1]).write_text(text, encoding="utf-8")
        print(f"wrote {sys.argv[1]}")
    else:
        sys.stdout.buffer.write(text.encode("utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
