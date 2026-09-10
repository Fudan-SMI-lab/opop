# S1 replay: the zero-GPU comparison did NOT clear S1, and the reason is instructive

**Measured** 2026-09-10 on box 1's corpus (5 L3 runs, 2559 trials, 64 spaces of which 31 usable)
via `scripts/replay_sampler.py`. Zero GPU. Raw output on box 1 at `/tmp/s1/replay_3arm.json`.

**Verdict: S1 is not refused, but it is not supported either — and the replay cannot currently
settle it.** Below is what the numbers say, what is my own instrumentation error, and what would have
to change for a replay to answer the question at all.

---

## 1. The result

| arm | mechanism | better | worse | within 1% | dead draws removed |
|---|---|---|---|---|---|
| `declare` | unconditional value removal | **0** | 0 | 31 | **6.9%** |
| `declare_cond` | conditional domain shrink (S1 path A) | **0** | **5** | 26 | **100.0%** |

Neither arm improved a single space's best latency under a 40-trial budget.

## 2. Two of my own errors, and the first one matters more than its fix

### 2.1 I reproduced the count-based retirement the plan forbids

The first `_dead_values` removed a value after **≥3 infeasible sightings** and never checked whether
that value also appeared in configurations that ran. That is exactly the mechanism `§2.5` of the plan
refuses, rebuilt inside the tool written to test the plan.

**Measured damage**: on `sp-6a0e4fe6` it removed `DOT_PRECISION=fp16`, which had **18 feasible
recorded configurations** and held that space's best latency of **2.6286 ms**. The `declare` arm then
read **37.6719 ms** — 14× worse — and the aggregate reported S1 losing **10 spaces to 0**.

**That entire "result" was an artefact of my rule.** Had I not checked the removed values against the
feasible side, I would have reported a strong negative result about S1 that was really a negative
result about my own code. The corrected rule requires ≥3 infeasible sightings **and zero feasible
ones**; the same table then shows 0 worse instead of 10.

This is the same shape as the history that refused count-based retirement in the first place:
**failure is conditional** (a value fails beside some partners and wins beside others) while
**removal is unconditional**.

### 2.2 The conditional arm is not comparable to the other two

`declare_cond` evaluates **24.4** points per space on average against the others' **27.9** — so it
was judged on ~13% fewer measurements. Cause: the recorded-pool arms walk a shuffled list of visited
points, while the conditional arm draws from the (shrunken) domain and burns asks on duplicates.

**So the "5 spaces worse" figure is not evidence about S1.** Fewer draws means a worse best-so-far,
independent of the mechanism. The arms differ in two things at once, which is the one thing a
controlled comparison may not do.

## 3. The deeper problem: a lookup-table replay may not be able to answer this at all

Beyond the bug, there is a structural obstacle that showed up in two independent ways.

**(a) The full domain is unusable as a draw universe.** These spaces reach **2.7e15** points while a
run records ~30. A uniform draw lands on a recorded point with probability ~1e-14. Run against the
full product, **17 of 24 spaces produced no comparison whatsoever** — every draw `missing`. That is
why the default universe is `recorded`.

**(b) But `recorded` makes the interesting part unobservable.** The recorded points are exactly the
ones the *old* sampler chose to visit. A domain shrink's whole purpose is to spend the freed budget
**elsewhere** — and "elsewhere" is by construction absent from the table. The replay can see the
waste a shrink avoids (and it does: **100% of dead draws removed**) but it cannot see what the freed
draws would have bought, because nothing measured them.

**This was written into the script's header from the start as a lower-bound caveat.** What the run
shows is that the caveat is not a footnote — on this corpus it is the dominant effect.

## 4. What the replay DID establish, and it is not nothing

**Conditional shrinking eliminates 100% of dead draws; unconditional removal eliminates 6.9%.**

That is a clean, mechanism-level result and it settles a design question inside S1: only **2 of 31
spaces** had any value that was infeasible beside *every* partner, which is why the unconditional arm
is nearly vacuous. Shared-memory infeasibility is a property of the **combination** (tile × depth ×
dtype width), so:

**S1 must implement the dependency-aware (ATF-style) shrink of path A. The simpler "drop a bad value
from the domain" variant is measurably not worth building.**

That is worth knowing before writing the code, which was the point of doing this first.

## 5. What this means for the plan

| | before | after |
|---|---|---|
| J1-1 (dead points vanish) | untested | **supported at the mechanism level**: 145 → 0 dead draws |
| J1-2 (more effective evaluations) | untested | **cannot be answered by replay** (§3b) |
| J1-3 (early-search quality) | untested | **cannot be answered by replay** (§3b) |
| J1-4 (endgame not worse) | untested | untested |
| which path to implement | A or C undecided | **A, conditional. Unconditional is vacuous (2/31 spaces)** |

**The control run is now the only route to J1-2/J1-3, and that was already the plan's own
position** ("the acceptance is the critical path, not the coding"). The replay was worth building
anyway: it cost no GPU, it killed one design variant, and it caught a mistake in my own reasoning
that would otherwise have gone into the implementation.

## 6. What I am NOT doing

**Not tuning the replay until it agrees with S1.** The obvious next move — give every arm the same
draw universe and the same duplicate handling — would make the 5-worse figure go away, and I would
then have a harness that produces a favourable number for a mechanism whose real effect it still
cannot see. Fixing the comparability bug is worth doing when the replay is next used for something it
*can* answer; it does not turn this into a verdict on S1.

**Not reporting "S1 removes 18% of wasted trials".** The 18% figure (180/1004 on L3:43) is what
motivated S1 and remains the measured waste. This replay does not confirm that a shrink converts it
into a better result.

## 7. Reproducing

```bash
# on box 1, where the corpus lives (box 1 is the v2 experiment machine; the script is stdlib-only
# and was copied to /tmp rather than into that checkout)
cd /root/autodl-tmp/opop-workspace/opop-glm/runs-l3
python /tmp/s1/replay_sampler.py run-l3-* --budget 40 --seeds 20 --out /tmp/s1/replay_3arm.json
python /tmp/s1/replay_sampler.py run-l3-* --universe full     # reproduces the 17-of-24 blank result
```

## 8. What this did not verify

- **Only shared-memory infeasibility.** The other hard wall (registers > 255) never appears as a
  recorded infeasible point in this corpus, so the replay says nothing about it.
- **Only L3, only box 1.** 31 spaces from three tasks on one card.
- **Random sampler, not TPE.** Deliberate — a TPE fitted to a table of mostly-missing points models
  the holes — but it means the replay says nothing about how a *surrogate* reacts to a shrunken
  domain, which is the actual mechanism in the live tuner.
- **The feasibility oracle is recorded support**, a subset of true infeasibility. It understates what
  S1 has to work with.
