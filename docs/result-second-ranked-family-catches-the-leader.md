# Result: the second-ranked family caught the leader, in two rounds

`run-l3-48-20260905-010737`, `fam-74c41d8d`. Recorded because it is the run's strongest
positive result and because the obvious way to describe it is wrong.

## The trajectory

| stage | candidate | result |
|---|---|---|
| seed | cand-cf0f07e7 | 3.55ms → **2.84ms** (K expansion) |
| round 1 H1 | cand-eed411d8 | rejected by the gate, never timed |
| round 1 H2 | cand-a04c3f52 | **2.46ms** |
| round 2 H1 | cand-dcf4e7e6 | rejected by the gate, never timed |
| round 2 H2 | cand-8a617cba | **2.09ms** |

Seed ranking after tuning: `fam-99aee6de` 2.09ms, **`fam-74c41d8d` 2.84ms**,
`fam-b1ee96ac` 3.80ms (`fam-dc0697c9` never passed correctness). So the second-ranked
family reached the leader's latency, while the leader stalled at 2.09 across all three of
its rounds.

Speedups at 2.09ms — `tuned_ms`, therefore a **lower bound**
(`finding-one-slow-sample-per-measurement.md`):

| baseline | ms | speedup |
|---|---|---|
| eager | 28.80 | 13.78x |
| eager_tf32 | 28.30 | 13.54x |
| torch_compile | 18.60 | **8.90x** |
| torch_compile_tf32 | 17.90 | 8.56x |

## What the anti-early-pruning change actually bought — stated correctly

My first reading was that greedy pruning would have cut this family before round 1. **That
is wrong**, and worth writing down so it does not get repeated: with
`max_families_active = 2`, the two lowest incumbents after seeding were `fam-99aee6de`
(2.09) and `fam-74c41d8d` (2.84). Ranking by latency alone would still have given this
family round 1.

The narrower, real counterfactual is about **round 2**. `active_families()` orders unproven
families first, then by *improvement slope*, with absolute latency only as a tie-break:

| family | history | slope entering round 2 |
|---|---|---|
| fam-99aee6de | 2.09 → 2.09 | **0%** (stalled) |
| fam-74c41d8d | 2.84 → 2.46 | **+13.4%** (moving) |

Under slope ranking the moving challenger keeps its budget. Under latency ranking the
stalled leader — still holding the better absolute number, 2.09 vs 2.46 — would have
outranked it, and round 2 would have gone to the branch that had already failed to move
twice. Round 2 is where 2.46 → 2.09 happened.

So the claim this run supports is not "a doomed branch was rescued". It is: **a family that
was behind on latency but ahead on slope converted its round into the run's joint-best
result, while the latency leader spent three rounds going nowhere.** That is the same shape
as the L3:43 evidence already cited in `active_families()`'s docstring — leader
`[19.6, 19.6, 19.6]`, challenger `[19.5, 17.9, 17.9]` — now reproduced on a second task,
and more sharply, because here the challenger started 0.75ms *behind* rather than level.

## The K expansion is part of the story

The seed only reached 2.84ms because improvement K widened `BLOCK_P` and `NUM_STAGES`
(3.55 → 2.84). At 3.55 it would have ranked third of three live families and, with
`max_families_active = 2`, would have been genuinely at risk. So the two mechanisms
compound: K made the family competitive on latency, slope ranking kept it in the budget
once it was moving.

Worth noting K's overall record on this run is mixed and honestly reported elsewhere: five
expansions, two of which were byte-identical no-ops costing 53 trials
(fixed in `4030458`), one flat, and this one — the only expansion that changed an outcome.

## What must not be claimed

Both H1 candidates in this family were rejected by the gate and never timed. So this family's
result says nothing about whether the tensor-core direction would have been faster still;
per `finding-unreachable-correctness-gate.md` that stays **unknown, not disproven**. The
2.09ms figure is the best of the *measurable* half of the search.
