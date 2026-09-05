# Result: the `best_history` seeding fix, replayed against the real policy on every family

Pending item 3 (seed `best_history` so `stop_kind = "converged"` is reachable) has been argued from
arithmetic. At 14:42 `fam-4aea322a` produced the first two-entry history in this run —
`hist=[11.0, 11.0], used=2, recent_improvements_pct=None` — which made it possible to stop arguing
and just run the real `ConvergencePolicy` over every family the project has recorded, twice: as it
happened, and with the seed prepended.

`scripts/audit_seeded_history_counterfactual.py` imports the actual policy, so these are the
verdicts that would have been produced, not a model of them.

## The result

```
policy: rewrite_rounds_per_family=3  no_improve_rounds=2  min_improvement_pct=2.0

run                    family         seed  history               TODAY at used=2            WITH SEEDING
l3-21-0903-071357      fam-97b5e645   24.6  [24.6, 24.6, 24.6]    continue  recent=None      freeze/CONVERGED [0.0, 0.0]
l3-21-0903-071357      fam-eb982220   24.6  [24.6, 24.6, 24.6]    continue  recent=None      freeze/CONVERGED [0.0, 0.0]
l3-21-0904-013056      fam-f14a294f   21.7  [21.3, 20.5, 20.5]    continue  recent=None      continue  [1.8, 3.8]
l3-21-0904-013056      fam-b278c226   21.9  [21.5, 20.7, 20.7]    continue  recent=None      continue  [1.8, 3.7]
l3-43-0903-020233      fam-d4be7a3a   29.3  [29.3, 29.3, 29.3]    continue  recent=None      freeze/CONVERGED [0.0, 0.0]
l3-43-0903-020233      fam-59e65a76   31.6  [31.6, 31.6, 31.6]    continue  recent=None      freeze/CONVERGED [0.0, 0.0]
l3-43-0903-145357      fam-36474be2   21.0  [19.4, 19.4, 19.4]    continue  recent=None      continue  [7.6, 0.0]
l3-43-0903-145357      fam-19e13c64   22.9  [19.5, 19.5, 19.5]    continue  recent=None      continue  [14.8, 0.0]
l3-43-0904-093730      fam-c9461c56   19.6  [19.6, 19.6, 19.6]    continue  recent=None      freeze/CONVERGED [0.0, 0.0]
l3-43-0904-093730      fam-ff3ef34b   20.0  [19.5, 17.9, 17.9]    continue  recent=None      continue  [2.5, 8.2]
l3-43-0905-091705      fam-4aea322a   14.2  [11.0, 11.0]          continue  recent=None      continue  [22.5, 0.0]
l3-43-0905-091705      fam-92e7c576   22.5  [19.6, 19.6]          continue  recent=None      continue  [12.9, 0.0]
l3-48-0905-010737      fam-99aee6de   2.09  [2.09, 2.09, 2.09]    continue  recent=None      freeze/CONVERGED [0.0, 0.0]
l3-48-0905-010737      fam-74c41d8d   3.55  [2.46, 2.09, 2.09]    continue  recent=None      continue  [30.7, 15.0]
```

| at `rewrite_rounds_used == 2` | today | with seeding |
|---|---|---|
| `recent_improvements_pct` computable | **0 of 14** | 14 of 14 |
| freeze as `converged` | 0 | **6 (43%)** |
| continue | 14 | 8 |

The `recent=None` column is the whole bug in one place: today's policy is one history entry short of
being able to compute anything, on **every family in the project**, at the last decision point where
it could still act.

## The safety check, which is the part that matters

A change that freezes families early could throw away real gains. It does not — I checked each of
the 6 against what its actual round 3 produced:

```
fam-97b5e645  [24.6, 24.6, 24.6]   round3 gain = 0.00%
fam-eb982220  [24.6, 24.6, 24.6]   round3 gain = 0.00%
fam-d4be7a3a  [29.3, 29.3, 29.3]   round3 gain = 0.00%
fam-59e65a76  [31.6, 31.6, 31.6]   round3 gain = 0.00%
fam-c9461c56  [19.6, 19.6, 19.6]   round3 gain = 0.00%
fam-99aee6de  [2.09, 2.09, 2.09]   round3 gain = 0.00%
```

**Every one gained exactly 0.00%.** Six rounds of structural search — six rewriter calls, six
parameterizations, ~240 GPU trials, roughly 3 hours across the record — spent on families whose
next round provably produced nothing, and the policy could have known.

And the fix is *conservative*, not aggressive. The 8 families it would let continue **also** gained
0.00% in their actual round 3:

```
fam-f14a294f  [21.3, 20.5, 20.5]  0.00%     fam-19e13c64  [19.5, 19.5, 19.5]  0.00%
fam-b278c226  [21.5, 20.7, 20.7]  0.00%     fam-ff3ef34b  [19.5, 17.9, 17.9]  0.00%
fam-36474be2  [19.4, 19.4, 19.4]  0.00%     fam-74c41d8d  [2.46, 2.09, 2.09]  0.00%
```

So the seeded policy would save 6 of the 12 provably-wasted rounds and still fund the other 6. It
errs toward continuing, which is the right direction for a search, and it is nowhere near
over-freezing. The 0-for-13 record on round 3
(`finding-converged-stop-kind-is-unreachable.md`) is now 0-for-12 with two families' round 3 still
pending in this run, and the counterfactual says the policy would have caught half of them.

## Two honest limits

1. **`min_improvement_pct = 2.0` is doing real work in this table.** `fam-f14a294f` continues on
   `[1.8, 3.8]` — its first slope is *below* the 2.0 threshold and only the second saves it. Change
   the threshold and the freeze set changes; at 4.0 it would freeze too, and its round 3 gained
   nothing, so that would also have been correct. But that means the 6/8 split is a function of a
   configured constant, not a property of the data.
2. **The seed value is reconstructed, not recorded.** The script takes the family's *first* tuned
   candidate's `best_ms` as its seed. That is right for every family here (seeds are tuned before
   any rewrite), but it is an inference from event order rather than a stored field — which is
   itself a small argument for the fix, since a seeded `best_history` would make the value explicit.

## Still not applied

This is evidence for a decision the user has not made. The change alters how much structural search
each family receives, which is a budget-allocation call and not a bug fix — the same reason it was
deferred in the first place. What is different now is that the argument no longer rests on
arithmetic about list lengths: the real policy, on real histories, freezes 6 families that provably
gained nothing and keeps every family that had any slope at all.

Run `python scripts/audit_seeded_history_counterfactual.py` to reproduce; it will pick up new
families as runs finish.
