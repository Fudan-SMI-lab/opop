# Measurement: improvement K re-buys the same widened range within a lineage

Found live. `run-l3-43-20260906-091019` expanded `cand-1129b4d9` on `BLOCK_N/max`, adding
`BLOCK_N=256`, and measured it at 17.9 ms against a 16.8 incumbent — worse, and unused by the
re-tune's winner. One rewrite round later it expanded that candidate's **child**,
`cand-3df5fd86`, on the same knob in the same direction, adding the same 256 to a kernel whose
ladder is nearly identical:

```
cand-1129b4d9  BLOCK_N  16:52.00  32:30.80  64:21.30  128:16.80   -> added 256 = 17.90 (worse)
cand-3df5fd86  BLOCK_N  16:52.60  32:31.50  64:20.70  128:17.00   -> added 256 again
```

Nothing in the expansion path consults what a related candidate already learned about the same
knob, so the second expansion pays a parameterizer call plus a fresh `trials_per_space` budget to
re-discover a measurement the run took twenty minutes earlier.

## How often, across every run on disk

`scripts/audit_expansion_repeats_within_lineage.py`, 18 runs, 57 expansions:

```
LINEAGE repeats (ancestor/descendant, same knob + direction)   16
FAMILY repeats  (same family, not directly related)            11
```

So roughly **half of all expansions repeat a (knob, direction) already expanded inside the same
family**, and 16 of them do it on a direct ancestor or descendant, where the earlier measurement
is most likely to transfer.

## But the repeats are not uniformly waste, which is why nothing is being changed yet

`scripts/audit_expansion_repeat_value.py` scores the EARLIER expansion of each repeated pair —
did its added values get reached, and were any better than that space's incumbent?

```
earlier added value reached and WORSE  -> the repeat re-buys a known dud   12
earlier added value BEAT the incumbent -> the repeat is arguably rational   9
earlier added value never sampled      -> no information to reuse           8
```

Twelve clear duds is a real cost — each one is a parameterizer call and 40 trials spent on a
value already shown not to help on a sibling. But nine repeats followed an expansion that
**did** help, and eight followed one that produced no information at all. A blanket
"don't repeat a knob within a family" rule would have suppressed those nine, including:

```
[HELPED] GEMM_BLOCK_N  incumbent 15.6 -> added best 14.7   (cand-80bf3097 -> cand-f66890d0)
[HELPED] GEMM_BLOCK_N  incumbent 15.9 -> added best 14.4   (cand-8510db1b -> cand-47371017)
[HELPED] NUM_STAGES    incumbent 19.0 -> added best 18.8   (cand-b9b38e21 -> cand-d842feb9)
[HELPED] BLOCK_D       incumbent 21.9 -> added best 21.0   (cand-c36d7820 -> cand-d257924a)
```

And the reason a repeat can be rational is structural: a rewrite **changes the kernel**. The whole
premise of the outer loop is that a different structure has a different optimum, so
`BLOCK_N=256` failing on a 128×128-tile parent genuinely does not prove it fails on a child that
repartitioned to 256×64 — which is exactly what `cand-3df5fd86` did. The value that transfers is
weaker than it looks.

What the numbers do NOT support is the strong claim I was tempted to make ("K wastes half its
budget re-testing known duds"). 12 of 57 expansions is 21%, and only if one accepts that a
sibling's measurement transfers across a structural rewrite.

## How the live repeat actually resolved: the tuner ignored the added value

The repeat completed while this was being written, and it is a cleaner outcome than the
retrospective categories allow for. `cand-3df5fd86`'s expanded space `sp-ecca8f39`:

```
BLOCK_N=16    n= 8  best 51.50
BLOCK_N=32    n= 6  best 31.20
BLOCK_N=64    n= 7  best 20.70
BLOCK_N=128   n=19  best 16.70   <- winner
BLOCK_N=256   n= 0  NEVER SAMPLED
```

Result 17.0 → **16.7 ms** (1.8%, below `min_improvement_pct`), winner
`BLOCK_M=128 BLOCK_N=128 BLOCK_K=32 NUM_WARPS=4 NUM_STAGES=3 COMPUTE_DTYPE=fp16` — every value
legal before the expansion.

So the added value was not merely worse this time; **TPE never tried it at all.** Given anchors
showing 128 fastest and a steep penalty below it, the sampler spent all 40 trials in the region it
already believed in. That is the sampler behaving correctly, and it makes the repeat's cost
concrete in a different way than the "reached and worse" category: the widened range was pure
overhead — one parameterizer call, zero trials spent on what it bought.

It also means the `NEVER SAMPLED` bucket in the table above (8 of 21) is not an absence of
evidence about repeats; it is evidence that a widened range **frequently goes unexplored** because
the tuner has already localized. Which weakens the case for expansion-by-widening generally, and
strengthens the reading that the fresh budget is the part that pays — consistent with all four of
this run's expansions:

```
cand-8c64ccc3  BLOCK_M=128 added   tried  5x, tied            19.8 -> 19.8
cand-a988ff79  5 values added      all 2-27% worse            21.3 -> 20.9
cand-1129b4d9  BLOCK_N=256 added   6.5% worse                 16.8 -> 16.8
cand-3df5fd86  BLOCK_N=256 added   NEVER SAMPLED              17.0 -> 16.7
```

Four expansions, correctly aimed every time, added value used zero times, and every gain
attributable to re-tuning.

## Why this is recorded rather than fixed

- Any rule strong enough to stop the 12 duds also stops the 9 helpful repeats, at n=21 total.
  A threshold fitted to that split would be fitted to noise.
- The information needed to tell them apart — whether this candidate's structure changed the
  knob's optimum — is only knowable after the trials are spent, the same objection that closed
  `measurement-expansion-spend-by-competitiveness.md`.
- The cheap partial remedy is not a gate but **evidence**: the parameterizer already receives an
  `expand_directive`, and a line naming what a related candidate measured for that knob
  ("`BLOCK_N=256` was tried on the parent at 17.9 vs 16.8 incumbent") would let the agent decide
  whether its rewrite changes the calculus. That is same-lineage information about the same knob,
  not cross-candidate report sharing, so it does not touch the decision recorded against that.
  Not implemented here: it is a prompt change on the live experiment's critical path, and the
  three L3 tasks need to finish on comparable settings first.

Recorded for the next round of improvement planning, with the split above as the thing any proposed fix
has to beat.

Reproduce with `python scripts/audit_expansion_repeats_within_lineage.py` and
`python scripts/audit_expansion_repeat_value.py`.
