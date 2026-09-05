# Measurement: K spent 40 trials confirming what the analyst had said two minutes earlier

`run-l3-21-20260905-071312`, `cand-d31b0474`. Not a bug — every component behaved as designed —
but a concrete instance of two mechanisms answering the same question independently, in the wrong
order, at a cost of 40 GPU trials.

## The sequence, from the event log

```
08:05:20  TUNING_DONE          sp-df786ac5   17.10 ms
08:07:04  BOTTLENECK_REPORTED                (analyst: no blocked headroom at these knobs)
08:09:22  SPACE_PUBLISHED      sp-1636fa27   (K expansion, v3)
08:09:22  SPACE_EXPANDED                     PW_BLOCK_N, PW_WARPS, FINISH_WARPS -- all "max"
08:14:35  TUNING_DONE          sp-1636fa27   17.10 ms
```

The analyst's report **precedes the expansion by 2 minutes 18 seconds**, and says, of exactly
those three knobs:

> "PW_BLOCK_N=128, PW_WARPS=8, and FINISH_WARPS=8 are search-space boundaries, but **no trial
> beyond them failed from registers, shared memory, threads, OOM, or compilation**. At the 17.1 ms
> best configuration, usage is balanced well below hardware limits: 78/255 registers per thread
> (30.6%), 32,768/101,376 B opt-in shared memory, 256/1024 threads, and zero spills… **neither
> registers nor shared memory is the performance block.**"

K then widened all three anyway: `PW_BLOCK_N` +{256, 512}, `PW_WARPS` +{16, 32},
`FINISH_WARPS` +{16, 32}.

## The re-tune confirmed the analyst, empirically

40 trials, and the result was **17.10 → 17.10 ms, exactly flat**. Breaking the trials down by
whether they used any newly-added choice:

| | trials | completed | best |
|---|---|---|---|
| used a new choice | 15 | 8 | **18.20 ms** |
| used only old choices | 25 | 17 | **17.10 ms** |

Every new choice is strictly worse than the incumbent. The best new-choice trial
(`PW_WARPS=16`, fp16) is 6.4% slower; the others land at 18.40, 19.90, 20.50, 34.50, 38.20,
41.90 and 130.00 ms. Eight completed trials is enough to say the widened region is genuinely
unproductive, not merely unlucky.

So the boundary was a boundary because **the optimum is interior** — which is the answer the
analyst gave from resource statistics before a single new trial ran.

## Why this happened

`boundary_knobs_to_expand` decides from the tuning statistics alone: a knob whose argmin sits at
an extreme choice with a monotone approach is a candidate for widening. That is a reasonable
heuristic and it is the mechanism the paper's improvement K describes.

The analyst's `BottleneckReport` answers a strictly stronger version of the same question — it
has the resource evidence (registers, shared bytes, threads, spills, per-value failure rates)
and can distinguish "at the boundary because the hardware stopped it" from "at the boundary
because that is simply the best value". Nothing connects the two: K never reads the report.

Note this is not the `SPACE_EXPANSION_REJECTED` case fixed in `4030458` — that catches an
expansion that returns *byte-identical* domains. Here the domains really did widen; the widening
was just useless.

## Cost, and what it is worth

40 trials at roughly 11 s each is about **7.5 minutes of GPU wall**, plus one parameterizer call.
Small in isolation. But this is the run's *best* candidate, the expansion budget is finite, and
the same 40 trials spent on an interior region — or given to another family — would have been
free of a known-negative prior.

Against that: the flat result is genuine evidence, and 8 completed new-choice trials
independently corroborate the analyst's claim from a different kind of measurement. A reader of
the report now knows the interior optimum is real rather than an artefact of a too-narrow domain.
That has some value; it is just expensive for a question that was already answered.

## Possible change — not applied, and not obviously correct

The obvious move is to let `boundary_knobs_to_expand` consult the latest `BottleneckReport` and
skip a knob the analyst explicitly cleared. Two reasons to be careful:

1. It makes a deterministic mechanism depend on an LLM judgement. The harness's design principle
   is that agents advise and the harness decides
   (`ConvergencePolicy` docstring, `plan §四/control`), and K is currently on the harness side of
   that line. Wiring the report in moves it across.
2. The analyst can be wrong. Here it was right and checkable, but a false "no headroom" would
   silently suppress a productive expansion, and that failure is invisible where the current
   false-positive is merely expensive.

A safer variant: keep K's decision, but **record** the disagreement — journal that the analyst
had cleared the knob when K expands it anyway, so the cost is attributable afterwards rather
than requiring a timeline reconstruction. That is observational and cheap, in the same spirit as
the `REPAIR_REVERTED` proposal in `finding-parameterizer-reverts-the-repair.md`.

K's overall record across runs stays mixed and is honestly reported elsewhere
(`result-second-ranked-family-catches-the-leader.md`): one expansion that changed an outcome
(3.55 → 2.84 ms), two byte-identical no-ops since fixed, one flat, and now this one — flat, with
the extra detail that the flatness was predicted.
