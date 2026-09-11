# The S2 switch reaches the agent, and the two arms differ in exactly one section

**First direct evidence.** Until now the arms' separation rested on `DIMENSION_STATE.prompt_mode`
(box 1 `label`, box 2 `vector`) — which is the *orchestrator saying what it intended*. This is the
document the agent was actually handed, read off both sandboxes on disk while the runs were live.

## What each arm receives

`analysis/bottleneck.md`, rendered by `_bottleneck_doc` in `agents/modules.py`. Both arms:

```
## What this task requires (measured on the REFERENCE, so it applies to every candidate)
## What this GPU can actually do (measured on THIS box, not a datasheet)
```

Then, and only then, they diverge:

| arm | third section | lines |
|---|---|---|
| box 1 (control) | `## Verdict: **resource_limited**` | 63 |
| box 2 (treatment) | `## Resource state, one line per dimension` | 53 |

## The separation is mutually exclusive, measured both ways

A one-directional check would pass on a document that carried both:

- box 2 contains `Verdict`: **0 times**
- box 1 contains `Resource state, one line`: **0 times**

That is what J2-3 requires — the digest *replaces* the label rather than being appended beside it.
Had both appeared, the treatment arm would be strictly more informed for two reasons at once and no
latency difference could be attributed to the vector.

## What the vector actually says

Not a metric dump. Each dimension carries a band and, where applicable, its own reason for having
none:

```
- **occupancy** — binding: too few resident warps to hide latency; the limiter field says which
  budget caps residency, and that budget is the knob to change
- **n_regs** — slack
- **shared_bytes** — slack
- **threads_launched** — measured; no band applies: no polarity: more threads is neither better
  nor worse, so no band applies to this reading
```

and a separate "room left" block that declines to invent headroom it cannot derive:

```
- n_regs — room left UNKNOWN: not derivable: register demand has no closed form in the knobs, and
  the SIGN is unreliable -- 13 non-monotone slices were measured, with BK 16->32 dropping 58
  registers while 32->64 added 87 and hit the 255 cap
- shared_bytes — room left UNKNOWN: not derivable: a candidate closed-form formula matched 0 of 96
  measured configurations, and the three points that appeared to confirm it were one family's
  coincidence
```

Those two refusals are the `resource-map-is-not-separable` finding reaching the agent as a stated
limit rather than as a silently absent field. `threads_launched` declaring *no polarity* is
`a-dimension-must-declare-its-binding-polarity` doing the same.

## Where the switch actually sits

Worth recording because it is not where I first looked. `_dimension_digest` returns prompt text only
in `vector` mode, and that text goes to the **analyst's** prompt as `digest_text`. The analyst's
report then becomes the rewriter's `analysis/bottleneck.json`. So the rewriter receives the treatment
*indirectly*, and its own sandbox shows only 2 of 8 dimension names with no `resource_state` key —
which is correct behaviour, not a defect. Checking the rewriter sandbox alone would have read as "the
vector never arrived".

## What this does and does not establish

Establishes: the independent variable is live, separated, and mutually exclusive on both arms.

Does not establish: that the vector *helps*. Box 2 entered Loop C at 4.91 h with its first rewriter
call; no `FAMILY_ROUND_RECORDED` exists yet on either arm, so there is no rewrite outcome to compare.
The endpoint comparison remains what `docs/result-arms-converge-to-the-same-answer.md` records — both
arms independently reached bf16/plain, 0.83% apart — which is a statement about the *tuning* phase,
before either arm had seen a rewriter prompt at all.
