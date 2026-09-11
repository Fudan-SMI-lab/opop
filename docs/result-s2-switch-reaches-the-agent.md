# The S2 switch reaches the agent, and the CONTROL arm is the one with the raw numbers

**First direct evidence.** Until now the arms' separation rested on `DIMENSION_STATE.prompt_mode`
(box 1 `label`, box 2 `vector`) — which is the *orchestrator saying what it intended*. This is the
document the agent was actually handed, read off both sandboxes on disk while the runs were live.

> **Read the "Correction" section before quoting the table below.** The two arms differ in one
> SECTION HEADING, which is what this document originally recorded — but the control arm's section
> also carries a 31–32-line raw metric dump that the treatment arm's does not. The independent
> variable is *judgement instead of a raw dump*, and the treatment arm gets LESS raw data, not more.

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

Not a metric dump — and that is now a measured claim rather than a description, since the control
arm's document IS one (see the Correction). Each dimension carries a band and, where applicable, its
own reason for having none:

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

## Correction: the control arm is NOT the number-free arm — it gets MORE numbers

Recorded after reading both documents field by field, because the earlier version of this file
described only the section headings and left the wrong impression.

The control document's `## Verdict` section is followed by `### The numbers behind it`: a **raw dump of
31-32 `- \`key\` = value` lines** — `n_regs`, `n_spills`, `occupancy`, `occupancy_limiter`,
`shared_headroom_bytes`, `shared_used_frac`, `reg_headroom_per_thread`, `achieved_tflops`,
`pct_of_compute_peak`, `pct_of_dram_peak`, `arithmetic_intensity`, `peak_alloc_mib`, and more. The
treatment documents carry **zero** such lines. Counted across every document on disk:
**8 of 8 control docs carry 31 or 32 raw metric lines; 10 of 10 treatment docs carry 0.** Clean
separation, no overlap, both directions.

| | control (`label`) | treatment (`vector`) |
|---|---|---|
| raw `key = value` metric lines | **31-32**, in 8 of 8 docs | **0**, in 10 of 10 docs |
| per-dimension judgements (binding/slack, with polarity) | 0 | 8 |
| explicit "not derivable" refusals | 0 | 2 (`n_regs`, `shared_bytes` room) |
| a single scalar verdict | `resource_limited` + prose | — |

So the independent variable is **judgement instead of a raw dump**, not *numbers versus no numbers*,
and the treatment arm is the one given LESS raw data. That is the KernelPro contrast the design cites
— their raw-counter arm reached 1.77x against 3.35x for no feedback at all — and it makes the
hypothesis falsifiable in the honest direction: if the vector is merely a lossy summary, the control
arm should win.

### A measurement that misled me, and why it is not evidence

Counting reasons that contain the word "measured" or a `before -> after` pair gives control 6 and
treatment 0. That looked like the control arm citing observations and the treatment reasoning
qualitatively — the opposite of the intended effect. It is not: the control's `measured 49152->24576`
is the agent's **own arithmetic** over tile shapes and dtypes (its document contains neither number;
`grep -c 49152` returns 0 across all 8 analyst docs). What it did have was `shared_headroom_bytes =
52224` and the 49152 static cap, from which those figures are derivable.

So the keyword count measured PROSE STYLE, not information. Recorded because it is the same shape as
the recorded `cross-dimension argmax` error — a statistic that tracks the presentation rather than the
quantity — and because a style difference between the arms is real but must not be reported as an
information difference.
