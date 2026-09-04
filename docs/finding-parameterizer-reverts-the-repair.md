# Finding: the parameterizer can silently revert the repair it was handed

Found live on the L3:21 rerun, 2026-09-05 07:22–07:43. **4 of 76** repair→parameterize hand-offs
across all runs, and when it happens the repair loop cannot converge, because the agent is asked
to fix a source that has had its fix removed.

> **Scope corrected twice as the run continued.** I first described this as one candidate's
> problem, then as a pattern that had hit two. The final shape on L3:21: it hit **both**
> tensor-core candidates and consumed **both of their entire repair budgets**.
>
> | candidate | minimal witnesses run | distinct sources | source |
> |---|---|---|---|
> | cand-6b313c39 | 3 | **1** | `ebb81c3f8116`, 3965 bytes, 4 dots |
> | cand-89fa74fe | 3 | **1** | `cdc899e62765`, 4178 bytes, 4 dots |
>
> Six GPU witness evaluations, **two distinct programs**. Every repair attempt on this task
> measured a source its own predecessor had already been rejected for. Cost so far: 4 wasted
> agent calls of 16, roughly **11.8 min of 23.7 min** of repair+parameterizer wall.
>
> It is triggered by the repair *pattern*, not the candidate: every instance is a
> split-precision (hi/lo residual) fix inside an fp16 branch, from two different rewriter
> lineages.

## Loop A's shape makes this possible

`_parameterize_with_repair` (`orchestrator.py:540`) iterates:

```
repair(broken) -> fixed  ->  parameterizer(fixed) -> parameterized  ->  validate(parameterized)
```

The thing **validated** is the *parameterizer's* output, not the repair's. The parameterizer's
job is to route tunable values through `PARAMS`, and it is a free-form LLM edit — so nothing
prevents it from rewriting the body it was given. When it does, the repair never reaches the
GPU.

## What happened

`cand-6b313c39`. Its two substantive repairs both implemented the same correct technique —
2-term split-precision (Dekker-style) emulation, which is the textbook answer to "fp16 operand
rounding loses accuracy":

```python
x_hi = x.to(tl.float16)
w_hi = w.to(tl.float16)
x_lo = (x - x_hi).to(tl.float16)      # the residual
w_lo = (w - w_hi).to(tl.float16)
acc += tl.dot(x_hi, w_hi, input_precision="ieee")
acc += tl.dot(x_lo, w_hi, input_precision="ieee")
acc += tl.dot(x_hi, w_lo, input_precision="ieee")
acc += tl.dot(x_lo, w_lo, input_precision="ieee")
```

The parameterizer that consumed it emitted:

```python
acc += tl.dot(x.to(tl.float16), w.to(tl.float16))     # back to a single cast
```

By signature, `(bytes, tl.dot count, @triton.jit count)`:

| stage | signature |
|---|---|
| repair `fixed.py` (a=1) | (4663, **10**, 1) |
| parameterizer emitted | (3965, **4**, 1) |
| repair `fixed.py` (a=2) | (4217, **7**, 1) |
| parameterizer emitted | (3965, **4**, 1) |

Both collapse to the *same* 3965-byte, 4-dot source — byte-identical to what a=1 already ran.

**Corrected as the run continued: it was three attempts, not two.** `cand-6b313c39` reached
a=3, and all three minimal witnesses ran source sha `ebb81c3f8116abe0`, 3965 bytes, 4 dots:

```
cand-6b313c39-wit-minimal-eval-127f3c41   sha=ebb81c3f8116abe0  3965 bytes  4 dots
cand-6b313c39-wit-minimal-eval-903fde0f   sha=ebb81c3f8116abe0  3965 bytes  4 dots
cand-6b313c39-wit-minimal-eval-8e16f314   sha=ebb81c3f8116abe0  3965 bytes  4 dots
```

3 witnesses, **1 distinct source**, and metrics identical to six decimals every time
(0.960593 / 0.978946). So the revert consumed the candidate's **entire** `repair_attempts`
budget of 3 — every attempt measured the same program, and the candidate was dropped as though
three repairs had failed.

## Why the loop then cannot converge

The next repair reads `broken.py` = the parameterizer's output. So repair a=2 was shown the
**reverted** source, not its own previous fix. Its diagnosis says so, accurately, about code
that no longer contained the fix:

> "In that branch, the expansion kernel rounded both FP32 operands to a single FP16 component
> before the dot product."

That is a correct reading of what it was given, and it is the exact defect a=1 had already
repaired. The agent is being asked to re-derive a fix that was silently discarded — and it
did, producing a=2's 7-dot version, which was reverted again.

`REPAIR_PRODUCED.source_sha` records the *repair's* output (`184ccf9b`, then `f7550424`), so
the event log shows two distinct repairs and gives no hint that neither was validated. Without
comparing sandbox files this reads as "repair tried twice and failed twice".

## Why it is rare, and what likely triggers it

72 of 76 hand-offs preserved the repair's structure exactly or near-exactly (small byte deltas
from re-indentation). All four failures are in the same run, across two candidates, and every
one shares a trait the survivors lack: **they violate the candidate contract.**

`candidate_contract.md` requires exactly one module-level `PARAMS` dict whose values are the
tunable knobs, and asks that a low-precision cast's dtype come *from* a knob. These repairs
hard-code a multi-term expansion inside the `fp16` branch — the dtype still comes from
`COMPUTE_DTYPE`, but the *number of dot products* is now branch-specific and unexpressed in
`PARAMS`. A parameterizer told to normalize a candidate onto the contract has a defensible
reason to rewrite that body. It just has no way to say "I am discarding your fix", and no
mechanism stops it.

That the trigger is consistent across two candidates makes this predictable rather than random,
and it means the fix has a clear target: **a repair that legitimately needs to add dot products
has no way to express that within the contract.** Option (2) below (tell the parameterizer to
preserve the body) addresses the symptom; the deeper answer is that the contract should let a
repair declare "this branch now has N terms" so the parameterizer has nothing to normalize
away.

## Cost

Small in frequency, larger in consequence than the count suggests: the 2 affected hand-offs are
2 of the same candidate's 3 repair attempts, so **one candidate's whole repair budget was spent
re-measuring one program**. The failure mode is bad out of proportion to how often it fires:

- it burns a repair attempt from a budget of 3, with no signal that it did;
- it makes the repair loop look incapable when it is being undone;
- it corrupts any analysis that reads `REPAIR_PRODUCED` diagnoses as evidence about what the
  repair agent can do. The a=2 diagnosis on this candidate is *correct about its input* and
  would read as a repeated misdiagnosis to anyone not checking sandbox shas. I have written
  several such analyses today.

## Proposed fix — not applied

Cheapest first, and none of these changes acceptance semantics:

1. **Journal the discrepancy.** Compare the parameterizer's output against its input on the
   axes that matter (`tl.dot` count, `@triton.jit` count, byte size) and append a
   `REPAIR_REVERTED` event when the output has *fewer* dots or is materially smaller. Purely
   observational, ~15 lines, and would have surfaced this in the event log instead of
   requiring a sandbox archaeology pass. Do this regardless of the others.
2. **Tell the parameterizer not to.** Its prompt does not currently say "preserve the body you
   are given; you are routing constants, not rewriting arithmetic". One sentence.
3. **Re-run the repair's own source as a witness on disagreement.** If (1) fires, validate the
   repair's `fixed.py` too and prefer it if it passes. Costs one GPU quick test on a rare path.

(1) and (2) together are cheap and low-risk. (3) is more invasive and should wait for evidence
that (2) is insufficient.

**Not applied because the L3:21 run is in flight** — all three are driver-side, so they cannot
affect it, and the right time is with the batch awaiting the user's decision. Recorded now
with the run's own evidence while the sandboxes are on disk.

`scripts/audit_repair_survival.py` runs the check over any set of runs.
