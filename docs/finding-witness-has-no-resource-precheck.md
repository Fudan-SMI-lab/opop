# Finding: the default witness has no resource-feasibility pre-check, and it cost the 256-row rewrite

`run-l3-43-20260905-091705`, 11:18:20. `cand-919059a0` — the rewrite designed to answer the
one open question in `result-row-tile-is-monotone-and-k-supplies-it.md` — was rejected at its
default witness before running a single tuning trial:

```
triton.runtime.errors.OutOfResources: out of resource: shared memory,
  Required: 131072, Hardware limit: 101376.
```

Not a correctness failure and not a bug in the kernel. The **default config alone** asks for
128 KiB of shared memory on a device with a 99 KiB opt-in limit.

## Why this one matters more than an ordinary rejection

`cand-919059a0` and `cand-13efdcd8` are the two rewrites from the same analyst hypothesis, both
children of `cand-cb7be6b4`, produced in the same second (11:06:46). They attack the register
ceiling from opposite directions:

| | approach | outcome |
|---|---|---|
| `cand-13efdcd8` | **serialize** multiple 128-row tiles through persistent CTAs (`QKV_M_CTAS`) | **11.0 ms** |
| `cand-919059a0` | **widen** to a logical 256-row CTA from two 128-row groups, 16 warps | rejected, 0 trials |

So the run has a clean answer for the serialize branch and *no data at all* for the widen
branch — which is exactly the controlled 128-vs-256 comparison the record has never had.

**Update after repair (see the follow-up section): that framing was wrong.** 256 logical rows
needs 131072 B of shared memory on a 101376 B device, so the widen branch is infeasible *by
arithmetic* on this hardware, not merely un-run. The rejection cost ~3 minutes of agent wall; it
did not cost the comparison, because the comparison was never available.

## The mechanism: the witness gate has no resource pre-check

`SpaceValidator.validate` (`paramspace/validation.py:198`) runs the default config on the GPU
and treats any `ok: False` as a space rejection. There is a constraint-feasibility check on the
sampling grid *before* that, and a `_looks_out_of_range` fallback *after* it — but only for the
**minimal** witness, and only for numeric overflow:

```python
if label == "minimal" and _looks_out_of_range(result):
    alt = self._next_witness(...)   # retry at another config
```

A `default` witness that fails for a **resource** reason gets no such treatment. Yet the two
cases are structurally identical: in both, one *config* is infeasible while the *space* may be
full of feasible ones. Here the space declares `ATTN_ROWS_PER_GROUP` and `ATTN_ROW_GROUPS`
(128 × 2 = 256 rows) with `ATTN_BLOCK_N: 64` and `ATTN_NUM_WARPS: 16` at its default — the most
expensive corner in shared-memory terms — and the gate concluded the candidate was unusable.

The declared domains almost certainly contain a feasible point: 131072 / 101376 = 1.29×, so
halving `ATTN_BLOCK_N` to 32, or dropping `ATTN_NUM_STAGES` from 2 to 1, would very likely fit.
Nothing tried.

## Scale: rare overall, but it has now happened twice — both in this run, both on "go bigger" rewrites

Across all runs, `witness_default_failed` fires 72 times. Categorising the details:

| cause | n |
|---|---|
| correctness (mismatch, incl. the older bare "Output mismatch" wording) | 57 |
| other runtime error | 13 |
| **shared-memory / register `OutOfResources`** | **2** |

Both resource cases are in `run-l3-43-20260905-091705`, and both are **rewrite children produced
from a "the tile is blocked, go bigger" hypothesis**:

| candidate | required | limit | ratio | its hypothesis |
|---|---|---|---|---|
| `cand-919059a0` | 131072 | 101376 | 1.29× | logical 256-row CTA from two 128-row groups |
| `cand-2d0194cc` | 139264 | 101376 | 1.37× | `STATS_BLOCK_M=256` "to reach beyond the parent's shared-memory-blocked 128-row region" |

`cand-2d0194cc`'s default witness is `STATS_BLOCK_M: 256, STATS_NUM_WARPS: 8, …` — again the most
expensive corner of its own declared space, and again 1.3–1.4× over the limit rather than wildly
out.

**This substantially strengthens the correlation argument I made when it was n=1.** I wrote then
that "resource-hungry structures are precisely the ones a register-pressure hypothesis produces, so
the failure mode is correlated with the most interesting rewrites rather than randomly
distributed." Two of two instances are now exactly that, and both arrived within 45 minutes of each
other in the first run whose analyst reports named a resource ceiling. It is still 2 of 72 overall,
so the *rate* claim stays modest — but the selectivity is no longer speculative.

Note also what the two share mechanically: a rewriter proposes a bigger tile, the parameterizer
makes that tile the **default**, and the witness gate — which only ever tries the default and the
minimal corner — hits the one configuration guaranteed to be worst-case for shared memory. The
larger the hypothesis, the likelier the default is infeasible.

## What the harness did next — correctly

`AGENT_CALL_STARTED repair cand-919059a0` at 11:18:20, one second after the rejection. The
repair path is the designed response and it may well fix this: the error text names the exact
byte counts, which is the kind of failure detail
`finding-failure-messages-must-carry-gate-criteria.md` argues for. So the run is not stuck, and
this finding is about efficiency, not correctness.

Whether repair *should* be the mechanism is a separate question. Repair rewrites the **source**;
what is wrong here is a **default value** in the space. Asking an agent to edit a kernel because
one corner of its own parameter grid does not fit is a category mismatch — and the parameterizer
that chose the default is the component that erred.

## What I would change — NOT applied

1. **Extend the existing fallback to resource failures on the default witness.** When the
   failure is `OutOfResources`, retry at another feasible config exactly as
   `_looks_out_of_range` already does for the minimal witness, and reject only if every
   alternative fails. This reuses `_next_witness` and adds no new machinery. It is the smallest
   change and it directly preserves the anti-inertness guarantee (two distinct sources must
   still both pass a real GPU test).
2. **Or: a static shared-memory estimate at parameterize time**, so the parameterizer's default
   is checked against `device.max_shared_bytes_optin` before a GPU job is spent. More
   preventive, much more work, and easy to get wrong for a nontrivial kernel.
3. **Or: tell the parameterizer to default to a cheap corner**, not an expensive one. A prompt
   change, and prompt changes to that module feed acceptance, so it is not a free move.

**Why I stopped:** (1) touches the acceptance path — the same file and the same function whose
threshold behaviour is item #1 in `decisions-awaiting-user.md`. I am not editing that gate while
five decisions about it are outstanding, on n=1 evidence, mid-run. Recorded instead, with the
repair attempt's outcome to follow.

## Follow-up: repair recovered it in 2 minutes, and its fix confirms the diagnosis

`REPAIR_PRODUCED` at 11:20:22 — 2m02s after the rejection — and the space republished at
11:23:21 as `sp-d0921062`. The repair agent's own diagnosis:

> "The default attention launch formed BLOCK_M = ATTN_ROWS_PER_GROUP × ATTN_ROW_GROUPS =
> 128 × 2 = 256. With D_PAD = 128, the flash-attention fp32 accumulator had 256 × 128 elements
> and required 131072 bytes of shared memory, exceeding the hardware limit of 101376 bytes."

Arithmetic exactly right (256 × 128 × 4 = 131072). And its change:

> "changed **only the default ATTN_ROWS_PER_GROUP from 128 to 64** … Python syntax and diff
> checks pass; the fixed file differs from the broken file only by this parameter value."

**This is the evidence the "follow-up to watch" section asked for.** Repair did not fix the
kernel — there was nothing wrong with the kernel. It changed **one default value**, which is
precisely what a resource fallback in the witness gate would have done automatically by trying
another config. Two agent calls (repair + re-parameterize, ~3 min of wall) were spent doing by
hand what option (1) does for free. That is now direct evidence for option (1), not just an
argument for it.

The cost was small and the recovery was clean, so this stays a low-priority efficiency item
rather than a defect. But the shape of the fix is unambiguous.

### The 256-row experiment survived — the space still reaches it

Worth checking rather than assuming, because a repair that shrinks a default could easily have
shrunk the whole experiment. It did not. The republished domains are
`ATTN_ROWS_PER_GROUP: [16, 32, 64]` × `ATTN_ROW_GROUPS: [1, 2, 4]`, and the logical row tile is
their product:

| RPG × RG | logical rows | fp32 accumulator | fits 101376? |
|---|---|---|---|
| 32 × 4 | 128 | 65536 | yes |
| 64 × 2 | 128 | 65536 | yes |
| **64 × 4** | **256** | **131072** | **no** |

So 256 logical rows is still *expressible* (`64 × 4`) but not *feasible* — it is the same
131072 B that caused the rejection.

**Correction, from the trials that followed:** I wrote here that "the tuner will sample it, get a
`runtime_error`, and learn to avoid it, which costs trials". It did not, and no trials were
wasted. The republished space carries a guard constraint —
`ATTN_ROWS_PER_GROUP * ATTN_ROW_GROUPS <= 128`, rationale "caps the fused attention row tile at
the known-working default to avoid explosive score and output state" — so the 256 corner was
never sampled in 40 trials. The guard did the work rather than the failure feedback.

I could not establish whether that constraint predates the repair: `SPACE_REJECTED` does not
carry the proposed space, and the parameterizer sandboxes keep only source files, not the emitted
`space.json`. If the cap *was* already there before the rejection, then the parameterizer emitted
a default that violated its own constraint, which would be a distinct and more interesting bug.
No evidence either way, so it is recorded as an open question, not a claim.

**That gap is itself worth noting as an observability limit:** a rejected space is the one case
where the space definition would be most useful for diagnosis, and it is exactly the case where
nothing is journalled. `SPACE_REJECTED` carries `candidate_id`, `attempt`, `reason`, `detail` —
not the domains or constraints. Adding the space to that payload would be purely observational
and is the same shape as the `REPAIR_REVERTED` proposal in
`finding-parameterizer-reverts-the-repair.md`. Not applied.

**Conclusion for the 256 question: this rewrite cannot answer it either.** Not because of the
rejection, but because 256 rows × 128 D_PAD × 4 bytes exceeds this device's shared memory
*as a matter of arithmetic*. The widen branch is not blocked by the harness; it is blocked by
the hardware, and no repair or fallback changes that. Both witnesses passed at 128 logical rows
(15.6 ms and 86.7 ms).

That also retires the concern in the section above about "losing the comparison": the comparison
was never available on this device via this structure. `cand-13efdcd8`'s serialization approach
(`QKV_M_CTAS`, 11.0 ms) is not a second-best substitute for widening — on a 99 KiB shared-memory
budget it is the *only* way to get 256 rows' worth of reuse, which is what the analyst's
hypothesis said and why it proposed distributing work across warps rather than enlarging the
accumulator.
