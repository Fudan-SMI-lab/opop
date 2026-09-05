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
branch — which is exactly the controlled 128-vs-256 comparison the record has never had. The
rejection did not merely lose a candidate; it lost the comparison.

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

## Scale: rare overall, but this is the first instance

Across all runs, `witness_default_failed` fires 71 times. Categorising the details:

| cause | n |
|---|---|
| correctness (mismatch, incl. the older bare "Output mismatch" wording) | 57 |
| other runtime error | 13 |
| **shared-memory / register `OutOfResources`** | **1** |

So this is **n=1** and I am not claiming a systemic pattern from it. What makes it worth writing
down is not frequency but *which* candidate it hit: the resource-hungry structures are precisely
the ones a register-pressure hypothesis produces, so the failure mode is correlated with the
most interesting rewrites rather than randomly distributed. A 1-in-71 event that selectively
removes the "go bigger" branch of every such experiment is worse than its rate suggests.

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

## Follow-up to watch

- Does repair recover `cand-919059a0`? If it does, the cost is ~2 agent calls; if it does not,
  the widen branch is lost for this run and the 256 question stays open.
- If repair's fix is to shrink the default rather than the kernel, that is direct evidence for
  option (1) or (3) over the current behaviour — the repair agent would be doing by hand what
  the fallback would do automatically.
