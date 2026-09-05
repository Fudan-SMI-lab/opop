# Result: the dtype DIRECTION decides whether a floor rejection is recoverable — 5/5 up, 0/3 down

`docs/finding-floor-rejection-sends-repair-after-the-dtype.md` records that a correctness
rejection with no reproducible logic bug leaves repair one lever it can always pull — precision —
and that **0 of 3** recovered that way. That reading was too pessimistic, and the reason is
visible once the *direction* of the move is separated from the fact of it.

Triggered by `cand-fe183b2d` on the clean L3:21 rerun at 16:15: rejected at
`frac_within_tol = 0.953277`, repaired, and **published**. The repair changed one thing —
`COMPUTE_DTYPE` default from `tf32` to `ieee`.

## The split

Every candidate whose correctness rejection recorded a dtype, and what its final default became:

| dtype move | candidates | published a space |
|---|---|---|
| `tf32` → **`ieee`** (more precision) | 5 | **5 (100%)** |
| `tf32` → **`fp16`** (less precision) | 3 | **0 (0%)** |

```
tf32 -> ieee    l3-21 cand-d31b0474   cand-7dcdbd99   cand-fdb4dac6   cand-fe183b2d
                l3-43 cand-cb7be6b4
tf32 -> fp16    l3-21 cand-6b313c39   cand-89fa74fe    (default stayed tf32, fp16 witness corner)
                l3-43 cand-90886b3c                    (oscillated fp16 -> tf32 -> fp16)
```

The direction is the whole story. Moving toward more precision reduces the deviation the
rejection complained about, and the gate is a deviation test — so it works, every time it was
tried. Moving toward less precision *increases* that deviation, and additionally walks into the
`witness_minimal_failed` corner (`memory: opop-v2-minimal-witness-is-fp16-corner`), so it fails
twice over.

## What was wrong with the earlier reading

The earlier finding counted "repair switched the dtype" as one behaviour and scored it 0/3. It
was drawing on three candidates that all happened to move *downward* — and `cand-90886b3c`,
which oscillated. Pooling those with the upward moves was the error; the corrected statement is
that the lever is not the problem, the sign is.

That also means the note's practical suggestion — suppress repair dispatch when the failure
detail already shows the candidate above the floor — would have prevented **five recoveries**,
including today's. Withdrawing that suggestion.

## The prompt is already right, and that is the useful part

`_repair_guidance("correctness_mismatch")` says, verbatim:

> A SMALL error just over tolerance is a precision issue: accumulate dot products/reductions in
> fp32 (**use `input_precision="ieee"` for `tl.dot` on fp32 refs**), check reduction order and
> masking of padded lanes, …

So the 5 successes are the guidance being followed and the 3 failures are it being ignored in
favour of the opposite move. Nothing in the harness needs changing for the successful path to
work — it already works. The open question is only why three cases went the other way, and with
n=3 I am not going to theorise about it.

**No fix applied.** A rule forcing the dtype upward after a correctness rejection would be a
plausible next step and I am not taking it on this evidence:

- It would override the agent's diagnosis on a case where the real cause *is* a logic bug and
  the dtype is incidental — and the guidance's own first paragraph is about exactly that case
  (a large systematic offset from train-mode BatchNorm, which no precision change fixes).
- `ieee` is the slowest path on this hardware. Forcing it converts a correctness rejection into a
  performance handicap silently, which is the shape of problem
  `finding-unreachable-correctness-gate.md` is about.
- 5/5 and 0/3 on n=8 is suggestive, not settled.

What it does change is the ledger: floor-adjacent rejections are **recoverable**, at least on
L3:21, and the recovery is already available through the existing prompt. That is a better
position than the earlier note implied.

## Cost, for completeness

`cand-fe183b2d`: rejected 16:12:37 → repair → published 16:15:59. **3.4 minutes** and one GPU
witness pair, for a candidate that is now in the search. Compare the never-published group: 25 of
37 candidates with a correctness rejection never published a space at all, most having burned all
4 repair attempts.

Reproduce with `python scripts/audit_noise_floor_rejections.py`.
