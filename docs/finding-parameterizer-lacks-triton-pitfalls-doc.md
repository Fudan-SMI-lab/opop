# Finding: the parameterizer is the only Triton-writing agent that never gets the pitfalls doc

Found at 13:47 when `cand-45c3fd7d`'s K expansion was rejected with `witness_minimal_failed`.
The proximate cause is a one-line omission in `seed_sandbox`; the interesting part is what the
agent did on the retry.

## The rejection

`SPACE_EXPANSION_REJECTED`, `cand-45c3fd7d`, attempt 0. The minimal witness carried
`PV_BLOCK_K: 8` and Triton refused to compile it:

```
triton.compiler.errors.CompilationError: at 21:19:
        if COMPUTE_DTYPE == "fp16":
            acc += tl.dot(probs.to(tl.float16), v.to(tl.float16))
                   ^
Input shapes should have M >= 1, N >= 1 and K >= 16
```

`PV_BLOCK_K` feeds the **contraction** dimension of the PV dot, and 8 < 16. The v1 space had
`PV_BLOCK_K: [16, 32, 64]`, so the illegal value exists only in the expansion — K invented it
while widening the knob downward.

## The omission

`_triton_pitfalls_doc()` already documents this rule, and `BLOCK_K = 8` is its literal bad
example. Checking which agents actually receive it:

| agent | writes Triton? | got `triton_pitfalls.md` |
|---|---|---|
| `CandidateGeneratorAgent` | yes | **yes** |
| `StructureRewriterAgent` | yes | **yes** |
| `NoveltyGeneratorAgent` | yes | **yes** |
| `RepairAgent` | yes | **yes** |
| **`ParameterizerAgent`** | **yes** | **NO** |
| `BottleneckAnalystAgent` | no (reads only) | no — correct |

The parameterizer rewrites the kernel body (that is its whole job in step 1) *and* chooses the
value set for every tile dimension. Every Triton rule that constrains a tile dimension binds it
exactly as it binds the rewriter. It was the only writer flying blind, and `seed_sandbox`
(`modules.py:312`) simply never wrote the file.

## Measured cost: small, and I want to say so

Searching every run for the six pitfall rules' error signatures:

| rule | trial failures | space rejections |
|---|---|---|
| 4 — `tl.dot` contraction dim | 0 | **1** |
| 3 — `constexpr` misuse | 0 | 5 |
| 1, 2, 5, 6 | 0 | 0 |

One rejection from rule 4 in the whole record, plus 5 from rule 3 (all in one early run). And
K has expanded a knob downward **17 times** across all runs; only this once did it cross a
`tl.dot` floor:

```
NUM_WARPS  [2,4,8] -> +1     (x5, all legal)
NUM_STAGES [1,2,3,4] -> +0   (x5, legal but useless -- separate finding)
BLOCK_N/BLOCK_M -> +16, +8, +32   (legal: M and N have no floor)
SOFTMAX_BLOCK [1024,2048] -> +512 (legal)
PV_BLOCK_K [16,32,64] -> +8       (ILLEGAL -- this one)
```

So this is a **latent** gap rather than an expensive one: the fix is a one-line omission worth
closing, not a fire. The honest framing is that the doc has been absent from the parameterizer
for the project's whole history and cost roughly one rejected attempt.

## The retry is the part worth reading

I expected the retry to drop the 8. It did not — `sp-f5e27f18` published with
`PV_BLOCK_K: [8, 16, 32, 64]`, the illegal value still in the domain. My first reading was that
it had slipped past a luckier witness. **That was wrong.** The agent changed the *kernel* so 8
became legal:

```python
DOT_BLOCK_K: tl.constexpr = 16 if BLOCK_K == 8 else BLOCK_K
for start_k in range(0, T, BLOCK_K):
    ks = start_k + tl.arange(0, DOT_BLOCK_K)
    probs = tl.load(..., mask=... & (ks[None, :] < start_k + BLOCK_K), other=0.0)
    v     = tl.load(..., mask=(ks[:, None] < start_k + BLOCK_K) & ..., other=0.0)
```

It pads the dot to a legal 16-wide contraction and masks the 8 lanes that must not contribute,
while keeping the loop stride at 8. That is a correct fix, it compiles, and both
`PV_BLOCK_K=8` trials completed. So the retry mechanism worked as designed — the agent read its
own failure, understood the constraint, and satisfied it by changing the code rather than
retreating on the domain.

Whether it was *worth* doing is a different question. The expansion's full 40 trials answer it,
and the answer splits by knob. K widened **seven** knobs here:

```
QKV_BLOCK_K   +128     QKV_M_CTAS  +16    QKV_NUM_WARPS  +2
SCORE_NUM_WARPS +16    OUT_BLOCK_M +256   OUT_NUM_WARPS  +16
PV_BLOCK_K    +8   <- the illegal one
```

The expansion **improved the candidate 22.0 → 20.3 (7.7%)**, and it is attributable — the winning
trial uses the newly-added `OUT_BLOCK_M=256`:

| | trials | complete | best |
|---|---|---|---|
| used a new choice | 18 | 10 | **20.3** |
| only old choices | 22 | 14 | 20.5 |

But the specific value that caused the rejection contributed nothing:

```
PV_BLOCK_K=8    n=4   complete=3   best=24.5
PV_BLOCK_K=16   n=2   complete=1   best=22.8
PV_BLOCK_K=32   n=20  complete=15  best=20.3   <- the winner
PV_BLOCK_K=64   n=14  complete=5   best=21.2
```

8 is the worst value in the domain, which is what padding a dot to double its useful width
predicts. The analyst had reported `PV_BLOCK_K` "trends downward in aggregate but the best uses
32" — weak evidence by its own account — and the downward expansion bought a strictly bad region
plus a masking branch in the hot loop, costing one rejected attempt, ~3 min of agent wall, and 4
trials.

So this expansion is a clean example of something the current metric cannot see: **one expansion,
seven knobs, one clearly productive (`OUT_BLOCK_M`) and one clearly counterproductive
(`PV_BLOCK_K`)**, scored as a single 7.7% success. `scripts/audit_expansion_outcomes.py` attributes
per-expansion, not per-knob, so the bad knob is invisible in the headline. Per-knob attribution
would need the same new-vs-old split done one knob at a time — read-only, and worth doing when the
run is over.

It is also a second instance of the pattern in
`result-analyst-hypothesis-refuted-by-control.md`: the boundary trend did not extrapolate past the
boundary, and measuring the point beyond the edge turned the curve.

## What I changed

1. **`seed_sandbox` now writes `docs/triton_pitfalls.md`** for the parameterizer, with a comment
   saying why (it rewrites the body *and* picks tile values).
2. **Both parameterizer prompts cite it.** Writing the file is inert if the prompt never names it
   — the other four agents all say "you MUST also read `docs/triton_pitfalls.md`". The fresh
   prompt now says the same and adds that the rule binds the *choices*, not just the body; the
   expansion prompt says an illegal added value makes the witness fail and the **entire**
   expansion is rejected, so check which dot dimension a knob feeds before widening it downward.
3. **Corrected the pitfalls doc's own rule 4.** It said "needs both dims % 16 == 0, and >= 16".
   Triton's actual message is `M >= 1, N >= 1 and K >= 16` — only the contraction dimension has a
   floor. This matters concretely: `cand-88e76051`'s expansion added `BLOCK_N: 8` in this same
   run and **5 of its 6 trials completed**, so an N tile below 16 is legal and the old wording
   would have taught the agent to refuse a legitimate expansion. The doc now states the
   asymmetry, and keeps the "multiples of 16 for all three dims" advice as a *performance*
   default (off-multiple M/N tiles get padded onto the MMA shape) rather than a legality rule.

Point 3 is a correction to my own first draft of the prompt text, which said "a `tl.dot` K
dimension must be >= 16 and a multiple of 16, so 8 is never a legal choice" — too broad, and it
would have suppressed the `BLOCK_N=8` case that measurably worked.

Driver-side, so the in-flight run is unaffected. 191 tests pass, 9 skipped. Verified both prompt
variants render and cite the doc, and that `_triton_pitfalls_doc()` still loads.

## Not changed

- **The samplable-illegal-value question.** `PV_BLOCK_K=8` violates nothing in
  `sp-f5e27f18`'s constraint set, so the guard would have sampled it even if the kernel had not
  been fixed. A "declared values must be legal for the dims they feed" check belongs in
  `validation.py`, not the prompt — but it needs the dot-dimension mapping, which only the agent
  knows. Worth a `SPACE_PUBLISHED` field (`knob_role: contraction|row|col|warps|stages`) and
  that is a schema change, so it joins the pending list rather than getting done now.
- **The `NUM_STAGES -> 0` expansions** (5 instances) have the same shape — a downward expansion
  into a value that is legal but useless — and are already recorded in
  `measurement-k-expands-downward-and-adds-zero-stages.md` as pending item 8.

## A side observation, and it is a good one

`sp-f5e27f18`'s constraints include this, written by the *unmodified* parameterizer:

```
((COMPUTE_DTYPE == "fp16" or COMPUTE_DTYPE == "bf16") and
  QKV_BLOCK_K * (QKV_BLOCK_M + QKV_BLOCK_N) * QKV_NUM_STAGES * 2 <= 101376) or
((COMPUTE_DTYPE == "tf32" or COMPUTE_DTYPE == "ieee") and
  QKV_BLOCK_K * (QKV_BLOCK_M + QKV_BLOCK_N) * QKV_NUM_STAGES * 4 <= 101376)
```

That is exactly the dtype-aware, per-kernel disjunction that
`finding-shared-memory-constraints-never-fire.md` argues for, produced spontaneously by the
prompt as it stood before my change. So the capability is clearly within the model's reach and
the failure documented there is one of *prompting*, not ability — which is the best possible
evidence that a prompt-only fix is the right instrument. Note it still covers only QKV and OUT,
not SCORE or PV, so the coverage half of that finding stands.
