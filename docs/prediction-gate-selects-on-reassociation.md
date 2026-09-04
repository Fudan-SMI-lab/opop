# Pre-registered prediction: the gate selects on reassociation, not correctness

**Written 2026-09-05 04:10, BEFORE the outcome was known.** Recorded in advance so the
result cannot be reinterpreted after the fact.

## Why pre-register

`docs/finding-unreachable-correctness-gate.md` claims the relaxed gate on level3/48
rejects candidates for *reassociating arithmetic* rather than for being wrong, because
`pass_frac = 0.99` sits above the reference's own ieee-vs-tf32 agreement (0.977760).

Its evidence is one controlled pair in `fam-99aee6de`: H1 (chunked parallel prefix with
tensor-core `tl.dot`) rejected at frac 0.975956; H2 (same sequence-level restructuring,
per-chunk scalar recurrence untouched) published first try. Two candidates is a
suggestive pair, not a test — and I found it *after* seeing the outcome, which is exactly
the shape of reasoning that produces confident wrong conclusions.

Round 2 supplies an independent instance from a different family, and I am writing the
prediction down before the witnesses run.

## The setup (all facts already on disk at 04:07:31)

`fam-74c41d8d`, parent `cand-cf0f07e7` (2.84ms, the K-expansion improver). The analyst's
bottleneck report for it says the wall is arithmetic throughput, not any resource limit:
80/255 registers, 512 B shared of 101,376, one warp, no spills, no OOM. Its recommendation
is explicitly to move scalar fp32 FMA work onto tensor cores.

The rewriter returned two candidates, and they split along the same axis as round 1:

| candidate | hyp | reassociates per-chunk arithmetic? | what it changes |
|---|---|---|---|
| `cand-eed411d8` | H1 | **yes** — `tl.dot(..., input_precision="tf32")` for the B-C interaction and its application to X | recurrence becomes causal prefix-weighted matrix products |
| `cand-a04c3f52` | H2 | **no** — scalar recurrence kept; fuses two passes into one | 192 recurrence steps -> 128, removes a duplicated first-chunk traversal |

## Prediction

1. `cand-eed411d8` (H1) is **rejected** at `witness_default_failed` or
   `witness_minimal_failed` with `frac_within_tol` in roughly **0.95–0.978** — below the
   0.99 gate but at or near the 0.977767 noise floor — and `cosine >= 0.9999`, i.e. the
   cosine agrees to ~7-8 decimals while frac does not clear the bar.
2. `cand-a04c3f52` (H2) **passes** its witnesses and gets tuned. Its latency is not
   predicted: fusing 192 steps to 128 could help or not.
3. If H1 is rejected, repair will diagnose a numerical cause that is *real but not a
   defect* (accumulation order, tf32 rounding), and its fix will make frac **worse**, not
   better — plausibly introducing non-finite output, as happened to `cand-dc4b6fec`
   (0.975956 -> 0.843332).

## What each outcome means

- **H1 rejected near the floor with a near-1 cosine, H2 passes.** Two independent
  families, same split, prediction made in advance. The finding stands as a property of
  the gate on this task, not an anecdote. It also means the run has now spent up to 8
  repair cycles (2 candidates x 4 attempts) on candidates that had no defect.
- **H1 passes.** The finding is materially weakened: reassociation is *not* sufficient to
  fail the gate, and round 1's H1 must have had a real defect the noise-floor comparison
  hid. I would have to withdraw the "selects on reassociation" claim and re-examine
  `cand-dc4b6fec` for an actual bug.
- **H1 rejected with frac far below the floor (< ~0.95), or a non-near-1 cosine.** Also
  weakens the finding — that is a genuinely wrong kernel, not one sitting in the witness
  gap, and the reject is correct.
- **Both rejected.** Ambiguous on the gate question; would point instead at the rewriter
  producing broken code for this family.

## Note on what this does and does not test

This tests whether the gate's *frac* threshold is what stops reassociating rewrites. It
does not test whether H1 would have been faster — that stays unknown either way, because
a rejected candidate is never timed. The write-up rule from the finding still applies: a
rejected structural direction is **unknown, not disproven**.

No change is being made to the gate to run this test. The prediction is read off events
the run produces on its own.
