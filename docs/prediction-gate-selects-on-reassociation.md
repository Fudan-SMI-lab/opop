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

---

# Resolution

**Prediction 1 confirmed at 04:10:05**, more sharply than the stated range required.

`cand-eed411d8` (H1, the reassociating rewrite) rejected `witness_default_failed`:

```
vs ieee ref: frac_within_tol=0.976029  cosine=0.99999996  median_rel_err=1.441e-03
vs tf32 ref: frac_within_tol=0.962411  cosine=0.99999990  median_rel_err=1.521e-03
noise floor: frac_within_tol=0.977767  cosine=0.99999996  median_rel_err=3.928e-04
```

Predicted frac 0.95–0.978 with cosine >= 0.9999. Observed 0.976029 with cosine
0.99999996, and **no** non-finite output — exactly the shape called in advance: sitting
0.0017 below the floor while the cosine matches the floor's own cosine to eight decimals.

Side by side with round 1's independent instance:

| | round 1 `cand-dc4b6fec` | round 2 `cand-eed411d8` |
|---|---|---|
| family | fam-99aee6de | fam-74c41d8d |
| parent | cand-c18203b6 (2.09ms) | cand-cf0f07e7 (2.84ms) |
| rewriter call | rewriter-... (03:03:57) | rewriter-9c437173 (04:07:31) |
| frac vs ieee | 0.975956 | 0.976029 |
| frac vs tf32 | 0.965382 | 0.962411 |
| cosine vs ieee | 0.99999996 | 0.99999996 |
| noise floor frac | 0.977767 | 0.977767 |

Two independent rewriter calls, different families, different parent source, landing
**7e-5 apart** in frac — and both cosines equal to the floor's cosine to eight decimals.
That is not two candidates each happening to be slightly wrong. It is the same structural
transformation reproducing the same numerical signature, which is what the finding
predicted and what a coincidence would not do.

## What this establishes, and what it does not

Established: on this task the frac threshold, not correctness, is what stops reassociating
rewrites. The claim was pre-registered with falsifiers and survived an independent
instance, so it is no longer an after-the-fact reading of one pair.

Still not established: whether either H1 would have been *faster*. Neither was ever
timed. Both remain **unknown, not disproven** — the write-up rule stands unchanged.

## Cost, now measurable

Two candidates x up to 4 attempts of repair + reparameterize, on candidates with no
defect to fix. Round 1's cost is known: `cand-dc4b6fec` degraded 0.975956 -> 0.843332 ->
0.844503 (non-finite) and was dropped after consuming ~40 min of wall. Round 2's is
accruing now.

Predictions 2 and 3 (H2 passing; repair damaging H1 rather than fixing it) resolve as the
run continues and are recorded above unchanged.

