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

## Prediction 3 confirmed at 04:13:51

`cand-eed411d8` attempt 1, after the repair, rejected `witness_minimal_failed`:
frac **0.836301** with **19,403,796 of 134,217,728 values non-finite** (16,070,921 NaN).

The repair's diagnosis was again correct-but-not-a-defect — TF32 truncation compounding
across sequential dot products, explicitly noting "high cosine similarity but only 97.6%
of elements within tolerance". It read the noise-floor evidence accurately and still
concluded there was a bug, because it has no channel to answer "this is the task's own
spread".

The two families now reproduce the whole trajectory, not just the rejection:

| | round 1 `cand-dc4b6fec` | round 2 `cand-eed411d8` |
|---|---|---|
| a=0 | 0.975956, finite, cosine 0.99999996 | 0.976029, finite, cosine 0.99999996 |
| a=1 | 0.843332 + non-finite | 0.836301 + non-finite (16.1M NaN) |
| repair's diagnosis | chunk boundaries + TF32 rounding | TF32 truncation across dot products |
| outcome | dropped after 4 attempts | in progress |

Independent rewriter calls, independent repair calls, different parents — same two-step
collapse: land ~0.0017 below the noise floor, then get destroyed by a fix for a bug that
was not there. All three pre-registered predictions hold.

This is the clearest statement of the cost. The gate does not merely fail to measure the
bolder structural direction; it reliably converts it into a broken kernel, and it does so
*reproducibly* rather than by accident. Both H1 candidates remain **unknown, not
disproven** on latency — neither was ever timed.

## The repair loop converges to a broken fixed point

Full attempt trajectories, both families:

| attempt | round 1 `cand-dc4b6fec` | round 2 `cand-eed411d8` |
|---|---|---|
| a=0 | 0.975956, finite | 0.976029, finite |
| a=1 | 0.843332, 18,424,816 non-finite | 0.836301, 19,403,796 non-finite |
| a=2 | 0.844503, 18,256,126 non-finite | 0.836300, 19,403,796 non-finite |
| a=3 | 0.844503, 18,256,126 non-finite | (pending) |

In both families the later attempts are *numerically identical* to their predecessor —
round 1's a=2 and a=3 agree to the last digit of frac and to the exact non-finite count;
round 2's a=1 and a=2 differ only in how 5 of 19.4M values split between NaN and Inf,
which is run-to-run nondeterminism, not a different kernel.

Yet each attempt was given a **different diagnosis**. Round 2's a=1 blamed TF32
truncation across sequential dot products; a=2 blamed `tl.sum` recomputation of the decay
prefixes instead of reusing `A_cumsum`. Different claims, functionally equivalent output.

Grouping every diagnosis in the run shows why: across all three affected candidates
(`cand-eb910a18`, `cand-dc4b6fec`, `cand-eed411d8`) they converge on one class of claim —
fp32 accumulation order, chunk boundaries, TF32 rounding. **As physics that class is
correct.** Those differences are exactly why the candidate's frac sits at 0.976. As a
defect it is wrong, because the reference's own two witnesses differ for the same reason at
0.977767. The agents are being asked to remove a difference that is not removable, so the
loop settles at whatever broken kernel it first reaches and then stops moving.

`prior_rejected` climbing 0 -> 1 -> 2 confirms the repair-history fix is live and each
diagnosis is being shown its predecessors as disproven. It stopped the oscillation it was
built to stop — no diagnosis repeats a rejected claim — but no amount of history helps when
every available answer is wrong.

This is the concrete case for the second concern in
`docs/finding-unreachable-correctness-gate.md`: a repair agent with a channel to answer
"within task noise, no change needed" would have preserved both kernels at a=0, under the
current threshold, with no change to what the harness calls correct.

## One thing the diagnoses do NOT show

I looked for a pattern where the loop diagnoses the damage it caused itself — noise-class
claims while the output is finite, real-bug claims (masking, overflow) once a repair has
introduced NaN/Inf. Classifying all 9 diagnoses in the run against whether the rejection
they answered actually reported non-finite output:

| | NOISE-class claim | REAL-BUG-class claim |
|---|---|---|
| rejection showed finite output | 3 | 1 |
| rejection showed non-finite output | 3 | 2 |

There is no such relationship. `cand-eb910a18`'s masking/overflow diagnosis was made while
the output was still finite — it anticipated the overflow rather than reacting to it — and
three diagnoses that *did* see millions of NaN went back to blaming accumulation order
anyway. With 9 points split like that, nothing is established either way, so this is
recorded as a non-finding rather than dressed up as one.

What survives is only the claim the evidence actually carries: every **a=0** diagnosis is
noise-class (3 of 3), which is the phantom-bug problem, and the later attempts converge to
a numerically fixed broken kernel. How the intermediate diagnoses are distributed is not
something this run determines.

## A third family, and the frac is effectively a constant

Round 3, `fam-b1ee96ac` (parent `cand-f4a2ce82`, 3.80ms). H1 `cand-741c2699` —
"reformulates each recurrence chunk into TF32 tensor-core matrix products … retaining fp32
accumulators" — rejected at a=0, finite output:

| candidate | family | parent | frac vs ieee | cosine | below floor |
|---|---|---|---|---|---|
| cand-dc4b6fec | fam-99aee6de | cand-c18203b6 (2.09ms) | 0.975956 | 0.99999996 | 0.001811 |
| cand-eed411d8 | fam-74c41d8d | cand-cf0f07e7 (2.84ms) | 0.976029 | 0.99999996 | 0.001738 |
| cand-741c2699 | fam-b1ee96ac | cand-f4a2ce82 (3.80ms) | 0.975839 | 0.99999996 | 0.001928 |
| *(reference's own noise floor)* | — | — | *0.977767* | *0.99999996* | — |

Three independent families, three independent rewriter calls, three different parent
kernels: total spread **1.9e-4**, every cosine identical to the floor's to eight decimals,
every one sitting 0.0017–0.0019 below the threshold it needed.

At this point the frac is not a property of any individual candidate. It is a property of
**the transformation** — reassociating this task's chunked scan onto tensor cores lands at
0.9759 ± 0.0001 against an ieee witness, whatever kernel performs it.

## The H1/H2 split is 3 for 3

| family | H1 (moves work to tensor cores) | H2 (keeps scalar arithmetic) |
|---|---|---|
| fam-99aee6de | **rejected**, then damaged to 0.844503 | published, 3.34ms |
| fam-74c41d8d | **rejected**, then damaged to 0.836300 | published, **2.46ms** |
| fam-b1ee96ac | **rejected** | pending |

Every rewrite that reassociates was rejected. Every conservative sibling was admitted. The
analyst independently identified the tensor-core path as the primary remaining headroom in
all three families — its bottleneck reports say the kernels are not resource-bound (80/255
registers, 512 B of 101,376 shared, one warp, no spills) and that the wall is scalar fp32
arithmetic throughput.

So the harness has now diagnosed the same optimisation three times, generated a candidate
for it three times, and refused to time it three times — on a threshold the reference
itself does not meet. The one direction the evidence points at is the one direction the
gate structurally cannot evaluate.

Worth stating plainly for the write-up: the run's best candidate so far (2.46ms, 7.6x
torch.compile) is an H2. Every H1 remains **untimed**. Whether tensor cores would beat it
is unknown, and this run cannot answer it.

## Final scoring of prediction 1, against every case the run produced

Prediction 1 named a numeric range in advance: `frac_within_tol` in **0.95–0.978** with
**cosine >= 0.9999**. The run went on to produce 13 finite rejections of `tl.dot`-bearing
candidates (8 candidates, 4 families, several arriving hours after this file was committed
at `e01708a`). Scored against all of them:

| test | result |
|---|---|
| frac inside 0.95–0.978 | **12 / 13** |
| cosine >= 0.9999 | **13 / 13** |

The single miss is `cand-61f768c8` a=2 at **0.978034**, which *overshoots* the upper bound —
it did better than predicted, cleared the reference's own 0.977767 floor, and was rejected
anyway. That is a miss on the stated interval and it is recorded as one, but it falsifies
the prediction in the direction that strengthens the finding rather than weakening it.

The 12 in-range values span 6.49e-4 (0.975775 to 0.976424), i.e. **0.067%**. Excluded from
this scoring, as it was from the fingerprint: `cand-eb910a18` at frac 0.9125, a genuinely
wrong kernel — which is the outcome branch "H1 rejected with frac far below the floor"
above, correctly identified by the gate, and the reason that branch was written down.
