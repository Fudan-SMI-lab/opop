# Result: on L3:43 the gate rejects bf16 at a *higher* agreement than the reference has with itself

Found live on `run-l3-43-20260905-091705`, 0.3h in. This is the strongest instance of the
correctness-gate finding on record, because on this task the arithmetic is
**unambiguously fine** and the rejection is decided entirely by the threshold.

## The numbers

The gate demands `frac_within_tol > 0.99`. L3:43's own reference, evaluated at ieee fp32
and at tf32, agrees with itself at only **0.976682**.

| | frac_within_tol | cosine |
|---|---|---|
| candidate (bf16) vs ieee ref | **0.987266** | 0.99999996 |
| candidate (bf16) vs tf32 ref | 0.972743 | 0.99999989 |
| **reference vs itself (ieee vs tf32) = the task's noise floor** | **0.976682** | 0.99999993 |
| **gate threshold** | **0.990000** | 0.99985 |

The candidate matches the ieee reference **better than the reference matches its own tf32
self** — 0.987266 against 0.976682, a margin of +0.0106 — and is rejected, because the
threshold sits 0.0133 *above* the floor.

Cosine is 0.99999996 against a requirement of 0.99985, so it passes that arm by four
orders of magnitude. `median_rel_err` is 1.781e-04 and `max_abs_diff` is 1.363e-03 on an
output whose `ref_absmax` is 0.8305. Nothing here looks like a broken kernel.

**17 of 17** bf16 mismatches on this candidate are above the floor. Not one is below it. (Was
14 of 14 when first written 25 minutes earlier; the tally grows as TPE keeps sampling the
branch, and every new instance lands in the same 0.987202–0.987266 band.)

On this candidate the other precisions are not producing mismatches at all: bf16 is the only
dtype with any `correctness_mismatch` in tuning, so this is not a general accuracy problem
with the candidate.

### It replicated on a second, structurally unrelated candidate — and bf16 split

By 1.18h the run had tuned a third seed, `cand-3bf724d6` (`fam-ea7bc8bb`, a
materialize-scores-then-separate-softmax design, structurally unrelated to
`cand-6476b4cb`'s fused online-softmax kernel). Its bf16 branch behaves the same way:

| candidate | family | bf16 best-arm frac | above floor 0.976682? | cosine |
|---|---|---|---|---|
| `cand-6476b4cb` | fam-92e7c576 | 0.987202 – 0.987266 (17 trials) | **all 17 yes** | 0.99999996 |
| `cand-3bf724d6` | fam-ea7bc8bb | 0.985624 (9 trials, identical) | **all 9 yes** | 0.99999995 |
| `cand-cb7be6b4` | fam-4aea322a | 0.811939 – 0.811958 (5 trials) | **no — 0.165 below** | 0.99999531 |

**26 of 34 mismatches in the run are above the floor; 8 are below.** That 26/8 split is the
part worth keeping, because it is no longer "bf16 is rejected unfairly" — it is
**dtype-independent**:

- bf16 is above the floor on two candidates (26 trials) and *far* below it on a third (5
  trials, frac 0.81, `median_rel_err` 3.09e-03 — an order of magnitude worse and plainly a
  real inaccuracy).
- tf32 is below the floor on `cand-cb7be6b4` (3 trials, 0.963953) and passes cleanly on the
  other two.

So the knob value does not predict the verdict; **the floor-relative comparison does.** A
dtype-based rule — the "ban bf16" shape the user already rejected as too dangerous — would
have thrown away 26 correct results *and* kept 5 genuinely wrong ones on the same task, in
the same run. That is the sharpest argument yet that the floor, not the dtype, is the right
discriminator, and it is also an independent confirmation that the user's call on the
dtype-ban was correct.

## Why this case is cleaner than the earlier ones

`finding-unreachable-correctness-gate.md` established the pattern on L3:48 and
`result-one-knob-decides-correctness.md` replicated it on L3:21. Both had a complication:

- On L3:48 the fp16 failures were genuine **overflow** (output > 65504), so a reader could
  argue the rejections were catching something real.
- On L3:21 the two failing branches split cleanly on merit — bf16 at 0.674 was *far* below
  that task's 0.955360 floor and deserved rejection, while tf32 at 0.9533 missed by
  2.1e-3. Having one deserving case alongside one marginal case muddies the argument.

L3:43 has no such complication. There is **one** failing branch, every instance of it is
above the floor, and the passing branches (fp16 6/6, tf32 2/2, ieee 2/2 completions) show
the kernel is otherwise correct. The only thing separating accept from reject is a
threshold chosen without reference to this task's measured precision spread.

### It also corrects something I wrote about bf16

`result-one-knob-decides-correctness.md` says of L3:21's bf16 cluster:

> "bf16 has 8 mantissa bits against fp16's 10, and on an O(1)-magnitude task like L3:21
> the extra range buys nothing while the lost mantissa costs accuracy."

That reasoning is sound for L3:21 (0.674 really is far off) but it does **not** generalize,
and I should not have implied it would. On L3:43 — also an O(1)-magnitude task,
`ref_absmax` 0.83 — bf16 lands at 0.987266, closer to the ieee reference than tf32 is.
Mantissa width predicts the *direction* of the error, not whether it clears a fixed
threshold; only the task's measured floor tells you that. This is the same lesson as
`opop-v2-noise-floor-is-not-predicted-by-output-range`, arriving from the other side.

## The cost, and what the older run hid

`bf16` is one value of one knob, and TPE keeps sampling it until the failures teach it not
to. Across the two L3:43 runs with a `COMPUTE_DTYPE`-style knob:

| run | bf16 trials | outcome |
|---|---|---|
| 09-04 | **190** | 190 fail (154 `correctness_mismatch`, 36 `runtime_error`) |
| 09-05 (in flight) | 16 | 16 fail (14 mismatch, 2 runtime) |

**190 trials on 09-04**, in an 11.66h run — every one rejected, and by the evidence above,
the mismatching ones were rejected while agreeing with the reference better than the
reference agrees with itself.

The reason nobody noticed is the failure message. On 09-04 it read, in full:

```
relaxed mismatch (max abs diff 0.001363) on trial 2
```

Today's message for **the identical value** (`max_abs_diff: '1.363e-03'`, byte-identical)
reads:

```
relaxed mismatch on trial 2; gate needs frac_within_tol>0.99 AND cosine>=0.99985
  vs ieee ref:  frac 0.987266  cosine 0.99999996  median_rel_err 1.781e-04
  vs tf32 ref:  frac 0.972743  cosine 0.99999989  median_rel_err 4.385e-04
  reference's OWN ieee-vs-tf32 spread (task noise floor, NOT a bug): frac 0.976682
```

Same kernel behaviour, same arithmetic, same measured discrepancy — but the 09-04 message
made it look like a small numerical bug, while the 09-05 message shows a candidate that
beats the floor being turned away by the threshold. That is the
`finding-failure-messages-must-carry-gate-criteria.md` fix earning its keep: it did not
change a single verdict, and it made a 190-trial pattern legible that had been invisible
for a day.

## What this does and does not argue

**Does not** argue the gate should be loosened to a specific number, or that bf16 should be
accepted here. That is the decision the user has twice deferred and it stays deferred; this
document adds evidence, not a change. Nothing in the acceptance path was touched.

**Does** argue that the *shape* of the rule is the problem rather than its strictness: a
fixed 0.99 cannot be satisfied on a task whose reference self-agreement is 0.9767, so on
L3:43 the bf16 branch is unreachable **by construction** — no kernel, however written, can
pass it, because the reference itself would not.

**Does** narrow the design question usefully. **All three tasks** have measured floors
below the threshold, and none is close to it — machine-verified by
`scripts/audit_noise_floors.py`, now extended to read per-trial failure details as well as
`SPACE_REJECTED` (reading only the latter missed L3:43 entirely, because its bf16
rejections happen inside tuning rather than at witness time):

```
task  runs     n  floor frac (min..max)       ref_absmax
L3:21      1  98  0.955360 .. 0.955360         5.749e+00
L3:43      1  34  0.976682 .. 0.976682         8.305e-01
L3:48      1  15  0.977767 .. 0.977767         1.038e+22

Gate threshold: 0.99
  L3:21   floor 0.955360  ->  gate is +0.0346 above the floor
  L3:43   floor 0.976682  ->  gate is +0.0133 above the floor
  L3:48   floor 0.977767  ->  gate is +0.0122 above the floor
```

Each floor is stable to six decimals across every measurement (n=98, 34, 15), so it is a
property of the task's arithmetic rather than a noisy estimate — and the harness already
measures it at witness time. Whatever is decided, the floor is available to decide with.

Note the L3:43 floor did not move a digit as n went 17 → 34 across three structurally
unrelated candidates. It is a property of the *reference*, which is why it is usable as a
threshold at all.

## The same run has a control case pointing the other way

At 09:38:02, `cand-cb7be6b4` was rejected at its **default** witness (`COMPUTE_DTYPE:
'tf32'`) with:

```
vs ieee ref: frac 0.963932   cosine 0.99999987   median_rel_err 2.056e-03
vs tf32 ref: frac 0.951209   cosine 0.99999975   median_rel_err 2.118e-03
floor:       frac 0.976682
```

Its best witness (0.963932) is **below** the floor by 0.0128, and its `median_rel_err` is
an order of magnitude worse than the bf16 case's (2.056e-03 vs 1.781e-04). Here the gate is
**right**: the candidate really does disagree with the reference by more than the
reference's own precision spread, and it went to repair as it should.

So within one task, one run, and half an hour:

| candidate | best witness frac | vs floor 0.976682 | gate verdict |
|---|---|---|---|
| bf16 trials of `cand-6476b4cb` | 0.987266 | **+0.0106 above** | reject — **wrong** |
| bf16 trials of `cand-3bf724d6` | 0.985624 | **+0.0089 above** | reject — **wrong** |
| `cand-cb7be6b4` default (tf32) | 0.963932 | **−0.0128 below** | reject — **right** |
| bf16 trials of `cand-cb7be6b4` | 0.811958 | **−0.165 below** | reject — **right** |

That pairing is what makes the finding actionable rather than merely a complaint about
strictness. The gate already discriminates correctly when the comparison is
floor-relative; it is the *fixed* 0.99 that produces the wrong verdict on the first two
rows. All four rows come from the same reference, the same task, the same measurement code,
and the floor separates them cleanly — 26 above, 8 below, with a gap of 0.17 between the
nearest above-floor value and the nearest below-floor one on the same dtype.

The four rows also rule out the two simpler rules one might reach for instead:

- **"Reject bf16"** — rows 1, 2 and 4 are all bf16, and it is right on one of them.
- **"Trust cosine"** — every row passes the 0.99985 cosine arm, including both genuinely
  wrong ones (0.99999531 and 0.99999987). Cosine is insensitive to the errors that matter
  here.

## Caveats

- 1.18h into a 12h run. Figures will keep moving as TPE samples; the floor has not moved
  and is not expected to.
- The 09-04 attribution is an inference, not a direct reading: those messages do not carry
  `frac_within_tol`, so I matched on `max_abs_diff` and on the same task, knob and failure
  kind. **Both 09-04 clusters are now located and they account for every mismatch there**,
  which makes that inference much stronger than when it was one cluster of two:

  | 09-04 `max_abs_diff` | trials | today's candidate with the identical value | today's frac | vs floor |
  |---|---|---|---|---|
  | 0.001363 | **123** | `cand-6476b4cb` (1.363e-03) | 0.987266 | **above** |
  | 0.001861 | **31** | `cand-3bf724d6` (1.861e-03) | 0.985624 | **above** |
  | | **154 = all of them** | | | |

  Every one of 09-04's 154 bf16 `correctness_mismatch` trials falls into one of two clusters,
  and both clusters reproduce today to six digits on candidates measured to be above the
  floor. Today's *below*-floor bf16 candidate has `max_abs_diff` 2.920e-03, which appears in
  09-04 not at all. So the 09-04 run most likely spent 154 trials rejecting kernels the
  reference could not distinguish from itself — and none on kernels that were genuinely wrong.
- `runtime_error` bf16 trials (36 on 09-04, 11 today) are a separate matter — tile/dtype
  incompatibilities, not gate rejections. Together with the 154 above, that is the full 190.
- What this does *not* establish: that an above-floor kernel is *correct*, only that the
  reference cannot distinguish it from its own precision spread. A floor-relative gate would
  admit anything the reference cannot resolve, which is a real and deliberate weakening of
  the guarantee. That is the substance of the user's "危险性过大", and these numbers do not
  answer it — they only show that the current rule's verdicts do not track accuracy.
