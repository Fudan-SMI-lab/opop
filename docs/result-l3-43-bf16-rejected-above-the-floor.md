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

**14 of 14** bf16 mismatches this run are above the floor. Not one is below it.

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
L3:43      1  17  0.976682 .. 0.976682         8.305e-01
L3:48      1  15  0.977767 .. 0.977767         1.038e+22

Gate threshold: 0.99
  L3:21   floor 0.955360  ->  gate is +0.0346 above the floor
  L3:43   floor 0.976682  ->  gate is +0.0133 above the floor
  L3:48   floor 0.977767  ->  gate is +0.0122 above the floor
```

Each floor is stable to six decimals across every measurement (n=98, 17, 15), so it is a
property of the task's arithmetic rather than a noisy estimate — and the harness already
measures it at witness time. Whatever is decided, the floor is available to decide with.

## Caveats

- One candidate, 0.3h into a run that has 11.7h to go. The 14/14 tally will grow and the
  floor is measured per witness evaluation, so the exact figures will move slightly (the
  four distinct ieee values seen span 0.987202–0.987266).
- The 09-04 attribution is an inference, not a direct reading: those messages do not carry
  `frac_within_tol`, so I matched on `max_abs_diff` being identical to six digits
  (0.001363 / 1.363e-03) and on the same task, knob and failure kind. The two distinct
  09-04 values (0.001363 and 0.001861) suggest two clusters, of which I have located one.
- `runtime_error` bf16 trials (36 on 09-04, 2 today) are a separate matter — tile/dtype
  incompatibilities, not gate rejections.
