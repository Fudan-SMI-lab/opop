# Finding: the minimal witness forces fp16 on a task whose outputs reach 1e22

**This corrects a claim in `finding-unreachable-correctness-gate.md`.** I had been reading
all 27 of this run's `SPACE_REJECTED` events as one phenomenon. They are two, and the larger
group has a different cause, a different fix, and — unlike the gate finding — needs no
decision about acceptance semantics.

## The mechanism

`validation.py:143-170` builds two witnesses and tests them **in order, returning on the
first failure**:

```python
for label, params in (("default", default_params), ("minimal", minimal_params)):
    ...
    if not result.get("ok"):
        return SpaceRejection(reason=f"witness_{label}_failed", ...)
```

So `witness_minimal_failed` means **the default witness passed**. That is the opposite of
how I had been reading those events, and it is the majority of them — 19 of 34 as of 06:57
(17 of 27 when this was first written; the run is still in flight).

The minimal witness is `{d.name: d.choices[0] for d in domains}` (`validation.py:136`) — the
first choice of every knob. `candidate_contract.md:96` tells the agent to write the precision
knob as:

```
PARAMS["COMPUTE_DTYPE"] = "tf32"  with choices ["fp16", "bf16", "tf32", "ieee"]
```

`"fp16"` is first, because the contract asks for choices ordered cheap→expensive. So on every
candidate that follows the contract, **the minimal witness is the fp16 corner**.

L3:48's reference output reaches `ref_absmax = 1.038e+22`. fp16's largest finite value is
65504. The ratio is 1.6e17. `exp` overflows fp16 for any argument above 11.09, and this task's
`cumsum` spans ±33.

## The evidence

Complete separation, **n=16, no exceptions** — read off the witness sources on disk:

| declared a `COMPUTE_DTYPE` knob? | candidates | outcome |
|---|---|---|
| yes | 9 (eb910a18, dc4b6fec, eed411d8, 741c2699, 61f768c8, dcf4e7e6, 8cb745ff, 2136993c, ef6f0748) | **9 rejected, 0 published** |
| no | 7 (c18203b6, f4a2ce82, cf0f07e7, 51dd1857, a04c3f52, 3bcc57ce, 8a617cba) | **7 published, 0 rejected** |

Every space that offered a precision knob died. Every space that did not, lived. (Counts
updated as the run continued; it began as 7/7 and the separation held at every step.)

The failure signature is overflow, not a wrong algorithm — all but two of the
minimal-witness rejections report **7.3% to 17.6% of the output non-finite**:

```
cand-61f768c8  a=1  17.6% of output non-finite   NaN/Inf
cand-dcf4e7e6  a=1  17.6% of output non-finite
cand-eed411d8  a=1  14.5% of output non-finite
cand-dc4b6fec  a=1  13.7% of output non-finite
```

And the `ref_absmax` field proves it is the *large* elements that die. `_relaxed_metrics`
computes its statistics over the finite subset (`worker_main.py:445-447`), so `ref_absmax`
reports the largest reference value **where the candidate stayed finite**:

| witness | ref_absmax over finite subset | full-output absmax |
|---|---|---|
| default (ieee) | 1.038e+22 | 1.038e+22 |
| minimal (fp16) | **7.696e+09** | 1.038e+22 |

Twelve orders of magnitude of the output are simply gone on the fp16 path. That is not a
kernel bug; 65504 is where fp16 stops.

### A second, independent fingerprint: the overflow positions are identical

`ref_absmax` was the first line of evidence. The non-finite *counts* are a second, and they
do not depend on how the metrics are computed at all. Four distinct programs, on their fp16
witness:

| candidate | source sha256 | bytes | non-finite | NaN | ±Inf |
|---|---|---|---|---|---|
| cand-61f768c8 | 2aca91f4… | 11082 | **23687808** | 19634497 | 4053311 |
| cand-dcf4e7e6 | d7f790b9… | 13787 | **23687808** | 19634561 | 4053247 |
| cand-ef6f0748 | ce7dadd5… | 9609 | **23687808** | 19634497 | 4053311 |
| cand-8cb745ff | 5348e86f… | 7420 | 23687872 | 19634561 | 4053311 |

23,687,808 of 134,217,728 elements — **17.6488%** — non-finite, identical across three
independent programs and within **64 elements** (4.8e-5%) on the fourth. The sources differ
in size by 1.9x and have four distinct SHAs.

Four separate kernels do not lose precisely the same 23.7M elements by coincidence. That set
is the positions where the reference's `|output|` exceeds 65504 — a property of the **task
data**, identical for any kernel that routes those values through fp16. The 64-element
difference is one tile boundary.

This is the fp16 counterpart of the gate finding's `frac` fingerprint, and it closes the same
argument from the other side: on the fp16 corner the failure is determined by the task, not by
the candidate, so no repair can change it.

## Why the repair loop could never win

Attributing each repair call to the rejection window it follows:

| triggered by | repair calls | wall |
|---|---|---|
| **minimal witness (fp16 corner)** | **10** | **1.33h** |
| default witness | 10 | 0.46h |

The affected candidates all show the same shape:

```
cand-dc4b6fec   default witness: 3/4 passed (first pass at attempt 1) | minimal: 0/3 passed
cand-eed411d8   default witness: 3/4 passed (first pass at attempt 1) | minimal: 0/3 passed
cand-61f768c8   default witness: 3/4 passed (first pass at attempt 1) | minimal: 0/3 passed
cand-dcf4e7e6   default witness: 3/4 passed (first pass at attempt 1) | minimal: 0/3 passed
```

Repair **fixed the real defect on its first attempt** — the default witness passes from
attempt 1 onward — and then spent its remaining two attempts against a configuration that
cannot pass at any level of correctness. The loop was asking the agent to make fp16 represent
1e22.

Worse, the agent is never told which witness failed. `SpaceRejection.detail` carries the
failure text but the *label* only reaches the event's `reason` field, not the repair prompt.
So the agent sees "your kernel produces NaN on 14% of elements" with no indication that this
happened only at the cheapest corner of a space its own default passes. Every diagnosis in
these chains reads as though the kernel were globally broken.

## What this changes about the gate finding

The gate finding still stands, on its own evidence, but **it is smaller than I reported**:

- the six tensor-core rewrites rejected at a=0 with frac 0.9758–0.9764 are `witness_default_failed`
  and remain gate artefacts — that analysis is unaffected;
- `cand-61f768c8` a=2 reaching **frac 0.978034** — above the reference's own 0.977767 floor —
  is `witness_minimal_failed` *without* non-finite output. So that one really is the gate
  rejecting something better than the floor, and it is the strongest single datapoint in the
  gate finding. Unchanged.
- but the bulk of the repair cost I attributed to the gate (0.85h of 1.37h) was in fact
  **1.33h against the fp16 corner**. The two overlap, since a chain's early attempts are
  default-witness rejections and its later ones minimal. The honest split is above.

Both findings share a root: **L3:48's 1e22 dynamic range breaks things that assume O(1)
outputs.** The gate assumes it (99% within 1% relative), and the minimal witness assumes a
cheap dtype is merely *slower*, not *out of range*.

## The fix

Unlike the gate, this needs no acceptance-semantics decision — the harness is asking a
question that has no correct answer, and can simply stop asking it.

Three options, cheapest first:

1. **Fall back instead of rejecting.** If the minimal witness fails, try the next feasible
   config before giving up — `_first_feasible` already exists (`validation.py:138`) for the
   constraint case. The witness's purpose is to prove the space is not inert, i.e. that *some*
   second config works; it does not need to be the cheapest one. This is ~10 lines and
   preserves the anti-inertness guarantee.
2. **Tell the agent which witness failed.** The label is already computed and thrown away.
   Even without (1), a repair agent told "your default config passes; this is the fp16 corner
   of your own space overflowing on a task whose outputs reach 1e22" would answer "narrow the
   knob", not "rewrite the recurrence". Cheap, and independently useful.
3. **Let a knob declare a choice out of range for the task.** Larger change, and (1) subsumes
   most of its value.

(1) and (2) are complementary and neither touches acceptance semantics: a candidate still has
to pass a real GPU correctness test at two distinct configurations. Recommend both.

### Applied, and narrowed — the fallback must not fire on 21/43

Both landed (`391e918`), and then a check against the historical 21/43 runs showed (1) was
too broad. Those runs *do* hit `witness_minimal_failed` — 2, 2, 3 and 1 times across four
runs — but their failures look nothing like L3:48's:

| | L3:48 minimal failures | 21/43 minimal failures |
|---|---|---|
| non-finite output | **7.3 – 17.6%** | **none** |
| max-abs-diff | 1.7e7 – 1.8e19 | **0.0013 – 0.0040** |
| finite-subset `ref_absmax` | collapses 1e22 → 1e9 | comparable across witnesses |

A finite, small-error mismatch at the cheap corner is plausibly a *real* defect that only
shows there — a kernel correct only at large block sizes, say. Falling back would step past
genuine evidence about the kernel. So the fallback is now gated on `_looks_out_of_range`,
which fires on either a non-finite population or a finite-subset `ref_absmax` collapsed by
≥1e4, and returns False for an ordinary mismatch. Verified against the real log tails from
both task families.

Worth recording why this is a matter of evidence quality rather than safety: **publishing a
space whose cheapest corner is wrong cannot promote that corner.** Every tuning trial runs
`quick_test` (3 correctness trials) before timing, and the worker only times `if correct
and num_perf`, so a wrong config comes back `status="fail"` with no latency and TPE learns to
avoid it. The reason to narrow the fallback is that hiding a real defect behind a working
alternative costs the repair agent its diagnosis, not that it could corrupt a result.

## Scope

Narrower than the gate finding, and for a reason that survives the gate finding's
falsification. **The fp16 claim rests on output MAGNITUDE, which is the right variable here:**
fp16 saturates above 65504, L3:48 reaches 1.038e22, L3:21's `ref_absmax` is **5.749** (measured,
07:19). An fp16 cast cannot overflow on a 5.749-magnitude output, so the corner really is
legitimate there.

That is exactly the distinction the gate finding got wrong: magnitude governs *overflow* (this
finding) but not *two-precision agreement* (that one). L3:21's floor is 0.95536 despite its
bounded range. Confirmed empirically for this finding: the 21/43 rejection histories show
max-abs-diff of 0.0014–0.004 with **no non-finite reports** in any of them, versus 7.3–17.6%
non-finite on L3:48.

## What is not claimed

That the fp16 corner is the *only* reason these seven died. Three of them (eb910a18 at
frac 0.9125, 741c2699 whose default witness failed three times, 8cb745ff) had genuine defects
too. The claim is narrower and fully supported: the correlation with declaring a dtype knob is
7/7 against 7/7, the signature is overflow at exactly the magnitudes fp16 cannot hold, and
repair fixed the default witness at attempt 1 in four of seven cases and then burned its
remaining budget on an impossible corner.
