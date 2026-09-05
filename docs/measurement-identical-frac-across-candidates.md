# Measurement: four unrelated L3:21 candidates fail correctness at the identical value

`cand-fe183b2d` was rejected at 16:12 on the clean L3:21 rerun with
`frac_within_tol = 0.953277`. That exact number has now appeared **four times**, on four
candidates with completely different kernel structures and different knob names:

```
cand-d31b0474   PW_BLOCK_M/PW_WARPS/REDUCE_BLOCK/APPLY_BLOCK...   tf32   0.953277
cand-7dcdbd99   PW_BLOCK_M/PW_WARPS/APPLY_BLOCK/FINISH_WARPS...   tf32   0.953274
cand-fdb4dac6   BLOCK_M/PW_NUM_WARPS/DW_BLOCK_X...                tf32   0.953277
cand-fe183b2d   EXPAND_BLOCK_M/PROJECT_BLOCK_M/PROJECT_NUM_...    tf32   0.953277
```

Agreement to six digits across four independent decompositions is not a coincidence and not
four separate bugs. **The failing elements are a property of the task and the precision, not of
any candidate's code.** At tf32 on L3:21, 4.67% of output elements fall outside the 1%
per-element tolerance; the reference's own tf32-vs-ieee comparison puts 4.46% outside.

## First, a correction to my own reading

When the rejection came in I read it as the deferred gate problem — a candidate *more*
consistent than the reference is with itself, rejected anyway. **That was wrong.**

```
candidate frac_within_tol   0.953277
L3:21 noise floor           0.955360
delta                      -0.002083     BELOW the floor, by 0.21%
```

It is 99.78% of the floor, not above it. A floor-relative gate with zero tolerance would reject
this candidate too, so it is *not* a clean instance of the over-strict-gate class and I should
not have described it as one. `scripts/audit_noise_floor_rejections.py` places it correctly, in
the BELOW group — which is why the ledger exists rather than reading individual failures by eye.

## What the four-way agreement does establish

The ledger already showed 12 rejections above the floor and 19 below, with margins as thin as
0.0013. The new information is about the **structure of the below-floor group**: at least 4 of
those 19 are the same measurement, not four independent near-misses.

That matters for the gate decision the user has twice deferred, in a way that cuts against a
naive fix:

- **A zero-tolerance floor-relative gate does not solve L3:21.** These four sit just below the
  floor, so they would still be rejected. Any gate that admits them needs a *tolerance below*
  the floor, and the audit's BELOW list shows what that costs: `-0.0013` at L3:48 and `-0.0128`
  at L3:43 are in the same band, and one of them (`cand-90886b3c` at `+0.0022`) is verified
  DEGRADED on the other three metrics despite clearing the gate metric.
- **The tolerance would have to be task-relative, not absolute.** 0.21% of the floor is a
  different absolute number on each task, and the three floors are 0.9554 / 0.9767 / 0.9778.
- **`cosine` is ≥ 0.999999 on every one of the four.** There is no directional error at all. The
  entire disagreement is per-element, on a task whose own two precisions disagree on 4.5% of
  elements.

So the honest summary is: this reinforces that `frac_within_tol > 0.99` is unreachable for a
tf32 candidate on L3:21 — **10 of 49 L3:21 candidates never published a space at all** — while
also showing that the simplest floor-relative repair would not have admitted today's candidate.
The gate change remains the user's call, and this narrows what a *correct* version of it would
have to look like rather than making the case easier.

## Cost, in this run

`cand-fe183b2d` was rejected, went to repair at 16:12:37, and re-entered parameterization at
16:14:01 — roughly 90 seconds plus a GPU witness pair, and the repair agent's only remaining
lever is precision (`finding-floor-rejection-sends-repair-after-the-dtype.md`: 0 of 3 recovered
that way, versus 4 of 4 on real resource errors). The historical pattern predicts it will switch
to fp16 and hit `witness_minimal_failed` instead.

The rest of the run is healthy: `cand-e9b995d0` tuned to **22.1 ms** (`improved_family: true`)
against baselines of `torch_compile_tf32` 16.4 / `eager_tf32` 21.2 / `torch_compile` 22.2 /
`eager` 25.5, and a second space published on attempt 1.

Reproduce with `python scripts/audit_noise_floor_rejections.py`.
