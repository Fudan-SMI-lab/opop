# Plan: the four fixes going into the next round, and the experiment that tests them

Written before implementation so the intent is on record and the next round's results can be
read against it. Four changes, three of them settled bugs and one an experiment.

## Status of the experiments when this was written

All runs stopped at the user's request. GPU verified idle (no `worker_main`, no compute apps).

| run | hours | state | reeval | precision | vs | speedup |
|---|---|---|---|---|---|---|
| l3-48 `010737` | 6.08 | FINISHED | **1.96 ms** | unknown | torch_compile | **9.49×** |
| l3-21 `071312` | 2.06 | FINISHED | **15.8 ms** | fp16 | torch_compile_tf32 | **1.03×** |
| l3-43 `091705` | 6.24 | FINISHED | **10.2 ms** | fp16 | torch_compile_tf32 | **1.80×** |
| l3-21 `153452` | 0.43 | ABANDONED | — | | | reward-hack contaminated |
| l3-21 `160156` | 0.89 | INTERRUPTED | — | | | 4 seeds tuned, best 19.4 ms |

## Fix 1 — an empty family must not end other families' rounds

**The bug.** `_rewrite_round` returns `progressed`, and a family with no correct candidate is
frozen without setting it (`orchestrator.py:1131`):

```python
if family.best is None:
    family.status = "frozen_budget"
    continue                      # progressed stays False
```

The outer loop reads `not progressed` as "nothing left to do anywhere" and freezes **every**
remaining active family (`orchestrator.py:329`). With `max_families_active: 2`, two empty
families fill both slots, so a single pass in which both slots happen to be empty ends the run
— including for families holding a good incumbent with rounds left.

**Evidence.** The threshold is exactly `max_families_active`:

| run | empty families | rounds used per family | elapsed |
|---|---|---|---|
| l3-48 `010737` | 1 | [3,1,0,3] | 6.08h |
| l3-21 `071312` | **2** | [0,1,0,1] | **2.05h of 12h** |
| l3-43 `091705` | 0 | [3,3,1,1] | 6.24h |

On l3-21 `071312` both empty families were activated for the first time at 09:16:16 — the same
second the run finished, with a 15.5 ms incumbent and 4 of 6 rewrite rounds unspent.

**The fix.** A family that cannot be rewritten should not occupy a rewrite slot. In
`FamilyManager.active_families()`, require a correct candidate:

```python
active = [f for f in self.families.values()
          if f.status == "active" and f.best is not None]
```

Generic: it states an invariant ("a family with nothing correct cannot be structurally
rewritten, so it does not compete for rewrite budget") rather than naming a task or a count.
Empty families are still frozen elsewhere; they just stop dragging others down. If *every*
family is empty the list is empty, `progressed` is False, and the run ends — which is correct.

## Fix 2 — `HARD_EDGE` must ask "would the extension cross a wall", not "is the range already at one"

**The bug.** Improvement K asks the parameterizer to widen knobs whose optimum sits at the edge
of the tried range. `HARD_EDGE` blocks requests that cannot be honoured, but only when the
range **already touches** the wall:

```python
return min(numeric) <= wall if direction == "min" else max(numeric) >= wall
```

Measured behaviour today:

| knob domain | direction | outcome | correct? |
|---|---|---|---|
| `NUM_WARPS [1,2,4]` | min | blocked | yes — 1 is already there |
| `NUM_WARPS [2,4,8]` | min | **requested** | no — agent is asked for warps=1 |
| `EXPAND_NUM_WARPS [2,4,8]` | min | **requested** | no — same, via suffix match |
| `BLOCK_K [16,32,64]` | min | **requested** | no — 8 is illegal for a `tl.dot` K dim |
| `BLOCK_M [32,64,128]` | max | requested | yes — legitimate |

Downward expansion is **0 for 9** post-filter. Two mechanisms: warp counts pushed to 1 (loses,
e.g. `EXPAND_NUM_WARPS=1` at +20% worse), and `BLOCK_K` pushed to 8, below the `tl.dot`
contraction floor of 16 — twice.

**Correction to what this section first said.** I wrote that `BLOCK_K=8` "fails to compile and
takes the whole expansion down with it". Only the first half of that happens. The first witness
is rejected on `CompilationError: Input shapes should have M >= 1, N >= 1 and K >= 16`, and then
the parameterizer *retries and changes the kernel* to make 8 legal —
`DOT_BLOCK_K = 16 if BLOCK_K == 8 else BLOCK_K`, masking the 8 lanes that must not contribute.
Two candidates invented that shim independently. So 8 compiles and completes trials, as the
widest dot the hardware has at half useful occupancy, and comes **last in its domain** both
times: `PV_BLOCK_K` 38.8ms vs 24.4 best, `QKV_BLOCK_K` 57.1ms vs 14.75 best.

That makes the filter's case stronger rather than weaker: below a hardware wall the agent can
only refuse (a wasted witness attempt plus a retry) or emulate (a masking branch in the hot loop
and a guaranteed-worst value in the domain). The request cannot win either way, so the floor
belongs in the table without the filter needing to predict which outcome it prevented. And it is
subtractive only where nothing was lost — on both historical expansions **6 of the 7** requested
knobs survive, including the `OUT_BLOCK_M=256` that earned cand-45c3fd7d's 7.7% gain.

**The fix, and a correction to my own first attempt.** I implemented the "would the next
value cross a wall" predicate first. Checked exhaustively over 6392 candidate domains it
gives an **identical verdict in every case** to the existing "is the edge at the wall" test,
because a ladder whose next step would cross a wall already has its edge at it. That
arithmetic was dead code, so it was removed and the docstring now says so.

The real fix is one table entry: add the `tl.dot` contraction floor
(`("BLOCK_K", "min"): 16`) beside the existing warp/stage floors, matched by suffix as they
already are. Verified: `BLOCK_K/QKV_BLOCK_K/EXPAND_BLOCK_K` at 16 are now blocked,
`PV_BLOCK_K=[32,64]` may still drop to 16 (legal), and `BLOCK_M`/`BLOCK_N` stay free —
`tl.dot`'s rule is asymmetric and only the contraction dim has a floor.

The warp-to-1 requests are NOT blocked by this and deliberately so: 1 warp is legal, merely
slow, and the 0-for-9 record is the tuner's business rather than this filter's.

## Fix 3 — seed `best_history` so `converged` is reachable and the slope is not a round stale

**The bug.** `stop_kind="converged"` has never fired and cannot, by arithmetic.
`family_verdict` checks the budget **before** convergence, and convergence needs
`len(best_history) >= no_improve_rounds + 1 = 3`, while the budget freezes at
`rewrite_rounds_used >= 3`. Per-round timing is verdict → rewrite → `record_round`, so:

| round | rounds_used at verdict | len(history) at verdict | branch |
|---|---|---|---|
| 1 | 0 | 0 | continue |
| 2 | 1 | 1 | continue |
| 3 | 2 | 2 | continue |
| 4 | 3 | **3** | FREEZE budget_exhausted (fires first) |

Confirmed over all 48 recorded families: `history_len` is 0, 1 or 3, never more, and every
status is `frozen_budget`.

The same off-by-one has a second cost. `_improvement_pct` needs two entries, so a family's
**first** rewrite round — the largest gains this project has recorded (22.5%, 12.9%, 15.0%,
23.9% on l3-43 `091705`) — scores 0.0% slope at the moment round 2 is allocated. The ranking
rule that is supposed to favour still-moving branches is blind exactly when it matters most.

**The fix.** Record the seed-phase tuned best as `best_history[0]`. Then round 3 has three
entries and `converged` can fire, and the first round's gain counts toward its own family's
ranking.

**Known side effect:** family activation order changes, so this round's search trajectory is
not comparable with earlier runs. Accepted deliberately — that is why all four changes land
together and the next round is a fresh comparison set.

## Fix 4 (the experiment) — an fp64-golden-reference relative correctness gate, DEFAULT ON

This is the one being tested rather than merely fixed, so it is a config flag and it is **on**
for the next round.

**Why.** Our gate applies one per-element tolerance to every candidate regardless of declared
precision, and requires 99% of elements to pass. All three reference implementations do
something different, verified by reading their source on this machine:

- **KernelBench** (`src/kernelbench/eval.py:83`) keys tolerance to declared precision:
  fp32 → 1e-4, fp16/bf16 → 1e-2, as `allclose(atol=rtol=tol)`. Applying its own rule to our
  rejections: **27 of 27** rejected (candidate, dtype) pairs pass the fp16/bf16 tolerance,
  0 of 27 pass the fp32 one. Every candidate we rejected is correct by the benchmark's standard.
- **torchbench + `torch._dynamo.utils.same()`** ship a genuine noise-floor-relative gate:
  cast the model to fp64 for a golden reference, then
  `res_error <= multiplier * ref_error + tol/10` on RMSE, with `multiplier = 3.0` for
  fp16/bf16 and 2.0 otherwise (8–10× for small tensors, "in the presence of noise, noise might
  dominate our error metric"). The candidate is *allowed to be worse than the reference*, by a
  documented factor.
- **KernelFoundry** applies `(frac on either witness) AND (cosine on either witness)`, where we
  apply `(frac AND cosine) on either`. On our data this changes nothing: cosine passes
  **279 of 279** and frac fails 279 of 279, so our gate is effectively single-criterion.

**Verified against our own numbers.** 512³ GEMM, MBConv-like values, fp64 golden reference:

```
reference's own rmse vs fp64:  torch@tf32 1.155e-03   <- the floor
                               torch@ieee 9.437e-07

candidate              rmse vs fp64   x floor   dynamo   our gate
fp32 ieee                 1.612e-06     0.00x    PASS      PASS
fp16 + fp32 accumulate    1.155e-03     1.00x    PASS      FAIL   (frac=0.9819)
tf32                      3.106e-03     2.69x    FAIL      FAIL   (frac=0.9807)
bf16                      9.466e-03     8.19x    FAIL      FAIL   (frac=0.8518)
```

The fp16 candidate is **exactly as accurate as the tf32 reference it replaces** and our gate
rejects it. This also independently confirms fp16-with-fp32-accumulate is 2.7× more accurate
than tf32 here despite the identical 10-bit mantissa — which is why "fp16 passes, tf32 fails"
was never a gate bug.

**Design, following dynamo rather than inventing one.**

```
accept  <=>  existing dual-witness relaxed gate passes
        OR   rmse(fp64_ref, candidate) <= multiplier * rmse(fp64_ref, tf32_ref) + tol/10
```

- the floor is measured against **fp64**, not against another low-precision run. This is what
  makes it a good floor and answers every objection I raised earlier against a
  "two-imprecise-results" floor: fp64's own error is 9.4e-07 versus the floor's 1.2e-03, three
  orders of magnitude below.
- **RMSE**, a single aggregate, so no second threshold per extra metric. RMSE is dominated by
  large deviations, which is what the multi-metric check was for.
- `multiplier` from the candidate's declared precision: 3.0 for fp16/bf16, 2.0 otherwise.
- the fp64 golden MODEL is built once per job, and an fp64 forward runs only on a trial the
  absolute gate has already failed — so the happy path pays nothing.
- the multiplier is chosen from the precision the candidate **computes** in, read from the
  MATERIALIZED source (the tuner's chosen knob value is already substituted into `PARAMS`).
  My first implementation keyed it off `out_kernel.dtype`, which is a live bug: a candidate
  doing `tl.dot(a.to(bf16), ...)` with an fp32 accumulator returns float32, so the
  low-precision multiplier never fired. Caught by running it — a bf16 candidate was scored
  with 2.0 instead of 3.0.
- fallback: if fp64 raises (unsupported op, OOM) the relative arm is skipped and the existing
  gate decides alone — dynamo's own fallback in that case is cosine, which we already require.

**Risks, stated.**

1. It changes which candidates are accepted, so results are not comparable with earlier runs.
2. fp64 on this GPU runs at ~1/64 rate; mitigated by caching per task.
3. It may admit a genuinely wrong kernel that RMSE happens to flatter. The multiplier is the
   only guard, and it is copied from PyTorch rather than chosen by me.
4. The L3:21 candidates sitting 0.21% *below* the two-precision floor may or may not pass under
   an fp64 floor — that is a different quantity and cannot be predicted from existing data.
   This is the main thing the next round measures.

**Not implemented and still deferred:** replacing the absolute gate outright, or lowering
`relaxed_pass_frac`. The flag adds an alternative acceptance path; it does not remove the
existing one.

### Verified live on the GPU before enabling

Run against a real rejected L3:21 candidate (`cand-fe183b2d`) materialized at two dtypes:

```
candidate   fp64 gate OFF          fp64 gate ON
tf32        0/3 correctness_mismatch  ->  3/3 PASS
bf16        0/3 correctness_mismatch  ->  0/3 still REJECTED
                                          ref rmse 7.03e-04, candidate 5.63e-03,
                                          multiplier 3.0, threshold 3.11e-03, ratio 8.00x
```

So the gate **discriminates** rather than merely loosening: the tf32 candidate, whose error
is comparable to the reference's own, is admitted; the bf16 candidate at 8× the floor is
still rejected even with the wider low-precision multiplier. That is the behaviour the
experiment is meant to test, confirmed to actually be wired up rather than a no-op flag.

## What the next round is expected to show

Pre-registered, so the results can falsify it:

- fixes 1+3 should raise rewrite-round utilisation on L3:21 from 2 of 6 toward 6 of 6, and the
  run should last well beyond 2.05h without hitting the 12h wall.
- `stop_kind="converged"` should appear for the first time on some family, or the arithmetic in
  fix 3 is wrong.
- fix 2 should eliminate `min`-direction warp requests and `BLOCK_K`-below-16 requests entirely;
  any that appear mean the wall prediction missed a knob-name shape.
- fix 4 should admit some of the 27 currently-rejected low-precision candidates. If it admits
  candidates that then fail the final 5-trial re-eval, the multiplier is too loose and the flag
  should go back off.

## Credential note

The API keys in `.opencode/opencode.jsonc`, `opencode_backup.jsonc`, `kimi-provider.yaml`, the
global `opencode.jsonc`, and `test.py:6` are plaintext, and the GitHub PAT has appeared in
session history. Both should be rotated at delivery. This is a user-side action.
