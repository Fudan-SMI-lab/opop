# Result: 10.2 ms verified on round 3 — the refuted hypothesis, re-attacked, wins

`run-l3-43-20260905-091705`, `cand-e3a5da01`. **9.73 ms `tuned_ms`** at 14:53:36, re-evaluated in a
fresh process at **10.2 ms** when the run finished at 15:31. Against the 18.40 ms same-precision bar
(`torch_compile_tf32`) that is **1.804×** — the project's best verified result on any task, and the
first to clear its same-precision baseline by a wide margin.

It also falsifies or corrects three things I documented earlier today, which is the more important
part of this note.

## The number, verified

```
tr-88446e7a   9.73 ms   std 0.444   min 8.9   max 10.1   n=20   reused_measurement: None (FRESH)
              regs 164   spills 0   shared 98304   warps 8   stages 5
              kernels: _flash_kernel, _linear_kernel, _row_microtiled_qkv_kernel
QKV_LOGICAL_M 512  QKV_MICRO_M 64  QKV_BLOCK_N 128  QKV_BLOCK_K 16
ATTN_BLOCK_M 128   ATTN_BLOCK_N 16  ATTN_NUM_STAGES 5   COMPUTE_DTYPE fp16
```

Not a cached witness (`reused_measurement` absent, job file `tr-88446e7a.py` on disk). The second-
and third-best trials are 9.85 and 10.00 from *different* `QKV_LOGICAL_M` values, so the result is
not one lucky sample sitting alone — three configurations land under 10 ms.

**RE-EVAL CONFIRMED at 15:31.** `final_reeval_ms = 10.2` (`final_reeval_ok: true`), a **+4.8%**
gap on the 9.73 tuned figure — within the −6.2%/+6.7% range on record. The verdict the harness
computed for itself:

```
precision                    fp16
compared_against             torch_compile_tf32   (18.40 ms, the same-precision bar)
same_precision_speedup       1.8039
beats_same_precision_baseline  true
excessive_speedup_flag       false
speedups   eager 4.078 | eager_tf32 2.824 | torch_compile 3.471 | torch_compile_tf32 1.804
```

So the headline is **10.2 ms, 1.804×** the same-precision baseline — not the 1.891× that
`tuned_ms` suggested. The pre-registered prediction ("at the worst gap on record this is 10.3 ms
and still 1.78× the bar, so the conclusion is robust to it") held: the gap moved the number and
did not move the conclusion. **1.804× is the figure to quote**; 9.73 is a tuning artifact and
`memory: opop-v2-reeval-gap-is-the-real-number` is why.

## Correction 1: my round-3 record was 0-for-12, and this breaks it

`finding-converged-stop-kind-is-unreachable.md` states, and I repeated several times today, that
round 3 has **never** improved a family — 0 of 12 (I had also written 0-for-13 before recounting).
`cand-e3a5da01` is `fam-4aea322a`'s **third** rewrite round:

```
CONV  11:00:08  hist=[]            used=0
ROUND 11:31:09  best 11.0                        <- round 1
CONV  13:18:43  hist=[11.0]        used=1
ROUND 14:02:50  best 11.0                        <- round 2, flat
CONV  14:42:59  hist=[11.0, 11.0]  used=2
                best 9.73                        <- round 3, -11.5%
```

So the record becomes **1 of 13**, and the one success is an 11.5% gain on the family holding the
project's best result. That materially weakens the strongest practical argument I was making for the
`best_history` seeding fix.

To be precise about what survives and what does not:

- **The unreachability proof is untouched.** `converged` still cannot fire; that is arithmetic about
  when `len(best_history)` reaches 3, machine-checked over 97 family decisions, and this result does
  not bear on it.
- **The seeding counterfactual is untouched as arithmetic** — 6 of 14 families would still have
  frozen at `used=2`, and all 6 still gained exactly 0.00% in their actual round 3. Those numbers
  are unchanged.
- **But the cost/benefit framing is weakened.** I wrote that the harness "is structurally obliged to
  spend a round that has a 0-for-13 record". It is now 1-for-13, and the single win is the largest
  single-round gain in this run's headline family. Crucially, `fam-4aea322a` is **not** in the freeze
  set of the counterfactual — the seeded policy would have let it continue on `[22.5, 0.0]` — so the
  fix would *not* have cost this result. That is the honest version: the fix remains safe on the
  evidence, and my rhetoric about round 3 being uniformly worthless was too strong.

## Correction 2: the failed-hypothesis channel has its first win, 30 minutes after I recorded 0-for-9

`measurement-failed-hypothesis-channel.md`, committed at ~14:50, reports "9 of 11 react, 0 of 9
won" and lists `cand-e3a5da01` as pending. It won.

`REWRITE_PRODUCED` records `hypothesis_id: 'H1'` — the *same hypothesis* the H1 refutation
(`result-analyst-hypothesis-refuted-by-control.md`) showed produced 11.8 vs 11.0. And the summary
explicitly contrasts with that failure:

> "**Unlike the failed narrow-column/warp-group rewrite**, it does not partition output columns or
> serialize column subtiles."

So the channel's value is now demonstrated, not just its plumbing: **1 of 10**. And it vindicates
the specific thing I declined to do in that same commit — I wrote that making the channel stricter
(forbidding re-attack of a failed axis) "would be wrong here… the H1 refutation showed 256 rows
*are* reachable and the gain was not there, which does not prove no decomposition reaches it
profitably". Thirty minutes later a different decomposition of the same axis produced the project's
best result. A stricter channel would have forbidden it.

## Correction 3: registers, not the row count, were the binding constraint

The mechanism is visible and it is not what H1's first implementation assumed:

| candidate | ms | regs | spills | shared | approach |
|---|---|---|---|---|---|
| `cand-cb7be6b4` (seed) | 14.20 | 255 | **12** | 65536 | — |
| `cand-13efdcd8` | 11.00 | 248 | 0 | 81920 | `QKV_M_CTAS` serialisation |
| `cand-aa016dfe` | 11.80 | **255** | 0 | 81920 | 256 rows × 64-col subtiles (H1, column-wise) |
| **`cand-e3a5da01`** | **9.73** | **164** | 0 | 98304 | 512 rows × 64-row microtiles (H1, row-wise) |

`cand-aa016dfe` reached 256 rows and stayed pinned at 255/255 registers — which is why it lost. The
analyst's original reading was that `QKV_BLOCK_M` was *register-blocked*, and that was correct; what
the column-wise decomposition failed to do was actually relieve the pressure. The row-microtiled
version drops to **164 registers**, a 33% reduction, and buys the latency with the freed budget
(shared memory rises 81,920 → 98,304, trading a resource that had headroom for one that did not).

The `QKV_MICRO_M` sweep shows the microtile is the operative knob, not the logical row count:

```
MICRO_M=16   best 27.20      MICRO_M=64   best  9.73
MICRO_M=32   best 12.20      MICRO_M=128  best 23.70
```

Interior optimum at 64, with both neighbours 25%+ worse — the same shape as `QKV_M_CTAS`'s optimum
at 4 (`inprogress-l3-43-11ms-rewrite.md`). And `LOGICAL_M` at the winning `MICRO_M=64` is nearly
flat (128 → 9.85, 256 → 10.00, 512 → 9.73), confirming that *how the rows are sliced* mattered and
*how many there are* barely did. H1 asked for more rows; what paid was a different slicing of them.

## What this says about the two-loop claim

This is the cleanest positive instance the project has produced, and worth stating precisely because
today's other analyst finding was negative:

1. The tuner's statistics identified `QKV_BLOCK_M` at its domain edge with `blocked_by: registers`.
2. The analyst named it, with a quantified prediction of +8.0%.
3. The first rewrite implemented the named change, delivered the stated *mechanism* (256 rows
   reachable), and **lost** — the prediction was refuted.
4. The failure was journalled and fed back.
5. A second rewrite attacked the *same* limit by a different decomposition, actually relieved the
   register pressure (248 → 164), and gained **11.5%** — more than the 8.0% originally predicted.

Step 3 is what makes this evidence rather than a coincidence: the loop's first attempt at this axis
failed, was recorded as failed, and the recorded failure is cited in the winning child's own
rationale. `measurement-predicted-gain-overshoots.md`'s conclusion holds unchanged — the magnitude
field is unreliable — while the *target selection* was right and eventually paid, which is exactly
the split that doc argued for.

## Open

- ~~`final_reeval_ms`~~ — **done: 10.2 ms, 1.804×, `final_reeval_ok: true`.** See above.
- ~~`fam-4aea322a`'s round-3 `FAMILY_ROUND_RECORDED`~~ — **fired.** History is
  `[11.0, 11.0, 9.73]`, `rewrite_rounds_used: 3`, frozen as `budget_exhausted`, exactly as
  pre-registered. That makes the record **14 of 14** `budget_exhausted` with `converged` never
  firing, and the twist stands: the round it "wastes" was the productive one. `fam-92e7c576`
  froze the same way on `[19.6, 19.6, 19.6]`, making it 15 of 15.
- The run finished at **6.24h of a 12h budget** with all four families frozen. Half the wall
  clock went unused — the empty-family/exhaustion interaction in
  `finding-run-stops-with-budget-unused.md`, still unfixed and now the largest single waste in
  a completed L3 run.
- 23 of 40 trials failed on this candidate (17 complete), so the space is still coarse; a fourth
  round is not budgeted.
