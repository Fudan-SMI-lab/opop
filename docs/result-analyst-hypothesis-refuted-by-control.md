# Result: the first analyst hypothesis tested against its own parent as a control — and refuted

`run-l3-43-20260905-091705`, `fam-4aea322a` round 2. This is the cleanest test of the paper's
central claim the project has produced, because for once the prediction, the rewrite, and the
control are all on the record and the control is the parent's own measured best.

## The prediction

The analyst read `cand-13efdcd8`'s 40 trials (best **11.0 ms**) and named one boundary knob with
a quantified gain (`BOTTLENECK_REPORTED`, 11:18:03):

```
param:              QKV_BLOCK_M
headroom_direction: increase
blocked_by:         registers
predicted_gain_pct: 8.0
evidence:           27.6 ms at 32, 20.9 at 64, 17.5 at 128 ... best-config peak usage is
                    already 248/255 registers with no spills ... doubling BLOCK_M without
                    changing decomposition is therefore register-blocked
```

And hypothesis **H1**, whose whole content is "make 256 rows reachable without a 256×128
accumulator":

> Reshape the persistent QKV matmul so a CTA covers more rows without holding a 256x128
> accumulator: use a 256x32 or 256x64 logical tile, partition rows/output columns among warp
> groups, and serialize output-column subtiles while retaining the current fp16 `tl.dot` with
> fp32 accumulation.

This is exactly the intended mechanism of the paper: a knob pinned at its domain edge by a
hardware resource is a *structural* signal, and the rewriter's job is to change the structure so
the knob can move. The analyst even predicted its own risk correctly ("narrower N tiles may
increase weight traffic").

## The rewrite did what was asked

`cand-aa016dfe` (13:24:17) implements H1 faithfully — a new `QKV_N_SUBTILES` knob serialises
64-column subtiles, so a 256-row logical tile holds a 256×64 accumulator: 16,384 elements, the
same count as the parent's 128×128. The claim is explicit in its `approach_summary`:

> Reshapes persistent QKV into a 256-row logical tile and serializes two 64-column subtiles,
> keeping each live fp32 accumulator at 256x64 (the same 16,384 elements as the parent's
> 128x128) while reaching the re[gister-blocked region]

And **256 became reachable, and won inside the child**:

| `QKV_BLOCK_M` | parent `cand-13efdcd8` | child `cand-aa016dfe` |
|---|---|---|
| 32 | 18.6 (n=3) | 16.9 (n=3) |
| 64 | 20.7 (n=3) | 24.7 (n=1) |
| 128 | **11.0** (n=21) | 12.2 (n=5) |
| 256 | *infeasible* | **11.8** (n=10) |

So the structural change worked as designed: the child's own optimum is at 256, the value the
parent could not hold. The monotone trend the analyst read off the parent continued into the
newly-opened region.

## And the result was 7% WORSE than the parent

`TUNING_DONE` 13:32:58: **11.8 ms**, `improved_family: False`. Against the parent's 11.0.

The predicted +8% did not arrive; the measured outcome is −7%. And the reason is visible in the
profile of the child's best trial:

```
child  best 11.8   QKV M=256 N=64 SUB=2 CTAS=8   regs=255  spills=0  shared=81920
parent best 11.0   QKV M=128 N=128 CTAS=4        regs=248  spills=0  shared=81920
```

The child reached 256 rows and is *still* at the register ceiling — 255 of 255, versus the
parent's 248. Holding the accumulator element count constant did not buy register headroom,
because the extra rows need their own addressing, masks, and pointer arithmetic. The decomposition
moved the cost rather than removing it, and the serialised subtile loop added the weight traffic
the analyst had flagged as the risk.

## Why this test is worth more than the 0.8 ms

Every previous rewrite in the record improved on its parent (`result-every-rewrite-round-improved.md`,
4-for-4 at the time), which is a pleasing number and a weak one: a rewrite that improves tells you
the child is better, not that the *reasoning* was right. A rewrite can win for reasons unrelated to
the hypothesis it was given — `cand-88e76051`'s kernel fusion is the clear case.

This one is different in three ways:

1. **The prediction was quantified before the measurement** (+8.0%), so it can be wrong.
2. **The mechanism was verified independently of the outcome.** 256 really did become reachable
   and really is the child's own optimum — so this is not "the rewriter failed to implement H1".
   The structural claim was delivered and the performance claim still failed.
3. **The control is the parent's own 21 trials at 128**, in the same run, same hardware, same
   session, same dtype. No cross-run comparison, no re-eval gap, no seed-cohort confound.

That combination isolates the failure to the *inference* — "this knob is register-blocked, so
relieving the block will pay 8%" — and not to the pipeline that acted on it. The knob was
correctly identified as blocked. Relieving the block was correctly implemented. The gain was not
there.

## What it says about `at_boundary` as a structural signal

The honest reading is narrow. `at_boundary` correctly located a real hardware wall: the parent
genuinely cannot hold 256 rows, and the child genuinely can. What it could not tell anyone is
whether the wall was *worth breaching* — and here the answer was no, because the resource that
actually binds (registers per thread) is not the resource the tile-shape change economises.

Two things follow that are worth holding onto:

- **A boundary knob's monotone trend does not extrapolate past the boundary.** 27.6 → 20.9 → 17.5
  → (11.0 at the best config) looks like a curve heading somewhere. It was not: the child measured
  the point beyond the edge and the curve turned. Every `predicted_gain_pct` in this project is an
  extrapolation of exactly this kind, and this is the first one measured against its own control.
- **The analyst's *own* caveat was the accurate part.** It wrote "the gain estimate is
  conservative because the trials are unpaired and resource reporting is the maximum across three
  kernels" — and unpaired-ness is precisely what went wrong. The 17.5 at `QKV_BLOCK_M=128` and the
  20.9 at 64 are different programs' configurations, not a controlled sweep. The report was more
  careful than its own headline number.

## Not a reason to change anything yet

n=1. One refuted prediction does not establish that `predicted_gain_pct` is systematically
optimistic — that needs the same treatment applied to every hypothesis with a parent control, and
most rewrites in the record do not have one because they changed several things at once.

What I would want before touching the analyst prompt or the `at_boundary` rule
(pending item 7): a count of how many `parameter_limits` entries ever got tested against their
own parent, and the signed error on each. `scripts/audit_expansion_outcomes.py` does the analogous
job for K's expansions and found K's attributable rate was 45% against a 70% headline — the same
question asked of the analyst's predictions is the obvious next audit, and it is a read-only one.

The round-2 sibling `cand-45c3fd7d` (a two-kernel split, 22.0 ms, `improved_family: False`) also
failed to beat 11.0. Neither child improved, so `fam-4aea322a`'s round 2 is on course to close with
its round-1 result intact.

**Pre-registered, since the round is not yet recorded** (its last `CONVERGENCE_DECIDED` at 13:18:43
shows `best_history: [11.0], rewrite_rounds_used: 1`, and no round-2 `FAMILY_ROUND_RECORDED` has
fired): the family's history should become **`[11.0, 11.0]`** — the first non-improving round in
this run, and the first `best_history` in the project long enough for `_improvement_pct` to compute
anything at all. It will compute **0.0%**, which for the first time is *true* rather than an
artifact of a one-entry history (`finding-converged-stop-kind-is-unreachable.md`). With
`no_improve_rounds: 2` that is one of the two rounds needed for a `converged` freeze — so if the
run has budget for a round 3 on this family, this is the first real chance to observe whether the
`converged` stop kind is reachable in practice. If instead `budget_exhausted` fires first, that is
the 14th confirmation of the existing finding.

`cand-45c3fd7d` is meanwhile in its own K expansion (a parameterizer call started 13:43:51), so the
round is not closed yet and these numbers can still move.
