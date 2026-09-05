# Result: kernel-count direction predicts a rewrite's outcome better than any prediction the harness makes

Found while reading `cand-03f69b0e` (14:15, 27.7 ms against its parent's 19.6). It is the fourth
rewrite in this run to *split* one kernel into more kernels, and the fourth to lose. Checking
every parent→child pair in the record turned up the strongest structural regularity I have
measured.

## The measurement

`TrialRecord.profile.kernel_names` comes from Triton's compile metadata, so the kernel count is
observed rather than claimed by the agent. Comparing each child's tuned best to its parent's:

| kernel-count change | n | median delta | improved |
|---|---|---|---|
| **FUSED** (child launches fewer) | 6 | **+14.9%** | **6 / 6** |
| same count | 38 | −0.5% | 14 / 38 |
| **SPLIT** (child launches more) | 11 | **−21.5%** | 2 / 11 |

Fusing has never once failed. Splitting fails 9 times in 11, and the failures are large.

Two checks before trusting it:

- **The count is stable per candidate.** `kernel_names` is unioned across a candidate's trials, so
  an unstable count would inflate it. Measured: **0 of 86 candidates** have a per-trial kernel
  count that varies. The count is a property of the program, not of the configuration.
- **Zero-kernel candidates excluded.** Three L3:21 candidates launch no Triton kernel at all —
  that is the dead-mode-branch bug (`finding-optimization-behind-a-dead-mode-branch.md`), not a
  fusion. Excluding them changes nothing, because they land in the `same count` bucket rather
  than `FUSED`.

## And it is task-dependent, which my first aggregate hid

Splitting by task reverses the sign:

| task | change | n | median | improved |
|---|---|---|---|---|
| **l3:21** (MBConv) | **SPLIT** | 3 | **+16.6%** | **2 / 3** |
| l3:21 | same | 13 | +0.0% | 6 / 13 |
| **l3:43** (attention) | FUSED | 5 | +14.8% | **5 / 5** |
| **l3:43** | **SPLIT** | 6 | **−37.0%** | **0 / 6** |
| l3:48 | FUSED | 1 | +15.0% | 1 / 1 |
| l3:48 | SPLIT | 2 | −30.3% | 0 / 2 |

So "fusing helps, splitting hurts" is **not** a universal rule — it is L3:43's and L3:48's rule.
On L3:21 the two best results in the project's history were both *splits*:

```
l3:21  SPLIT 20.5 -> 17.1   regs 40->78   shared 8->32768    3 kernels -> 5
l3:21  SPLIT 20.5 -> 15.5   regs 40->80   shared 8->32768    3 kernels -> 4
```

`cand-7dcdbd99` at 15.5 ms is L3:21's best kernel ever (`opop-v2-l3-21-best-result`), and it got
there by splitting into *more* kernels. Had I stopped at the aggregate I would have recorded a
rule that contradicts the project's own best L3:21 result.

## The mechanism is in the profile — and it is REGISTERS, not shared memory

My first reading of this was that shared-memory headroom was the separator. Widening the profile
table to all 17 pairs shows that is wrong, and the actual separator is cleaner.

Every split/fuse pair, sorted by the **parent's** register count:

```
task   move     delta   p_regs  p_shared  p_spill  parent
l3:48  FUSED   +15.0%       98       512        0  cand-a04c3f52
l3:43  FUSED   +15.0%      142     98304        0  cand-de802450
l3:43  FUSED    +7.9%      156     36864        0  cand-f954e424
l3:43  FUSED   +14.8%      156     36864        0  cand-f954e424
l3:43  FUSED    +2.5%      255     49152        4  cand-0c3b5820
l3:43  FUSED   +23.9%      255     73728       60  cand-3bf724d6
------------------------------------------------------------------
l3:21  SPLIT    -2.4%       35         0        0  cand-080f8c60
l3:21  SPLIT   +16.6%       40         8        0  cand-1eee8139
l3:21  SPLIT   +24.4%       40         8        0  cand-1eee8139
l3:48  SPLIT   -59.8%       80         0        0  cand-c18203b6
l3:48  SPLIT    -0.8%       80         0        0  cand-f4a2ce82
l3:43  SPLIT   -84.5%      248     81920        0  cand-13efdcd8
l3:43  SPLIT   -32.6%      255     24576       32  cand-372b59bd
l3:43  SPLIT  -132.9%      255     24576       32  cand-372b59bd
l3:43  SPLIT   -21.5%      255     49152        2  cand-2def3d6c
l3:43  SPLIT   -13.8%      255     49152        0  cand-f2e09b3b
l3:43  SPLIT   -41.3%      255     90112        2  cand-ec53c32b
```

Splitting a parent by its register occupancy:

| parent registers | n | improved | median |
|---|---|---|---|
| **≥ 200** (of 255) | 6 | **0** | **−37.0%** |
| < 200 | 5 | 2 | −0.8% |

Against my original shared-memory hypothesis:

| parent shared bytes | n | improved | median |
|---|---|---|---|
| ≥ 70,000 | 2 | 0 | −62.9% |
| < 70,000 | 9 | 2 | −13.8% |

**Registers separate the outcome; shared memory does not.** Six of eleven splits came from
parents at 248–255 of 255 registers and all six lost, with a −37% median. The shared-memory cut
puts only 2 pairs on the high side and leaves 7 failures on the low side — so the two L3:43
splits I originally cited (81,920 B and 90,112 B) were selected examples, not the rule. Three
other L3:43 splits failed from parents at 24,576 and 49,152 B, well below any shared-memory
threshold, and all three sat at 255 registers.

(The script's per-pair `delta` column is computed from the two sides' best *trial* latencies, so
two rows read +0.5% and −3.2% where the tables above — computed from `TUNING_DONE.best_ms`, which
can be a cached witness — read +2.5% and −0.8%. Neither the grouping nor the separator test
changes; the discrepancy is the `tuned_ms` vs best-trial distinction, not an error.)

That is a better mechanism anyway. A register-saturated kernel has all its live state in
registers; splitting it forces that state out to HBM as an intermediate tensor and back, and the
child re-materialises addressing for a second launch. There is no register budget to pay with, so
the transfer is pure cost — which is what the child-side spills show (0→50, 2→18, 0→6 in the
earlier table).

By contrast L3:21's winning splits came from parents at **35–40 registers and 0–8 bytes of
shared memory** — kernels barely using the hardware. There, a split lets the child acquire a real
tile (8 B → 32,768 B, 40 → 80 regs, zero spills). And the caveat that keeps this honest:
L3:48's two splits also came from low-register parents (80 regs) and *both* failed, one at
−59.8%. So low occupancy makes a split *possible*, not *good* — the 5-pair low-register group is
2 improved, 3 not.

The corrected claim: **high register occupancy in the parent is a reliable negative signal for
splitting (0 for 6), and low occupancy is not a positive one.** That is weaker than what I wrote
first, and it is what the 17 pairs support.

## Why this matters more than the analyst's magnitude field

`measurement-predicted-gain-overshoots.md` shows `predicted_gain_pct` has a −5.0% median bias and
the wrong sign 8 times in 21. The parent's register occupancy, as a *veto* on splitting, is 0 for
6 — a cleaner signal than anything the harness currently computes, from a field already recorded
on every trial (`profile.n_regs`).

Concretely, on this run: a rule reading "do not split a parent at ≥200 of 255 registers" would
have declined `cand-45c3fd7d` (11.0→20.3, parent 248 regs) and `cand-03f69b0e` (19.6→27.7, parent
255 regs), while permitting every L3:21 split that won. That is 2 rewrite children of this run —
roughly 25 minutes of agent and GPU wall — spent on a structural move whose parent profile
predicted failure before it ran.

## What I am NOT doing

**Not implementing any of it**, and the reason is specific rather than general caution:

1. **It would gate what the rewriter may attempt**, which is the search's core freedom. A rule
   that forbids splitting is exactly the kind of early-pruning the user has already ruled out for
   seeds ("no early-pruning/greedy seed selection"), and the same objection applies with more
   force here — a forbidden structural move cannot be recovered later, whereas a slow candidate
   merely loses a round.
2. **n is small where it matters.** 6 fusions and 11 splits total; L3:21's positive result is
   n=3, and the register veto rests on 6 pairs. The mechanism is plausible and the profiles agree,
   but "200 of 255 registers" is a threshold I would be picking from 17 data points — and note
   that every one of those 6 parents is at 248–255, so the data cannot distinguish a threshold of
   200 from one of 240.
3. **The honest version is a prompt hint, not a gate.** The rewriter's prompt could carry the
   parent's occupancy and the observed regularity — "this parent uses 248 of 255 registers; splits
   from parents above ~200 registers have lost 6 of 6" — and let the agent decide. That is
   information, not prohibition, and it is the only version I would propose. It joins the pending
   list rather than going in mid-run.

Note the rewriter *already* receives the profile at the best config, so this would be a framing
change rather than new data — which is another reason to prefer the hint over a gate.

The immediately useful thing is that this is now measurable: `scripts/audit_kernel_count_moves.py`
prints all three tables, so the next run's splits and fusions extend the count instead of needing
to be reconstructed.
