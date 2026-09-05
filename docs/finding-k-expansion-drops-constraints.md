# Finding: every K expansion silently drops the space's constraints — 26 of 26

Found at 12:31 while checking the fifth K expansion of `run-l3-43-20260905-091705`. This is the
most consequential mechanical defect I have found in improvement K, and it is systematic rather
than occasional.

## The observation

`cand-2d0194cc`'s expansion, `sp-77866f89` → `sp-03bd504f`:

```
BEFORE (5 constraints)                              AFTER (1 constraint)
  STATS_BLOCK_M % 16 == 0 and STATS_BLOCK_N % 16 == 0        OUTPUT_BLOCK_M * OUTPUT_BLOCK_N <= 4096
  OUTPUT_BLOCK_M % 16 == 0 and OUTPUT_BLOCK_N % 16 == 0
  STATS_NUM_WARPS * 32 <= MAX_THREADS_PER_BLOCK and OUTPUT_NUM_WARPS * 32 <= ...
  STATS_NUM_WARPS * 8 <= STATS_BLOCK_M
  OUTPUT_NUM_WARPS * 8 <= OUTPUT_BLOCK_M
```

All five original constraints gone, one unrelated new one in their place. And it happens on
**every expansion in the record**:

| | |
|---|---|
| expansions whose constraint list changed | **26 of 26** |
| expansions that ended with **zero** constraints | **11** |
| largest single loss | 7 constraints (`cand-3bf724d6`, `sp-57e98792` → `sp-f50bba31`) |

Not one expansion in any run preserved its constraint set.

## Why it happens

An expansion is not a patch to the existing space — it is a **fresh parameterizer call**.
`orchestrator.py:825-844` invokes `self.deps.parameterizer.invoke(...)` with an
`expand_directive`, gets back a whole new source and a whole new space, and accepts it:

```python
crun.space = verdict.space
self.store.append("SPACE_PUBLISHED", {"space": verdict.space.model_dump()})
```

The guard immediately above it checks only **domains**:

```python
prev_choices = {d.name: tuple(d.choices) for d in crun.space.domains}
new_choices  = {d.name: tuple(d.choices) for d in verdict.space.domains}
if new_choices == prev_choices:   # no-op expansion -> reject
```

Nothing compares `constraints`. So whatever the agent happens to re-derive on this call becomes
the new constraint set, and since it is writing a space from scratch under a directive that says
"widen these knobs", it routinely emits fewer.

## What it costs — measured

Across all runs, trials in post-expansion spaces that lost at least one constraint:

| | trials | failed |
|---|---|---|
| violate a **dropped** constraint | **116** | **71 (61%)** |
| violate nothing | 860 | 238 (28%) |

A configuration that the original space forbade fails at **61%, against a 28% baseline** — 2.2×
the rate. So the dropped constraints were carrying real information, and 116 trials were spent
exploring regions a previous version of the same space had ruled out. At ~10 s per trial that is
roughly 20 minutes of GPU, but the more important cost is opportunity: those are trials taken from
the 40-trial budget of the re-tune.

## The sharpest instance: K re-added the value repair had just removed

`cand-2d0194cc` reached this expansion by way of three repair attempts, whose entire purpose was
to get `STATS_BLOCK_M` off 256 (`finding-witness-has-no-resource-precheck.md`): at 256 the stats
kernel needs 139,264 B against a 101,376 B limit. Repair fixed it by setting the default to 128.

**Nine minutes later, K put 256 back in the domain:**

```
STATS_BLOCK_M: [16, 32, 64, 128] -> [16, 32, 64, 128, 256]
```

Two trials sampled it immediately and one failed with
`out of resource: shared memory, Required: 147456, Hardware limit: 101376` — a *worse* violation
than the one repair spent three attempts fixing, because the widened `OUTPUT_*` knobs add to it.

### Then the re-tune's winner turned out to be `STATS_BLOCK_M=256` — and K was right

The 40-trial re-tune reported **21.1 ms** (7.5% better than the pre-expansion 22.8), and its
θ_best is:

```
STATS_BLOCK_M: 256   <- the value repair removed
STATS_BLOCK_N: 64, STATS_NUM_WARPS: 4, STATS_NUM_STAGES: 2
OUTPUT_BLOCK_M: 128, OUTPUT_BLOCK_N: 32, OUTPUT_NUM_WARPS: 4, OUTPUT_NUM_STAGES: 3
COMPUTE_DTYPE: fp16
```

`shared_bytes: 98304` — it fits, at 97% of the opt-in limit. So **re-adding 256 was correct and I
was wrong to imply otherwise.** The resolution is `COMPUTE_DTYPE`:

| dtype at `STATS_BLOCK_M=256` | trials | complete |
|---|---|---|
| **fp16** (2 bytes) | 9 | **7** |
| bf16 | 4 | 0 |
| ieee (4 bytes) | 7 | 0 |
| tf32 (4 bytes) | 2 | 0 |

The failing witness used `COMPUTE_DTYPE: tf32` — four bytes per element — which is why its 256-row
tile needed 139,264 B. At fp16 the same tile needs half that and fits. `STATS_BLOCK_M=256` is not
infeasible; it is infeasible *at 4-byte precision*, and feasible at 2-byte.

Two corrections to what I wrote above, both mine:

1. I said repair's three attempts established "a hard fact about this kernel". They did not — they
   established a fact about one *config*. Repair moved the default to a value that works at every
   dtype, which is the right call for a witness, and K then correctly re-opened a value that works
   at the dtype the tuner actually prefers.
2. My earlier table of "same STATS knobs, both complete and fail" looked like non-determinism. It
   was my own omission: the rows differed in `COMPUTE_DTYPE`, which I had dropped from the columns.
   `tr-dccc6077` (fp16) completes and `tr-95f802bd` (ieee) OOMs on otherwise identical vectors.

**What this does not change:** the constraint-dropping bug is unaffected — 26 of 26 expansions
still discard constraints, and the 116-trial / 61%-failure measurement stands, since none of those
constraints concerned `STATS_BLOCK_M`. What it does change is the framing of this instance: it is
an example of K's *domain* widening working, not of K undoing a repair. The correct criticism of
this expansion is narrower — it re-opened a value whose infeasibility at 3 of 4 dtypes had just
been demonstrated, and paid for that with 6 OOM trials, while finding a genuine 7.5% at the fourth.

## What I would change — not applied

1. **Carry the previous constraints forward.** Require the expanded space's constraint set to be a
   superset of its predecessor's, or simply union them and re-check feasibility. The union is
   sound: a constraint that was true of the narrower space is still true of the wider one, since
   expansion only adds choices. This is the direct fix and it is small.
2. **Journal the diff.** `SPACE_EXPANDED` records `knobs` and `prev_best_ms`; adding
   `constraints_dropped` would have surfaced this on day one. Purely observational.
3. **Feed the expansion the candidate's failure history**, so a value that has already been proven
   infeasible is not re-offered. Bigger, and it edges toward the cross-candidate sharing the user
   has ruled out — though *within* one candidate it does not.

   **Weakened by the `STATS_BLOCK_M=256` outcome above.** A history-aware expansion would likely
   have declined to re-add 256, and 256 turned out to hold the best result. "Already failed" is not
   "infeasible" when the failure was at one dtype and the win is at another. So this option needs
   to be per-(value, dtype) at minimum, which makes it considerably less attractive than it looked.

**Why I stopped:** (1) changes what K produces, and K's expansion rule is already the subject of
two pending items (`at_boundary` gating in `measurement-analyst-median-on-one-sample.md`, per-knob
floors in `measurement-k-expands-downward-and-adds-zero-stages.md`). (2) is safe and I would do it
on a word. Mid-run, with the run's headline result still unverified, I am not touching the
expansion path.

`scripts/audit_expansion_outcomes.py` prints the attribution table; the constraint counts above
come from comparing consecutive `SPACE_PUBLISHED` payloads per candidate.

## Caveat

The 61%-vs-28% comparison is observational, not causal. Trials that violate a dropped constraint
tend to be at extreme knob values, and extreme values fail more often anyway — the constraint
existing is *evidence* that the region is bad, not proof that the failures were caused by crossing
it. The honest claim is that the original space encoded 116 trials' worth of avoidable exploration
and the expansion discarded it, which stands regardless of the causal question.
