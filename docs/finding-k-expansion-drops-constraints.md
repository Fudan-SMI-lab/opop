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

Two trials have already sampled it, and one failed with
`out of resource: shared memory, Required: 147456, Hardware limit: 101376` — a *worse* violation
than the one repair spent three attempts fixing, because the widened `OUTPUT_*` knobs add to it.

This is not the constraint-dropping bug by itself — no constraint forbade `STATS_BLOCK_M=256`,
since the original space expressed the limit through its *default* rather than a constraint. But
it is the same root cause: **the expansion has no memory of what the candidate's history
established.** Repair learned a hard fact about this kernel and wrote it into a default; K, calling
a fresh parameterizer with no knowledge of that episode, re-opened it.

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
