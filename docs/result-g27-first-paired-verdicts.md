# G27's first paired evidence: the two arms produced OPPOSITE conversion verdicts

Both arms have now recorded their first `FAMILY_ROUND_RECORDED` on the same task (L3:43, same config
apart from the two S2 switches), so `conversion_verdict` can be compared rather than merely observed.

| arm | verdict | latency gain | resources that improved |
|---|---|---|---|
| box 2 (treatment, `vector`) | **`improved`** | **+10.93%** | `candidate_aten_bytes`, `candidate_aten_ops` |
| box 1 (control, `label`) | **`no_conversion`** | **+1.67%** | `candidate_aten_ops` |

1 of 1 rounds on each arm carries both `conversion` and `resource_deltas` — G27's wiring works in
production on both arms, which until today had zero production evidence of any kind.

## The `no_conversion` verdict is the one worth reading

Box 1's own note, verbatim from the event:

> `candidate_aten_ops` improved but latency moved only 1.67% (below the 2.0% floor), so those resources
> were NOT the limit for this structure. This is evidence about where the limit is not…

That is the whole point of the conversion idea stated by the harness itself: a resource that improved
while latency did not is a *measurement that the resource was not binding*. Without this verdict the
round reads as "rewrite produced a 1.67% gain, roughly nothing"; with it, the round produced a
falsification — `candidate_aten_ops` is not where L3:43's limit lives on that structure.

Both arms agree on the direction for `candidate_aten_ops` (it improved on both). They differ in whether
that came with a latency gain: box 2's rewrite also cut `candidate_aten_bytes` by 36.2% and got 10.93%;
box 1's cut ops alone and got 1.67%.

## What this does and does not support

**Does not** support "the vector helps". n = 1 round per arm, and the arms' rewrites are different code
attacking different hypotheses — box 2's winner was H1+H3, box 1's was H1+H2+H3. A single round cannot
separate the switch from the draw, and `wall-clock-is-always-the-binding-budget` records that 5 finished
runs produced only 9 rewrite rounds in total, so per-round mechanisms accumulate across runs, not within
one.

**Does** establish three things that were previously unevidenced:

1. `conversion_verdict` reaches production and fires on both arms (G27, previously 0 for 9 rounds
   because the corpus predated the fix).
2. Both of its interesting verdicts occur in real data — `improved` and `no_conversion` — so the
   distinction is not decorative.
3. The floor is load-bearing: 1.67% against `min_improvement_pct: 2.0` is what turns a "small gain" into
   a stated non-result. A run without the floor would have called it an improvement.

## The registers are pinned at the cap on box 1, and that is a task finding

Box 1's family is worth recording separately. **Both** rewrites declared `n_regs`, `n_spills` and
`shared_bytes` would fall; **all six predictions missed.** Every candidate in the family — the two
rewrites and the four seeds — sits at exactly:

```
n_regs = 255      n_spills = 6      shared_bytes = 49152      occupancy = 0.1667 (limiter: registers)
```

255 is the hardware register cap and 49152 the static shared-memory limit. Verified as a real reading
rather than a flat probe: each candidate saw **29–57 distinct** `(n_regs, n_spills, shared_bytes)`
triples across its trials, so the profiler discriminates fine — the *winning* configurations all land on
the cap. Consistent with `opop-triton-caps-regs-instead-of-spilling`: Triton pins registers rather than
spilling, so `occupancy` is the dimension that moves and `n_regs` is the one that cannot.

This is why the rewriter kept predicting register relief and kept being wrong: the structures it wrote
are all register-bound at the cap, and no amount of restructuring at this level moves a number the
compiler is clamping.

## Caveat carried from the pooled-ledger defect

Both arms' entries are POOLED (`candidate_id: null`), so the hit/miss counts printed by
`check_wrapup.py` — 7/6/3 on box 2, 6/10/0 on box 1 — mix two candidates. The per-candidate figures
above and in `result-s2d-pooled-ledger-defect.md` are re-derived offline with
`scripts/rederive_per_candidate_ledger.py`. The conversion verdicts and latency gains in this document
are NOT affected: those are per-round quantities and the round has one incumbent.
