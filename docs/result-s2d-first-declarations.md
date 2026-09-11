# S2d's first production declarations: 16 predictions, made before any measurement

**Status: the declaration half (S2d(a)) is confirmed working on live data, and the winning rewrite's
declarations are now MEASURED — 4 hits / 1 miss / 3 vacuous, every falsifiable direction correct (see
the last two sections). S2d(c), the ledger reaching the next round's prompt, is still pending** — it
needs a round 2, and the round-0 entry itself is affected by the pooling defect recorded in
`docs/result-s2d-pooled-ledger-defect.md`.

## Why the ordering is the whole point

`REWRITE_PRODUCED.payload.expectations` is journalled **at declaration time**, before any measurement
of the rewrite exists (`orchestrator.py:2337`). That ordering is what makes the later reconciliation a
*prediction check* rather than a description: a declaration recorded after the fact could have been
shaped by the outcome, and nothing in the log would show it.

Box 2 (`run-l3-43-20260911-052630`, treatment arm), both rewrites at 10:40:31, **8 declarations
each** across all 8 dimensions:

### `cand-2d8eaf9a`, hypothesis H1+H3 — fuse the two cuBLAS projections into Triton

| dimension | expect | the reason given |
|---|---|---|
| `candidate_aten_bytes` | **down** | the `x.to(dt)` (~201 MB) and `out.to(float32)` (~201 MB) round trips no longer exist |
| `candidate_aten_ops` | **down** | two `F.linear` calls and two big dtype casts leave the aten path |
| `shared_bytes` | **up** | the new GEMM stages a 128×32 fp32 A-tile plus a 32×128 bf16 W-tile per stage (2 stages ≈ 48 KB) against the attention kernel's 17.4 KB |
| `peak_alloc_bytes` | **down** | the 67 MB bf16 `x` copy and the 67 MB bf16 `c_proj` intermediate are never allocated |
| `threads_launched` | **up** | two large Triton GEMM grids (~6k and ~2k programs × 256 threads) where cuBLAS launches were not Triton-thread-counted |
| `n_regs` | *unknown* | two new GEMM kernels join the unchanged 155-reg attention kernel; which dominates the aggregate is not predictable |
| `n_spills` | *unknown* | newly compiled GEMM code paths, no prior measurement to reason from |
| `occupancy` | *unknown* | GEMM blocks are smem-heavy at 8 warps while the attention kernel is unchanged; direction depends on how the per-kernel figures aggregate |

### `cand-3760b4d7`, hypothesis H2 — change register/mask shape at the same tile family

| dimension | expect | the reason given |
|---|---|---|
| `n_regs` | **down** | Q lives in its loaded bf16 (2 elems/register) instead of a scaled fp32 tile, per-iteration converts disappear, and `EVEN_T` compiles out the bounds-mask registers |
| `n_spills` | **unchanged** | parent best ran 0 spills; the claim is that the BM=64/128 4-warp region becomes spill-*reachable*, not that zero drops below zero |
| `occupancy` | **unchanged** | at the 2-warp default the smem limit still binds at 5 blocks/SM (~21%); it rises only if the tuner takes the now-reachable 4-warp knob |
| `shared_bytes` | **unchanged** | K/V staging per (BM, BN, stages) tile is untouched |
| `candidate_aten_bytes` / `_ops` / `peak_alloc_bytes` / `threads_launched` | **unchanged** | host-side op sequence intentionally identical to the parent |

## What is notable, beyond the count

**The declarations quote the S2 vector's own readings.** `155-reg`, `17.4 KB`, `5 blocks/SM (~21%)`,
`0 spills` — these are the per-dimension numbers the treatment arm's `## Resource state, one line per
dimension` section carries. A label-only arm has none of them to cite. So this is the treatment being
*used*, not merely delivered (`docs/result-s2-switch-reaches-the-agent.md` established delivery).

**3 of 16 are explicit `unknown`, with a reason for declining.** These become `vacuous` in the ledger
— the agent *declined to predict*, which is not a wrong call. That distinction is the one
`check_wrapup.py`'s `NO JUDGEMENT` verdict exists to preserve: a ledger of nothing but vacuous rows
ran and judged nothing, and must not read as success.

**`unchanged` is a real prediction, not an abstention.** H2 declares it 7 times *with reasons* — and
one of them is unusually careful: `n_spills` unchanged "because the claim is that the BM=64/128 4-warp
region becomes spill-reachable, not that the default config's zero spills drop below zero". That
distinguishes a claim about the *default configuration's* profile from a claim about the *space*, which
is exactly the confusion that would make a hit or miss uninterpretable.

## The expected ledger shape, stated before it exists

16 declarations, **13 falsifiable and 3 vacuous**. So the reconciliation should produce ~13 judgeable
per-dimension rows, not 0 hits / 0 misses. Recording that here, ahead of the measurement, so the
comparison cannot be adjusted to whatever turns up:

- `n_declared` = 16 (8 per rewrite), and **not 0** — that would be the "empty entries" failure where
  the ledger runs and judges nothing.
- `vacuous` ≥ 3 from the explicit `unknown`s, plus any dimension the profile does not measure.
- `hits + misses` should be > 0. If it is 0 with 16 declarations, `check_wrapup.py` reports
  **NO JUDGEMENT**, which is the defect case.

One prediction is already checkable against the parent profile on disk: H2's `shared_bytes: unchanged`
is against a parent best of 17408 bytes, and H1+H3's `shared_bytes: up` against the same 17408.
Directly opposed claims about the same dimension from the same round — which is what makes this round
a usable test rather than two agreeing guesses.

## What the measurement then showed, checked against the shape above

The winning rewrite `cand-2d8eaf9a` (H1+H3) landed at **2.8969 ms against the parent's 3.2128 ms, a
9.83% gain**, and it is the run's best. Its declarations against its own measured profile:

| dimension | declared | measured | |
|---|---|---|---|
| `candidate_aten_bytes` | down | 2 544 918 528 → 1 624 786 944 | **hit** |
| `candidate_aten_ops` | down | 15 → 10 | **hit** |
| `shared_bytes` | up | 17 408 → 32 768 | **hit** |
| `threads_launched` | up | 1 048 576 → 4 194 304 | **hit** |
| `peak_alloc_bytes` | down | 2 207 064 064 → 2 107 581 952 | miss (−4.5%, under the 5% floor) |
| `n_regs` / `n_spills` / `occupancy` | *unknown* | 155→155, 0→0, 0.2083→0.2083 | vacuous, as declared |

**4 hits, 1 miss, 3 vacuous.** Every declaration whose *direction* was falsifiable was right; the one
miss is a materiality artefact, not a wrong direction — `peak_alloc_bytes` did fall, by 99 MB, which
is 4.5% of its own value and therefore `flat` under `_MATERIAL_RESOURCE_DELTA = 0.05`. Worth stating
precisely rather than rounding to "5 of 5": the agent's *sign* was right five times out of five, and
its claim counted as a miss once because the harness requires a 5% move before it will call a
dimension moved at all.

The three `unknown` declarations were all correct to decline — every one of those dimensions came back
flat, so there was nothing there to predict.

### The prediction in "the expected ledger shape" was right about the count and wrong about the unit

`n_declared = 16` and `vacuous ≥ 3` held. `hits + misses > 0` held. But writing the shape down as a
single per-round row was itself the error the fix corrected: **16 declarations is two candidates'
worth**, and the round has no single hits/misses figure, because H1+H3's 4/1 and H2's 2/5 are about
different code. Pooling them gives 6/6 — see `docs/result-s2d-pooled-ledger-defect.md`. The count
recorded in advance was the right thing to fix in advance; the granularity was not, and the
measurement is what exposed it.
