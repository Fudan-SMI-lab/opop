# In progress: L3:43 reached 11.0 ms from a rewrite — and this one is not an outlier

`run-l3-43-20260905-091705`, `cand-13efdcd8`, 1.97h of 12h. **Provisional: `tuned_ms`, no
`final_reeval_ms` yet.** Recorded now because the *shape* of the evidence is materially
different from the 14.2 ms case in `inprogress-l3-43-14ms-outlier.md`, and that difference is
the point.

## The number

| | ms | vs 18.40 bar |
|---|---|---|
| `cand-13efdcd8` (rewrite) | **11.0** | **1.673×** |
| `cand-cb7be6b4` (its parent, seed) | 14.2 | 1.296× |
| `torch_compile_tf32` (same-precision denominator) | 18.40 | 1.0 |
| best L3:43 ever published | 19.1 re-eval | 0.963× |

## Why this is stronger evidence than the 14.2

The 14.2 was one trial with a 2.1× gap to second place and nothing between. This is the
opposite distribution:

```
11.0  13.0  13.2  13.4  13.6  13.8  14.2 | 17.1 17.4 17.4 17.5 18.0 18.6 ...
```

**Seven configurations under 15 ms**, gap to second place only **1.18×**, from 27 completed
trials of 40. The sub-15 region is densely populated rather than a single spike:

| ms | ATTN_M | ATTN_N | QKV_M_CTAS | regs | spills | std |
|---|---|---|---|---|---|---|
| **11.0** | 128 | 16 | 4 | 248 | **0** | 0.205 |
| 13.0 | 128 | 32 | 4 | 255 | 12 | 0.146 |
| 13.2 | 128 | 16 | 2 | 248 | 0 | 0.242 |
| 13.4 | 128 | 128 | 2 | 248 | 0 | 0.363 |
| 13.6 | 128 | 32 | 4 | 162 | 0 | 0.385 |
| 13.8 | 128 | 16 | 4 | 154 | 0 | 0.399 |
| 14.2 | 128 | 16 | 2 | 244 | 0 | 0.659 |

Six *independent* parameter vectors land within 25% of the best. The "measurement skipped work"
hypothesis, which needed real ruling out for the 14.2, does not get off the ground here — an
artifact would not reproduce across `ATTN_BLOCK_N` ∈ {16, 32, 128}, `QKV_M_CTAS` ∈ {2, 4}, and
register counts from 154 to 255.

**It also spills nothing.** The 14.2 case's one uncomfortable detail was 12 spills at 255
registers; this winner is at 248 registers with **0 spills** and 81,920 B shared (80.8% of the
101,376 B opt-in limit). The register pressure the analyst named as the blocker was in fact
relieved.

## In-process control

From `jobs/cand-13efdcd8-tr-df92ec3e-eval-bc237fb0.out.json`:

```
correct                   true     trials_passed 3/3
correctness_mode          "dual_witness_relaxed"
excessive_speedup         false
latency_ms                {mean 11.0, std 0.205, min 10.4, max 11.3, n 20}
ref_latency_ms            {mean 40.6, std 0.809, min 39.5, max 41.9, n 10}
speedup_vs_ref_in_worker  3.682
```

The worker timed the reference at **40.6 ms in the same process**, against the independently
measured `eager` baseline of 41.6 — agreement to 2.4%, so the timing harness was working during
this exact measurement. `std 0.205` on a 11.0 mean is the tightest of any fast trial in the run.

## Physical plausibility

L3:43 is 412.3 GFLOP (B=128, T=512, C=768, H=8, head_dim=96):

| | TFLOP/s | % of fp16 tensor-core peak |
|---|---|---|
| **11.0 ms** | **37.5** | **15.5%** |
| 14.2 ms | 29.0 | 12.0% |
| `torch_compile_tf32` 18.4 ms | 22.4 | 9.3% |
| reference 40.6 ms | 10.2 | 4.2% |

15.5% of peak on an attention kernel with a causal mask is unremarkable — no physical limit is
approached, and it is 1.67× a tf32 `torch.compile` path, not 10×. Nothing here needs a
FLOP-ratio defence.

## Where it came from — the loop worked as designed

This is the first L3:43 result where the full chain is traceable end to end:

1. `cand-cb7be6b4` (seed) was **rejected below the noise floor** at its default tf32 witness,
   correctly (0.963932 vs floor 0.976682).
2. **Repair** changed `COMPUTE_DTYPE` default tf32 → ieee; the parameterizer preserved the body
   (16 `tl.dot` intact).
3. Tuning found fp16 + `ATTN_BLOCK_M=128` → 14.2 ms.
4. The **analyst** identified the row tile as the only credible blocked knob, `registers` as the
   blocker, with shared memory at 64.6% and 256 of 1024 threads used — and proposed distributing
   a larger query tile across more warps rather than shrinking the accumulator.
5. The **rewriter** produced `cand-13efdcd8`: persistent column-local QKV GEMM CTAs that
   serialize multiple 128-row tiles via a new `QKV_M_CTAS` knob, "avoiding a register-prohibitive
   256×128 live accumulator".
6. Tuning found `QKV_M_CTAS=4` → **11.0 ms, 0 spills**.

The analyst's diagnosis was register pressure; the rewrite targeted register pressure; the
result has zero spills and is 22.5% faster. That is the paper's central claim — tuning feedback
steering structural search — with every step on disk.

Note `QKV_M_CTAS` is a **new knob invented by the rewriter** — machine-checked against the
parent's published spaces: the child's domain list is the parent's 15 knobs plus exactly this
one, with none removed. It is also the knob that decides the result:

| `QKV_M_CTAS` | trials | fails | best | median |
|---|---|---|---|---|
| 1 | 5 | 1 | 19.2 | 20.5 |
| 2 | 22 | 8 | 13.2 | 19.2 |
| **4** | 8 | 1 | **11.0** | **13.8** |
| 8 | 5 | 3 | 20.7 | 33.5 |

An interior optimum at 4 — best *and* best-median, with a clear penalty on both sides. That is a
knob worth having: the serialization factor trades register pressure against parallelism, and
neither extreme wins. The rewriter proposed it, the parameterizer exposed it, the tuner found its
value; none of the three could have produced this alone.

## What would still change the verdict

1. **`final_reeval_ms`** — 100 samples, fresh process, 5/5 correctness. The re-eval gap has run
   +1.9% (L3:21) and −6.2% (L3:48), so direction is not predictable. At +6% this is 11.7 ms and
   still 1.57× the bar; the verdict is robust to the gap in a way the 14.2 case was not.
2. **The `beats_same_precision_baseline` flag**, which compares against `torch_compile_tf32`
   rather than eager — already the honest denominator, and 11.0 clears it by a wide margin.
3. Whether the second rewrite (`cand-919059a0`, the 16-warp 256-row CTA) does better still.

## What this does NOT resolve

- The 14.2's own n=1 status. It remains a single trial; it is simply no longer load-bearing,
  because its child is faster and densely reproduced.
- Whether `ATTN_BLOCK_M=256` helps. Every sub-15 config here uses 128, and `QKV_M_CTAS`
  serializes rather than enlarging the tile — so this rewrite *sidestepped* the 256 question
  rather than answering it. The controlled 128-vs-256 comparison the record lacks
  (`result-row-tile-is-monotone-and-k-supplies-it.md`) is still missing.
