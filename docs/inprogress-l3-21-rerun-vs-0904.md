# In progress: L3:21 rerun vs the 09-04 run — same latency, opposite structural class

`run-l3-21-20260905-071312`, written at 0.60h elapsed with 2 candidates tuned. **Provisional:
no `final_reeval_ms` yet, so no honest verdict.** Recorded now because the comparison against
yesterday's completed run is unusually clean and depends on data that will be overwritten as
the run proceeds.

## The two runs' best-so-far

| | 09-04 (finished) | 09-05 (in flight, 0.60h) |
|---|---|---|
| best candidate | `cand-05f1118a` | `cand-1eee8139` |
| `tuned_ms` | **20.5** | **20.5** |
| `final_reeval_ms` | 20.8 | — |
| structural class | **4 `tl.dot`, fp16** | **0 `tl.dot`, pure fp32** |
| detected precision | `fp16` | (will be `ieee_fp32`) |

Identical tuned latency from **opposite** structural classes. Yesterday's winner reached the
tensor cores; today's does not use them at all — verified from source: 0 `tl.dot`, no
`float16`/`bfloat16` casts, no `input_precision`, and a `PARAMS` dict of pure block/warp knobs
(`REDUCE_BLOCK`, `APPLY_BLOCK`, `REDUCE_WARPS`, `FINISH_WARPS`, `APPLY_WARPS`).

That is the acceptance path changing *which* structural class produces the result, not the
result itself — the L3:48 pattern
(`result-every-tensor-core-candidate-was-rejected.md`) reproducing on a second task:
**0 of 2 tensor-core candidates published, 2 of 2 scalar published.**

## The baseline comparison cuts differently for the two

This is the part worth recording, because it is easy to state wrongly.

Today's candidate is pure fp32, so its same-precision denominator is `torch_compile` (ieee):

| baseline | 09-05 ms | vs 20.5 |
|---|---|---|
| eager | 25.30 | 1.234x |
| eager_tf32 | 20.90 | 1.020x |
| **torch_compile** (its denominator) | 22.20 | **1.083x** |
| torch_compile_tf32 | 16.30 | 0.795x |

Yesterday's candidate was **fp16**, so the harness judged it against `torch_compile_tf32` —
and it **lost**:

```
honest_verdict = {"candidate_precision": "fp16",
                  "compared_against": "torch_compile_tf32",
                  "same_precision_speedup": 0.7885,
                  "beats_same_precision_baseline": false}
```

So on L3:21 the two runs are heading for the same latency but *opposite verdicts*, purely
because of which baseline the candidate's own precision selects:

- fp16 candidate at 20.5 → compared to 16.4 → **0.79x, fails**
- fp32 candidate at 20.5 → compared to 22.5 → **1.08x, passes**

Both are honest applications of the same rule. It means L3:21's real difficulty is
`torch_compile_tf32` at 16.3–16.4 ms, which **neither** run has approached, and that a
tensor-core candidate on this task is held to a 26% harder standard than a scalar one at
identical latency.

## What this does and does not say

**Does not say** the harness is wrong to compare like with like — that rule exists so a tf32
candidate cannot claim credit for beating an ieee baseline, and it is the right rule.

**Does say** something about where the headroom is: on L3:21 a scalar candidate that ties the
tf32-compile path would need 16.3 ms, and the best measured scalar result across two runs is
20.5. The gap is 26%, and the class that could plausibly close it — tensor-core kernels
reaching the same hardware `torch.compile` uses — is the class the acceptance path is rejecting
(4 above-floor rejections so far, `finding-unreachable-correctness-gate.md`).

**Provisional.** `tuned_ms` runs optimistic against `final_reeval_ms` (+1.5% here yesterday:
20.5 → 20.8), the run has 3 more families to go, and the numbers above will move. The
structural-class observation is the durable part.

---

## Superseded at 0.87h: a third candidate reached 17.1 ms

`cand-d31b0474` — the below-floor candidate that one repair fixed
(`finding-unreachable-correctness-gate.md`) — tuned to **17.10 ms**, the best latency any L3:21
run has produced. That **corrects the claim above** that "the class that could plausibly close
the gap is the class the acceptance path is rejecting": this candidate closed most of it, and it
was *published*, not rejected.

| | ms | vs 17.1 |
|---|---|---|
| eager | 25.30 | 1.480x |
| eager_tf32 | 20.90 | 1.222x |
| torch_compile | 22.20 | 1.298x |
| **torch_compile_tf32** | 16.30 | **0.953x** |

Its winning configuration is **`COMPUTE_PRECISION=fp16`**, and the tuner chose that on its own:
22 of 28 completed trials selected fp16, and the best ieee trial was **21.6 ms** — so the fp16
path is 26% faster on this very candidate. The knob did what the contract intends.

### Under the same-precision rule it still does not beat its baseline

Because it computes in fp16, its honest denominator is `torch_compile_tf32` (16.30), not
`torch_compile`. All three L3:21 bests, judged by the rule the harness actually applies:

| run | candidate | ms | precision | denominator | verdict |
|---|---|---|---|---|---|
| 09-04 | cand-05f1118a | 20.5 | fp16 | 16.40 tf32-compile | **0.789x fails** |
| 09-05 | cand-1eee8139 | 20.5 | fp32 | 22.20 ieee-compile | **1.083x passes** |
| 09-05 | cand-d31b0474 | **17.1** | fp16 | 16.30 tf32-compile | **0.953x fails** |

So the run's fastest kernel is one the harness will report as *not* beating its baseline, while a
kernel 20% slower passes. That is the rule applied correctly in all three cases — and it is
still the right rule — but it means **raw latency and the harness verdict rank these candidates
in opposite orders.**

The honest summary of L3:21 so far: 17.1 ms is a real improvement over 20.5 (a 16.6% gain,
closing the gap to the tf32-compile path from 26% to 4.7%), achieved by a *tensor-core* kernel
that the acceptance path let through. The gate finding stands on its own four above-floor
rejections; this candidate is evidence that the path is not uniformly closed to tensor-core work
on this task.

Still provisional: `tuned_ms`, 20 samples, and this run has four families and no rewrite rounds
completed yet.

---

## 1.2h: 16.9 ms, and the rewrite round is the reason

`cand-7dcdbd99` tuned to **16.90 ms** at fp16 — a second improvement, and the trajectory is now
worth stating as a whole:

| candidate | ms | class | origin |
|---|---|---|---|
| cand-080f8c60 | 25.00 | scalar fp32 | seed |
| cand-1eee8139 | 20.50 | scalar fp32 | seed |
| cand-d31b0474 | 17.10 | **fp16 tensor-core** | rewrite of `cand-1eee8139` |
| cand-7dcdbd99 | **16.90** | **fp16 tensor-core** | rewrite of `cand-1eee8139` |

**17.6% better than anything L3:21 has produced before** (the 09-04 run's best was 20.50), and
the gap to `torch_compile_tf32` has gone from **26% to 3.6%**.

The mechanism is exactly what the two-loop design is for. Both 17.10 and 16.90 are rewrites of
the *same* parent — the 20.50 ms scalar seed — under one structural hypothesis: move the
pointwise convolution onto the tensor cores. That hypothesis produced two independent kernels
(different source shas, different signatures, 7830 vs 9636 bytes, 3 vs 6 `tl.dot`) which both
beat the parent by ~18%. The tuner then found fp16 inside each of them on its own.

So on this task the structural loop supplied the idea, the parameter loop supplied the precision,
and neither alone would have got from 20.50 to 16.90.

### Both are still reported as not beating their baseline

Unchanged and worth repeating, because it is the thing a reader will misread:

| candidate | ms | precision | denominator | verdict |
|---|---|---|---|---|
| cand-1eee8139 | 20.50 | fp32 | 22.20 ieee-compile | **1.083x passes** |
| cand-d31b0474 | 17.10 | fp16 | 16.30 tf32-compile | 0.953x fails |
| **cand-7dcdbd99** | **16.90** | fp16 | 16.30 tf32-compile | **0.964x fails** |

The run's two fastest kernels both report `beats_same_precision_baseline: false`, while the
20.50 ms one passes. That ordering is the correct application of the rule and it will be what the
final report says. The honest headline for L3:21 is therefore two sentences, not one: *the search
improved the best known kernel for this task by 17.6%, and it still does not beat
`torch.compile`'s tf32 path on equal precision terms.*

Provisional as before: `tuned_ms`, 20 samples, no `final_reeval_ms` yet, and `fam-f069ef3c` has
rewrite rounds remaining.

---

## 1.4h: 15.5 ms — the first L3:21 candidate to beat its own same-precision baseline

The K re-tune of `cand-7dcdbd99` returned **15.50 ms**, and this one crosses the line the
previous two could not:

| baseline | ms | vs 15.50 |
|---|---|---|
| eager | 25.30 | 1.632x |
| eager_tf32 | 20.90 | 1.348x |
| torch_compile | 22.20 | 1.432x |
| **torch_compile_tf32** (its denominator) | 16.30 | **1.052x BEATS** |

The candidate computes in fp16, so `torch_compile_tf32` is the honest comparison, and 15.50
beats it by **5.2%**. Every earlier L3:21 best reported
`beats_same_precision_baseline: false`; this is the first that would not.

### The gain is the RE-TUNE, not the expansion — attribution matters here

Easy to credit improvement K, and wrong. The winning configuration is
`PW_BLOCK_M=64, PW_WARPS=8, PW_BLOCK_N=128, fp16` — **every value in it already existed** in the
pre-expansion space `sp-d127a691`. The two knobs K widened (`PW_BLOCK_M` +256, `PW_WARPS` +1)
do not appear in the winner, and the best trial that *does* use a new choice is **16.00 ms**
(`PW_BLOCK_M=256`), worse than 15.50.

What did change: that combination was sampled **0 times in the previous 40 trials** and **9 times
in the new 40**, reproducing 15.50–15.70 ms. So TPE reached a region its earlier budget missed,
on a space that was only incidentally wider.

This is a different effect from the one already on file.
`measurement-retune-repeatability.md` documents re-tuning an unchanged space swinging up to
**2.1%**. Here the swing is 16.90 → 15.50 = **8.3%**, four times larger, and it is not noise —
nine trials at the winning config agree to ±0.2 ms. It is genuine search coverage: 40 TPE trials
do not exhaust an 9-knob space, so a second 40 can find materially better ground without any
domain changing.

Two consequences worth stating:

1. **K's record on this run is now 0 for 2 on its own terms** — both expansions produced their
   improvement (or lack of it) from pre-existing choices. The first was flat
   (`measurement-k-expansion-vs-analyst.md`); this one improved 8.3% for reasons unrelated to the
   widening.
2. **A plain re-tune may be worth more than an expansion on this task.** That is a testable claim
   and the cheaper intervention, but it is n=1 and must not be turned into a config change on
   this evidence.

### Still provisional, and the caveat is now load-bearing

`tuned_ms`, 20 samples per trial, no `final_reeval_ms`. The re-eval gap on this task yesterday
was **+1.5%** (20.5 → 20.8); if that repeats, 15.50 → ~15.73, which still beats 16.30. But
L3:48's best went the other way (−6.2%), so the direction is not reliable, and until the final
re-eval runs the "beats its baseline" claim is **provisional, not established.**
