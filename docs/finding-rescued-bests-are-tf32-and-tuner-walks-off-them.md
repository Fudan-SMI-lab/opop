# Finding: every rescued space-best is tf32, and the tuner walks off them onto clean fp16

All 12 space-bests on `run-l3-21-20260905-195615`, with the dtype their winning trial used and
whether that trial needed the fp64 relative arm:

```
sp-ad4f50cd  cand-52aaee73  19.4ms  (no dtype knob)  rescued=0
sp-488b1916  cand-6f96a754  20.4ms  (no dtype knob)  rescued=0
sp-fb745e6e  cand-e2cd07de  19.4ms  (no dtype knob)  rescued=0
sp-607e4351  cand-e2cd07de  19.4ms  (no dtype knob)  rescued=0
sp-154b1a6b  cand-61759130  22.8ms  tf32   rescued=3
sp-68475458  cand-61759130  22.8ms  tf32   rescued=3
sp-9ba61a54  cand-80bf3097  15.6ms  fp16   rescued=0
sp-274540fa  cand-80bf3097  14.7ms  fp16   rescued=0
sp-8784d130  cand-f66890d0  11.1ms  fp16   rescued=0
sp-d36a6e38  cand-f66890d0  11.0ms  fp16   rescued=0
sp-72cb5ea2  cand-8510db1b  15.9ms  tf32   rescued=3
sp-6c0aca30  cand-8510db1b  14.4ms  fp16   rescued=0
```

The split is total: **every tf32 best needed the rescue, every fp16 best passed unaided.** With
`relaxed_elem_tol = 0.01` on a task whose ieee-vs-tf32 floor is 0.9554, that is arithmetic rather
than luck — the measured RMSE ratios are tf32 1.047 and fp16 1.000 against the tf32 reference, and
0.99 of elements within 1% is the line between them.

## The interesting part: `cand-8510db1b` walked off its own rescue

That candidate's two spaces are the same kernel, one expansion apart:

```
sp-72cb5ea2   15.9 ms   tf32   rescued=3    <- admitted only by the relative arm
sp-6c0aca30   14.4 ms   fp16   rescued=0    <- faster AND passes the absolute gate
```

The expansion added `GEMM_BLOCK_N=[128,256]` and the tuner's new optimum uses `128` **with fp16
instead of tf32**. So the search did not merely improve the latency by 9.4%; it moved the
candidate from a configuration that needed relaxed correctness onto one that does not.

This is the strongest argument so far for keeping the gate on, and it is not the argument I
expected. The gate's value here is not that it admitted a winner — it did not. It is that it kept
the candidate **alive through the tf32 configuration long enough for the tuner to find the fp16
one**. Without the rescue, `cand-8510db1b`'s first space fails, the candidate is discarded, and
the 14.4 ms configuration is never sampled. The rescue bought a search path, not a result.

Whether that generalises is untested: n=1, and it needs a candidate whose dtype knob spans both
classes with the fp16 corner faster, which is task-dependent (it requires outputs below the fp16
ceiling — `opop-v2-minimal-witness-is-fp16-corner` documents the tasks where it does not hold).

## Both of that candidate's numbers lose or tie on measurement

Timed fresh at 100 samples with 20 warmup:

```
                    tuned_ms   measured   baseline (same run)   verdict
sp-72cb5ea2 tf32      15.9       16.36          15.10           0.923x  LOSES
sp-6c0aca30 fp16      14.4       15.00          14.96           0.997x  TIE
```

Against the run's recorded 16.4 ms baseline both look like wins (1.03x and 1.14x). Measured
alongside their own fresh baseline, one loses by 8% and the other is a statistical tie — the
baseline's `std` of 0.64 makes 0.997x indistinguishable from parity.

## Correction: the harness baseline is NOT 8% too slow — the compiled baseline is bimodal

I first wrote here that the harness's `torch_compile_tf32` figure of 16.4 is "the outlier,
consistently, by about 8%", having measured 15.08 / 15.10 / 14.96 three times, and proposed too
little warmup as the cause (KernelBench uses `num_warmup=5`; I used 20). **Both the claim and the
proposed cause are wrong.** Sweeping warmup on one compiled instance
(`scripts/audit_baseline_warmup_sensitivity.py`):

```
 warmup    mean  median     min     max    std
      5   14.63   14.74   13.80   15.02   0.30    <- stable FAST mode
     10   21.55   17.66   13.50   35.12   6.21    <- transition
     20   27.03   29.53   14.10   38.50   7.78    <- transition
     40   16.75   16.69   16.30   19.28   0.31    <- stable SLOW mode
     80   16.72   16.68   15.41   17.11   0.22    <- stays slow
fresh instance, warmup=5:  16.83   16.74  16.13  18.79  0.39   <- slow mode

eager_tf32 for contrast:  warmup 5 -> 22.22 (std 3.17),  warmup 20 -> 21.30 (std 0.15)
```

More warmup makes it **slower**, not faster, which is the opposite of a warmup deficit. The
compiled model has two stable regimes, ~14.7 and ~16.7, and it migrates from the fast one to the
slow one as it keeps running — with a violent transition in between (std 6–8, max 35–38 ms) that
looks like recompilation, a cudagraph re-record, or an autotune pass landing it in a different
plan. Eager shows nothing of the kind.

So the harness's 16.4 is a real number, not an artefact: it is the slow regime. My three earlier
measurements happened to catch the fast regime. **What I can claim is only that the compiled tf32
baseline is bimodal at the ~12% level (14.7 vs 16.7), which makes any single-number comparison
against it unreliable by that much.** What I cannot claim, and now retract, is that the harness
measures it 8% too slow.

The consequence for reporting is different from what I wrote, and worse in one way and better in
another:

- **Worse:** a candidate within ~12% of this baseline cannot be called a winner or a loser from
  one measurement of either side. `cand-8510db1b`'s two configurations (16.36 and 15.00) both sit
  inside that band, so neither verdict above is safe.
- **Better:** `cand-f66890d0` at 11.40 ms beats even the fast regime (14.63) by 1.28×, so the
  run's headline result is unaffected by which regime the baseline is in.

The right fix is to report the candidate against the baseline's **fast** regime, since that is the
strongest thing torch.compile does on this task and the honest bar to clear. On that basis:
`cand-f66890d0` 1.28× (a win), `cand-8510db1b` 0.98× and 0.89× (not wins).

Reproduce the dtype/rescue table by joining `STATS_DONE.best.trial_id` against
`jobs/*.out.json`'s `fp64_rescued_trials`; timings with
`python scripts/verify_candidate_timing.py <trial.py> <reference.py> 100`.
