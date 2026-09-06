# L3:21 result — 7.21 ms, 2.09× the strongest baseline

`run-l3-21-20260905-195615`, the gpt-5.6-sol arm. Best result of the project so far, and the
first run to reach its wall-clock budget rather than freezing on family budgets.

## The number

```
final_reeval_ms        7.21   (100 samples, fresh process, PASS)
tuned_ms               6.90   (+4.5% optimistic, inside the known 1.5-6.7% band)
```

Quote **2.09×**, not 2.27×. The run records `torch_compile_tf32 = 16.4`; I measured the same
baseline independently at **15.08**, and its fast regime at 14.63 (it is bimodal — see
`docs/measurement-baseline-is-bimodal.md`). The harder comparison is the honest one:

```
vs run-recorded 16.4   ->  2.27x   (what the summary claims)
vs my 15.08            ->  2.09x   <- report this
vs fast regime 14.63   ->  2.03x
```

The candidate computes in fp16, so `honest_verdict` compares it against `torch_compile_tf32`
rather than the slower ieee `torch_compile` (3.09×) or eager (3.51×). Comparing an fp16 kernel
to an ieee baseline would be a strawman.

Correctness needed no relaxation: `fp64_rescued_trials = 0`, 3/3 on the absolute dual-witness
gate, `excessive_speedup_flag = false`, zero eager fallbacks, zero register spills.

## What produced it

`cand-e6e7c9c5`, in `fam-a2688942`: per-channel sum and sum-of-squares emission fused **into the
producer kernels**, eliminating three standalone `torch.var_mean` reduction passes. The BN
statistics become a byproduct of work already touching the data. 7 Triton kernels, 327 lines.

## The finding that matters for the method

```
fam-a2688942   6.90   [20.4, 15.1, 10.0, 6.9]   <- WINNER; worst seed of the four
fam-4286a3be   8.13   [19.4, 9.42, 8.13]
fam-a4a8353c  10.70   [19.4, 11.0, 10.7]
fam-fd92a2d8  15.20   [22.8, 15.2]
```

**The winning family had the worst seed**, and the runner-up was flat across its first two
tunings (19.4 → 19.4). Either would have been killed first by an early-stopping or greedy-pruning
policy. This run is therefore direct evidence for keeping families alive on a bounded budget
rather than concentrating it on the early leader — the decision taken deliberately earlier in
the project.

Two structural gaps between siblings from a single rewrite round: 6.92 vs 9.45 (34%) and
15.3 vs 22.1 (31%). Two hypotheses, both tuned, the structurally better one wins by a wide
margin — the paper's central claim in miniature.

Progression: **19.4 → 15.6 → 14.7 → 11.1 → 11.0 → 9.78 → 9.42 → 9.14 → 8.13 → 6.92 → 6.90**,
final re-eval 7.21. −63% from the best seed.

## Stop condition

`budget_exhausted` on the 12 h wall clock, at a measured 12.78 h — the budget is checked only at
the top of each outer round, so a run can overshoot by up to one round
(`scripts/audit_wall_clock_overshoot.py`). `stop_kind = "converged"` has still **never** fired in
17 runs.

## Reproduce

```
kernel-opt --config configs/experiments_l3.yaml run --task level3:21
kernel-opt --config configs/experiments_l3.yaml report --run runs/run-l3-21-20260905-195615
```

`report.md` here was regenerated purely from `events.jsonl` after the run ended, which is the
check that the event trace is complete: 1657 trials, full lineage, every verdict.

## Cross-check against the GLM arm's independent baseline

The glm-5.3 arm re-measured the same four baselines on the same GPU a few minutes after this run
ended, which is a free replication of the measurement floor:

```
kind                   gpt     glm     diff
eager                 25.3    25.3   +0.00%
eager_tf32            20.9    20.9   +0.00%
torch_compile         22.3    22.2   -0.45%
torch_compile_tf32    16.4    16.2   -1.22%
```

Max divergence 1.2%, so a model-to-model comparison of candidate latency is not confounded by
baseline drift.

Note what this does *not* fix: my own measurement of `torch_compile_tf32` was 15.08, **below**
both arms' recorded values. The baseline is bimodal, and both arms happened to record the slow
regime — so both arms' reported speedups are ~8% optimistic **in the same direction**. That
cancels when comparing gpt against glm; it does not cancel for an absolute claim, which is why
2.09× rather than 2.27× is the number above.
