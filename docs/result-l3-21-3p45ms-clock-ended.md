# L3:21 result — 3.45 ms final, 4.51x eager / 3.30x same-precision, ended by its clock

Run: `run-l3-21-20260908-232211`, box 1 (RTX 4090), commit `e2d3d32`, agent model
`zhipuai/glm-5.3`. Task: KernelBench level3/21_EfficientNetMBConv, train-mode BN.
All numbers below are from the run's own `events.jsonl` (`RUN_FINISHED` summary seq 1186)
and `report/report.md`; nothing is quoted from notifications.

## Headline

| | ms | source |
|---|---|---|
| **final_reeval median** | **3.4545** | independent re-eval, new process |
| final_reeval mean | 3.5 | same job |
| tuned_ms (winner's best trial) | 3.6050 | tuning loop — reeval was 4.2% FASTER this time |
| eager | 15.6 | 100 samples |
| eager_tf32 | 13.9 | |
| torch_compile | 14.2 | |
| **torch_compile_tf32** | **11.4** | strongest baseline |
| previous incumbent (this task) | 6.92 tuned | run of 09-04, `final` never produced |

Speedups (median-based): **4.51x eager, 4.10x torch_compile, 3.30x torch_compile_tf32.**
The report's honest verdict is stated correctly: the winner computes in fp16, so the
same-precision comparison is against `torch_compile_tf32` and reads **3.30x** —
`beats_same_precision_baseline: true`, `excessive_speedup_flag: false`. The re-eval landing
*faster* than the tuning number (−4.2%) is the opposite sign of the usual +1.5–6.7% gap;
both directions exist, which is exactly why only `final_reeval_ms` is quotable.

Winner: `cand-8a5dcef3` — a SEED, not a rewrite. Fully-fused 3-stage Triton pipeline
(expand GEMM + BN-stats epilogues, depthwise 5x5/s2 with folded BN+ReLU6, project GEMM),
`COMPUTE_DTYPE=fp16, STORE_DTYPE=fp16`, fp32 accumulators. Its two rewrites tuned to
3.6741 and 5.8389 — neither beat their parent.

## The seven checks

1. **final_reeval_ms** = 3.4545 median (3.5 mean), `final_reeval_ok: true`. Quoted above.
2. **Ended by its own clock**: `budget_exhausted` at 12.28 h of 12 h (2.4% overrun — the
   overrun is one in-flight tuning loop draining, same mechanism as L3:48's 31% but far
   smaller because no rewrite round straddled the deadline). This is only the **second of
   20 runs** ever ended by wall clock. All 4 families still `active`, each having used
   **1 of 5** rewrite rounds: the budget raise (3→5) did not bind; wall clock is the
   binding constraint on this task. 10 candidates × 80 trials = 800 trials.
3. **fp64 rescue on the winner**: best trial has `fp64_rescued_trials: 3` — **all 3 of 3**
   quick-correctness trials passed via the fp64-relative arm (the candidate is *closer to
   the fp64 golden than the reference is*, ratio < 1.0 territory). Run-wide, 465 of 800
   trials used a rescue somewhere; the gate's fp64 arm is load-bearing on this task, and
   without it (pre-F-series behaviour) this entire fp16 result would have been rejected.
4. **Per-precision table** (final, 800 trials): fp16 343✓/49✗, tf32 181✓/35✗,
   ieee 64✓/28✗, **bf16 0✓/100✗** (92 mismatch + 8 shmem). D2's bf16 pattern held to the
   end across all 10 candidates and both knob names — 0-for-100 is now the strongest
   dead-value evidence in the project. Dead-value waste (D1): 70/800 = 8.8% of trials.
5. **F5 prescreen**: net **−30.5 min** final (99 guard avoidances × 18.6 s = 30.7 min saved
   vs 61+ min spent screening 20 spaces). Worse than the mid-run −9.6 min. D5 stands, and
   its fix surface is confirmed as items 1+2 only (cache does hit across expansions —
   verified from job payloads, mean ~10% hits, the rest are genuinely new configs).
6. **Loop D**: zero novelty calls. Not the old `max_families_total` gate (4 < 6 here) —
   the run spent its whole clock in Loops B/C. On a 12 h budget with 10 candidates' tuning
   at ~18.6 s/trial plus 48 agent calls, D never gets scheduled. Structural: D is last in
   priority and the clock binds first.
7. **Backend counts** (merged into D3's correction): 6 rewrites, all declared+detected
   triton, 0 `BACKEND_DECLARATION_MISMATCH`, 0 CUDA. This run DID have the switch available
   (reflog-verified — the earlier "control run" claim was wrong and is corrected in the
   deferred-findings doc).

## Quality signals

- **Winner separation is real**: 2 of 588 completed trials within 1 combined SEM
  (0.0025 ms = 0.07% of best); per-trial CV median 0.6%. Unlike L3:48's earlier run, the
  timing here can rank.
- **Agent reliability**: 48 calls, 1 transport ReadTimeout — and the artifact rescue
  recovered the finished rewrite from the sandbox (`AGENT_ARTIFACT_RESCUE, rescued: true`),
  so nothing was lost. Both repair calls produced accepted diagnoses (the cudnn legacy-flag
  crash; the cross-call stale BN-stat buffers).
- **Rewrites improved two families substantially** — fam-4687b53c 6.93→4.05 (−41.5%),
  fam-4b0cfad6 5.27→4.42 (−16.2%) — but not the winning family (3.605→3.605): the seed
  was already the best structure the run found. The worst seed's family (10.20) never got
  a rewrite round before the clock ran out; round scheduling is best-family-first.

## What this run adds to the deferred batch

- **D1/D2 upgraded**: bf16 0-for-100, 8.8% budget waste. The FAIL→PRUNED escalation now
  has a second run's worth of evidence, same shape.
- **D5 upgraded**: −30.5 min. The screen's *correctness* half is still perfect (all 62
  shmem trials refused pre-launch, zero `out of resource` raises).
- **New, minor**: report's bottleneck section prints each candidate's verdict twice
  (space + expanded space) — cosmetic, dedupe at report time. Occupancy on the winner is
  8% (shared_memory-limited) with 19.6% fp16-tensor-core utilization; the analyst's
  resource_limited verdicts were consistent across all 20 reports.

## Comparison discipline

The 6.92 ms incumbent was `tuned_ms` from a run that stopped early (2.05 h of 12) and never
produced a final re-eval, so "3.4545 vs 6.92" mixes measurement kinds; the honest statement
is: this run's own end-to-end chain produced **3.4545 ms final** on the same task, same box,
same baselines, and its tuned-vs-final gap was −4.2%.
