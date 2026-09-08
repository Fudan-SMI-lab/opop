# What is broken now, and what to fix next

Written 2026-09-08 from `run-l3-43-20260908-053708` (11.0 h in, 1004 trials, still running) plus
`run-l1-42-20260908-023039`. The L3 chain loop was stopped after L3:43 so that these land before
L3:21 and L3:48 spend 24 h reproducing the same waste. Every number below is from `events.jsonl` or
from a probe run on the box; where something is inferred rather than measured, it says so.

Ranked by measured cost, not by how interesting the bug is.

---

## P1. 18% of every trial budget is spent on configs that cannot possibly run

**Measured.** 181 of 1004 trials (18.0%) failed, and **180 of them for one reason**: Triton
`out of resource: shared memory`. Cost: **0.93 h of 11.0 h wall clock (8.5%)**, enough for ~204 more
real measurements at the 16.4 s median cost of a passing trial. It is not one bad candidate — the
failure rate is 21–33% spread across all 16 candidates.

**The failures are exactly predictable before touching the GPU.** Every one of the 180 error strings
carries the two numbers that decide it:

```
required 107008 > limit 101376  (106%)
required 221184 > limit 101376  (218%)
```

And `metadata.shared` is populated by the Triton **compiler**, not by the launch. Probed directly on
the box with a real pipelined `tl.dot` matmul, 6 configurations:

| BLOCK_M | BLOCK_N | BLOCK_K | stages | `metadata.shared` at compile | actual launch |
|---|---|---|---|---|---|
| 64 | 64 | 32 | 3 | 16384 | ok |
| 128 | 128 | 32 | 3 | 32768 | ok |
| 128 | 128 | 64 | 4 | 98304 | ok |
| 256 | 128 | 64 | 4 | 147456 | `Required: 147456` |
| 128 | 256 | 64 | 5 | 196608 | `Required: 196608` |
| 256 | 256 | 64 | 5 | 262144 | `Required: 262144` |

6/6 agreement, and the compile-time figure equals the runtime `Required:` byte-for-byte. So the
information needed to refuse these configs is available from `warmup()` alone.

**Why the existing guard does not catch it.** `check_config` evaluates arithmetic constraint
expressions over PARAMS values. But shared usage is **not a closed-form function of the knobs** —
audited across all 16 candidates, `BLOCK_M * BLOCK_N * stages` gives failing-min *below* passing-max
in **15 of 15** candidates that had both, i.e. the classes overlap and no product bound separates
them. (This also re-confirms the already-disproved flash-attention shared formula: shared bytes must
be read from the compiler, never derived.) A constraint expression therefore cannot express this, and
asking the parameterizer agent to write one would be asking it to guess.

**Worse than wasted time: the budget vanishes silently.** `tpe.py:107` reports a failed trial as
`TrialState.FAIL`, and Optuna **excludes** failed trials from the TPE model rather than treating them
as bad objectives. So those 181 samples produce neither a measurement nor an avoidance signal — TPE
will happily propose the same doomed region again. That is the mechanism behind the 21–33% rates
persisting for a candidate's whole tuning run instead of decaying.

### Fix: a compile-only feasibility screen in front of the GPU

Add a worker job type that compiles a materialized candidate at a given config and returns
`metadata.shared` / `n_regs` **without launching** (`kernel.warmup(...)`, grid unused). Then in
`_tune`'s loop (`orchestrator.py:1017`), before `_run_trial`:

1. Compile-probe the config. When `shared > device.max_shared_bytes_optin`, record a trial with a new
   `failure_kind="infeasible_shared_memory"` carrying required/limit, and `tell()` it as
   `TrialState.PRUNED` rather than `FAIL` — Optuna keeps pruned trials in the model, so the region is
   actually learned instead of discarded.
2. Cache the probe by `(source_sha, params.key())`, so a re-tune or an expansion re-ask is free.

This is generalizable, not per-task: it reads the compiler's own number against the device's own
limit, with no formula and no task knowledge. It also makes this class of waste visible in the log
rather than as an opaque `runtime_error`.

**Expected gain:** ~18% more measured configs per budget on this task, and a TPE model that is no
longer blind to a fifth of its samples. The screen costs a compile, which the failing path already
pays — the saving is the launch, the correctness pass, and the process spin-up.

**Risk:** low. When the probe cannot compile (an unrelated error), fall through to the normal path and
let the real trial report it — the screen must never be the thing that rejects a candidate.

---

## P2. `launch_bound` cannot fire, and its ratio is computed with the wrong denominator

Already written up in `plan-next-round-and-deferred-fixes.md`; repeated here because it is P2 by cost
and belongs in the same round. Summary: `measure_launch_overhead` is set only by `full_eval`, whose
sole caller is the final re-eval, so **848/848** trial profiles carry `cpu_issue_ms: None` and the
calibrated `launch_bound_cpu_ratio = 0.8737` has never been compared against anything. Behind it,
`classify()` divides `cpu_issue_ms` by the L2-flushed harness latency instead of the overhead probe's
own warm-cache `gpu_ms`, which biases the ratio low.

Fix: measure once per candidate on its tuned-best config (not per trial), and give `classify()` the
probe's own denominator, separate from the `gpu_ms` used for throughput fractions. Then verify on a
task that MUST be launch-bound — an L1 elementwise op — because until one of `launch_bound` /
`overhead_floor` fires on a task chosen for that reason, "the classifier has eight classes" is a claim
about the code and not about the harness.

---

## P3. The fp16/bf16 compute ceiling is still missing, and it now affects the winner

Also already documented, unchanged in priority, but the count has grown again: **13 of 25** verdicts
in L3:43 carry `impossible_fraction`, all fp16/bf16 on tensor cores. The current best reads **140.7%**
of the tf32 ceiling. The classification KIND stays sound (arithmetic intensity 990 against a ridge of
60 independently puts it compute-side), and `impossible_fraction` + `disagreement` warn the agent, but
the percentage — the one number that answers "how much headroom is left" — is not usable for any
candidate good enough to lead.

Fix: measure fp16 and bf16 matmul ceilings in `run_calibrate` alongside fp32/tf32, add them to
`Calibration`/`DevicePeaks`, and have `classify` select the denominator from `_candidate_precision`
(which already distinguishes fp16/bf16/tf32/ieee_fp32 — only the ceiling is absent).

**Must land together with P1/P2 and before the next task, not between tasks of a comparison set:**
changing calibration moves every derived threshold, so runs either side of it are not comparable on
throughput fractions.

---

## P4. Loop D still has not fired on an L3 task, for the documented reason

`NOVELTY_ROUND_STARTED = 0` in L3:43, with 4 seed families against `max_families_total = 3`. This is
the already-diagnosed gate, and the budget rise has been applied — but the evidence that it works is
from L1:42 (which fired twice), not from an L3 task. Worth watching rather than fixing: if L3:21 and
L3:48 also produce 4 seeds, Loop D stays unexercised on the tasks the paper reports.

---

## P5. Two things to keep measuring rather than fix

**Spills rank negatively against speed** (rho = -0.516, n = 13): the fastest candidate spills 52, a 4x
slower one spills 2, while `compiled_kernel.md` warns that spills "can cost more". One task is not
enough to invert the guidance. Collect the same statistic on L3:21 and L3:48; if it holds, the
paragraph should say that spills at large tile sizes are often the price of tensor-core throughput and
should be judged on measured latency.

**55% of wall clock is inside agent calls** (6.05 h of 11.0 h: analyst 2.61, parameterizer 1.68,
rewriter 1.57, generator 0.19). Not a defect — the loops are the method — but it bounds what any
GPU-side optimization can buy. P1's 0.93 h is 8.5% of the total and about 20% of the GPU-side half,
which is the right way to read that saving.

---

## Order

P1 → P2 → P3 in one batch, then re-run L3:43 to compare against this run (same task, same calibration
change, so the comparison is clean), then resume the chain with L3:21 and L3:48. P1 alone justifies the
restart: it returns ~18% of the trial budget and repairs the TPE model, and it is the only one of the
three whose effect shows up within a single task.
