# One slow `ptxas` voids a 40-configuration prescreen batch, and it happened twice

**Status: MEASURED, NOT FIXED.** Recorded for the post-experiment analysis. The harness behaves
correctly at every step — nothing is mis-cached, no candidate is rejected, no verdict is wrong — so
this is a *cost* defect, not a correctness one, and the fix is not low-risk (see below).

## What was measured

Box 3 (`run-l3-48-20260911-052647`, A800) sat 15 minutes with the **GPU at 0%** and 1853 MiB used.
Nothing was in flight on the agent side. The cause, from `ps --forest`:

```
110184 kernel_optimizer.cli resume
  151826 \_ worker_main.py --job cand-2926f7cd-compile-screen-986d1a43.json      15:35 elapsed, do_wait
    152098    \_ .../triton/backends/nvidia/bin/ptxas -lineinfo -v --gpu-name sm_80 /tmp/tmpggzmfi4j.ptx
              09:14 elapsed, 99.5% CPU
```

`/tmp/tmpggzmfi4j.ptx` is **152,185 lines / 7.5 MB**.

**This is NOT the recorded 111 GiB OOM case.** `VmRSS` was flat at 680 MiB across a 10 s window and
the box had 979 GB available, so `ptxas` is compute-bound here, not memory-pathological. Different
failure, same origin: a candidate whose PTX is enormous.

## The cost, per batch, off the event log

`SPACE_PRESCREENED` carries `configs_probed` and `infeasible`, so the payoff is measurable directly:

| seq | candidate | probed | infeasible | preceding gap |
|---|---|---|---|---|
| 39 | `cand-d2cf7928` | 40 | 0 | 610.6 s |
| 104 | `cand-d2cf7928` | 40 | **17** | 490.6 s |
| 181 | `cand-2926f7cd` | 40 | 0 | **1200.5 s** |
| 234 | `cand-2926f7cd` | 40 | 0 | **1201.0 s** |

**1200.5 and 1201.0 are `build_timeout_s: 1200` firing exactly.** Two batches of 40 configurations
each burned the full timeout and screened out **nothing**. Total: 0.97 h of the run's ~7 h on
prescreening, of which 0.67 h produced zero information.

Whole-run context: **3.16 h of non-agent blocking** in gaps ≥ 120 s (agent-call spans excluded, or the
largest "screen cost" would just be a rewriter call). 22 `CONFIG_SCREENED_INFEASIBLE` against 131
`TRIAL_DONE`.

## Why one slow config voids forty

`prescreen_batch` puts the whole batch in ONE worker process under ONE timeout:

```python
job = make_compile_probe_job(str(task.ref_path), str(first), backend=backend,
                             extra_kernel_src_paths=[str(p) for _, p in pending[1:]])
probe = self.worker.run_job(job, self.cfg.build_timeout_s, f"{tag}-prescreen", ...)
```

That batching is deliberate and well-justified — its docstring measures a per-process probe at a median
16.7 s against 11.02 s for **48 configurations in one process**, a marginal 7 ms each, 73× cheaper. The
design is right. What it does not survive is one configuration whose `ptxas` runs for 20 minutes: the
timeout is per *batch*, so that single config takes the other 39 answers with it.

The intended cost is ~11 s for 48 configs. The measured cost was 1200 s for 40 and no answers.

## Everything downstream is correct, which is why nothing was restarted

- **The failure is not cached.** `prescreen_batch` and `compile_screen` both cache only *answers*, and
  the comment says why: "a FAILURE is a fact about this one probe attempt … and caching it makes a
  transient permanent". Verified in the source. So the timeout does not poison the space.
- **No candidate is rejected.** `compile_screen` "REFUSES ONLY on the compiler's own figure exceeding
  the device's own limit … This screen must never be the thing that rejects a candidate." A probe that
  cannot answer returns `None` and the real trial decides.
- **`_shared_memory_ok` treats unknown as feasible**, so an unscreened configuration is still sampled.
- No `SCREEN_FAILED` / `WORKER_TIMEOUT` event exists under those names — the timeout is *silent* in the
  log, which is the one part worth changing (see below).

So the only damage is wall clock, on a run whose budget is the binding constraint in 5 of 5 finished
runs.

## Why the fix is deferred rather than applied now

Each candidate option can over-prune, and over-pruning is the failure this project has already paid
for twice (`opop-s1-dropped-s1b-redesigned`, `retirement-of-a-value-is-unconditional-but-failure-is-not`):

1. **Per-config timeout inside the worker.** Correct in principle, but it means abandoning a compile
   mid-`ptxas` and recording "no answer" for that config — and a config that merely compiles slowly is
   not infeasible. Needs care that a slow compile never reads as a rejection.
2. **Split the batch on timeout and retry the halves.** Recovers the other 39 answers, costs another
   1200 s in the worst case, and the worst case is exactly when the budget is already tight.
3. **Skip prescreening for a candidate whose PTX exceeded some size.** A per-case threshold on a
   quantity nobody has characterized; this is the hardcoded-special-case shape the project forbids.
4. **Lower `build_timeout_s` for prescreens only.** Cheapest and most defensible — the measured
   marginal cost is 7 ms per config, so 48 configs finishing in 11.02 s means a 60–120 s prescreen
   ceiling is ~10× headroom over the measurement, against the 1200 s a *build* legitimately needs. But
   it changes a config the running arms share, so it cannot be applied mid-experiment without breaking
   parity.

**The one change that is unambiguously safe and should happen regardless: emit an event when a screen
times out.** Right now a 1200 s prescreen that answers nothing is indistinguishable in the log from a
fast one that found nothing — the two entries at seq 181 and 234 look identical to the seq 39 entry.
`SPACE_PRESCREENED` should carry the elapsed time and whether the probe returned `ok`, so this is
countable rather than something I had to catch by finding the GPU idle.

## Guard

None yet — this is a measurement, not a fix. `scripts/screen_cost.py` reproduces the table from any
run's `events.jsonl`, excluding agent-call spans so the gaps are attributable to GPU-side work.
