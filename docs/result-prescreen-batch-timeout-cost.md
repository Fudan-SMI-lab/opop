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

### The batch path times out 35× more often than the per-trial path

The event log understates this, and the `jobs/` directory settles it: a job that never completed leaves
a `*.json` spec with no matching `*.out.json`. Counted on disk:

| path | jobs | no output | rate |
|---|---|---|---|
| batch `prescreen` | 5 | **3** | **60%** |
| per-trial `compile-screen` | 117 | 2 | 1.7% |

```
cand-2926f7cd-prescreen-6962ceba.json   12:23:33  NO OUTPUT (timed out)
cand-2926f7cd-prescreen-7063a72c.json   11:02:38  NO OUTPUT (timed out)
cand-d2cf7928-prescreen-543236f5.json   06:43:46  NO OUTPUT (timed out)
cand-d2cf7928-prescreen-74fe8317.json   08:30:48  COMPLETED
cand-d2cf7928-prescreen-ee95164b.json   06:30:47  COMPLETED
```

**Three of five, not two of four** — the event log shows only 4 `SPACE_PRESCREENED` entries because
the third timed-out batch (06:43:46) never reached its event either. That is the same invisibility from
a second direction: a batch that dies produces neither an output file nor a log line.

60% against 1.7% is the batching hypothesis confirmed by measurement rather than by reading: a single
configuration whose `ptxas` runs 20 minutes fails the whole batch, so batching 40 configurations
multiplies the chance that *some* member is that configuration. The per-trial path screens one config at
a time and almost never times out.

### Measured on all three boxes, and the arms differ

`scripts/screen_cost.py` reproduces the count from any run directory:

| box | task | batch prescreens | timed out | per-trial screens | timed out |
|---|---|---|---|---|---|
| box 1 (control) | L3:43 | 10 | **3 (30%)** | 357 | 1 (0.3%) |
| box 2 (treatment) | L3:43 | 13 | **0 (0%)** | 447 | 0 (0%) |
| box 3 | L3:48 | 5 | **3 (60%)** | 117 | 2 (1.7%) |

Two things follow, and the second matters more than the first.

**It is not the task.** Box 1 and box 2 run the *same task* with the *same config* and differ 30% to 0%.
So the cause is the candidates' own code — how large a PTX their kernels compile to — not the problem
being solved.

**It is not one bad candidate either.** Box 1's three timeouts are spread across three different
candidates (`cand-52e0e567`, `cand-70cbf6bc`, `cand-b937d22f`) out of five, and two of those three had
*another* prescreen that completed fine. So a per-candidate exclusion would not have caught them, which
rules out option 3 below on evidence rather than on principle.

**This touches arm parity.** Box 1 lost three prescreens; box 2 lost none. The runs are still comparable
on the metric that matters — `check_search_effort` confirms both arms sit at exactly 40 trials per space,
and a failed screen never rejects a candidate, so neither arm's *search* was narrowed. What differs is
wall clock: box 1 spent up to 3 × 1200 s = 1 h on screens that produced nothing, and box 2 spent none.
On runs whose binding budget is the wall clock in 5 of 5 finished cases, that is an hour of tuning the
control arm did not get. **Report it as a caveat on any wall-clock comparison between the arms**; it does
not invalidate the latency comparison, which is per-trial and budget-capped.

**A live confirmation of the recovery, and of the gap.** While this was being written the worker for
`cand-2926f7cd-compile-screen-986d1a43` was replaced by a fresh process (new PID, new `/tmp/*.ptx`), so
the timeout does fire and the run does move on — nothing hangs permanently. But the event count stayed
at **248 across the whole replacement**: no event marked the timed-out screen at all.

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
   **Now also ruled out on evidence**: box 1's three timeouts hit three *different* candidates, and two
   of those three had another prescreen that completed normally — so there is no "bad candidate" to
   exclude, and the same candidate is sometimes fast and sometimes not depending on which 40
   configurations the sampler drew.
4. **Lower `build_timeout_s` for prescreens only.** Cheapest and most defensible — the measured
   marginal cost is 7 ms per config, so 48 configs finishing in 11.02 s means a 60–120 s prescreen
   ceiling is ~10× headroom over the measurement, against the 1200 s a *build* legitimately needs. It
   also bounds the damage rather than trying to predict it, which suits a cause that is per-draw rather
   than per-candidate. But it changes a config the running arms share, so it cannot be applied
   mid-experiment without breaking parity. **This is the recommended fix for the next run.**

**The one change that is unambiguously safe and should happen regardless: emit an event when a screen
times out.** Right now a 1200 s prescreen that answers nothing is indistinguishable in the log from a
fast one that found nothing — the two entries at seq 181 and 234 look identical to the seq 39 entry.
`SPACE_PRESCREENED` should carry the elapsed time and whether the probe returned `ok`, so this is
countable rather than something I had to catch by finding the GPU idle.

## Guard

None yet — this is a measurement, not a fix. `scripts/screen_cost.py` reproduces the table from any
run's `events.jsonl`, excluding agent-call spans so the gaps are attributable to GPU-side work.
