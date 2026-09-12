# D8 option 1 — record job wall clock even when the job is killed

**Status** prepared 2026-09-12 10:5x · **not applied** while any run is in flight (it changes what is
written to `events.jsonl`, which is additive and would not change search behaviour — but it lands in
the same pass as D9, which *does*, so it waits with it).

**Why this is a hard prerequisite, not the cheapest step.** D8's option 2 wants to cap PTX size. The
threshold has to come from *time*, and time is unmeasurable today on exactly the cases that matter:
`compile_s` is written only when a job returns, and the six box-1 trials that paid 20–50 minute
compiles all timed out, so they carry no profile at all. Measured: `compile_s` reads p50 0.3 s /
max 1.6 s across 198 profiles while `ptxas` was observed live at 25:35. Choosing a size threshold on
that record would be picking a threshold for a proxy of a proxy — the 10x `excessive_speedup`
mistake. (`a-timeout-censors-the-metric-that-would-price-it`.)

## Where it goes

`gpu/worker_client.py::run_job` already knows the wall clock on every exit path, including the
timeout — it is the only place that does, because the worker cannot report a time it did not survive
to write. Four exits, all currently anonymous as to cost:

| exit | line | today |
|---|---|---|
| timeout, killed | 271 | `failure_result("timeout", ...)` |
| no out.json | 277 | `failure_result("worker_crash", ...)` |
| normal | 282 | `json.loads(out_path.read_text())` |
| unparseable out.json | 284 | `failure_result("worker_crash", ...)` |

Verified while preparing this: `import time` is already present (line 29) and `time.monotonic()` is
already the idiom in this file (lines 99, 115), so no new import and no clock-choice inconsistency.

**A trap to avoid, found by checking rather than assuming.** `ProfileRecord`
(`models/core.py:169`) sets only `frozen=True`, so it inherits pydantic's default
`extra="ignore"` — it will **silently drop** `job_wall_s` rather than reject it. That is the failure
shape of `occupancy-is-nested-and-a-flat-read-fakes-unmeasured` and
`a-fixture-invented-to-match-the-reader-proves-nothing`: everything green, field gone, and the
report reads "not measured". So `job_wall_s` must **not** be routed through `ProfileRecord`. It
belongs on the trial record beside the profile, and the guard below has to assert it survives a real
round trip through whatever model actually carries it — not merely that `run_job` returns it.

```python
# at the top of run_job, after job_id is formed
t_start = time.monotonic()

# helper, module-level next to failure_result's import
def _with_wall(result: dict[str, Any], t_start: float, timed_out: bool) -> dict[str, Any]:
    """Stamp every job result with the wall clock the DRIVER measured.

    The worker cannot report this on the path that matters: a killed job writes no out.json, so
    the only record of a 20-50 minute compile is the driver's own clock. `job_wall_s` is therefore
    additive to, not a replacement for, `profile.compile_s` -- the latter still measures the
    in-worker compile when the job survives, and the two disagreeing is itself the signal (a
    cache hit is 0.3 s of a 13 s job; a pathological ptxas is 25 min of a 30 min job).

    Never overwrites a key the worker set, so a future worker-side timing cannot be silently
    clobbered by the driver's coarser figure.
    """
    out = dict(result)
    out.setdefault("job_wall_s", round(time.monotonic() - t_start, 3))
    out.setdefault("job_timed_out", timed_out)
    return out
```

applied at all four exits — `_with_wall(failure_result(...), t_start, True)` on the timeout,
`False` on the other three.

## What consumes it

Nothing, at first — same discipline as `CONVERSION_RATES` in `842e2a6`. It must reach the event log
and the report, and **must not** enter ranking, allocation or acceptance:

1. `TRIAL_DONE.payload.trial` already carries the worker's dict, so `job_wall_s` arrives with no
   plumbing — **but not through `ProfileRecord`**, which drops unknown keys (see the trap above). It
   has to be threaded onto the trial record explicitly, and the guard must prove it survives the
   model, not just `run_job`.
2. A **per-candidate cost table in the report**: total job wall, median, max, and count of
   `job_timed_out`. This is D8's "nothing in the event log states that one candidate consumed 43% of
   a run; it took a bespoke script to find".
3. A **`JOB_COST_OUTLIER` event** when a single job exceeds some multiple of the run's own running
   median — the multiple derived from the corpus once (2) exists, not guessed now.

## Guards to write with it

- A test that a **timed-out** job's result carries `job_wall_s` ≈ the deadline and
  `job_timed_out: True`. This is the case the whole change exists for, so it must fail on the
  unfixed code.
- A test that a **normal** job carries `job_wall_s` and `job_timed_out: False`.
- A test that a worker-supplied `job_wall_s` **survives** (`setdefault`, not assignment).
- A revert-check variant that stamps only the success path, which must be CAUGHT by the first test.
- A revert-check variant using `time.time()` instead of `time.monotonic()`, which must be CAUGHT —
  a clock that can step backwards makes a long compile read as negative.
- A test that no ranking, allocation or acceptance path reads `job_wall_s`, by grep, mirroring the
  test that guards `CONVERSION_RATES`. Cheap-but-slow must never become a reason to reject a
  candidate: that is `never-narrow-the-search-space-to-control-cost` and
  `speed-guard-must-not-override-correctness`.

## What this does NOT do

It does not cap, refuse, throttle or reorder anything. It makes the cost **visible**, which is the
precondition for deciding whether option 2 (a PTX-size screen) or option 3 (cost-aware allocation) is
justified — and D8's own evidence already withdrew option 3 in its per-candidate form, since the
straggler's median trial was 1.1 min and only 5 of 21 were pathological.
