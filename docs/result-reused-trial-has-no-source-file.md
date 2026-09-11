# A reused measurement wrote no source file, so `launch_bound` was unreachable for a fifth of candidates

**Status: FIXED (`80a4c14`), NOT restarted.** Driver-side, so it cannot reach the three live runs. What
it costs them is measured below and it is a diagnostic loss, not a validity loss.

## How it was found

Not by reading code. Box 2's `BOTTLENECK_CLASSIFIED` events were being checked for the `cpu_issue_ms`
fix (the recorded defect where the launch probe ran only in `full_eval`, so `cpu_issue_ms` was `None` in
848 of 848 trial profiles and the `launch_bound` branch could never be entered). The fix works — 11 of
14 classifications now carry a real ratio, on the sound `overhead_probe_gpu_ms` denominator:

```
cpu_over_gpu = 0.026 .. 0.059   against launch_bound_cpu_ratio = 0.85
```

So `launch_bound` is now a **measured negative** rather than an unmeasured one. But three of the
fourteen carried `cpu_issue_ms: None` **with no `LAUNCH_OVERHEAD_FAILED` beside them** — the
classification ran without the probe and nothing in the log said why.

## The defect

`_tune`'s reuse branch: when a TPE ask lands on an already-measured param set, the record is copied with
a **new `trial_id`** and `_run_trial` is skipped entirely. `_run_trial` is the only writer of
`trials/<trial_id>.py`. So the file for that id never existed.

`_measure_best_overhead` resolves the winning trial id back to source, and its `path.exists()` guard
returned **silently**: no event, no `cpu_issue_ms`, and the whole `launch_bound` branch unreachable for
that candidate — indistinguishable in the log from a kernel that is simply not launch bound.

## Measured, and it is one cause

| run | reused records | with no `.py` |
|---|---|---|
| `run-l3-43-20260911-053020` (live control) | 26 | **26** |
| `run-l3-43-20260911-052630` (live treatment) | 33 | **33** |
| `run-l3-43-20260908-121539` | 12 | **12** |
| `run-l3-43-20260910-083647` | 9 | **9** |
| `run-l3-48-20260907-202457` | 41 | **41** |
| `run-l3-48-20260909-115701` | 40 | **40** |
| total | **161** | **161, 100%** |

And the correspondence with the lost diagnostics is exact. For every `BOTTLENECK_CLASSIFIED` across four
runs that have any, comparing "was this classification unmeasured?" against "was the candidate's best
**complete trial at that moment** a reused record?":

```
run-l3-43-20260911-053020   classifications=11  agree=11  DISAGREE=0
run-l3-43-20260911-052630   classifications=13  agree=13  DISAGREE=0
run-l3-48-20260909-115701   classifications=17  agree=17  DISAGREE=0
run-l3-43-20260910-083647   classifications=3   agree=3   DISAGREE=0
```

**44 of 44, no disagreements.** One cause, no second mechanism, so the one-line fix is the whole fix.

A first version of that comparison used the run's **final** best and reported one disagreement on box 2
(`cand-2d8eaf9a`). That was my reader, not a second cause: `_measure_best_overhead` scans `crun.trials`
as they stand at the call, and classification fires **twice per candidate** (before and after
K-expansion), so the pre-expansion classification must be compared against the pre-expansion best. With
the timing right the disagreement disappears. Worth recording because "one case doesn't fit" is exactly
where a real second mechanism would show up, and here it was an artefact of comparing across a moment.

## What it cost, per arm

| arm | candidates classified | with an unmeasured classification |
|---|---|---|
| box 1 control | 7 | **3 (43%)** |
| box 2 treatment | 8 | **3 (38%)** |
| `run-l3-48-20260909-115701` | 10 | 1 (10%) |

**Both arms, at similar rates — so this is not an arm-parity defect.** It is a diagnostic loss of about a
fifth to two fifths of each run's launch-overhead readings, and it touches nothing that selects,
ranks, accepts or times a candidate. `launch_bound` was never reached by any candidate in any run
anyway (measured ratios 0.026–0.059 against 0.85), so no verdict changed; what was lost is the evidence
for stating that as a measured negative on those candidates rather than an unmeasured one.

For the write-up: report `launch_bound` as **not observed on the candidates where the probe ran**, with
the count, rather than as not observed at all.

## The fix

Two halves, both generic:

1. The reuse branch writes `materialize(crun.source, params)` — the same bytes `_run_trial` writes, not a
   reconstruction — and journals `TRIAL_ARTIFACT_FAILED` rather than swallowing a write failure.
2. The `path.exists()` return emits `LAUNCH_OVERHEAD_FAILED` naming **what became unreachable**. That
   silence is how this survived five runs.

The `best is None` return stays silent **deliberately**: a candidate with nothing correct has no winner
to measure, and firing there would put a FAILED event on every all-failed candidate and make the event
meaningless. That asymmetry is tested.

## Guard

`tests/test_reused_trial_artifact.py`, 4 tests, A800-only (imports optuna). They drive the **real**
`_tune` loop and the **real** `_measure_best_overhead`, substituting only the evaluator, tuner and store,
and assert on **whether the file exists** — not on the branch's source text. The pre-existing
`test_reused_measurement_is_journalled_with_flag` is a source-text assertion and would pass on an
orchestrator that writes the file to the wrong place; that is the recorded
`source-text-assertions-can-encode-the-bug` failure mode.

Revert-checked on the A800, all four variants CAUGHT:

| variant | caught by |
|---|---|
| artifact write removed (the original defect) | 2 tests |
| the silent return restored | `no_source_for_the_winning_trial_is_recorded_not_silent` |
| write points at the CACHED trial_id, not the new one | `a_reused_measurement_writes_the_same_py...` |
| `best is None` also fires an event (over-reporting) | `a_winner_with_no_complete_trial_is_still_silent` |

Variant A's first run reported "4 passed" **having patched nothing** — its literal anchor was mangled by
the shell heredoc. It was re-run with a regex anchor that asserts the match before reporting. A patch
that does not apply is a non-result, not a pass, and this is the second time in this project that a
non-applying edit nearly read as a verdict.

Full suite on the A800 with the fix: **804 passed, 1 skipped**.
