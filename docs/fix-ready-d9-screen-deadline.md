# D9 fix — ready to apply, NOT applied while any run is in flight

**Status** prepared 2026-09-12 08:5x · **must not be applied** until all three E1/E3 runs have
`RUN_FINISHED` on disk. Changing the screen deadline changes which configurations get a screen
verdict versus falling through to a real trial, so a run started after it is not comparable with one
before, and E1's two arms must stay comparable with each other.

## What the fix is

`compile_screen` (`evaluation/correctness.py:268`) passes `self.cfg.build_timeout_s` — a real trial's
1200 s compile budget — to a screen whose non-answer is free by construction. `prescreen_batch`
(line 180) already passes a purpose-built deadline via `prescreen_timeout_s(cfg, n)`. The fix is to
bring the single-config path under the same function.

```python
# correctness.py, in compile_screen, replacing the run_job deadline
probe = self.worker.run_job(job, screen_timeout_s(self.cfg),
                            f"{tag}-compile-screen", lock_mode="shared")
```

with a new module-level function next to `prescreen_timeout_s`:

```python
def screen_timeout_s(cfg: EvalConfig) -> float:
    """How long a SINGLE-config compile screen may run, in its own right.

    Same argument as `prescreen_timeout_s`, which this deliberately reuses rather than restates:
    a screen's timeout removes NOTHING from the search, because only answers are cached, so an
    unanswered configuration still gets a real trial with the full `build_timeout_s`.

    The floor exists because `prescreen_timeout_s(cfg, 1)` is 33 s at the shipped defaults, which
    sits BELOW the measured p90 of 26.8 s by too small a margin to be safe. Measured across 1243
    single-config screens on three tasks and two GPU models (box1 L3:43 4090, box2 L3:43 4090,
    box3 L3:48 A800): p50 11.4-13.6 s, p90 26.5-28.5 s, p99 38.7-105.0 s. A 120 s deadline still
    answers 99.0% / 99.6% / 99.7% of the screens that answered at all.
    """
    return max(prescreen_timeout_s(cfg, 1), float(cfg.screen_floor_timeout_s))
```

and one config field:

```python
    # A SINGLE-config compile screen's floor deadline. Derived from 1243 measured screens
    # (p99 38.7-105.0 s across three tasks and two GPU models), not guessed: at 120 s the screen
    # still answers 99.0-99.7% of what it answers today. See `screen_timeout_s`.
    screen_floor_timeout_s: float = 120.0
```

`min(..., build_timeout_s)` is not needed on top: 120 s is far below 1200 s, and
`prescreen_timeout_s` already clamps its own term.

**Verified while preparing this** (so the fix does not carry a second defect):

- `prescreen_timeout_s(cfg, 1)` = 30 + 3×1 = **33 s** at the shipped defaults, which is 4.5–6.5 s
  *below* the measured p90 band of 26.5–28.5 s. Too small a margin — the floor is genuinely
  necessary, not decoration.
- Neither `screen_timeout_s` nor `screen_floor_timeout_s` exists anywhere in `src/`, `configs/` or
  `tests/` today, so both names are free.
- `EvalConfig` extends `StrictConfig` with `extra="forbid"`. A new field with a default is therefore
  **safe** — existing configs that omit it keep working — but any config that *sets* it must spell it
  correctly. This is the same strictness that required removing `suspicious_speedup` from every
  config file in `c399640`.
- **Pre-existing docs defect to fix in the same commit**: `config.py:264` points the reader at
  `Evaluator._prescreen_timeout_s`, which does not exist — the function is module-level
  `correctness.prescreen_timeout_s`. A comment naming a symbol that isn't there is exactly the kind
  of thing that let this whole defect hide, so it should be corrected while the file is open.

## Why 120 s and not 33 s or 300 s

| box | task / GPU | screens | p50 | p90 | p99 | max | answered ≤120 s |
|---|---|---|---|---|---|---|---|
| 1 | L3:43 / 4090 | 222 | 11.8 s | 28.5 s | 105.0 s | 991.0 s | 206/208 = 99.0% |
| 2 | L3:43 / 4090 | 602 | 11.4 s | 26.8 s | 56.9 s | 167.2 s | 524/526 = 99.6% |
| 3 | L3:48 / A800 | 419 | 13.6 s | 26.5 s | 38.7 s | 144.7 s | 366/367 = 99.7% |

33 s (the `n=1` formula value) would cut into the p90 band on all three boxes. 300 s buys nothing
over 120 s: no box has an answered screen between 167 s and 991 s.

## Measured payoff

Time recoverable, counting **both** hung screens and screens that answered past the cap — the second
category matters and the first version of this accounting missed it (one box-1 screen answered at
1202 s and another at 991 s, both counted as successes):

| box | recoverable | share of a 12 h budget |
|---|---|---|
| 1 | **~2.1 h** (+0.3 h for the 1202 s screen) | **~17-20%** |
| 2 | 0.02 h | 0.2% |
| 3 | 0.01 h | 0.1% |

The cost is at most three screen verdicts across 1243 screens, each of which then falls through to a
real trial exactly as an unscreened configuration does today.

## What the fix must NOT do

1. **Not touch `build_timeout_s`.** A legitimate candidate whose `ptxas` genuinely needs ten minutes
   must still get its full compile budget in the real trial
   (`never-narrow-the-search-space-to-control-cost`).
2. **Not touch the refusal criterion, the caching rule, or which configurations get screened.**
   Measured: across both E1 arms' 225 refusals, none is marginal (closest asks for 5% more shared
   memory than the device has, median 62% more) and there are zero false negatives. The decision
   logic is right; only the deadline is wrong.
3. **Not cache a non-answer.** Unchanged by this fix, and load-bearing: a cached failure would
   convert one transient into a permanent blind spot
   (`caching-a-probe-failure-makes-a-transient-permanent`).

## Guards to write with it

- A test that `screen_timeout_s(cfg)` is **strictly less than** `cfg.build_timeout_s` at the shipped
  defaults, so the screen can never again borrow a trial's budget.
- A test that it is **at least** the measured p99 floor, so a future edit to
  `prescreen_per_config_timeout_s` cannot silently drop the single-config screen below the useful
  band.
- A revert-check variant restoring `build_timeout_s` at the call site, which must be CAUGHT.
- A revert-check variant setting the floor to 0, which must be CAUGHT by the p99 test.
- A grep-based test that `build_timeout_s` appears at **no** `run_job` call for a screen — the
  generalizable form of this defect, since `59d5a71` fixed one caller and left the sibling
  (`a-fix-applied-to-one-caller-leaves-its-siblings`).

## Deeper fix, out of scope here

The screen and the eval compile the same source twice; measured, the screen does not even warm the
cache (a 991 s screen was followed by a 978.8 s eval). Screens cost 23-34% of each run's span even
when they answer. Making one compile serve both, or dropping the per-config screen in favour of the
batch one, is B2's territory and needs its own design. The deadline fix is the cheap, low-risk half
and is independent of it.
