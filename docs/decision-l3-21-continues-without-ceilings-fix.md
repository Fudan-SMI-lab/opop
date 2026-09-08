# Decision: L3:21 keeps running without the ceilings fix

`run-l3-21-20260908-232211`, box 1, started 2026-09-08 23:22 on commit `e2d3d32`.

## The situation

While the run was in its first hour I found that `docs/device.md` reaches every agent with the
card's hard limits and **none of its measured ceilings** — eight lines, no DRAM figure, no TFLOP
figure — in a run whose own `CALIBRATION_MEASURED` event had just recorded fp16 at 158.6 and
bf16 at 164.2 TFLOP/s. Fixed in `9e8066d`.

That fix is in `agents/modules.py`, which is **driver-side code**: it cannot reach a running
orchestrator. Only the prompt `.md` files are re-read per agent call. So the choice was restart or
continue.

## Why continue

The question is not "is the fix good" but "did its absence damage THIS run". Audited on disk with
`scripts/audit_precision_reachability.py`:

| seed | precision knob | default | dot precisions in body |
|---|---|---|---|
| cand_1 | `COMPUTE_DTYPE` | tf32 | ieee, tf32, + fp16 cast |
| cand_2 | `GEMM_PRECISION` | tf32 | from knob |
| cand_3 | `COMPUTE_DTYPE` | tf32 | ieee, tf32, + fp16 cast |
| cand_4 | `COMPUTE_DTYPE` | fp16 | ieee, tf32, + fp16 cast |

Published space: `COMPUTE_DTYPE = ['fp16','bf16','tf32','ieee']`, `STORE_DTYPE = ['fp16','fp32']`.

**All four seeds made the tensor-core path reachable and the tuner has the full precision
domain.** The behaviour the ceilings were meant to produce is already present — the contract's own
guidance ("treat dot-product precision as a first-class design choice", "fp16 is roughly 2x tf32
on this class of card") was sufficient without the per-box numbers. Both were verified present in
this run's sandboxes: `grep -c 'roughly 2x'` and `grep -c 'Two backends are supported'` on the
generator's contract copy each return 1, so **all eleven contract fixes did reach this run**.

Restarting would cost the 38 minutes already spent, a fresh 22-minute generator call, and would
discard four seeds that are behaving correctly — to add numbers whose intended effect is already
observable. The honest expected gain is *better-informed* precision choices, not *newly possible*
ones, and that is not worth resetting an experiment that is already ahead of its incumbent.

## What this run can and cannot be evidence for

**Can:** the 11 contract fixes, F5's compile screen, the calibration-cache fix, the median
objective, and the L3:21 number itself.

**Cannot:** whether the measured ceilings change agent behaviour. This run is the *control* for
that, not the treatment. L3:43 (box 2, starting on `9e8066d` or later) will be the first run whose
writers see the ceilings, and the comparison is confounded by the task, so it is suggestive at
best. A clean answer needs the same task both ways, which is a next-round experiment, not a fix.

**Also cannot:** whether the rewriter backend switch fires. `RewriteCandidate.backend` and the
`_detect_backend` override landed in `1ef142d`, after this run started, so its rewrites still
inherit the parent backend. Count CUDA candidates on box 2's run, not this one.

## Early signals, for the record (not conclusions — the run is 38 min into 12 h)

- **F5 refused 3 of 40 configs pre-launch** on a real L3 space, and two trials were then refused
  by the post-materialize screen with the compiler's own figures (147456 B and 116224 B against
  the 101376 B limit, 145% and 115%). This is the screen working in production for the first
  time; on `run-l3-43-20260908-053708` these same failures cost 180 of 1004 trials.
- **Best trial 5.4774 ms** against a 6.92 ms incumbent, at 6 completed trials. Far too early to
  quote — `final_reeval_ms` is the only number that counts and the run has 11 hours left.
- Calibration re-measured rather than served from the stale cache, with real fp16/bf16 figures.
  The `CALIBRATION_SCHEMA_VERSION` fix is confirmed working live.
