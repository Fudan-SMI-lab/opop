# Measurement: repair times out on 9% of calls; the parameterizer never does

Noticed from a live `AGENT_CALL_FAILED` on `run-l3-43-20260905-091705` at 09:58:03
(`prompt transport error (ReadTimeout): timed out`), which recovered at 10:00:07. Not a new
bug — it has happened **10 times** across the runs on disk, always non-final, and the
harness always recovers. Recorded because the cost is measurable and the distribution
across modules is lopsided in a way that points somewhere specific.

> **Corrects my own first pass.** I initially reported a bimodal repair-duration
> distribution with a 59.5-minute maximum and 4 calls over 20 minutes. That was an artifact
> of a pairing bug in my throwaway script: it paired each `AGENT_CALL_FINISHED` with the
> nearest preceding `AGENT_CALL_STARTED` *without resetting on the intervening
> `AGENT_CALL_FAILED`*, so a 20-minute timeout plus a 2-minute retry read as a
> 22-minute call. There is no bimodality and no 59-minute call. The real distribution is
> tight; see below. `scripts/audit_agent_call_durations.py` does the pairing correctly.

## The cost

| | |
|---|---|
| transport timeouts across all runs | **10** |
| of those, on the `repair` module | **8** |
| wall clock spent waiting on timeouts | **~140 min = 2.3h** |
| any of them final (candidate dropped)? | **no — 0 of 10** |

Each timeout burns the full `request_timeout_s: 1200` (20 min) before failing, then
`AGENT_SESSION_RESET` fires and the retry runs. One case burned 40 min (two consecutive
timeouts on the same call, `run-l3-43-20260904-093730` 10:49–11:49).

The retries themselves are **fast** — 1.2, 2.1, 4.2, 4.7 min for the four single-timeout
cases. So the loss is almost entirely the wall itself, not repeated work.

## It is not exposure, and not prompt size

The obvious explanation — repair is called most, so it times out most — is wrong:

| module | finished | timeouts | rate | median | p90 | max |
|---|---|---|---|---|---|---|
| **repair** | 80 | **8** | **9.1%** | 2.4 | 5.1 | 8.9 |
| generator | 14 | 1 | 6.7% | 8.0 | 8.9 | 9.6 |
| rewriter | 43 | 1 | 2.3% | 3.0 | 4.1 | 6.4 |
| **parameterizer** | **231** | **0** | **0.0%** | 1.7 | 2.8 | 6.5 |
| **analyst** | **107** | **0** | **0.0%** | 1.2 | 1.8 | 3.5 |

The parameterizer is called *three times as often as repair* and has never timed out once.
Neither has the analyst, at 107 calls. So this is a property of the repair call rather than
of call volume.

Prompt size does not explain it either — average sandbox input per call is generator
69.6 KB, rewriter 58.9, **repair 57.2**, parameterizer 49.4, analyst 38.8. Repair sits
mid-pack, and the generator (largest input, 14 calls) is not where the timeouts cluster.

## What the corrected distribution actually shows

Successful `repair` durations, n=70: **max 8.9 min**, median 2.4, p90 5.1. Nothing above
10 minutes. Same for every other module — the highest maximum anywhere is the generator's
9.6 min.

That is the interesting part, and it is the opposite of what I first claimed. **No agent
call has ever legitimately needed more than ~10 minutes**, yet the wall is at 20. So a
timeout is not "a call that needed more time"; it is a call that produced nothing for at
least twice the longest observed completion. That reads like a stuck request rather than a
slow one — consistent with the retry then finishing in 1–5 minutes on the same input.

## What this suggests, and what I have not done

I have **not changed the timeout.** The change the data supports is lowering
`request_timeout_s` — if no successful call has exceeded 9.6 min, a wall at ~12 min would
detect a stuck request in 12 min instead of 20, with no observed call at risk of being cut
off. Across 10 occurrences that is roughly an hour saved, and the recovery path is already
proven (10 of 10).

Two reasons it still needs a decision rather than an edit:

1. `request_timeout_s` is **global**, not per-module. Lowering it also applies to the
   generator, whose median is 8.0 min and max 9.6 — uncomfortably close to a 12-minute
   wall. A per-module timeout is the safer shape, and that is a config-schema change.
2. n=70 successful repair calls on three tasks is not a strong basis for asserting a
   ceiling on legitimate duration. A harder task could plausibly need longer, and cutting
   off a genuine 15-minute repair to save 8 minutes of waiting would be a bad trade.

The harness behaves correctly today: non-final failure, session reset, retry, no candidate
lost. This is a cost-efficiency observation, not a correctness one, which is why it sits
behind the five decisions already pending.

## Caveats

- `AGENT_CALL_STARTED` carries no module field, so durations pair each terminal event with
  the nearest preceding start, resetting on *both* finish and fail. That is exact while
  calls run sequentially; a concurrent call would mis-attribute one duration. **The
  reset-on-fail is the part I got wrong the first time.**
- The ~140 min figure assumes each timeout consumed its full 20-minute budget, which
  matches every observed gap (exactly 20.0 or 40.0 min).
- n=8 repair timeouts. The 0-of-231 parameterizer record is the stronger half of the
  comparison.
