# Measurement: expansion direction ignores per-value failure rate, and it costs ~3x

`SPACE_EXPANDED` picks a widening direction from the winning trial's position (the
`at_boundary` / `boundary_direction` fix in
`fix-boundary-direction-follows-the-winning-trial.md`). It does **not** consult how often the
edge choice it is widening past actually *fails*, even though `TuningStats.param_stats` already
carries `failure_rate_by_value` for every knob.

Measured on `run-l3-43-20260906-091019` (12 expansions, 24 widened knobs, 1120 trials):

```
candidate       knob              dir  edge edge_fail  added         added_fail
cand-8c64ccc3   BLOCK_M           max    64      21%   ['128']        7/12
cand-a988ff79   SCORE_BLOCK_M     max   128       0%   ['256']        0/1
cand-a988ff79   SCORE_NUM_WARPS   min     2       0%   ['1']          0/6
cand-a988ff79   VALUE_BLOCK_M     max   128       0%   ['256']        0/1
cand-a988ff79   VALUE_BLOCK_D     max   128       0%   ['256']        0/1
cand-a988ff79   VALUE_NUM_WARPS   max     8       0%   ['16']         0/5
cand-1129b4d9   BLOCK_N           max   128       3%   ['256']        0/2
cand-3df5fd86   BLOCK_N           max   128       5%   ['256']        0/0
cand-9df6f133   BLOCK_M           max   128      45%   ['256']        1/3   <--
cand-36cda636   BLOCK_M           max   128      41%   ['256']        2/3   <--
cand-36cda636   NUM_WARPS         max     8      39%   ['16']         3/6   <--
cand-a7cb7970   VALUE_BLOCK_D     max   128       2%   ['256']        1/10
cand-dda162c7   BLOCK_M           max    64       7%   ['128']        4/21
cand-718483d0   NUM_STAGES        max     4       0%   ['5', '6']     0/12
cand-5b1bf2d1   BLOCK_M           max   128      34%   ['256']        6/8   <--
cand-60fdcae9   QKV_BLOCK_M       max   128      25%   ['256']        0/0   <--
cand-60fdcae9   QKV_BLOCK_N       max   128      24%   ['256']        1/2
cand-60fdcae9   QKV_WARPS         max     8      26%   ['16']         0/2   <--
cand-60fdcae9   QKV_STAGES        max     4      40%   ['5']          0/2   <--
cand-60fdcae9   PROJ_BLOCK_M      max   128      29%   ['256']        1/2   <--
cand-60fdcae9   PROJ_BLOCK_N      max   128      30%   ['256']        2/4   <--
cand-60fdcae9   PROJ_WARPS        max     8      33%   ['16']         1/7   <--
cand-93d4a2ff   SCORE_NUM_WARPS   max     8       0%   ['16', '32']   0/5
cand-93d4a2ff   OUT_NUM_WARPS     max     8       0%   ['16', '32']   0/6
```

**10 of 24 widened knobs pushed past an edge that was already failing >= 25%**, and the values
added there fail far more often than the rest:

```
added values past a FAILING edge (>=25%) : 16/37 = 43% fail
added values past a HEALTHY edge (<25%)  : 13/84 = 15% fail
```

Roughly **3x the failure rate**, and a failed trial buys nothing: it consumes a compile plus a
GPU slot and returns no latency. `cand-60fdcae9` is the clearest case — seven of its eight
widened knobs sat past edges failing 24-40%, so the whole expansion was aimed into a region the
parent had already shown to be hostile.

This is also visible in the run's overall trial health, which is strongly bimodal rather than
uniformly noisy:

```
1120 trials: 942 complete / 178 fail  = 15.9% overall
failure kinds: runtime_error 131, correctness_mismatch 47

per-space failure rate (>= 20 trials):
  cand-9df6f133  38%      cand-dda162c7  10%
  cand-36cda636  34%      cand-bfab4d37   8%
  cand-4a3d3538  32%      cand-1129b4d9   5%
  cand-8f66c41c  32%      cand-3df5fd86   2%
  cand-8c64ccc3  31%      cand-a7cb7970   2%
  cand-60fdcae9  31%      cand-93d4a2ff   2%
  cand-5b1bf2d1  30%      cand-a988ff79   0%
                          cand-053f3dc6   0%
                          cand-718483d0   0%
```

Seven spaces at 30-38% against seven at 0-5%. The high-failure group is where the resource
limits actually bite (a 256-row tile with an fp32 accumulator, or 16-32 warps, exceeds
registers or shared memory on sm_120), and it overlaps the expansions above.

## Why this is not the same claim as "expansion is useless"

It is not. Over all 63 historical expansions, TPE samples added values in 98% of spaces and 26%
have their *winning* trial land on an added value. Expansion pays. The claim here is narrower
and about *aim*: the direction is chosen from where the best trial sits, with no regard for
whether that direction is already failing, so a subset of expansions is systematically pointed
into a wall.

A clean live example of the mechanism working *correctly* is example 4 below.

## The full outcome distribution: 67 expansions, four kinds

Classifying every expansion in the completed runs that has at least two tuning results (before
and after), by whether the family's best improved >= 2% and whether the *winning* trial used a
value the expansion added:

```
no change at all (|gain| < 0.5%)      26  (39%)
marginal (0.5-2%)                     19  (28%)
improved, WINNER USED an added value  12  (18%)
improved via re-tune only             10  (15%)
```

**Only 18% of expansions pay off through the mechanism they exist for** — a wider range whose new
value wins. Another 15% improve, but from re-exploring the domain the space already had, which
40 fresh trials would have done without any widening. And **67% produce no meaningful change**,
at 40 trials each.

At the measured ~19 s per trial that is roughly `45 x 40 x 19 s ≈ 9.5 h` of GPU time across the
project spent on expansions that moved nothing. This does not mean the feature is bad — 12
expansions genuinely found a better configuration outside the original range, and one of them
(`cand-9f6af7bd`, below) is on the run's best hand-written result — but it does mean the aim is
what needs work, which is exactly what the failure-rate veto addresses.

## Four worked examples, one per outcome

### 1. Improved, winner used the added value (18%)

...is the intended case; `measurement-expansion-direction-yield.md` covers the 26%-of-winners
statistic across all history.

### 2. Improved via re-tune only (15%) — `cand-9f6af7bd`, and it produced the run's best hand-written kernel

```
expansion requested 5 knobs widened:
  QKV_BLOCK_M max, QKV_BLOCK_N max, ATTN_BLOCK_M max, ATTN_NUM_WARPS max, ATTN_NUM_STAGES max
actually gained a value: QKV_BLOCK_N only, [32,64,128] -> [32,64,128,256]

v1  best 10.7 ms
v2  best  9.43 ms   (-12%)

winning params: QKV_BLOCK_N = 128   <- NOT the added 256
what actually moved: PROJ_BLOCK_M 32->128, QKV_NUM_STAGES 3->4, ATTN_NUM_WARPS 8->4
```

A 12% improvement from an expansion whose added value the winner never used — the gain came
entirely from 40 fresh trials re-exploring the *existing* domain. Note also that 4 of the 5
requested knobs gained nothing, so **the expansion requested and the expansion delivered are far
apart**; do not attribute an improvement to the knobs an agent asked for.

Resource context for this winner, which bears on defect 2 in
`analysis-framework-defects-and-next-steps.md`:

```
n_regs 255 (the hard limit)   shared_bytes 98304 of 101376 opt-in   n_spills 4
```

It is pressed against both ceilings, consistent with its family carrying a high trial-failure
rate: most neighbouring configurations do not fit.

### 3. No change at all (39%) — `cand-e29aa508`

```
expansion: PROJ_BLOCK_N max, [16,32,64,128] -> [16,32,64,128,256]
v1 best 16.1 ms  ->  v2 best 16.1 ms      (exactly identical)
winning params: PROJ_BLOCK_N = 128, COMPUTE_DTYPE = bf16
```

40 trials, the added value not used, and the same incumbent returned to the millisecond. This is
the `memory: opop-v2-K-retune-cannot-disconfirm-incumbent` shape: a re-tune cannot dislodge its
own incumbent, so a zero-gain expansion is indistinguishable from having skipped it.

### 4. A cheap negative — `cand-9c8d066a`

Counted above as "marginal (<2%)", but it is the qualitatively distinct case: the expansion buys
real information, just not an improvement.

```
space v2  ATTN_BLOCK_M = [16, 32, 64, 128]        best 19.2 ms
expand    ATTN_BLOCK_M direction=max
space v3  ATTN_BLOCK_M = [16, 32, 64, 128, 256]   best 19.1 ms  (0.5%, below the 2% threshold)

pooled trials by ATTN_BLOCK_M:
   16: 13 trials  best  22.9
   32:  9 trials  best  19.4
   64:  6 trials  best  23.0
  128: 50 trials  best  19.1   <- incumbent, unchanged
  256:  2 trials  best 166.0   <- the added value: 8.7x WORSE
```

The added value was sampled, was catastrophic, and TPE stayed at 128 — spending only 2 of 40
trials to establish it. The edge at 128 was not unprobed potential. This knob's edge was healthy
(128 rarely failed), so it is not one of the 10 flagged above.

The two behaviours coexist: aiming at a **healthy** edge yields an improvement or a cheap
negative; aiming at an **already-failing** edge yields mostly failed trials, which return nothing
at all. That asymmetry is the whole argument for the veto.

## Proposed change (next round, not now)

Consider `failure_rate_by_value` when choosing the expansion direction. Two variants, cheapest
first:

1. **Veto** — skip a widening whose edge choice fails above a threshold (say 30%). Costs
   nothing to compute; the data is already in `TuningStats`.
2. **Redirect** — prefer the opposite direction, or a different knob, when the intended edge is
   failing but another candidate knob's edge is healthy.

Both are generic: they read a statistic the harness already produces for every knob, and they
encode no task-, knob-, or model-specific values. Neither changes what the agent is asked for,
only which of its named knobs the harness widens.

**Not implemented in this round.** The boundary-direction fix is mid-flight and its prospective
record (11 consecutive correct predictions) should not be perturbed while the L3 runs are still
producing comparable data. Recorded here so the next round starts from measurement.

## Reproduce

The analysis is ad hoc; the event nesting is the part worth writing down, because three of my
attempts got it wrong:

* `TRIAL_DONE` -> `payload.trial.candidate_id`, and the params are at
  `payload.trial.params.values` (not `payload.trial.params`). `payload.trial.space_id` is
  `None`, so trials cannot be attributed to a space version through it.
* `SPACE_PUBLISHED` -> `payload.space.candidate_id` (there is no top-level `candidate_id`);
  domains at `payload.space.domains[].choices`.
* `SPACE_EXPANDED` -> `payload.knobs[] = {name, direction}` and `payload.prev_best_ms`. It
  records the *request*, not the values added — those must be diffed out of the next
  `SPACE_PUBLISHED` for the same candidate.
* Trial success status is `"complete"`, not `"ok"`.
* `CONVERGENCE_DECIDED` -> the verdict is nested at `payload.decision.verdict` /
  `payload.decision.stop_kind`, with `payload.family_id` beside `decision` (not inside it).
  Reading `payload.verdict` silently yields `None` for every event, which makes every family
  look permanently `active` — I reported a frozen family as active this way before noticing the
  live notification disagreed. `payload.decision.evidence.rewrite_rounds_used` is the
  **per-family** round count; the `round` in `FAMILY_ROUND_RECORDED` is the orchestrator's
  global counter and is not comparable across families.
