# Framework analysis: where v2 spends its time, where it loses candidates, and what to fix next

Scope: the **harness itself** — not the performance baselines (that is
`analysis-performance-baseline-attribution.md`). Read from all 18 L3 run directories on disk
(8x L3:21, 7x L3:43, 3x L3:48; 17 with >=50 events), plus the live
`run-l3-43-20260906-091019`. Numbers come from `events.jsonl`, never from notification text.

**This is a preliminary pass.** `run-l3-43-20260906-091019` is still running, so it will be
re-checked against final numbers before any of this is acted on.

## 1. Where the wall clock goes: 83.7 h across 17 runs

```
phase                    hours   % of wall
tuning trials            45.01     53.8%
  |- completed           33.63     40.2%
  |- FAILED              11.39     13.6%      <- returns no latency at all
all agent calls          20.68     24.7%
  |- parameterizer        6.87      8.2%
  |- repair               5.08      6.1%
  |- rewriter             4.53      5.4%
  |- generator            2.19      2.6%
  |- analyst              2.01      2.4%
quick-test / witness      2.32      2.8%
witness rejections        1.25      1.5%
baselines                 0.66      0.8%
```

The single largest controllable loss is **13.6% of all wall clock spent on trials that fail**:
1840 of 8199 trials (22.4%), mean 22.3 s each — *slower* than a completed trial (19.0 s), because
a compile error or a correctness mismatch still pays compile plus launch.

```
failure kinds: runtime_error 969, correctness_mismatch 862, timeout 4, excessive_speedup 5
```

Clustering the 1840 `failure_detail` texts:

```
correctness (relaxed gate)         862  (46.8%)
shared memory / out of resource    526  (28.6%)
no diagnosable cause recorded      452  (24.6%)   <- see below: already fixed
```

The 452 with no diagnosable cause are **entirely historical**, and this is worth stating because
I initially mistook it for an open defect. `failure_detail` used to be truncated from the FRONT,
so a traceback's first 500 chars — harness call frames — were kept and the exception line at the
end was cut. `error_excerpt()` (`paramspace/validation.py:69`) fixed it by keeping the tail, and
`orchestrator.py:861` already applies it to the trial path with an 800-char limit. Per run:

```
run                        fails   no_diag   mean detail len
run-l3-43-20260903-020233     69        69       500     <- front-truncated
run-l3-43-20260903-145357    118       118       500
run-l3-43-20260904-093730    408       254       331
run-l3-43-20260905-091705    477         0       806     <- error_excerpt in effect
run-l3-43-20260906-091019    185         0       827
run-l3-21-20260905-195615    262         0       814
```

**Every run from 2026-09-05 onward records a diagnosable cause.** So the live failure split is
roughly 62% correctness / 38% resource-limit, and the resource half is the actionable one: a
`shared memory / out of resource` failure is a knob combination the space should have excluded,
which is what defect 3's veto would address.

That 22.4% is not uniform noise — it is concentrated. On L3:43's live run seven spaces sit at
30-38% failure while seven sit at 0-5% (see `measurement-expansion-ignores-failure-rate.md`),
and the high-failure group is where resource limits bite.

## 2. Where candidates are lost: the witness gate dominates every other cause

Across the 18 runs, 183 candidates registered and 214 spaces published, against 146 space
rejections. Grouped by cause:

```
witness_default_failed              83     <- 57% of all rejections
witness_minimal_failed              36     <- 25%
expansion:key_mismatch               7
expansion:witness_minimal_failed     6
agent_call_failed:parameterizer      2
agent_error                          2
expansion:constraint_invalid         2
expansion:degenerate_domain          2
expansion:infeasible_space           2
infeasible_space                     1
agent_call_failed:repair             1
inert_space                          1
novelty_rejected                     1
expansion:no_new_choices             1
```

**119 of 146 (82%) are witness failures.** Everything else combined is 27.

The recovery picture, per candidate that hit at least one rejection:

```
candidates hitting >=1 space rejection : 52
  recovered (a space eventually published) : 26  (50%)
  lost outright                            : 26  (50%)

by first rejection reason:
  witness_default_failed   n=49  recovered 24 (49%)  lost 25   119 total rejections
  infeasible_space         n= 1  recovered  1        lost  0
  agent_error              n= 1  recovered  0        lost  1
  inert_space              n= 1  recovered  1        lost  0
```

Two things follow:

* **Half of all candidates that trip the witness are lost permanently.** Not deferred — gone.
* **119 rejections against 49 candidates = ~2.4 repair attempts each.** With
  `repair_attempts: 3`, the gate is consuming nearly the whole repair budget of every candidate
  it touches. Repair is 6.1% of wall clock, and most of it is spent here.

This also explains the shape of the largest agent cost. Counting `AGENT_CALL_STARTED` for the
parameterizer per candidate (it is the only one of the pair that carries `candidate_id`):

```
1 call :  15 candidates      candidates parameterized : 74
2 calls:  42 candidates      extra (re-)calls         : 81
3 calls:  12 candidates      total calls              : 155
4 calls:   5 candidates      re-parameterization share: 52%
```

**Only 15 of 74 candidates (20%) were parameterized acceptably on the first attempt**, and 52% of
all parameterizer calls are retries. The parameterizer is 8.2% of wall clock — the single largest
agent cost — so roughly 4% of every run is spent re-parameterizing candidates whose first space
the witness gate refused. The distribution caps at 4 calls, consistent with `repair_attempts: 3`.

(Method note, because I got this wrong once: `AGENT_CALL_FINISHED` does **not** carry
`candidate_id`, so keying a per-candidate counter on both events collapses every finish into a
single `None` bucket and inflates the retry share — it produced a bogus 77% before I checked the
per-candidate distribution and found counts of 48/33/29 that no `repair_attempts: 3` could
explain.)

`witness_minimal_failed` (36 + 6) is a *known design defect*, already documented in
`memory: opop-v2-minimal-witness-is-fp16-corner`: the minimal witness is every knob's
`choices[0]`, which is the fp16 corner, and on a task whose outputs exceed 65504 it must fail
however correct the candidate is. On L3:48 it accounts for 21 of 42 rejections — exactly half.

## 3. Confirmed framework defects, ranked by measured cost

Ordered by wall clock or candidates lost, not by how easy they are to fix.

| # | Defect | Measured cost | Status |
|---|---|---|---|
| 0 | **Loop D (novelty) has never executed, in any run** | 0 of 18 runs; a named paper contribution is untested | **found this session** |
| 0b | **`rewrite_rounds_per_family: 3` cuts families off mid-improvement while ~30% of wall clock is unused** | 5 of 18 budget-frozen families (28%) were still gaining >=2%; in `run-l3-43-20260906-091019` **all 4 of 4** were, and it ended at 8.15 h of 12 h | **fixed this session** (3 -> 5) |
| 0c | `stop_kind="converged"` structurally unreachable | 13 of 13 freezes were `budget_exhausted`, 8 with flat histories | **fixed this session**, as a side effect of 0b |
| 1 | Witness gate rejects correct candidates; 50% never recover | 26 candidates lost, ~2.4 repair attempts each | partly mitigated by the fp64 relative gate |
| 2 | 22.4% of trials fail, consuming 13.6% of wall clock | 11.39 h of 83.7 h — but this is dominated by runs without the fp64 gate; the live rate is **13.5%** | largely addressed by the fp64 relative gate (see section 5) |
| 3 | Expansion aims at already-failing edges | 10 of 24 widened knobs; added values fail 43% vs 15% | measured, fix deferred |
| 4 | `minimal` witness is the fp16 corner | 42 rejections; half of L3:48's | documented, not fixed |
| 5 | Agent-call timeout was 20 min | 8 real calls killed across 5 runs | **fixed this session** (1800 s) |
| 6 | `precision: "unknown"` for no-`tl.dot` kernels | 1 of 3 tasks mislabelled | reporting only, not fixed |
| 7 | `AgentModuleConfig.timeout_s` is dead config | none (never read) | documented this session |
| 8 | **47% of families never get a single rewrite round, and runs end using 10-57% of the wall clock** | 17 of 36 families across 7 finished runs; L3:21 ended at 2.06 h of 12 h | **quantified this session**; root cause in `memory: opop-v2-run-stops-with-budget-unused` |
| 9 | Repeated param sets re-measured instead of served from cache | 175 of 304 repeats; 34.4 min (0.7% of wall) | **measured this session — minor** |

### Defect 0 in detail: loop D is structurally unreachable

Counting candidate origins across all 18 runs:

```
origin:seed       72
origin:rewrite   111
origin:novelty     0      <- loop D produced nothing, ever
NOVELTY_REJECTED   1
AGENT_CALL_STARTED module=novelty : 0
```

Not "rarely useful" — **never invoked**. The cause is a budget interaction, in
`orchestrator._novelty_round`:

```python
if len(self.deps.families.families) >= self.cfg.budgets.max_families_total:
    return False
```

with `max_seed_candidates: 4` and `max_families_total: 3`. The generator emits 4 seeds, each
seed registers its own family, so the run reaches **4 families before loop D is first
consulted** — and `4 >= 3` is true at every subsequent check. Verified per run:

```
all 17 runs with >=50 events:  families=4, max_families_total=3  -> gate closed
```

The single `NOVELTY_REJECTED` is the novelty *acceptance* check rejecting a candidate produced by
another path, not loop D running.

Why this matters beyond a wasted feature: **loop D is one of the four loops the paper claims**,
and the mechanism it tests (do distinctly-different families beat rewriting existing ones?) has
zero evidence behind it. Any claim about it in the paper is currently unsupported.

The fix is a budget question, not a code bug, so it needs a decision: either raise
`max_families_total` above `max_seed_candidates` (e.g. 3 seeds + up to 6 families, using the
existing `max_families_total_hard: 6`), or lower `max_seed_candidates` so seeds leave room. Both
change how search effort is divided, which is the user's call — I have not changed either.

### Defect 0b in detail: the round budget binds before the time budget

`rewrite_rounds_per_family: 3` freezes a family after 3 rewrite rounds regardless of whether it is
still improving. Across all runs, 18 families froze on `budget_exhausted` with >=2 recorded
rounds, and **5 of them (28%) were still gaining >=2% on the round where the budget ran out**:

```
run              family         best_history          final-round gain
20260906-091019  fam-6eea8eac   [18.6, 15.4,  8.1]         47.7%
20260906-091019  fam-8fb9b2b8   [18.5, 17.0,  9.4]         44.5%
20260906-091019  fam-94add40d   [19.4, 18.6, 16.1]         13.4%
20260905-091705  fam-4aea322a   [11.0, 11.0,  9.7]         11.5%
20260906-091019  fam-e6706893   [16.7, 15.9, 15.4]          3.1%
```

**Those are the four largest single-round gains in the entire project**, and all four happened on
the last round the family was allowed. The trajectories are accelerating, not plateauing —
`fam-6eea8eac` went 18.6 -> 15.4 -> 8.1, improving *faster* each round, and was then stopped.

`run-l3-43-20260906-091019` alone contributes 4 of the 5: **all four of its families were cut off
mid-improvement**, none of them plateaued, and the run ended at 8.15 h of a 12 h budget. That is
not an occasional unlucky freeze — at this task the round cap is what ends every family.

And the time budget was not the constraint:

```
run-l3-43-20260906-091019: froze its two best families at 6.62 h of a 12 h budget
run-l3-43-20260905-091705: froze its best family    at 6.24 h of a 12 h budget
```

**~45% of the wall-clock budget went unused** while the most productive families were being shut
down by a round counter. Note this is a *different* mechanism from defect 8
(`memory: opop-v2-run-stops-with-budget-unused`, where empty families fail to set `progressed`):
here the families are highly productive and the per-family round cap is what stops them.

The convergence machinery already distinguishes the two cases — `no_improve_rounds: 2` with
`min_improvement_pct: 2.0` exists precisely to freeze a family that has stopped improving, and
would have let all three of these continue. The round cap fires first and overrides it.

Fix options, both budget decisions rather than code fixes:

1. **Raise `rewrite_rounds_per_family`** (say 3 -> 5) and let `no_improve_rounds` do the stopping
   it was designed for. The 12 h wall clock remains the hard backstop.
2. **Make the round cap conditional on progress** — allow extra rounds only while the family's
   last round gained >= `min_improvement_pct`. This spends the extra budget on exactly the
   families that are earning it.

**Applied: option 1, `rewrite_rounds_per_family` 3 -> 5** (`config.py` field default plus the
three L3 configs; `smoke_l1*` left alone).

Option 2 was rejected on the user's reasoning, which is correct and worth recording because it is
the opposite of what "spend the budget where it is earned" suggests at first glance:
`min_improvement_pct` is a **relative** threshold, so a 9.43 ms family must find 0.19 ms to
qualify for another round while an 18.6 ms family needs only 0.37 ms. A progress-conditional cap
therefore systematically penalises the *fastest* families — and those are the ones most likely to
produce the run's best result. It would cut off exactly the families we most want to continue.
The user's framing: any spend that raises the final best is worth it.

Measured cost of the change, from 56 observed rewrite rounds:

```
one rewrite round: 38.9 min median / 46.3 min mean  (L3:21 43.3, L3:43 38.9, L3:48 34.6)

+2 rounds x 2 concurrently-active families : +2.6 h median   6.6 h -> ~9.2 h   fits 12 h
+2 rounds x all 4 families (pessimistic)   : +5.2 h median   6.6 h -> ~11.8 h  at the edge
```

`max_families_active: 2` makes the 2-family figure the likely case. Overshoot is safe: the wall
clock check sits at the **top** of the outer loop (`orchestrator.py:428`), so exhausting it exits
cleanly and `_finalize()` still runs the final re-eval and report — no work is discarded, at worst
one in-flight round completes past 12 h.

**Expect L3 runs to get noticeably longer** (~6.5 h -> 9-12 h). That is the intended trade.

The live `run-l3-43-20260906-091019` is unaffected: its `manifest.json` pins
`rewrite_rounds_per_family: 3`, which is where a run's actual budget can be verified (the
`RUN_CREATED` event carries only `run_id`, not the config). It froze **all four of its
families** mid-improvement — 8.06 ms (+47.7% on the final round), 9.43 ms (+44.5%), 16.1 ms
(+13.4%) and 15.4 ms (+3.1%) — and finished at 8.15 h of its 12 h budget, so this single run
supplies 4 of the pattern's 5 instances.

### Defect 8 in detail: half the families are never rewritten at all

Defect 0b is about families stopped *too early*; this is about families never started. Counting
`rewrite_rounds_used` at freeze across the 7 finished runs:

```
families across finished runs        : 36
  with ZERO rewrite rounds used      : 17  (47%)

run              starved  families  elapsed_h  of 12 h
20260902-113144        3         4       2.58      22%
20260903-210650        2         2       1.98      16%
20260905-071312        4         4       2.06      17%
20260902-140823        2         2       6.81      57%
20260902-213608        2         2       1.22      10%
20260905-091705        2         4       6.24      52%
20260905-010737        2         4       6.08      51%
```

**Nearly half of every family the generator produced was seeded, parameterized, tuned once, and
then never rewritten** — the entire outer loop, which is the paper's mechanism, never touched
them. And no run has ever used more than 57% of its wall clock; three used under 25%.

`20260905-071312` is the starkest: 4 of 4 families starved, run over at 2.06 h of 12 h. Loop C
effectively did not run.

The root cause is recorded in `memory: opop-v2-run-stops-with-budget-unused` — `_rewrite_round`
returns `progressed=False` when the families it *can* act on are empty, the orchestrator then
tries `_novelty_round`, which (defect 0) always returns `False`, so every remaining active family
is force-frozen as `frozen_budget` (`orchestrator.py:440-442`). **Defects 0 and 8 compound**:
loop D being unreachable is what turns "nothing to rewrite this round" into "end the run".

This also means raising `rewrite_rounds_per_family` to 5 (defect 0b) will *not* help these
families — they never reached round 1. The two fixes are independent and both are needed before
the wall-clock budget is actually spent.

Not fixed: the interaction sits in the orchestrator's loop-control flow rather than in a budget
number, so it needs a design decision about what "no family made progress" should mean when
`max_families_active` is throttling which families are eligible.

### Defect 0c: raising the round cap also un-blocked `stop_kind = "converged"`

An unplanned but welcome side effect, caught by an existing test that was written to detect it.
`converged` requires `len(best_history) >= no_improve_rounds + 1` = 3, while budget froze at
`rewrite_rounds_used >= 3` — the same round. The budget test runs first, so `converged` could
never be emitted: **13 of 13 freezes were `budget_exhausted`, 8 of them with completely flat
histories** like `[25.2, 25.2, 25.2]`, which misreports "there may be headroom left" when there
is none. With the cap at 5, a flat family reaches `converged` two rounds before the budget test.
See `finding-converged-stop-kind-is-unreachable.md`. Not yet observed live.

## 4. Next-step improvements, cheapest first

Each is generic — none is special-cased to a task, knob, or model.

1. **Veto expansions that widen past a failing edge.** `failure_rate_by_value` is already computed
   in `TuningStats`; skip a widening whose edge choice fails above a threshold. Reads existing
   data, no new measurement. Addresses defect 3 and part of 2.
2. **Classify no-dot kernels as `ieee_fp32`** rather than `unknown`. One branch in
   `_detect_candidate_precision`. Reporting only; cannot change what runs. Defect 6.
3. **Report the honest same-precision verdict above the four raw speedups.** Presentation only.
4. **Make the minimal witness a *legal* corner rather than `choices[0]`** — e.g. the cheapest
   configuration that the default's dtype admits, so a task with large outputs is not forced
   through fp16. Defect 4. Needs care: it changes what the gate accepts, so it must be validated
   against the historical rejections before it is trusted.
5. ~~**Investigate the 3.3% cache hit rate.**~~ Done, and it is **not** a defect worth fixing.
   The 3.3% is near its ceiling: only 304 of 8204 trials (3.7%) repeat a parameter set for the
   same candidate at all, because TPE rarely proposes duplicates. Of those 304 repeats, 129 were
   served from cache and **175 were re-measured**, costing **34.4 min = 0.7% of the 83.7 h**.
   Real but small. (`space_id` is `None` on `TRIAL_DONE`, so whether a repeat crossed a space
   version could not be determined — worth fixing if this is ever revisited, since it is the
   field that would say whether re-tunes or anchors are the source.)

Deliberately **not** proposed: early pruning / greedy seed selection, cross-candidate report
sharing, and dtype bans — all three were ruled out by the user. The change in
`finding-unreachable-correctness-gate.md` also stays unimplemented per the user's explicit
decision ("问题4现阶段不要实施, 危险性过大").

## 5. Checked and NOT defective

Recorded so these are not re-investigated, and so the analysis is not read as implying more is
broken than is:

* **The analyst -> rewriter channel works.** 216 `BOTTLENECK_REPORTED` events, 207 carrying
  hypotheses and 136 carrying `parameter_limits`, and the full report is written into the
  rewriter's sandbox as `analysis/bottleneck.json` (`agents/modules.py:684`). The analyst's 2.4%
  of wall clock buys the rewriter's actual input.
* **`suggested_action` being advisory-only is correct, not a bug.** It is read in exactly one
  place — `reporting/report.py:329`, which prints it — and by no control logic. That matches the
  design decision that the harness decides and the agent only suggests. Worth noting though that
  the analyst asked for `tune_more` 25 times and `stop` 7 times (against `rewrite` 184), and every
  one of those was overridden by the round counter rather than considered.
* **The fp64 relative gate is the single most effective fix in the project so far.** It was
  enabled for the first time in `run-l3-43-20260906-091019`, and the same task's trial failure
  rate fell from 45.9% to **13.5%** while the run did 50% more trials than any before it:

  ```
  run                        fp64_gate  trials  fail%  witness_rej  rescued trials
  run-l3-43-20260902-140823       off      152  17.1%            9        0
  run-l3-43-20260903-020233       off      418  16.5%            0        0
  run-l3-43-20260903-145357       off      456  25.9%            7        0
  run-l3-43-20260904-093730       off      912  44.7%            3        0
  run-l3-43-20260905-091705       off     1040  45.9%           10        0
  run-l3-43-20260906-091019        ON     1560  13.5%            2      663
  ```

  **663 trials were rescued, and all 663 are `complete`** — each would otherwise have failed the
  absolute gate. That is 42.5% of the run's trials. Witness rejections also fell 10 -> 2.

  (Reading note: the event field `fp64_rescued_trials` sums to 1989, but it counts *correctness
  reps*, not trials — every rescued trial reports exactly 3, one per rep. 663 is the trial count.)

  This also revises defect 2 downward for any future run: the 22.4% pooled failure rate is
  dominated by the five runs without the gate. The live rate is 13.5%.
* **The trial-measurement cache is near its ceiling** (defect 9 above): the 3.3% hit rate looks
  alarming, but only 3.7% of trials repeat a parameter set at all.
* **Resource metrics do not predict per-value failures** — see the negative result below. The
  pooled table looks exactly like the predictor one would want, and it does not survive being
  asked the question that matters.

## 6. What this analysis does not yet cover

* **The GLM arm has no data.** Five of six agent modules have never run on glm-5.3, so nothing
  here is validated across models. `run-l3-21-20260906-084636` died at its first call.
* **L3:43's live run is unfinished**, so defect 3's numbers (24 widened knobs) will grow.
* ~~**No per-task attribution of the 22.4% trial failure rate** to specific resource limits.~~
  Measured, and the answer is **negative — do not build the defect-3 veto on resource metrics.**

  Pooled across all runs, the profile metrics of the *succeeding* trials look beautifully
  predictive: group every (candidate, knob, value) by its failure rate and the shared-memory
  usage rises monotonically with it.

  ```
  failure rate   n knob-values   median regs   median shared   median spills
  0%                      974           255          40960               4
  1-24%                  1124           255          73728             280
  25-49%                  966           255          81920             137
  50%+                    619           255          86016              44
  ```

  That table is misleading in two ways. `n_regs` is **255 in every bucket** — the hard limit, so
  it carries no information at all; and `n_spills` is non-monotone (280 in the low-failure bucket,
  44 in the highest), so it is not a signal either.

  More importantly the shared-memory trend **does not survive being asked the question that
  matters**. The veto would have to rank values *within one candidate's knob*, and there:

  ```
  knob axes with >=3 measurable values                         : 935
  highest-shared value also has the higher failure rate        : 458 (49%)
  ```

  **49% — a coin flip.** The pooled monotone trend is an artifact of aggregating across
  candidates (kernels that use more shared memory are also more fragile overall), and it says
  nothing about which value of a given knob will fail.

  So defect 3's veto should stay keyed on the thing that is measured directly —
  `failure_rate_by_value`, where the observed 43%-vs-15% split lives — and **not** be "upgraded"
  to a resource model. This is worth recording precisely because the pooled table looks like
  exactly the predictor one would want.
* **`novelty` never fired in 18 runs** — see defect 0. Whether loop D earns its budget cannot be
  measured until the gate is opened.
