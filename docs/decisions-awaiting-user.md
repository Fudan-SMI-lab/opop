# Decisions awaiting the user — consolidated, with what each one blocks

Five items are recorded as "not fixed, needs a decision", spread across separate findings.
They have accumulated over two days and three of them now interact, so this collects them
in one place with the evidence strength and the cost of leaving each as-is.

**Nothing here has been implemented.** Each entry says what I would do and why I stopped.

---

## 1. The correctness gate is a fixed 0.99 on tasks whose references self-agree less

**Evidence: strongest of the five, and it grew again on 09-05.**

`scripts/audit_noise_floors.py`, machine-verified across all runs:

```
task    n   floor frac    gate 0.99 sits above the floor by
L3:21  98   0.955360      +0.0346
L3:43  36   0.976682      +0.0133
L3:48  15   0.977767      +0.0122
```

All three tasks. Each floor stable to six decimals. The floor is the reference's own
ieee-vs-tf32 disagreement — the harness already measures it at witness time.

The sharpest case (`result-l3-43-bf16-rejected-above-the-floor.md`): an L3:43 bf16
candidate matches the ieee reference at **0.987266** while the reference matches its own
tf32 self at only **0.976682** — the candidate agrees with the reference *better than the
reference agrees with itself* — and is rejected. Cosine 0.99999996 against a 0.99985
requirement.

**The decisive new evidence: the verdict is dtype-independent, and only the floor predicts
it.** Four groups within L3:43's 09-05 run, one task, one reference, one measurement path:

| group | trials | best-arm frac | vs floor | gate is |
|---|---|---|---|---|
| bf16 on `cand-6476b4cb` | 17 | 0.987266 | +0.0106 | **wrong** |
| bf16 on `cand-3bf724d6` | 9 | 0.985624 | +0.0089 | **wrong** |
| tf32 on `cand-cb7be6b4` | 3 | 0.963953 | −0.0128 | right |
| bf16 on `cand-cb7be6b4` | 5 | 0.811958 | −0.165 | right |

26 above the floor, 8 below, with a 0.17 gap between the nearest members of each side on
the *same* dtype. This kills the two cheaper alternatives outright:

- **A dtype rule** ("ban bf16") would discard 26 correct results *and* keep 5 badly wrong
  ones — bf16 appears on both sides. This is independent confirmation that the user's
  earlier call against a dtype-ban was correct, reached from the data rather than from
  caution.
- **Trusting cosine** fails too: all four groups pass the cosine arm, including both
  genuinely-wrong ones (0.99999531, 0.99999987). Cosine cannot see these errors.

**Cost of leaving it:** on L3:48, 8 of 8 tensor-core candidates rejected while 7 of 7
scalar published (`result-every-tensor-core-candidate-was-rejected.md`); on L3:43's 09-04
run, all **154** bf16 mismatch trials are now traced to two `max_abs_diff` clusters that
both reproduce above the floor today (123 + 31 = every one of them), while today's
below-floor value appears there zero times. Repair budget is spent "fixing" kernels that
are not broken, and the fix sometimes makes them worse (0.976 → 0.844 + NaN on L3:48).

**What I would do:** compare against `max(threshold_floor_relative, cosine_arm)` where the
floor is the already-measured per-task value, keeping cosine unchanged.

**Why I stopped, and the honest argument against my own position:** the user deferred this
twice, explicitly ("问题4现阶段不要实施, 危险性过大"), and that concern is not answered by any
number above. A floor-relative gate admits *anything the reference cannot resolve*, so on a
task with a loose floor it weakens the correctness guarantee by exactly that much — and a
wrong kernel that happens to land inside the floor would be published, not repaired. What
the evidence establishes is narrower than "the gate should be loosened": it establishes that
the current rule's verdicts **do not track accuracy** — they track whether the task's
arithmetic happens to be well-conditioned. Both facts are true at once, which is why this
stays a decision rather than a fix.

---

## 2. Empty families end the outer loop early

**Evidence: n=2, on opposite sides of a threshold — mechanism confirmed.**

`_rewrite_round` freezes a family with `best is None` **without setting `progressed`**, and
the outer loop reads `not progressed` as "nothing left to do", freezing everything.
With `max_families_active: 2`, two empty families fill both slots and end the run.

| run | empty families | rounds used | elapsed |
|---|---|---|---|
| L3:48 09-05 | 1 | `[3, 1, 0, 3]` | 6.08h |
| L3:21 09-05 | **2** | `[0, 1, 0, 1]` | **2.05h of 12h** |

L3:21's two empty families were activated for the first time at 09:16:16 — the same second
the run ended. Its winning family was frozen with 2 of 3 rounds left while still improving
24% per round (20.50 → 15.50 inside one round).

**Cost of leaving it:** L3:21's headline result came from roughly one third of the intended
structural search. Any future run with two failed seed families stops early the same way.

**What I would do:** make the loop's continue-condition depend on whether any family is
*eligible* for another round, not on whether this pass attempted one; and/or admit a family
to the active cap only once it has a correct candidate. **Why I stopped:** it interacts
with #3 — both change how rounds are allocated and counted.

---

## 3. `stop_kind = "converged"` is unreachable

**Evidence: 13 of 13 `family_verdict` freezes are `budget_exhausted`; `converged` has
never fired on any L3 run.**

(13, not 15 — I mis-stated this earlier. L3:21 09-05's four families were frozen by the
outer loop's cleanup path in #2, not by `family_verdict`, so they emitted no freeze
decision at all. That is itself a symptom of #2: when the loop exits early, the convergence
policy never gets to judge those families.)

Two compounding off-by-ones (`finding-converged-stop-kind-is-unreachable.md`):
`best_history` excludes the seed, and the budget check runs before the improvement check.

**Cost of leaving it:** the run cannot distinguish "this structure is exhausted" from "we
ran out of rounds", so the report's `stop_kind` carries no information. Six families with
completely flat histories were still labelled `budget_exhausted`.

**What I would do:** seed `best_history` with the seed candidate's tuned latency.
**Why I stopped:** it changes budget semantics — with the seed included, the
`no_improve_rounds: 2` window closes one round earlier, so runs would freeze *sooner*. That
interacts with #2 (which makes runs stop too early already) and with the 2.1% retune swing
against `min_improvement_pct: 2.0`, which means noise alone can trip the converged test.

---

## 4. The parameterizer can silently revert the repair it was handed

**Evidence: 4 of 79 hand-offs, and the consequence is larger than the count.**

All four are on L3:21, all split-precision fp16 fixes, and they consumed **both**
tensor-core candidates' entire repair budgets — every attempt measured a source whose fix
had been removed (`finding-parameterizer-reverts-the-repair.md`).

**Newly discovered today:** those two candidates are exactly the seeds of the two empty
families in #2. So this bug did not merely waste ~11.8 min of agent wall — it ended the
L3:21 run about 10 hours early via #2.

**What I would do:** (a) journal a `REPAIR_REVERTED` event when the parameterizer's output
has fewer `tl.dot` than its input — purely observational, ~15 lines; (b) add one sentence to
the parameterizer prompt: preserve the body, you are routing constants. **Why I stopped:**
(a) is safe and I could have done it; (b) changes an agent prompt whose output feeds
acceptance. The deeper fix — letting a precision branch declare its term count in the
contract — is a contract change and clearly a decision.

*(This is the one item where I think the observational half, (a), is worth doing regardless.
Say the word and it is ~15 lines.)*

---

## 5. Per-sample timing retention

**Evidence: the L3:48 best went −6.2% from `tuned_ms` to `final_reeval_ms`**, against the
usual +1.5–6.7% optimism, and the explanation (one slow sample diluted over 100 draws vs 20)
is unverifiable without the per-sample values.

**Cost of leaving it:** the re-eval gap's *direction* stays unpredictable, so every
"beats the baseline" claim needs the full re-eval to settle — which is correct but slow.

**What I would do:** keep the raw sample vector in the job output alongside the summary
stats. **Why I stopped:** it is worker-side and changes the job output schema, so it would
take effect mid-run on a running experiment, and it grows every event payload.

---

## A sixth, smaller one surfaced today

`request_timeout_s` is global at 20 min, but **no successful agent call on any module has
exceeded ~10 min** while repair times out on 9.1% of calls (2.3h of wall lost across 10
occurrences, all recovered). A ~12-minute wall would detect a stuck request 8 min sooner
with no observed call at risk — but the setting is global and the generator's median is
already 8.0 min, so the right shape is a per-module timeout, which is a config-schema
change (`measurement-repair-agent-transport-timeouts.md`).

---

## Suggested order, if it helps

1. **#4(a)** — observational, ~15 lines, no semantics change, and it would have saved a day
   of misattributed analysis.
2. **#2** — the clearest defect with the largest measured cost (10 hours), and it does not
   touch acceptance.
3. **#1** — strongest evidence, biggest effect on results, highest risk. Wants the most
   thought, and it is the one you have twice said to leave alone.
4. **#3** with **#2** — same subsystem; deciding them together avoids a run that neither
   converges nor exhausts sensibly.
5. **#5** and **#6** — efficiency and instrumentation; no result depends on them.

Everything above is recorded in its own finding doc with the run IDs and event evidence.
The acceptance path, the tuner, and the convergence policy are all untouched.
