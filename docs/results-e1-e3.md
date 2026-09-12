# E1 / E3 results — verified from on-disk `events.jsonl`

**Read** 2026-09-12 12:2x · every number below comes from parsing `events.jsonl` on the box that
produced it, never from a notification or a report tool (`report-tool-conflates-running-with-crashed`).
**All three runs are finished**, each with `RUN_FINISHED` on disk and 0 orchestrator processes.

**Two monitor defects bit during this batch, in opposite directions, from the same one-line mistake.**
A finish-detector using `grep -c RUN_FINISHED || echo 0` reported boxes 2 and 3 finished when neither
was; `watch_run.sh`'s probe using `pgrep -c ... || echo 0` could never report ENDED at all, and
called box 2 "alive but stalled" 37 minutes after it had finished. Both because those commands print
`0` *and* exit 1 on no match, so the fallback appends a second `0` and the variable becomes `"0\n0"`.
Fixed in `e9e7c8e` with three-way verification. Every figure below was re-derived by parsing the
`type` field — which is why the rule is to verify against disk.

---

## 1. Box 2 — E1 treatment arm (`reserve_round_for_reconciled: true`), L3:43 on a 4090

`run-l3-43-20260911-230736` · **finished normally at 12.346 h**, `RUN_FINISHED` on disk, 0
orchestrator processes remaining · 12 rewrites, 4 families, 1536 events.

### Verified best

| | value |
|---|---|
| winner | `cand-eb28dbe5`, family `fam-a47a797a` |
| origin | **rewrite**, parent `cand-b010505b` — itself a rewrite of seed `cand-9ccccb91` ⇒ **second-generation rewrite** |
| `tuned_ms` | 2.9471 ms |
| **`final_reeval_median_ms`** | **2.8785 ms** (`final_reeval_ok: true`) |
| reeval vs tuned | **2.33% FASTER** |

The reeval being *faster* than tuned is the opposite direction from
`reeval-gap-is-the-real-number`, which measured tuned as systematically optimistic by 1.5–6.7%.
Worth noting rather than smoothing over: on this run the independent re-measurement was kinder, not
harsher.

### Speedups on the verified number

| baseline | median | speedup |
|---|---|---|
| eager | 21.4528 ms | **7.453x** |
| eager_tf32 | 18.2415 ms | 6.337x |
| torch_compile | 13.9500 ms | 4.846x |
| torch_compile_tf32 | 10.9896 ms | **3.818x** |

**4.45% faster than the previous best on this task** (3.0126 ms, `l3:43 final result 3.01ms`).

### Loop C: 4 of 4 families improved by rewriting

| family | seed | final | improvement | rounds |
|---|---|---|---|---|
| fam-550b5696 | 5.9525 | 3.3372 | **43.9%** | 1 |
| fam-d375b4d4 | 4.8072 | 3.1836 | **33.8%** | 2 |
| fam-a47a797a | 4.1938 | **2.9471** | **29.7%** | 2 |
| fam-75181090 | 3.4821 | 3.0853 | 11.4% | 1 |

**Every family's best is a rewrite**, and 6/6 `FAMILY_ROUND_RECORDED` report
`conversion=improved` (gains 11.4 / 20.8 / 31.7 / 43.9 / 3.1 / 11.2%). The two late small gains
show diminishing but non-zero returns rather than a cliff.

All four families ended `status: active` — the run was ended by the **wall clock**, not by
convergence (`wall-clock-is-always-the-binding-budget`, now 6 for 6).

The winner's stated hypothesis matches what it did: *"both cuBLAS projections become one shared
Triton tensor-core GEMM that fuses the dtype casts the parent paid ~402 MB/call of copy traffic
for"* — i.e. the task-level fusion headroom (`task-cost-fusion-headroom`), acted on.

### S2d ledger — the significance the prior run did not have

12 `EXPECTATIONS_RECONCILED` (6 rounds × 2 candidates, as
`reconciliation-must-be-per-candidate-not-per-round` requires):

```
04:13  3/4/1    04:13  4/4/0     06:23  7/1/0    06:23  6/2/0
07:34  7/1/0    07:34  4/3/1     08:28  7/1/0    08:28  3/4/1
10:00  7/1/0    10:00  7/0/1     11:28  3/2/3    11:28  0/0/0
                                          (hits/misses/vacuous)
```

**Totals: 58 hits, 23 misses, 7 vacuous ⇒ 58/81 = 71.6%** non-vacuous accuracy, vacuous rate
7/88 = 8.0%.

| null | one-sided binomial p |
|---|---|
| 50% coin | **0.00006** |
| 1/3 (up/down/unchanged) | 2.2e-12 |

The prior paired run gave **p = 0.1585**, and "ledger accuracy has no significance" was an explicit
open item. This closes it under either null.

**The statistical caveat, stated rather than buried.** The binomial treats 81 declarations as
independent, and they are not: they come from 6 rounds × 2 candidates on one task with one agent, so
declarations within a round share a prompt and declarations across rounds share a task. The
conservative reading is to count **rounds** rather than declarations — 5 of 6 round-pairs are
hit-majority, which at a 50% null is p = 0.109 and *not* significant. So the honest claim is:
**71.6% per declaration with a strong nominal p, and a directionally consistent but not
independently significant 5/6 at the round level.** A second arm or a second task is what would
settle it, not more declarations from this one.

The low vacuous rate (8.0%) is what A2 was for — a transport timeout that lost the declarations
would have inflated it.

### The confound that must accompany every number above

Box 2 lost **1.00 h** to jobs that produced no output; box 1 lost **4.75 h** (D9). The arms'
*answered* jobs cost the same to within 5% (median 12.9 s vs 13.6 s), so this is a harness artifact,
not a property of either search. Any statement about search *volume* must carry it
(`equal-configs-do-not-imply-equal-search`).

---

## 2. Box 3 — E3, reproduction of L3:48 on the A800

`run-l3-48-20260911-231217` · **finished at 13.159 h** (self-reported 13.152), `RUN_FINISHED` on
disk, 0 orchestrator processes · 6 rewrites, 3 family rounds, 1043 events.

### Verified best

| | value |
|---|---|
| winner | `cand-207eabd1`, family `fam-a5484ff4` |
| origin | **rewrite**, parent = seed `cand-2e142acc` |
| params | `fp16, DOT_MODE=plain, BLOCK_L=16, BLOCK_P=32, NUM_WARPS=1, NUM_STAGES=4` |
| `tuned_ms` | 1.0097 ms |
| **`final_reeval_median_ms`** | **1.0532 ms** (`final_reeval_ok: true`) |
| reeval vs tuned | **4.31% SLOWER** |

**Two corrections to figures quoted while this run was in flight**, kept visible because both were
wrong in the direction that flatters the result:

1. **The verified number is 1.0532 ms, not 1.0097.** I quoted the best in-flight `tuned_ms`. The
   reeval came in **4.31% slower**, which is the direction `reeval-gap-is-the-real-language` predicts
   (tuned is systematically optimistic by 1.5–6.7%). Box 2 went the *other* way on the same day
   (reeval 2.33% faster), so **the sign of that gap is not fixed** — which means neither run's
   direction can be used to predict another's, and only `final_reeval_median_ms` may be claimed.
2. **It ran 13.159 h, not ~12 h.** So "a clean 12 h on this box buys ~1.01 ms" — which I wrote
   earlier — is not licensed. `WALL_CLOCK_REACHED` is 0 and the per-candidate budget check let the
   in-flight candidate finish, giving a **9.7% overrun**. The honest form is "a clean **13.2 h**
   buys 1.0532 ms".

### Speedups on the verified number

| baseline | median | speedup |
|---|---|---|
| eager | 13.9930 ms | **13.286x** |
| eager_tf32 | 13.4758 ms | 12.795x |
| torch_compile | 8.5862 ms | 8.153x |
| torch_compile_tf32 | 8.0609 ms | **7.654x** |

DRAM floor 0.8011 ms ⇒ **76.1% of the physical roofline**; A1's derived ceiling is 17.48x and the run
reached 13.29x.

### Loop C: 3 of 4 families improved

| family | seed | final | improvement | rounds |
|---|---|---|---|---|
| fam-a5484ff4 | 1.8514 | **1.0097** | **45.5%** | 1 |
| fam-aac749d4 | 1.7367 | 1.0245 | **41.0%** | 1 |
| fam-47830987 | 1.1484 | 1.1320 | 1.4% | 1 |
| fam-1b92f176 | 5.4753 | 5.4753 | — | **0** |

The three families that ran a round are all won by a **rewrite**; the best seed anywhere was
1.1484 ms, so rewriting produced the top two results. But the spread — 45.5%, 41.0%, **1.4%** — is
the honest picture: **rewriting is not uniformly productive**, and L3:48 is the task already shown to
sit on a physical plateau (`l3:48 plateau confirmed three runs`).

The winner's hypothesis was taken from the bottleneck report: *"remove the full-T cumsum bookkeeping
and the per-chunk O(LC*T) masked where/sum (~64 regs/thread of intermediate plus 51-barrier /
101-shared-load class reductions)"* — a register/barrier pressure argument, acted on and measured.

### Against the run it reproduces

1.0532 vs **0.9728 ms** ⇒ **8.26% slower**. That is the point of the run, not a failure of it: the
0.9728 came with 1 resume, 2 agent timeouts (0.83 h) and a 13.67 h span while self-reporting
12.383 h, and `a-resumed-run's-own-clock-restarts` means its self-reported hours were never
comparable. E3 has **1 `RUN_CREATED`, 1 agent failure** (parameterizer `ReadTimeout` at 00:01) and a
13.159 h span that matches its self-report to 0.007 h. So E3 replaces an unverifiable number with a
verifiable one — at the cost of the number being larger.


---

## 3. Box 1 — E1 control arm (`reserve_round_for_reconciled: false`), L3:43 on a 4090

`run-l3-43-20260911-230217` · **finished at 12.506 h** (self-reported 12.501), `RUN_FINISHED` on
disk, 0 orchestrator processes · **0 rewrites, 0 family rounds**, 473 events.

### Verified best

| | value |
|---|---|
| winner | `cand-e254236c`, family `fam-bf95261c` |
| origin | **seed** |
| `tuned_ms` | 4.2440 ms |
| **`final_reeval_median_ms`** | **4.0852 ms** (`final_reeval_ok: true`) |
| speedup vs eager (21.5588 ms) | 5.277x |

### Every family is seed-only

| family | best | rounds | history | members |
|---|---|---|---|---|
| fam-bf95261c | 4.2440 | **0** | `[4.244]` | 1 seed |
| fam-676e0814 | 4.8937 | **0** | `[4.8937]` | 1 seed |
| fam-742fa210 | 5.5209 | **0** | `[5.5209]` | 1 seed |
| fam-366e1492 | 5.9715 | **0** | `[5.9715]` | 1 seed |

A `history` of one element is the whole finding: **no round ever ran.** `WALL_CLOCK_REACHED` is 0
because the budget check is per candidate (`orchestrator.py:2790`), so the candidate being tuned when
12 h passed finished its whole space — hence the 12.506 h.

### Why it never entered loop C

Between the analyst call at 02:35:49 and the next agent call at 08:41:46 there is a **6.10 h gap with
no agent call**, containing exactly 40 `TRIAL_DONE` — all for `cand-941ea454` — at 9.2 min/trial.
That is **50.8% of the budget in one tuning pass**.

Localised further: that candidate's K-expansion gave it a second space, and the two are a controlled
comparison — **space 1: 40 trials, 5.99 h, 9.2 min/trial, 6 hung screens; space 2: 34 trials, 1.43 h,
2.6 min/trial, 2 hung.** 3.5x apart, same candidate, same settings ⇒ the pathology is **per
configuration**, not per candidate.

---

## 4. The E1 arm comparison — what the gap does and does not mean

| arm | verified `final_reeval_median_ms` | families | rewrites |
|---|---|---|---|
| control (box 1) | **4.0852 ms** | 4, **all seed-only** | **0** |
| treatment (box 2) | **2.8785 ms** | 4, **all rewrite-won** | 12 |
| | **treatment 29.5% faster** | | |

**This is not the experiment's answer.** The independent variable was
`reserve_round_for_reconciled`, which acts *on a rewrite round*; the control ran zero. So the 29.5%
gap measures **"rewriting happened vs it did not"** — an accident of budget consumption (D8/D9), not
the treatment.

**It is not a box effect either**, which is worth establishing rather than assuming. The two 4090s'
baselines agree to **0.49% on eager and 0.20% on torch_compile_tf32**, so they are equivalent
measuring instruments:

| baseline | box 1 | box 2 | difference |
|---|---|---|---|
| eager | 21.5588 | 21.4528 | 0.49% |
| eager_tf32 | 18.2784 | 18.2415 | 0.20% |
| torch_compile | 13.9755 | 13.9500 | 0.18% |
| torch_compile_tf32 | 11.0111 | 10.9896 | 0.20% |

**What the gap does support**, as an observational rather than controlled comparison: **loop C is
worth its budget.** Four seed-only families at 4.24 / 4.89 / 5.52 / 5.97 ms against four
rewrite-won families at 2.95 / 3.09 / 3.18 / 3.34 ms, same task, same hardware, same settings,
same 12 h. That is the sharpest available answer to `framework diagnosis`'s "rewrites are flat" —
and it arrived as a side effect of the pair failing at its intended purpose.


---

## 4. What this settles, and what it does not

All three runs ended by **wall clock**, none by convergence: every family on every box is
`status: active`. That is now **8 of 8** completed runs, and it keeps
`wall-clock-is-always-the-binding-budget` unbroken. All three also **overran** their budget, by
4.2% / 2.9% / 9.7%, because the check is per candidate (`orchestrator.py:2790`) — so a stated
"12 h" result is really 12.3–13.2 h and must be quoted as the measured span.

**Settled.**
- **Loop C produces the winner on all three runs.** Box 2: 4/4 families improved, winner a
  second-generation rewrite. Box 3: 3/4 improved, winner a rewrite. Box 1: 0 rounds ran and the
  winner is a seed — which is the same point from the other side. This is direct counter-evidence to
  `framework diagnosis`'s "rewrites are flat".
- **But rewriting is not uniformly productive**, and the spread is large: L3:43 gave 11.4–43.9%
  (4/4), L3:48 gave 1.4–45.5% (3/4, one family never entered a round). The task's remaining headroom
  governs it — L3:48 is the known plateau.
- **S2d(a)/(b) has production evidence**: 71.6% of 81 non-vacuous declarations, nominal one-sided
  p = 0.00006, vacuous rate 8.0%.
- **A1's derived ceiling behaved correctly on all three boxes** and flagged nothing; the ceilings
  were 9.15–9.20x (L3:43, compute-bound) and 17.48x (L3:48, DRAM-bound).
- **The reeval gap's sign is not fixed**: box 2's reeval came in 2.33% *faster* than tuned, box 3's
  4.31% *slower*. So only `final_reeval_median_ms` may ever be claimed, and no run's direction
  predicts another's.

**Not settled.**
- **S2d(c) has no control observation.** The control arm ran 0 rewrite rounds, so
  `reserve_round_for_reconciled` is untested. **E1 must be re-run** — a fact on disk, not a
  projection.
- The ledger's significance is per-declaration; at the round level it is 5/6, p = 0.109 —
  directionally consistent, not independently significant. Needs a second arm or a second task.
- Whether E3's 1.0532 ms or the old 0.9728 ms is the better estimate of this box's capability. E3 is
  the verifiable one, but it is also 8.26% slower, and the two differ in budget as well as
  cleanliness.

**Before the re-run:** D9 (`docs/fix-ready-d9-screen-deadline.md`) and D8 option 1
(`docs/fix-ready-d8-job-wall-clock.md`), together in one pass — both change comparability, and D9
alone would not have bought loop C on box 1, since the 6.10 h pass was mostly *answered* jobs.

