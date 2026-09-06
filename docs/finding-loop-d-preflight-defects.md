# Loop D (novelty) pre-flight audit: three defects beyond the budget interlock

**Status: D1/D2/D3 FIXED 2026-09-06**, plus `max_families_total` 3 -> 6 and
`max_families_active` 2 -> 3. **The D2 fix then introduced D4 (§D4 below), fixed the same
day.** Verification runs recorded at the bottom.

Written while running the L1 novelty smoke, before enabling Loop D on L3.
Every claim is from reading the shipped code plus the 19 runs on disk.

**Context**: Loop D had never executed **on an L3 run** (`origin:novelty` = 0, `module=novelty`
agent calls = 0 across all 14 completed L3 runs), which is what matters because L3 is the
experiment. It *had* run once on an L1 smoke: `run-l1-19-20260902-093018` (2026-09-02) called
the agent and accepted `cand-16c51d16` into `fam-e13bb739`, because that config used
`seeds=1, total=2` and so cleared the interlock. An earlier version of this document said
"never executed, in any run" and put the first execution on 2026-09-06 — that generalized an
L3 finding to all runs and was wrong; the Sept 2 call is in `events.jsonl`. What 2026-09-06
was actually first for is Loop D at the **shipped L3 budget shape** (4 seeds), and the D1/D2/D3
defects below are all still real. The known blocker was the budget interlock
(`max_seed_candidates: 4` >= `max_families_total: 3`). Raising the cap opens that gate — but
it opens it onto a path with three further problems, two of which end runs early.

---

## D1 — The two novelty gates use different rules, and the stricter one runs first

**Severity: medium. Fix: 1 line. Risk: low.**

`_novelty_round` (outer, decides whether to CALL the agent) counts every family:

```python
if len(self.deps.families.families) >= self.cfg.budgets.max_families_total:
    return False
```

`accept_novel_seed` (inner, decides whether to KEEP the result) counts only productive ones:

```python
if self.productive_family_count() >= self.max_families_total:
```

and `productive_family_count()` exists specifically to exclude dead families:

> *"A family that was dropped (nothing correct, already frozen) is dead and should NOT
> consume a slot — otherwise a batch of failed seeds permanently blocks novelty exploration
> (the level3:43 failure, improvement E)."*

So improvement E was implemented in the inner gate but not the outer one that runs first.
The outer gate counts the dead families, returns False, and the rule written to prevent this
never gets consulted.

Measured — the gates disagree in **5 of 14** completed L3 runs, and in 4 of those the inner
rule would have allowed novelty while the run ended with most of its clock unspent:

```
run                        productive  inner  outer    elapsed
run-l3-21-20260902-113144            1  ALLOW  BLOCK    2.57h / 12h  (21%)
run-l3-21-20260903-210650            0  ALLOW  BLOCK    1.97h / 12h  (16%)
run-l3-21-20260905-071312            2  ALLOW  BLOCK    2.05h / 12h  (17%)
run-l3-43-20260902-213608            0  ALLOW  BLOCK    1.22h / 12h  (10%)
run-l3-48-20260905-010737            3  BLOCK  BLOCK    6.08h / 12h  (51%)
```

**Two runs had ZERO productive families** — every seed died — and still could not reach
novelty. That is exactly the scenario improvement E names.

**Fix**: use one rule.

```python
if self.deps.families.productive_family_count() >= self.cfg.budgets.max_families_total:
    return False
```

`accept_novel_seed` still applies both this and the hard cap, so the inner gate stays the
authority; the outer one merely stops being stricter than it.

**Risk**: novelty can now be called on runs where every seed died. That is intended, but it
spends an agent call plus a pipeline on a task four seeds already failed. Bounded by
`max_families_total_hard` (6) and the wall clock.

---

## D2 — One failed novelty attempt ends the whole run

**Severity: high — this is the one that will bite on L3. Fix: small. Risk: low.**

The outer loop:

```python
progressed = self._rewrite_round(round_no)
if not progressed:
    added = self._novelty_round(round_no)
    if not added:
        for fam in self.deps.families.families.values():
            if fam.status == "active":
                fam.status = "frozen_budget"
        continue
```

and `global_verdict` freezes the run when nothing is active:

```python
if families and all(f.status != "active" for f in families):
    return ConvergenceDecision(scope="global", verdict="freeze", ...)
```

So `added=False` freezes every remaining family, and the next iteration ends the run —
however many hours remain. Today that is harmless, because `added` is always False at the
interlock *before* anything is frozen that mattered. Once Loop D is live it is not.

The decisive detail is that **the last novelty attempt always fails**, and it is the OUTER
gate that fails it. Each accepted family raises `len(families)`, so with seeds=4 and
total=6 (simulated against the shipped bookkeeping):

```
attempt 1: ACCEPTED, len(families) now 5
attempt 2: ACCEPTED, len(families) now 6
attempt 3: OUTER gate fires (len=6 >= 6) -> agent NOT called, added=False
           -> freeze ALL active families -> next global_verdict ends the run
```

So the run cannot end any way *except* through a novelty attempt that returns False, and
every such ending is premature by construction. (An earlier draft of this section attributed
the final rejection to `accept_novel_seed`'s `family_budget` check; the simulation shows the
outer gate gets there first, which matters because it means the fix must sit in the caller's
handling of `added=False`, not in the rejection reasons.)

Other routes to `added=False`, with measured likelihood:

| route | measured |
|---|---|
| `family_budget` after the last accept | **certain**, once D is enabled |
| `AgentCallError` | 0–1.1% per module (0/19 generator, 0/74 rewriter) |
| `too_similar` (>=0.85) | 1.9% of genuinely-different pairs (2/108 seed pairs) |
| `duplicate_signature` | not observed |

The similarity gate is **well calibrated** and is not the problem: across 108 pairs of
deliberately-different seed anchors, max similarity is 0.236–0.898 and only 2 pairs reach
0.85. I expected this to be the main rejection risk and it is not.

**Fix**: `added=False` means "no NEW family this round", not "the search is over". The sweep
is simply removed. That is safe because `progressed=False` already implies every active
family either

* froze inside `_rewrite_round` (its own `family_verdict`, or `best is None`), or
* incremented `rewrite_rounds_used` without rewriting (a missing bottleneck report),

so the first group is already frozen and the second is bounded — the counter grows on every
pass until `family_verdict` freezes it on the round cap. Termination therefore comes from that
counter plus the wall-clock check at the top of the loop, and `global_verdict` still ends the
run the moment nothing is active. The shipped code keeps only a diagnostic
`OUTER_LOOP_EXHAUSTED` event for the genuinely-nothing-rewritable case.

**Risk considered and resolved**: my first draft of this fix froze families in *both*
branches (with a "spin guard"), which was no fix at all — it did the same thing by two
routes. The spin concern it was guarding against does not exist, because the unrewritable
path increments `rewrite_rounds_used`, so an iteration can never be identical to its
predecessor forever.

---

## D3 — Novelty's step key is not resume-safe

**Severity: low (only affects resumed runs). Fix: 1 line. Risk: very low.**

```python
key = f"novelty:{round_no}"
```

`round_no` is a local in `_run`, initialised to 0 and never restored —
`_restore_family_control_state` rebuilds `best_history`, `rewrite_rounds_used` and
`failed_hypotheses`, but nothing rebuilds `round_no`. So after a resume the counter restarts
and re-derives `novelty:1`, `novelty:2`, … If `novelty:1` is already in `steps_done`,
`_novelty_round` returns False immediately — which, per D2, freezes everything and ends the
run.

Contrast `_rewrite_round`, which keys on replayed state and is therefore correct:

```python
key = f"rewrite:{family.family_id}:{family.rewrite_rounds_used}"
```

**Fix**: key novelty on persisted state — the number of `novelty:*` steps already in the log:

```python
key = f"novelty:{sum(1 for k in state.steps_done if k.startswith('novelty:'))}"
```

No "already done, skip" check accompanies it, and that is deliberate: the count is derived
from the log, so the key is by construction one the log does not contain. Idempotency comes
from the count itself — N recorded attempts means the next is numbered N, and the work of
attempts 0..N-1 is already reflected in the families rebuilt on resume. Re-running an attempt
whose acceptance is recorded is prevented by the D1 gate, since an accepted family raises
`productive_family_count()`.

`round_no` is kept as a parameter and now recorded in a new `NOVELTY_ROUND_STARTED` event
(with the productive/total family counts), so an attempt can still be tied to the outer
iteration that triggered it — which is what made D1 and D2 legible in the log at all.

---

## D4 — My D2 fix made the outer loop non-terminating (introduced and fixed 2026-09-06)

**Severity: high — it burned the whole verification run. Found by running it.**

The D2 fix removed the blanket sweep. My comment justifying that removal claimed an
exhaustive case analysis:

> *"`progressed=False` already means every active family either froze inside
> `_rewrite_round` (its own verdict, or `best is None`) or incremented
> `rewrite_rounds_used` without rewriting."*

**Both branches of that "either/or" live inside `_rewrite_round`'s `for` loop, and that loop
iterates `active_families()` — which excludes families whose `best is None`.** So the one
case the sweep uniquely covered is the case my analysis assigned to a code path that cannot
reach it. `active_families()` says so in its own docstring, which I had read and quoted
elsewhere in this document:

> *"Empty families are still frozen (in `_rewrite_round`, when reached, **and by the outer
> loop's sweep**)"*

I removed the second half of that sentence's mechanism while citing the first half as
sufficient.

**What it cost.** `run-l1-19-20260906-192759` — the run started to *verify* D1/D2/D3:

```
fam-3fd4ad41  frozen_budget   (seed, tuned to 251.0 ms)
fam-53f42e15  frozen_budget   (novelty #1, tuned to 246.0 ms)
fam-92c506b3  active          <-- novelty #2; space rejected twice, so best is None
```

With that third family pinned `active` forever:

* `global_verdict` sees an active family → `continue`
* `productive_family_count()` counts it (it tests `status == "active" or best is not None`)
  → the novelty gate refuses
* `active_families()` filters it out → `progressed = False`

Two events per iteration, no GPU work, no agent calls: **2,054,908 iterations in 13 minutes,
4,109,882 events, 991 MB**. The log has been trimmed to its 104 real events plus 40 spin
events; see that run's `TRIMMED.md`.

**Why the tests passed.** Two of the six D-tests re-implemented the outer loop's post-miss
handling *inside the test body* rather than calling the loop:

```python
    # Replicate the outer loop's post-miss handling as shipped.
    if not orch.deps.families.active_families():
        for fam in fm.families.values():
            if fam.status == "active":
                fam.status = "frozen_budget"
```

That copy still froze the family, so the tests asserted termination against a loop that no
longer existed. A test of a loop has to execute that loop. Both now drive the shipped code
through a `_drive_outer_loop` helper with a hard iteration ceiling, so a non-terminating loop
fails the test instead of hanging it.

**Fix, in two parts:**

1. **The cause** — `_freeze_unrewritable_families()`, called at the top of `_rewrite_round`,
   freezes any active family with `best is None`. This puts the decision where the
   un-rewritable predicate already is, so it holds however the caller is written — rather
   than depending on a sweep that also destroyed families with budget left (D2). Emits
   `FAMILY_FROZEN_UNREWRITABLE`.
2. **The class of bug** — an `idle_rounds` counter in `_run`. Three consecutive rounds that
   neither rewrite nor add a family end the loop with an `OUTER_LOOP_STUCK` event carrying
   every family's status. Every legitimate ending already comes from `global_verdict` or a
   family's own freeze, so reaching this bound means a defect — and the bound makes the next
   such defect cost a handful of events instead of hours of clock. A correct run can have
   **zero** consecutive barren rounds: a round that changes no status and adds nothing is by
   construction identical to its predecessor.

Part 2 is the part I should have written the first time. The D2 reasoning was a proof sketch
in a comment, and I shipped it as if a proof sketch were a test.

Tests: `test_a_family_with_no_rewrite_parent_gets_frozen_rather_than_spinning` (reproduces
the exact three-family live shape; fails without part 1) and
`test_the_outer_loop_cannot_spin_even_if_a_family_never_freezes` (monkeypatches the freeze
into a no-op, so only the guard can end the loop).

**Verified live — `run-l1-19-20260906-202221`, 38.0 min, `RUN_FINISHED` reached:**

```
TUNING_DONE            cand-fd0c81b9  251.0 ms        (seed)
CONVERGENCE            family freeze budget_exhausted fam-9004456c
NOVELTY_ROUND_STARTED  key=novelty:0  productive=1    <- D1's count, D3's log-derived key
CANDIDATE_REGISTERED   cand-64d201e0  origin=novelty  fam-cc8854cd
TUNING_DONE            cand-64d201e0  249.0 ms        (the novel family WON)
CONVERGENCE            family freeze budget_exhausted fam-cc8854cd
NOVELTY_ROUND_STARTED  key=novelty:1  productive=2    <- D2: pre-fix the run ended HERE
NOVELTY_REJECTED       too_similar                    <- a real miss, not the interlock
OUTER_LOOP_EXHAUSTED   round 2                        <- ONE of them, not 2,054,908
CONVERGENCE            global freeze budget_exhausted <- terminated
```

Event counts, before the D4 fix vs after, for the same config:

| | 192759 (spun) | 202221 (fixed) |
|---|---|---|
| `CONVERGENCE_DECIDED` | 2,054,912 | **5** |
| `OUTER_LOOP_EXHAUSTED` | 2,054,908 | **1** |
| `OUTER_LOOP_STUCK` | n/a | **0** (the guard was not needed) |
| total events / size | 4,109,882 / 991 MB | 58 / 40 KB |

Both families froze through their own `family_verdict`, so `_freeze_unrewritable_families`
correctly did nothing — both had a correct incumbent. The guard never fired, which is the
right outcome: it is a backstop, and the real fix ended the loop before it was reached.

`scripts/audit_loop_d_fix_effect.py` on the pair: novelty calls 1 -> 2, the second attempt
reached at all being the D2/D3 proof. And the novel family produced the run's winner
(249.0 ms vs the seed's 251.0 ms), which is the first direct evidence that Loop D does what
the paper claims for it.

---

## What I checked and found NOT defective

* **The 0.85 similarity gate** — well calibrated, see D2's table. Do not loosen it.
* **`max_families_active: 3` does not lengthen a run.** Rounds are serial
  (`_rewrite_round` iterates families, each `_do_rewrite` blocks on the GPU) and the total
  available is `max_seed_candidates * rewrite_rounds_per_family` = 20 either way. At a
  38.8 min median a 12 h budget affords ~18, so **wall clock binds in both settings** —
  `run-l3-21-20260905-195615` already overshot at 12.82 h with active=2. active=3
  redistributes rounds (more families reach a first round, each gets fewer) rather than
  adding work. This corrects an earlier estimate of mine that predicted +2.6 h.
* **Novelty sandbox seeding and prompt rendering** — exercised directly: seeds
  `task/ref.py`, `docs/{candidate_contract,triton_pitfalls,device}.md`,
  `task/eval_semantics.md`, and per-family `anchor.py` + `summary.json`; prompt renders to
  1350 chars with the right file references. `docs/device.md` correctly carries
  "RTX 5080 Laptop (sm_120)" rather than the old "unknown".
* **Loop D can fire repeatedly when it succeeds** — an accepted family becomes active, so
  the next `_rewrite_round` returns True and D is only re-entered when that family is
  exhausted. The repeat path is fine; only the failure path (D2) is broken.

---

## Recommended order

1. **L1 novelty smoke** — proves the live agent call, schema and accept path work at all.
2. **D2** — the only one that will visibly damage an L3 run, and it is guaranteed to fire.
3. **D1** — one line, and it is what makes novelty reachable in the runs that need it most.
4. **D3** — one line, resume-only exposure.
5. Then raise `max_families_total`.

## What was done (2026-09-06)

All five steps, in that order.

**Step 1 — smoke passed.** `run-l1-19-20260906-183211`, 28 min: candidate `cand-afb67e8b`
accepted into new family `fam-eb8bd915`, parameterized, witnessed and tuned. The whole path
works. (This was the *second* `module=novelty` call on disk, not the first — see Context;
`run-l1-19-20260902-093018` beat it by four days. The smoke is still what exercised the path
end to end under observation.) That run also produced D2's live evidence (below) and a
candidate that computed no
ReLU at all — investigated separately in
`docs/finding-relu-memcpy-is-a-benchmark-property.md`, where it turned out to be a property
of KernelBench's own `get_inputs()` rather than a framework defect, and **not** a blocker for
Loop D on L3 (the ReLU6 sites in L3:21 clamp ~50% of their elements, so the substitution has
no purchase there).

**Steps 2-4 — D2, D1, D3 fixed** in `control/orchestrator.py`, with tests:

| test | what it pins |
|---|---|
| `test_the_novelty_gate_counts_productive_families_not_corpses` | D1: 4 dead families no longer refuse the call |
| `test_the_novelty_gate_still_refuses_once_productive_families_fill_the_budget` | the looser count is still a count |
| `test_a_novelty_miss_does_not_freeze_families_that_still_have_budget` | D2: 3 families with 4 of 5 rounds left survive a miss |
| `test_the_outer_loop_still_ends_when_nothing_is_rewritable` | D2 did not remove termination |
| `test_the_novelty_step_key_survives_a_resume` | D3: a resumed run advances the counter |
| `test_loop_d_is_reachable_at_the_shipped_l3_budget` | step 5: seeds < total in both L3 configs |

**Step 5 — budgets raised.** `max_families_total` 3 -> 6 (4 seeds + room for 2 novel
families, matching `max_families_total_hard`), `max_families_active` 2 -> 3, in `config.py`
and all three shipped configs. 251 tests pass.

A note on `max_families_active`: raising it does **not** lengthen a run. Rounds are serial
(`_rewrite_round` iterates families, each `_do_rewrite` blocks on the GPU) and the total
available is `max_seed_candidates * rewrite_rounds_per_family` = 20 either way; at a 38.8 min
median a 12 h budget affords ~18, so wall clock binds in both settings —
`run-l3-21-20260905-195615` already overshot at 12.82 h with active=2. It redistributes
rounds rather than adding them. This corrects an earlier estimate of mine that predicted
+2.6 h.
