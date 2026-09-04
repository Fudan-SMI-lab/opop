# The bottleneck has inverted: agent wall now dominates, and repair owns most of it

Measured on `run-l3-48-20260905-010737` at 4.62h elapsed. Not a proposal — a
measurement, recorded because it changes what the next optimisation should target.

## The numbers

| run | elapsed | agent wall | agent share | repair wall | trials |
|---|---|---|---|---|---|
| l3-43 (09-04, pre-fix-A) | 11.67h | 2.88h | 25% | 1.11h | 912 |
| l3-21 (09-04, pre-fix-A) | 8.10h | 2.10h | 26% | 0.04h | 654 |
| **l3-48 (09-05, post-fix-A)** | **4.62h** | **3.15h** | **68%** | **1.51h** | 414 |

> **Corrected 2026-09-05.** An earlier version of this table reported 33% / 30% / **89%**
> with repair at 2.11h / 1.81h. Those were inflated. The ad-hoc script summed every
> terminal event against the same start, and a call retried after a transport timeout emits
> both `AGENT_CALL_FAILED` *and* `AGENT_CALL_FINISHED` — so the two retried repair calls in
> l3-48 were each counted twice. (`scripts/verify_fixes.py` already keeps one span per
> `call_id` for exactly this reason; the throwaway script did not, which is an argument for
> using the checked-in tool.) Figures above use one span per `call_id`, taking its last
> terminal event. The direction of the finding holds — agent wall went from ~25% to 68% of
> the run — but the magnitude was overstated.

Within l3-48, by module:

| module | calls | wall |
|---|---|---|
| repair | 16 | **1.51h** |
| parameterizer | 34 | 0.98h |
| rewriter | 5 | 0.33h |
| analyst | 11 | 0.20h |
| generator | 1 | 0.13h |
| total | 67 | 3.15h |

GPU + everything else: **1.47h of 4.62h (32%)**.

## Why this happened

Fix A moved the WSL venv off the 9p `/mnt` mount onto ext4, cutting per-job wall from
~29s to ~11.2s — 2.6x on trial jobs, which are 92% of jobs. That was the right fix and it
worked. But it removed the constraint that was hiding the agent cost. GPU work is no
longer the bottleneck on this task; **agent calls are, by a bit over 2x**.

Note this is partly task-specific: l3-48's trials are fast (11.2s median) where l3-43's
were ~30s. A slower task would shift the ratio back. But the direction is now clear and
will only sharpen as the remaining GPU overhead comes down.

## Repair is the single largest cost, and most of it is waste

1.51h of 3.15h agent wall, from 16 calls — the largest single module. Two of those calls hit
the 1200s transport timeout and were retried.

Attributing every repair call to the candidate whose rejection window contains it (0 calls
unattributed, unlike the earlier "nearest following event" heuristic that failed) shows how
much is avoidable:

| candidate | repair calls | repair wall | class |
|---|---|---|---|
| cand-eb910a18 | 3 | 0.51h | genuinely wrong (frac 0.9125) |
| cand-dc4b6fec | 3 | 0.41h | **gate artefact** |
| cand-61f768c8 | 3 | 0.15h | **gate artefact** |
| cand-741c2699 | 3 | 0.12h | **gate artefact** |
| cand-eed411d8 | 3 | 0.12h | **gate artefact** |
| cand-dcf4e7e6 | 1 | 0.05h | **gate artefact** |

**0.85h of 1.37h attributed repair time — 62% — went to candidates with no defect to fix**,
per `docs/finding-unreachable-correctness-gate.md`. That is 18% of the whole run's wall clock
spent repairing kernels already as accurate as their arithmetic allows, and in four of five
cases the repair destroyed them.

So the largest line item in the run's budget is a loop that, for these candidates, converts
correct kernels into broken ones. Three effects compound:

- repair is called on candidates that need no repair (gate finding)
- each such call costs ~5 min of agent wall, x up to `repair_attempts` (3) + 1
- repair is also the module that times out, adding up to 1200s each

## What this implies for the next round of work

Ordered by expected wall saved, not by how interesting each is:

1. **The gate decision (already documented, awaiting the user).** Option 1 in the finding
   would have skipped all five repair chains outright — 0.85h here. Largest single lever,
   and a correctness-semantics change, so it is the user's call.
2. **A "within task noise, no change needed" channel for repair.** Also in the finding.
   Independent of the threshold decision: even under the current gate it would have stopped
   attempt 1 from wrecking a working kernel. Cheaper and less invasive than 1.
3. **The repair transport stall.** Two 1200s stalls in this run. Not fixable in the harness
   beyond the retry already in place and demonstrably working — belongs with whoever owns
   the model endpoint. `candidate_id` on `AGENT_CALL_STARTED` (added in `fc51f18`) makes the
   repeat-repair hypothesis testable from the next run.
4. **B2, shared-lane concurrency for correctness jobs.** Previously ranked higher on the
   assumption GPU work dominated. At 32% of wall it is worth more than the earlier inflated
   11% figure implied, but still less than the gate decision; it stays deferred, and its
   real value is on slower tasks like l3-43.

The general point: fix A was worth doing and is verified, but its success means further
GPU-side optimisation now has a 32% ceiling on this task, while agent wall — 62% of whose
largest component is chasing phantom defects — is where the hours are.
