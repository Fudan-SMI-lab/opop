# The bottleneck has inverted: agent wall now dominates, and repair owns most of it

Measured on `run-l3-48-20260905-010737` at 3.15h elapsed. Not a proposal — a
measurement, recorded because it changes what the next optimisation should target.

## The numbers

| run | elapsed | agent wall | agent share | repair wall | trials |
|---|---|---|---|---|---|
| l3-43 (09-04, pre-fix-A) | 11.67h | 3.88h | 33% | 2.11h | 912 |
| l3-21 (09-04, pre-fix-A) | 8.10h | 2.44h | 30% | 0.04h | 654 |
| **l3-48 (09-05, post-fix-A)** | **3.15h** | **2.80h** | **89%** | **1.81h** | 300 |

Within l3-48, by module:

| module | calls | timeouts | wall | tokens |
|---|---|---|---|---|
| repair | 8 | 2 | **1.81h** | 210,705 |
| parameterizer | 19 | 0 | 0.56h | 331,206 |
| rewriter | 2 | 0 | 0.16h | 70,732 |
| analyst | 8 | 0 | 0.15h | 128,973 |
| generator | 1 | 0 | 0.13h | 28,242 |
| total | 38 | 2 | 2.80h | |

GPU + everything else: **0.35h of 3.15h (11%)**.

## Why this happened

Fix A moved the WSL venv off the 9p `/mnt` mount onto ext4, cutting per-job wall from
~29s to ~11.2s — 2.6x on trial jobs, which are 92% of jobs. That was the right fix and it
worked. But it removed the constraint that was hiding the agent cost. GPU work is no
longer the bottleneck on this task; **agent calls are, by a factor of eight**.

Note this is partly task-specific: l3-48's trials are fast (11.2s median) where l3-43's
were ~30s. A slower task would shift the ratio back. But the direction is now clear and
will only sharpen as the remaining GPU overhead comes down.

## Repair is the single largest cost, and much of it is waste

1.81h of 2.80h agent wall, from **8 calls**. Two of those are the 1200s transport
timeouts (0.67h between them, 24% of all agent wall in the run). The rest is genuine work
— but on this task a large share of it is spent on candidates that had **no defect to
fix**, per `docs/finding-unreachable-correctness-gate.md`: both reassociating rewrites sat
inside the reference's own ieee-vs-tf32 spread and were sent to repair anyway, which then
destroyed them (0.976 -> 0.836, 19.4M non-finite).

So the largest line item in the run's budget is a loop that is, for these candidates,
converting correct kernels into broken ones. Three effects compound:

- repair is called on candidates that need no repair (gate finding)
- each such call costs ~5 min of agent wall, x up to `repair_attempts` (3) + 1
- repair is also the module that times out, at 36.4% historically, adding 1200s each

## What this implies for the next round of work

Ordered by expected wall saved, not by how interesting each is:

1. **The gate decision (already documented, awaiting the user).** Option 1 in the finding
   would have skipped both repair chains on this run outright. Largest single lever, and
   it is a correctness-semantics change, so it is the user's call.
2. **A "within task noise, no change needed" channel for repair.** Also in the finding.
   Independent of the threshold decision: even under the current gate, this would have
   stopped attempt 1 from wrecking a working kernel. Cheaper and less invasive than 1.
3. **The repair transport stall.** 0.67h in this run alone. Not fixable in the harness
   beyond the retry that is already in place and demonstrably works — belongs with whoever
   owns the model endpoint. `candidate_id` on `AGENT_CALL_STARTED` (added in `fc51f18`)
   makes the repeat-repair hypothesis testable from the next run.
4. **B2, shared-lane concurrency for correctness jobs.** Previously ranked higher on the
   assumption that GPU work dominated. At 11% of wall it can save at most a few minutes
   here; it stays deferred, and its real value is on slower tasks like l3-43.

The general point: fix A was worth doing and is verified, but its success means further
GPU-side optimisation now has an 11% ceiling on this task. Agent wall is where the hours
are.
