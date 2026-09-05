# Result: every rewrite round on L3:43 09-05 improved its family, 4 for 4 — then round 2 broke the streak

**Superseded in part.** The 4-for-4 record below is round 1 for all four families and stands as
recorded. Round 2 on `fam-4aea322a` then failed twice: `cand-aa016dfe` at 11.8 and `cand-45c3fd7d`
at 22.0, both `improved_family: False` against the 11.0 incumbent. So the streak is 4 of 6, and the
one refuted case is documented in detail in `result-analyst-hypothesis-refuted-by-control.md` —
worth reading alongside this file, because it is the *stronger* piece of evidence about the paper's
claim despite being the negative result.

Why the negative one carries more weight: the four wins below show that rewriting helps, which is
compatible with the rewrites succeeding for reasons unrelated to the tuning feedback that prompted
them. The round-2 failure had a quantified prediction (+8.0%), a verified mechanism (256 rows
became reachable and won *inside* the child), and the parent's own 21 trials as a same-run control —
so it isolates the analyst's *inference* as the thing that failed, which no win here does.

`run-l3-43-20260905-091705` at 2.66h of 12h. Recorded because the *consistency* is new — earlier
L3:43 runs had families that sat flat across all three rounds — and because each gain has a
mechanism verifiable in the trial data rather than only a latency delta.

## The tally

| family | seed | after round 1 | gain | mechanism, and how it was verified |
|---|---|---|---|---|
| fam-4aea322a | 14.2 | **11.0** | **22.5%** | `QKV_M_CTAS` serialization; **0 spills** at 248 regs (parent: 12 spills at 255) |
| fam-7f682a54 | 23.4 | 20.3 | 13.2% | two kernels fused into one; **`kernel_names` went from 2 to 1** |
| fam-92e7c576 | 22.5 | 19.6 | 12.9% | single-stage K/V staging; shared fell 90112 → 81920 at the large tile |
| fam-ea7bc8bb | 28.0 | — | — | no round yet |

Three rounds, three improvements, 12.9–22.5%. For contrast, `finding-converged-stop-kind-is-unreachable.md`
lists eight families across earlier runs whose histories were *completely flat* for three rounds
(`[25.2, 25.2, 25.2]`, `[2.09, 2.09, 2.09]`, …).

## Each mechanism is independently checkable, which is the part worth keeping

A latency drop after a rewrite is weak evidence on its own — the tuner might simply have gotten
luckier. In all three cases the profile shows the *specific* thing the hypothesis targeted:

**fam-7f682a54 is the cleanest.** The rewrite claimed to fuse the two-pass pipeline into one
online-softmax kernel. `TrialRecord.profile.kernel_names` across all trials:

```
parent  cand-de802450 -> ['_attention_recompute_kernel', '_attention_stats_kernel']
child   cand-88e76051 -> ['_attention_online_kernel']
```

Two launches became one, exactly as advertised, and `shared_bytes` at the winner is 40960 —
well under the parent's 98304 at its own best. The claim is not taken on the agent's word; the
launched-kernel list is recorded by the worker.

**fam-4aea322a**: the analyst named registers as the blocker; the child's winner has 0 spills
against the parent's 12, at 248 vs 255 registers.

**fam-92e7c576**: the analyst named shared memory; the child's `BLOCK_N=128` trial dropped shared
from 90112 to 81920 — the change worked mechanically even though the resulting latency was 32%
*worse* (`measurement-analyst-median-on-one-sample.md`). Mechanism confirmed, prediction refuted.

That last row is why "3 for 3" needs care: the family improved 12.9%, but **not** for the reason
the hypothesis gave. The gain came from better staging at tile sizes that already fit, while the
predicted win at `BLOCK_N=128` went the wrong way. So the correct summary is *three rounds, three
improvements, two hypotheses confirmed as stated, one improved despite a wrong prediction.*

## What this does not show

- **Not** that round 2 will continue the pattern. Every one of these is a *first* round. Reading
  the recorded histories across all runs — and remembering that `best_history` excludes the seed,
  so `h[0]` is already the post-round-1 incumbent — the later rounds look like this:

  | step | improved (>0.5%) | flat |
  |---|---|---|
  | round 1 → round 2 | **4 of 13** | 9 |
  | round 2 → round 3 | **1 of 13** | 12 |

  Round 3 has improved a family **once** — `cand-e3a5da01`, 11.0 → 9.73, which happened in this run
  after the paragraph below was written asserting it never had. Round 2 improves about a third of
  the time (3.8%, 3.7%, 8.2%, 15.0%). Today's three first-round gains of 12.9–22.5% still say
  nothing about whether rounds 2 and 3 will add anything; the base rate says round 3 usually will
  not, and the one time it did was worth more than every round-2 gain combined.

  That is a sharper statement than the one I first wrote here ("the flat histories are mostly
  rounds 2 and 3") — it is specifically round 3 that is nearly dead, and it interacts directly with
  `finding-converged-stop-kind-is-unreachable.md`: a family always *spends* its third round
  because `converged` cannot fire, and the third round has never paid off. At ~22 min per round
  and four families, that is roughly 1.5 hours per run spent on a step with a 1-for-13 record
  (0-for-13 when written; `cand-e3a5da01` improved fam-4aea322a 11.0 -> 9.73 on round 3 at
  14:53, see `result-l3-43-973ms-round-three-win.md`).

- **Not** a claim about the harness's average. Three rounds in one run on one task; the
  cross-run picture is `scripts/audit_convergence_stop_kinds.py`, which is far less flattering.
- **Not** verified end-to-end. All four numbers are `tuned_ms`. The 11.0 has no
  `final_reeval_ms`, and until it does, L3:43 has no verdict
  (`inprogress-l3-43-11ms-rewrite.md`).
- **Not** independent of the seed quality. `fam-4aea322a`'s 22.5% starts from a seed
  (14.2) that was itself the run's outlier, so the largest gain sits on the least reproducible
  base.

## Why it is still worth recording now

The paper's claim is that parameter-tuning feedback usefully steers structural search. The
strongest possible form of that evidence is: *the tuner's statistics identified a specific
resource limit, the analyst named it, the rewriter attacked exactly that limit, and the profile of
the result shows that limit relieved.* This run has that chain three times over in one hour, with
the mechanism visible in `n_spills`, `shared_bytes` and `kernel_names` rather than inferred from
latency alone.
