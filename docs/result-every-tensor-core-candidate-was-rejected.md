# Result: the harness rejected 100% of tensor-core candidates on level3/48

`run-l3-48-20260905-010737`. This is the run's most important number, and it is a perfect
partition with no exceptions, n=16 (counts as of 06:57; the run is still in flight and the
separation has held at every step, from 7/7 through 9/7):

| candidate class | published | rejected |
|---|---|---|
| **uses `tl.dot`** (tensor-core path) | **0** | **9** |
| no `tl.dot` (scalar FMA path) | **7** | **0** |

Read off every candidate's own source on disk, not from event labels:

```
candidate         tl.dot  lowp_cast  dtype_knob  published
cand-61f768c8         20       True        True      False
cand-dc4b6fec         16       True        True      False
cand-dcf4e7e6         16       True        True      False
cand-8cb745ff         12       True        True      False
cand-ef6f0748         12       True        True      False
cand-eb910a18          8       True        True      False
cand-2136993c          4       True        True      False
cand-741c2699          4       True        True      False
cand-eed411d8          4       True        True      False
cand-3bcc57ce          0      False       False       True
cand-51dd1857          0      False       False       True
cand-8a617cba          0      False       False       True
cand-a04c3f52          0      False       False       True
cand-c18203b6          0      False       False       True
cand-cf0f07e7          0      False       False       True
cand-f4a2ce82          0      False       False       True
```

`tl.dot` count, low-precision cast, and dtype knob are perfectly collinear here — the nine
tensor-core candidates all declared a `COMPUTE_DTYPE` knob and all cast to fp16/bf16,
exactly as `candidate_contract.md:87-104` instructs. The seven survivors did none of it.

## Why this matters more than either individual finding

The run's founding question was **why candidates never beat `torch.compile`**
(`opop-v2-root-cause-no-speedup`). The answer was: the harness was ieee-locked, so candidates
ran on the scalar FMA path while `torch.compile` used tf32 tensor cores. Improvements H1/H2/H3
and the dual-precision witness gate were built to unlock that path.

They did unlock it — the agent *generated* tensor-core kernels, nine of them, unprompted and
correctly following the contract. **Every single one was then rejected by the harness**, by
two independent mechanisms that both trip on this task's 1e22 dynamic range:

- `finding-minimal-witness-forces-fp16.md` — the minimal witness is `choices[0]` of every
  knob, i.e. the fp16 corner, which overflows by construction. 17 of 27 rejections.
- `finding-unreachable-correctness-gate.md` — `pass_frac = 0.99` exceeds what the reference
  achieves against its own tf32 witness (0.977767). 10 of 27.

So the joint-best 2.09ms result — 8.90x `torch.compile` — was achieved **entirely without
tensor cores**, by the only candidates the harness would accept. That is a genuine result and
the speedup is real. But it means the run measured one half of its search space and rejected
the other half wholesale, and the rejected half is the half the analyst nominated as primary
headroom in every family.

## What this does and does not say

**Does not say** the tensor-core candidates were faster *on this task*. None was ever timed —
that is the point, and it stays unknown rather than disproven. Three of the nine also had
genuine defects alongside the two harness problems (eb910a18 at frac 0.9125, 741c2699 whose
default witness failed three times, 8cb745ff).

**But on the one task where both classes were measured, the rejected class won.** L3:21 timed
14 tensor-core candidates and 2 scalar-FMA ones:

| L3:21 | best | median | n |
|---|---|---|---|
| tensor-core | **20.50 ms** | **21.50 ms** | 14 |
| scalar-FMA | 25.00 ms | 31.60 ms | 2 |

The scalar-FMA candidates are the two slowest of the sixteen. That is a small n on the
scalar side and a different task, so it is not evidence about what 48's rejected kernels
would have clocked. It does mean the class the harness discarded on 48 is not a class that
loses on the merits where it has been measured — which is the assumption a reader would
otherwise have to make to treat the 0/9 rejection as harmless.

**Does say**, and this is fully supported: the acceptance path, not the generator and not the
tuner, is what kept tensor cores out of this run's results. The 0/9 vs 7/0 split is not a
tendency, it is every case.

**For the paper.** The two-loop argument does not depend on tensor cores — the 2.09ms result
and the slope-ranking evidence (`result-second-ranked-family-catches-the-leader.md`) stand on
their own. But any claim about *what structures the search explored* on this task has to say
that a whole structural class was generated and never measured. Reporting the 8.90x without
that caveat would overstate what the search covered.

## Scope

Task-specific, and this is now measured rather than inferred. Running the same partition
against the most recent L3:21 and L3:43 runs:

| run | uses `tl.dot` | no `tl.dot` |
|---|---|---|
| **l3-48** (09-05) | **0 published, 9 rejected** | 7 published, 0 rejected |
| l3-21 (09-04) | **14 published, 0 rejected** | 2 published, 0 rejected |
| l3-43 (09-04) | **14 published, 0 rejected** | — |

28 tensor-core candidates on 21/43, all published. 9 on 48, all rejected. So the rejection
is not a property of tensor-core kernels as such — those runs measured plenty of them.

> **Corrected 2026-09-05 07:19.** I originally explained this difference by saying the gate
> "does not fire" on 21/43 because their outputs are bounded. **That explanation is wrong.**
> The first L3:21 rerun rejection reports the task's own ieee-vs-tf32 floor as **0.95536**
> with `ref_absmax` 5.749 — bounded, and a floor 3.5 points *below* L3:48's 0.977767, i.e.
> 4.5 points below the 0.99 gate. Bounded magnitude does not imply witness agreement; the
> spread is set by reduction depth and cancellation, not output range.
>
> The 28-vs-9 fact stands as an observation about those runs. Its **cause is not the gate being
> harmless on 21/43.** Do not cite the bounded-output argument. See the falsification note in
> `finding-unreachable-correctness-gate.md`.

### A candidate mechanism for the 28-vs-9 split: how many precisions it must clear

The L3:21 rerun initially looked like it was reproducing the L3:48 pattern: 0 of 2 tensor-core
published against 14 of 14 yesterday.

> **That reproduction did NOT hold, 08:00.** `cand-d31b0474` — 3 `tl.dot`, fp16 cast, dtype knob
> — was published and tuned to **17.1 ms**, the best latency any L3:21 run has produced. The
> task's tally is now **1 of 3 tensor-core published, 2 of 2 scalar published**, and
> `audit_structural_coverage.py` no longer reports a wholly-rejected class for this run.
>
> So the acceptance path is **not** uniformly closed to tensor-core work on L3:21, and the
> mechanism below explains a tendency rather than a wall. The L3:48 result (0 of 9) stands as
> measured; it should not be generalized to other tasks, and the wholesale-rejection framing in
> this document's title applies to L3:48 alone.

Same task, same gate, same 0.95536 floor — so the difference is in the candidates. It is:

| run | tensor-core candidates | `default == choices[0]` | published |
|---|---|---|---|
| l3-21 (09-04) | 14 | **12 of 14** (all fp16) | 14 of 14 |
| l3-21 (09-05) | 3 | 0 of 3 | **1 of 3** |

Publication requires **both** witnesses to pass. When a candidate's `PARAMS` default equals
`choices[0]`, both witnesses run the same dtype and the gate has to be cleared at **one**
precision. When they differ, it must be cleared at **two**. Yesterday 12 of 14 were in the easy
case; today none of the three are.

Stated as a mechanism, not a cause — and the run has already produced the counterexample that
keeps it a tendency: `cand-d31b0474` had `default=ieee != minimal=fp16`, needed to clear the gate
at two precisions, and **did** (after one repair fixed a real defect). Together with the two
such candidates that published yesterday, the honest reading is that a two-precision pair is
harder on L3:21, not impossible.

What it is **not**: evidence that fp16 is the accurate configuration. It is worth noting
separately that on `cand-6b313c39` the fp16 config outscored its own tf32 default (0.978946 vs
0.927816 against the tf32 reference) — but that is one candidate, and the mechanism above does
not depend on it.

One consequence worth stating plainly: **the earlier 21/43 runs already accepted
tensor-core kernels**, so their results are not affected by this, and the reruns are a test
of the H1/H2/H3 speedup rather than of the acceptance path. Worth watching there instead:
whether a `tl.dot` candidate becomes the family best, which is the question 48 could not
answer.
