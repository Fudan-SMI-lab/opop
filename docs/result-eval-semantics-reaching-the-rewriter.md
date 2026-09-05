# Result: `eval_semantics.md` reaching the rewriter is what stopped the train-mode BN failures

`opop-v2-mbconv-train-mode-bn` recorded that L3:21 candidates kept failing because the reference
runs in **train mode** — BatchNorm must use batch statistics, not `running_mean`/`running_var` —
and the agents did not know. The probe (`SEMANTICS_PROBED` → `task/eval_semantics.md`) was added
to tell them. This measures whether it worked, and the answer identifies a specific module.

## The trend, per run

Counting L3:21 candidates on disk that carry either failure signature — a runtime
`if ....training:` branch (the dead-mode-branch trap,
`opop-v2-dead-mode-branch-strands-optimization`) or a `running_mean`/`running_var` reference:

```
run            started        cands  affected   SEMANTICS_PROBED
260902-113144  09-02 11:31        7      0             0
260903-071357  09-03 07:13       11     11             0
260903-210650  09-03 21:06        4      4             0
260904-013056  09-04 01:30       16      9             1
260905-071312  09-05 07:13        8      1             1
260905-153452  09-05 15:34        4      0             1
260905-160156  09-05 16:01        4      0             1
260905-195615  09-05 19:56        6      0             1
```

11/11 → 4/4 → 9/16 → 1/8 → 0/4 → 0/4 → 0/6. Two things stop this from being a clean story, and
both are worth stating rather than smoothing over:

- **The 09-02 run had 0 affected with no probe at all.** So "no probe" does not imply failure; it
  only removes the guard. That run produced 7 candidates and happened not to reach for BN
  internals.
- **`260904-013056` had the probe and still produced 9.** That is the interesting one.

## The mechanism: the probe existed but the rewriter never saw it

Counting which sandboxes actually received the file:

```
260904-013056        sandboxes / with eval_semantics.md
  generator                1 / 1
  repair                   2 / 2
  rewriter                 6 / 0     <-- 
  parameterizer           33 / 0
  analyst                 18 / 0

260905-195615 (current)
  generator                1 / 1
  rewriter                 1 / 1     <-- 
  analyst                  6 / 6
  parameterizer            8 / 0
```

And attributing that run's affected candidates by **origin**:

```
260904-013056, affected by origin
   rewrite     9 / 12 affected
   seed        0 /  4 affected
```

The seeds came from the generator, which had the doc: **0 of 4** affected. The rewrites came from
the rewriter, which did not: **9 of 12**. Same run, same task, same model, same day — the only
difference between the two populations is which of them was told what mode the reference runs in.
That is about as close to a controlled comparison as a run log offers, and it makes the
attribution a measurement rather than a correlation over dates.

**A second instance of the same comparison, found by scripting the scan.** `260905-071312` has
the identical asymmetry — `generator: 1/1` sandboxes with the doc, `rewriter: 0/2` without — and
the identical direction:

```
260905-071312   by origin -> seed: 0/4   rewrite: 1/4
```

So two independent runs show seeds clean and rewrites affected, with the doc's presence the only
systematic difference between the two populations. And the run before the probe existed at all
(`260903-071357`, `SEMANTICS_PROBED=0`) shows **both** populations failing — `seed: 4/4`,
`rewrite: 7/7` — which is the control the other direction: with nobody informed, the generator is
no better than the rewriter.

Pooled across every L3:21 run:

```
rewrite     17 / 28  (61%)
seed         8 / 32  (25%)
```

The pooled figure understates the effect, since it mixes pre-probe runs where both populations
fail; the per-run splits above are the ones that separate the variable.

## The two candidates produced right now

`fam-a4a8353c`'s first rewrite round (the one that cost a 20-min ReadTimeout before succeeding
on attempt 2) produced two candidates whose own summaries name the constraint unprompted:

- `cand-80bf3097` (H1) — "retaining the **TRAIN-mode** two-pass current-batch BatchNorm path"
- `cand-f66890d0` (H2) — "Keeps the required **TRAIN-mode** BatchNorm reductions"

Checked against the source rather than the summary: **0 `.training` branches and 0
`running_mean`/`running_var` references in either file.** So neither carries the dead-mode-branch
trap nor the train/eval mismatch.

## What is not established

- **Whether `parameterizer` and `analyst` need the doc.** `parameterizer` still receives it 0
  times in both runs and has produced no BN-mode failure — but it rewrites the kernel body, so the
  same argument that justified giving it `triton_pitfalls.md`
  (`finding-parameterizer-lacks-triton-pitfalls-doc.md`) applies here. Not changed: there is no
  observed failure to point at, and adding context to 8 calls per run is not free.
- **Whether the remaining 1 affected candidate on `260905-071312` matters.**
  `cand-c0b3b7cd` carries both signatures; that run is the one that stopped at 2.05h on the
  empty-family bug, so it has other explanations available and n=1 does not separate them.
- **Any latency claim.** This note is about a correctness trap, not speed. Whether avoiding it
  produces faster kernels is a separate question the current run is still answering.

Reproduce with `python scripts/audit_eval_semantics_reach.py`.
