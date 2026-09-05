# Measurement: where improvement K's expansion budget goes, by candidate competitiveness

`measurement-expansion-budget-economics.md` measured what an expansion buys on average. This
asks whether the spend is *targeted*: does K expand whatever hits a boundary, regardless of
whether the candidate is anywhere near competitive?

Prompted by `run-l3-21-20260905-195615` expanding `cand-2b4d5338` at 22.8 ms while the run's
leader stood at 9.42 — 2.4× away, in a family pinned at exactly 22.8 across two structurally
different candidates, and *slower than the eager_tf32 baseline* (20.9).

## The split

38 expansions, bucketed by how far the candidate stood behind the run's leader at the moment
its pre-expansion tuning finished:

```
within 10% of the leader   n= 16  trials= 625  improved=10  ended as the run's best=5
10-50% behind              n=  9  trials= 348  improved= 7  ended as the run's best=0
50-100% behind             n=  8  trials= 314  improved= 5  ended as the run's best=0
more than 2x behind        n=  5  trials= 200  improved= 4  ended as the run's best=0

trials spent expanding candidates >50% behind the leader: 514 of 1487 (35%)
```

Every one of the 5 expansions that produced a run winner came from the within-10% bucket.

## The confound, which narrows the claim substantially

"Ended as the run's best" is close to tautological for a candidate already near the lead, and
far too strong a test for a lagging one: an expansion can be worthwhile without winning the run.
Re-scored against the number the paper actually reports — does the resulting kernel beat the
strongest `torch.compile` baseline?

```
lagging expansions (>50% behind): 13
  closed to within 50% of the leader         : 1
  produced a kernel beating the best baseline: 3
      cand-cf0f07e7   3.55 -> 2.84 ms  (20.0%, baseline 17.9)   <- largest gain in the dataset
      cand-faa8862d  15.10 -> 15.10 ms ( 0.0%, baseline 16.4)
      cand-51dd1857   3.34 -> 3.34 ms  ( 0.0%, baseline 17.9)
```

The largest single expansion gain anywhere in the dataset came from a candidate 1.70× behind its
leader. So the strong reading — "K wastes a third of its budget on hopeless candidates" — is not
supported.

**What is defensible:** no lagging expansion has produced a run *winner*, and all 5 winners came
from candidates already within 10% of the lead. Expansion value concentrates near the front.

## Why this is not being changed

- K's precondition already includes **idle resources** (`space_expansion_idle_frac`). It is a
  use-spare-capacity mechanism by design, so expanding a lagging candidate is only waste if that
  capacity had a better use. With GPU timing strictly serialized, "idle" does not mean free —
  but it does mean the alternative use is not obvious.
- A competitiveness gate would have suppressed `cand-cf0f07e7`'s 20% gain, the best expansion
  outcome on record.
- Selecting on the outcome is not available to a policy: whether a widened range is reachable is
  only known after the trials are spent.
- n=13 in the lagging bucket, 3 useful. Any threshold fitted to this would be fitted to noise.

Recorded as a distribution worth knowing when the budget split is next revisited, not as a
defect. The one actionable finding from this line of work — that *direction* matters because
value concentrates in expansions that reach a new value — is already implemented in
`fix-boundary-direction-follows-the-winning-trial.md`.

Reproduce with `python scripts/audit_expansion_on_hopeless_candidates.py`.
