# Measurement: what improvement K's expansions cost, and where the gain comes from

Prompted by two live expansions landing on opposite sides within twenty minutes of each other in
`run-l3-21-20260905-195615`:

```
cand-47371017   9.78 -> 9.42 ms   improved, but the winner used only pre-existing values
cand-faa8862d   15.1 -> 15.1 ms   no improvement; the added BLOCK_N=256 ran 17.4 ms and failed once
```

The second one is the well-directed case — `BLOCK_N`'s median pick and winning value agree, so
`docs/fix-boundary-direction-follows-the-winning-trial.md` leaves it untouched — and it still
returned nothing. That raises a question the direction work does not answer: **is an expansion
worth its budget at all?**

## What an expansion costs

An expansion buys a whole fresh tuning budget for one candidate. Across every run on disk:

```
expansions with a before/after tuning pair: 38
  improved the candidate's best :  26  (68%)
  left it unchanged             :  10  (26%)
  ended WORSE                   :   2  (5%)

trials spent on expansion re-tunes: 1487   (median 40 per expansion)
when it improved: median 2.1%   mean 4.4%   max 20.0%
```

1487 trials is a substantial share of the project's GPU time, and a 2.1% median gain is close to
`min_improvement_pct` (2.0) — i.e. often not enough to count as family progress.

## Where the gain comes from — the split that decides whether K earns its slot

68% improve, but `audit_expansion_direction_yield.py` shows only 12–16% end with the best actually
*using* an added value. So the improvement could be coming from the extra 40 trials rather than
from the widened range — and if so, a plain re-tune of the unchanged space would deliver the same
thing without an agent call.

It is not. Splitting the same 38 expansions by whether the post-expansion winner used an added
value:

```
best USED an added value   n=12   improved 12 (100%)   median gain 4.9%
best used NO added value   n=26   improved 14 ( 54%)   median gain 1.3%
```

Fisher two-sided on the improvement rates (12/12 vs 14/26): **p = 0.0065**.

The two rows are far apart in both rate and size. When an expansion reaches a genuinely new value
it always helps, and by ~4× the median amount; when it does not, the re-tune is a coin flip worth
about 1.3% — consistent with TPE noise on a fresh budget rather than with real progress.

**So K is vindicated, and aiming it correctly matters more than these numbers first suggest.** The
value is concentrated in the ~1/3 of expansions that land on a reachable new value; the rest are
paying 40 trials for noise. That is exactly the ratio the direction fix moves.

## What this does NOT establish

- **Not a controlled comparison against rewriting.** As a scale reference only, consecutive
  candidate-to-candidate steps in the same runs improve 47% of the time with a 10.2% median gain
  when they do — but consecutive candidates are not all rewrites of each other, so this is a
  coarse proxy, not a like-for-like trade.
- **Not evidence that the 26 no-hit expansions should not have happened.** Whether a value is
  reachable is only known afterwards; a policy cannot select on it.
- **No causal claim about the 2 that ended worse.** `best_ms` is a minimum over a fresh sample, so
  a worse post-expansion best means the re-tune got unluckier, not that the wider space hurt.

## Not acted on

Changing the re-tune / expand / rewrite budget split alters what every candidate in every run
receives, and nothing measured here is a defect: K works, and it works better than the raw 68%
suggests once the split is taken into account. The one actionable finding — that direction
matters, because value is concentrated in expansions that reach a new value — is already
implemented in `fix-boundary-direction-follows-the-winning-trial.md`.

Reproduce with `python scripts/audit_expansion_budget_economics.py`.
