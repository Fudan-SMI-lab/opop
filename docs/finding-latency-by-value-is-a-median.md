# Finding: `latency_by_value` is a median, and it disagrees with the minimum on 45% of knobs

Found while checking whether `cand-80bf3097`'s expansion passed the strict K test. It does — the
14.7 ms winner uses the newly-added `GEMM_BLOCK_N=256`, a 5.8% gain over 15.6 — but the stats
table the analyst reads says something else about that value:

```
GEMM_BLOCK_N   latency_by_value (median)
   16   32.95      256 -> 20.55  ranks WORST of five
   32   22.90
   64   19.15
  128   17.20      <- median's pick
  256   20.55
```

The per-value **minimum** tells the opposite story:

```
 value   n   median      min
    16   4    34.65    31.90
    32   3    22.60    22.00
    64   4    19.65    18.00
   128  19    17.20    15.60
   256   4    20.55    14.70   <- the best trial in the entire run
```

`_param_stat` computes `statistics.median(vals)` per choice (`tuning/stats.py:47`), so
`latency_by_value` is a median by design. `best_value` and `at_boundary` are both derived from
that same median table, which is why `GEMM_BLOCK_N` reports `at_boundary=False` after the
expansion even though its best configuration sits on the new upper edge.

## The hypothesis I formed, and the measurement that refuted it

I proposed a mechanism: the median systematically penalises **newly added** values, because a
value the TPE has just started sampling has mostly exploratory trials while an established value
has been refined. `GEMM_BLOCK_N=256` (n=4) versus `128` (n=19) fits that story exactly.

It does not hold. Over 1035 knobs with ≥3 measured values
(`scripts/audit_latency_by_value_median_bias.py`):

```
knobs where the median's best value differs from the minimum's: 466  (45%)

the minimum's winner had FEWER trials than the median's :  106
the minimum's winner had MORE trials                    :  336
```

The direction is the **reverse** of my hypothesis: the median's winner is usually the
*less*-sampled value, not the better-sampled one. A small-n value can catch a lucky median as
easily as a lucky minimum, so there is no low-n penalty to fix. `GEMM_BLOCK_N=256` is a real
instance of the shape I described; it is not the dominant pattern, and I would have shipped a
"fix" for a bias that is not there.

## What actually stands

**A 45% disagreement rate on which value is best.** That is a large number and it is what the
analyst consumes when it writes hypotheses about which knobs have headroom. Worst instances:

```
260903-020233  sp-59baedfe  STATS_NUM_WARPS
   median picks 1 (n=5,  med 104.00, min 98.70)
   minimum picks 4 (n=10, med 146.50, min 73.60)
260905-091705  sp-7e04deb1  ATTN_BLOCK_N
   median picks 128 (n=4, med 39.60, min 30.90)
   minimum picks  64 (n=9, med 54.50, min 14.20)
```

Whether the median is the *wrong* statistic is genuinely arguable, and I am not proposing a
change:

- **For the median.** Latency is noisy on a laptop GPU (this run's baselines show std up to 1.78
  on 100 samples). A minimum is the extreme order statistic — it rewards whichever config got
  the quietest moment on the card, which is exactly the noise-mining the project's
  `final_reeval` exists to catch. Ranking by minimum would import that bias into the analyst's
  view of every knob.
- **Against the median.** The objective being optimised *is* the minimum: `best_ms` is
  `min(trials)`, families are compared on their best, and the reported result is a single best
  configuration. So the analyst is shown a statistic that does not match the thing being
  optimised, and can conclude a knob is exhausted when its best is the run's best.

Both readings are defensible, which is precisely why this is a finding rather than a fix. A
change here alters what every analyst call sees on every run, and it is not a defect with a
demonstrated cost — `GEMM_BLOCK_N`'s expansion **still produced the win** despite the
misranking, because the tuner samples the space directly and does not read this table.

The honest cost statement: the median can cause the analyst to *stop pushing* a knob whose best
value it just found. Whether that has actually happened is not established here, and would need
tracing analyst hypotheses against subsequent expansion requests — a separate measurement.

Reproduce with `python scripts/audit_latency_by_value_median_bias.py`.
