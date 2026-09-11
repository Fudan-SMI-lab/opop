# Box 3 will probably not reach Loop C, and the reason is 1.58 h of witness repair

**Status: recorded, NOT acted on.** The only knobs that would change the outcome break arm parity or
change the correctness gate mid-experiment, both of which the standing rule defers to a unified
post-experiment analysis.

## The arithmetic

Measured at 4.78 h wall on `run-l3-48-20260911-052647` (A800, task L3:48):

| | value |
|---|---|
| wall clock elapsed | 4.78 h |
| budget clock elapsed | 3.49 h (the run was resumed once, so the budget clock restarted) |
| budget remaining | 8.51 h of 12 h |
| candidates finished | 1 of 4 |
| that candidate's cost | 3.85 h |
| projection to first `FAMILY_ROUND_RECORDED` | **8.84–10.95 h** |

The range is the tail adjustment: 0.70 h of the finished candidate's 3.85 h was a single
`runtime_error` trial (18% of the whole candidate), and a one-off tail must not be charged to every
future candidate. Either endpoint brackets or exceeds the 8.51 h remaining, so Loop C is unlikely to
be reached.

Both arms on L3:43 are healthy by contrast: box 1 projects ~3.5 h to Loop C against 7.94 h left, box
2 ~0.14 h against 7.88 h.

## Where box 3's time went

Not to search. **1.58 h of parameterize + repair agent calls after a witness rejection**, and no
trial is measured during any of it:

| candidate | rejections | repairs | agent time |
|---|---|---|---|
| `cand-d2cf7928` | 2 | 2 | 51.8 min |
| `cand-2926f7cd` | 2 | 2 | 42.9 min |

Box 1 and box 2: **zero** rejections, zero repair time. This is not a box-speed difference — it is
the same failure hitting both L3:48 candidates.

## The failure is one shape, twice

Every one of the four rejections is the same pair, in the same order:

1. `witness_default_failed` — the DEFAULT config is `COMPUTE_DTYPE: tf32`, and it fails with
   `frac_within_tol` 0.19 against a task whose own ieee-vs-tf32 spread is 0.9798. The fp64-relative
   arm also fails, at `ratio_to_reference: 6.786`. So this one is the candidate's own numerics, not
   the gate: both arms agree, and the relative arm exists precisely to rescue a candidate the
   absolute gate is too strict for.

2. `witness_minimal_failed` — and this one carries its own explanation in the payload:
   `the DEFAULT config passed; only this one failed`, with
   `19403796 of 134217728 candidate values are not finite (16070970 NaN, 3332826 +/-Inf)`.

The minimal witness is every knob's `choices[0]`, which is the **fp16 corner**. L3:48's
`ref_absmax` is `1.099e+11`, four orders of magnitude above fp16's 65504 ceiling. The overflow is
arithmetically certain before the kernel runs.

The two candidates' non-finite counts are 19403796 both times, differing only in the NaN/Inf split
(16070970/3332826 vs 16070921/3332875). Two independently generated candidates producing the same
count to eight significant figures means the overflow is a property of the TASK and the fp16 corner,
not of either candidate's code — which is exactly why repairing the candidate cannot fix it.

## Why nothing is being changed now

- Skipping or relaxing the minimal witness on a high-`absmax` task would be a **correctness-gate
  change mid-experiment**, and it is the gate that catches real fp16 overflow elsewhere. High risk of
  over-acceptance.
- Cutting `repair_attempts` for box 3 alone breaks the config parity the three runs were launched
  under.
- Nothing here is a low-risk generalizing fix, which is the bar for acting during a run.

## Consequence for the paper

If box 3 stops before Loop C, its G27 `conversion` evidence is unavailable and G27 rests on the two
L3:43 arms alone — single-task, two-arm. That has to be stated as such rather than implied to be
three-task. The S3/S4′ per-dimension evidence is unaffected: box 3 has already produced 16 dimension
records over 2 diagnoses.

See also `docs/result-box3-loop-c-reachability.md` (the earlier 3.53–9.66 h estimate, before the
second candidate's repair cost was known) and the `opop-v2-minimal-witness-is-fp16-corner` memory,
which recorded this failure mode 7/7 on an earlier corpus.
