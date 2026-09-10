# The pre-change corpus: what the OLD prompts reached, and the three mechanisms behind
# `correctness_mismatch` at low precision

Corpus: box 2's four L3 runs, fetched to `external_files/box2-runs/runs-l3/`.
**None of them has a `v3` config block**, so this corpus is evidence about the low-precision
contract change and about nothing in S2/S2d. Provenance, from each run's `manifest.json` and
`events.jsonl` on disk:

| run | span | events | trials | ended |
|---|---|---|---|---|
| `run-l3-21-20260909-154359` | 12.75 h | 1219 | 760 | `RUN_FINISHED` |
| `run-l3-43-20260908-053708` | 11.73 h | 1485 | 1080 | `AGENT_CALL_STARTED` (interrupted) |
| `run-l3-43-20260909-015247` | 13.52 h | 1322 | 840 | `RUN_FINISHED` |
| `run-l3-48-20260910-083700` | 3.02 h | 122 | 55 | `RUN_INTERRUPTED` |

2735 trials, all four on `RTX 4090 (sm_89)`, every trial carrying a `COMPUTE_DTYPE` knob (48/48
candidate `source.py` files have one).

## 1. A correction: the old prompt was NOT precision-blind

The change was motivated partly by a belief that the pre-change prompts left low precision
unreached. **That is wrong, and this corpus refutes it.** Sampling share and pass rate over all
2735 trials:

| `COMPUTE_DTYPE` | trials | share | ok | pass rate |
|---|---|---|---|---|
| fp16 | 1087 | 39.7% | 953 | 87.7% |
| bf16 | 874 | 32.0% | 705 | 80.7% |
| tf32 | 490 | 17.9% | 348 | 71.0% |
| ieee | 284 | 10.4% | 167 | 58.8% |

Tensor-core precisions were 89.6% of all sampling, and a tensor-core precision won the best trial
in **4 of 4 runs** (fp16 3.73 ms / fp16 3.26 ms / bf16 2.77 ms / fp16 1.57 ms). Per-candidate
winners: fp16 23, bf16 17, tf32 3, ieee 0.

So the earlier framing — "the contract change lets low precision into the search space" — is not
what the change did. What it actually did is narrower and is stated in §3.

## 2. Three mechanisms, separable on disk

Pooling all `correctness_mismatch` rows makes the error look dtype-independent. It is not; the
pooling mixed two populations. Partitioning candidates by their **per-precision** pass rate
(shared-memory and runtime failures excluded — those are resource verdicts, not numerical ones,
and including them changes the answer):

**(B) bf16 fails, every ≥10-mantissa-bit precision passes ≥95%** — 6 candidates, **64 bf16
trials, zero passes**, all in L3:21:

| candidate | bf16 | fp16 | tf32 | ieee |
|---|---|---|---|---|
| `cand-6303ca6f` | 0/6 | 39/39 | 6/6 | 2/2 |
| `cand-75d032aa` | 0/9 | 15/15 | 48/48 | 5/5 |
| `cand-876f9cde` | 0/11 | 43/43 | 6/6 | 9/9 |
| `cand-8ede5e28` | 0/11 | 46/46 | 9/9 | 7/7 |
| `cand-e9d34283` | 0/18 | 29/29 | 16/16 | 2/2 |
| `cand-fd2bb2b6` | 0/9 | 14/14 | 28/28 | 5/5 |

This is cause (a), a mantissa shortfall, by elimination:
- **not RANGE** — bf16 has 8 exponent bits against fp16's 5, so anything bf16 overflows, fp16
  overflows worse; fp16 passes 186/186 here.
- **not a code path** — fp16 and bf16 take the *same* branch in all six sources (both are the
  "cast to a 16-bit type" arm); only ieee/tf32 differ. A branch cannot separate them.
- leaves the one thing that does separate them: 7 explicit mantissa bits against 10.

**(A) bf16 fails AND a ≥10-bit precision is also degraded** — 4 candidates. Precision is not the
mechanism; these have algorithm bugs. Two sub-shapes: L3:21's `cand-277a15ac` / `cand-781a8932`
degrade *everywhere* including ieee (80–93%), and L3:48's two candidates fail tf32 0/14 while
fp16 and ieee pass 100% — which is cause (b), since nothing but a code path can fail tf32 while
passing both its neighbours.

**(other) bf16 passes** — 33 candidates, including all 31 across the two L3:43 runs.

## 3. What the contract change actually buys, stated as the corpus supports it

Not access to low precision — that was already there. Two things:

1. **A remedy for population B.** Searched five independent ways (hi/lo dot operands, an fp32
   residual subtraction, three `tl.dot` calls accumulated in one expression, split/hi/lo naming,
   a `DOT_MODE` knob), **0 of 48 pre-change candidates contain a compensated dot**. The 64
   zero-pass bf16 trials had no reachable fix; the tuner could only learn to avoid bf16. Post-change
   candidates carry `DOT_MODE: ["plain","split3"]` — 8/8 inspected on boxes 2 and 3.
2. **A diagnosis that separates A from B.** Both arrive as `correctness_mismatch`. Pre-change the
   repair prompt's advice for a small numerical error was `input_precision="ieee"`, which is the
   wrong fix for both.

## 4. Two readings of mine that did not survive, kept because the near-miss is the lesson

**`ratio_to_reference = 8.000` is not the mantissa ratio.** bf16 has 3 fewer mantissa bits than
tf32, 2³ = 8, and 17 failing trials across 4 unrelated candidates reported exactly `8.000` with
`log2 = 3.0000`. It is a coincidence of rounding: the same partition shows **ieee at 7.592×** and
fp16 at 7.658×, and ieee has no mantissa deficit at all. Those rows come from population A, where
the error is an algorithm bug that any precision inherits. A power of two landing on the predicted
exponent is not evidence when a precision that cannot have the deficit lands there too.

**"The cast gemm feeds a normalization" does not predict the failure.** It scores 8/10 on L3:21
and 40/43 overall, but its two L3:21 misses are the two bf16 *passers*, i.e. the rule mispredicts
exactly the cases that discriminate. The real difference is which kernels are cast: the 8 failers
cast the expand gemm, the 2 passers cast only the proj gemm — but with n=2 on the passing side
that is one candidate's worth of evidence, and the same 2 candidates are the entire basis of the
"perfect" 10/10 2×2. Not reported as a finding.

**A reader defect found in my own probe.** `TRIAL_DONE`'s payload nests as
`payload.trial.params.values`; reading `payload["params"]` returns `None` for every trial and
prints one plausible-looking all-`None` row instead of failing. The analysis scripts now abort
when every trial reads `(None, None)`.
