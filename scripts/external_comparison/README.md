# External CUDA vs our Triton — comparison scripts

Everything used to produce `docs/finding-external-cuda-vs-our-triton.md`. Kept together because the
conclusion rests on several measurements taken the same way on the same card, and re-deriving any
one of them in isolation is how the two errors recorded at the end of that doc happened.

The external `.cu` files themselves are **not** copied here — they live in
`external_files/l3_best_candidates_21_43_48_20260908/` and each wrapper sha256-checks the file it
loads against `external_manifest.json`, so a silently substituted kernel fails loudly rather than
producing a plausible number.

| script | what it answers |
|---|---|
| `ext_eval_l3_43.py` | the external L3:43 kernel through our job builder (5 correctness + 100 timed) |
| `ext_eval2.py` | the same for L3:21 and L3:48 |
| `ours_on_box2.py` | **our** best L3:21 and L3:48 candidates on the same box, same builder |
| `split_l3_43.py` | splits our total into Triton GEMM vs Triton attention, against cuBLAS |
| `why_cuda_wins.py` | standalone cuBLAS projection cost at ieee and tf32 |
| `fp32_search.py` | per-tile sweep of our candidate at tf32 and ieee (P1 screen refuses 3 configs) |
| `gemm_gap.py` | our Triton GEMM vs cuBLAS across 4 precisions, with/without the L2 swizzle |
| `best_total.py` | end-to-end totals, shipped tile vs per-precision retuned tile |
| `verify_claims.py` | re-measures the derived numbers in the write-up; caught two of them wrong |
| `ieee_recheck.py` | resolves a 12.339-vs-6.737 ms contradiction, with an in-run control |

## Running them

They are written for box 2's absolute paths (`/root/autodl-tmp/...`) because that is where the GPU
is, and they were run over ssh rather than through the harness's own worker client. That is
deliberate for `split_l3_43.py`, `gemm_gap.py`, `best_total.py` and `ieee_recheck.py` — those launch
individual Triton kernels directly to time one piece of a candidate, which the harness has no path
for. The four that measure a whole candidate (`ext_eval*.py`, `ours_on_box2.py`) do go through
`make_relaxed_correctness_job`, so their numbers are directly comparable to any run's.

```sh
scp -r scripts/external_comparison/* autodl2:/root/autodl-tmp/ext-eval/
ssh autodl2 'cd /root/autodl-tmp/ext-eval && python ext_eval2.py'
```

`EXT_KERNEL_CU` points each wrapper at its `.cu`; without it the wrappers fall back to a literal
box-2 path. They must **not** use `pathlib.Path(__file__)` — the harness loads a candidate by
`exec`ing its source text, so `__file__` is undefined there, and the first L3:43 attempt died with a
`compile_error` that looked like a defect in the external kernel.

## One caveat on the two long sweeps

`gemm_gap.py` and `best_total.py` take longer than a 2-minute foreground budget. Launch them
detached (`nohup ... &`) and poll the log; a foreground `timeout 2400` gets killed by the *client's*
cap, not the one in the command, which looks like the script failing.
