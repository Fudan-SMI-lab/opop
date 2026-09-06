# Run report — run-l3-21-20260905-195615

## Baselines

- **eager**: 25.3 ms (std 1.46, n=100)
- **eager_tf32**: 20.9 ms (std 1.78, n=100) (tf32 matmul reference)
- **torch_compile**: 22.3 ms (std 1.25, n=100)
- **torch_compile_tf32**: 16.4 ms (std 0.16, n=100) (tf32 matmul reference)

## Best result

- candidate: `cand-e6e7c9c5` (family `fam-a2688942`)
- tuned latency: 6.9 ms
- final independent re-eval: PASS at 7.21 ms
- candidate arithmetic precision: **fp16**
- speedup vs each baseline (baseline_ms / candidate_ms, >1 = faster):
  - vs `eager`: **3.509x**
  - vs `eager_tf32`: **2.8988x**
  - vs `torch_compile`: **3.0929x**
  - vs `torch_compile_tf32`: **2.2746x**
- **honest same-precision verdict**: candidate is fp16; vs same-precision baseline `torch_compile_tf32` = **2.2746x** — ✅ beats the same-precision baseline
- best params: `{"EXPAND_BLOCK_M": 64, "EXPAND_BLOCK_N": 128, "EXPAND_BLOCK_K": 16, "EXPAND_WARPS": 4, "EXPAND_STAGES": 3, "PROJECT_BLOCK_M": 64, "PROJECT_BLOCK_N": 128, "PROJECT_BLOCK_K": 16, "PROJECT_WARPS": 4, "PROJECT_STAGES": 3, "COMPUTE_DTYPE": "fp16", "DW_BLOCK": 256, "DW_WARPS": 4, "DW_STAGES": 3, "MOMENTS_WARPS": 4, "FINAL_BLOCK": 1024, "FINAL_WARPS": 2}`

### Search budget actually used

- rewrite rounds used: **10** across 4 families ([2, 3, 3, 2])
- elapsed: **12.816 h**

## Families / lineage

### `fam-a4a8353c` — active, best 10.7 ms
- rewrite rounds used: 2
- best history: [19.4, 11.0, 10.7]
  - `cand-52aaee73` (seed): Uses a deterministic two-stage hierarchical sum/squared-sum reduction for each train-mode BatchNorm, followed by a Triton kernel that fuses normalizat
  - `cand-80bf3097` (rewrite, parents ['cand-52aaee73']): Replaces both dense 1x1 F.conv2d operations with NCHW-aware Triton GEMM kernels using tunable tl.dot input precision defaulted to tf32 and fp32 accumu
  - `cand-f66890d0` (rewrite, parents ['cand-52aaee73']): Keeps the required TRAIN-mode BatchNorm reductions, but removes intermediate BN apply tensors by normalizing and applying ReLU6 while loading the foll
  - `cand-dd62990b` (rewrite, parents ['cand-f66890d0']): Reworks the 1x1 GEMM into a 256x128 logical tile formed from two 128-row warp groups and launches 8 warps, distributing the larger accumulator footpri
  - `cand-5a3182cb` (rewrite, parents ['cand-f66890d0']): Remaps depthwise work channel-major and emits per-program sum and sum-of-squares partials while writing convolution output, removing the standalone Ba

### `fam-a2688942` — active, best 6.9 ms
- rewrite rounds used: 3
- best history: [20.4, 15.1, 10.0, 6.9]
  - `cand-6f96a754` (seed): Delegates stable channel-wise moment reduction to torch.var_mean, then fuses BatchNorm normalization, affine transformation, and ReLU6 in one Triton s
  - `cand-faa8862d` (rewrite, parents ['cand-6f96a754']): Replaces both executed NCHW 1x1 F.conv2d calls with tiled Triton tl.dot kernels using tunable TF32 input precision and fp32 accumulation; retains curr
  - `cand-1376c224` (rewrite, parents ['cand-6f96a754']): Replaces torch.var_mean with explicit two-stage fp32 per-channel sum/sum-of-squares reductions and fuses final projection normalization with the resid
  - `cand-0a02b061` (rewrite, parents ['cand-faa8862d']): Replaces standalone BatchNorm statistics scans with convolution-produced per-channel sum/sumsq partials and a small finalization kernel for all three 
  - `cand-8e225f0a` (rewrite, parents ['cand-faa8862d']): Eliminates normalized intermediate tensors by applying expand BatchNorm/ReLU6 inside depthwise loads, depthwise BatchNorm/ReLU6 inside project-GEMM lo
  - `cand-e6e7c9c5` (rewrite, parents ['cand-8e225f0a']): Fuses per-channel sum and sum-of-squares emission into all pointwise/depthwise producer kernels and replaces three standalone torch.var_mean activatio

### `fam-4286a3be` — active, best 8.13 ms
- rewrite rounds used: 3
- best history: [19.4, 9.42, 8.13, 8.13]
  - `cand-e2cd07de` (seed): Uses channel-partitioned block reductions with atomic cross-block accumulation into compact per-channel moment buffers, then recomputes mean and varia
  - `cand-8510db1b` (rewrite, parents ['cand-e2cd07de']): Replaces both 1x1 F.conv2d operations with explicit Triton GEMMs using tunable TF32 tensor-core dot precision and fp32 accumulation; each GEMM also co
  - `cand-47371017` (rewrite, parents ['cand-e2cd07de']): Restructures the full TRAIN-mode pipeline around required moment barriers: expand BatchNorm/ReLU6 is normalized while loading the depthwise consumer, 
  - `cand-0d0dcd49` (rewrite, parents ['cand-47371017']): Replaces producer-side global atomic statistics with collision-free per-program partial sum/square workspaces and a compact per-channel second-stage r
  - `cand-d4fe9dfd` (rewrite, parents ['cand-47371017']): Splits the shared pointwise GEMM configuration into independently tunable EXPAND and PROJECT M/N/K tile, warp, and stage parameters, allowing each fix
  - `cand-2dc4fe01` (rewrite, parents ['cand-0d0dcd49']): Uses an internal NHWC activation pipeline: the expansion pointwise kernel writes NHWC, depthwise convolution tiles contiguous channels, projection con
  - `cand-83954796` (rewrite, parents ['cand-0d0dcd49']): Replaces one-program-per-channel statistics reduction with 2D part-by-channel tiles that coalesce partial-stat loads and store contiguous channel grou

### `fam-fd92a2d8` — active, best 15.2 ms
- rewrite rounds used: 2
- best history: [22.8, 15.2, 15.2]
  - `cand-61759130` (seed): Replaces the expansion 1x1 convolution with a tiled NCHW-aware Triton GEMM using an fp32 accumulator and a tunable tf32/ieee dot-precision path. It pr
  - `cand-2b4d5338` (rewrite, parents ['cand-61759130']): Uses a logical M=256 expansion tile split into sequential M=128 accumulator subtiles, committing each before allocating the next so the blocked larger
  - `cand-66d76774` (rewrite, parents ['cand-61759130']): Implements a TRAIN-safe expansion boundary: TF32 expansion is followed by current-batch mean/variance computation, then a Triton depthwise kernel fuse
  - `cand-8eda41e5` (rewrite, parents ['cand-66d76774']): Materializes expand BatchNorm plus ReLU6 once per activation in a Triton kernel, then runs the depthwise convolution on that normalized tensor so its 
  - `cand-8bd31d35` (rewrite, parents ['cand-66d76774']): Extends the TF32 expand-pointwise kernel to emit per-tile channel sums and squared sums, finalizes current-batch statistics in a second Triton reducti

## Tuning

- trials: 1657 total, 1395 complete, 262 failed
- `cand-52aaee73` [`sp-ad4f50cd`]: best 19.4 ms (40 asked)
- `cand-6f96a754` [`sp-488b1916`]: best 20.4 ms (17 asked)
- `cand-e2cd07de` [`sp-fb745e6e`]: best 19.4 ms (40 asked)
- `cand-e2cd07de` [`sp-607e4351`] (expanded space): best 19.4 ms (40 asked)
- `cand-61759130` [`sp-154b1a6b`]: best 22.8 ms (40 asked)
- `cand-61759130` [`sp-68475458`] (expanded space): best 22.8 ms (40 asked)
- `cand-80bf3097` [`sp-9ba61a54`]: best 15.6 ms (40 asked)
- `cand-80bf3097` [`sp-274540fa`] (expanded space): best 14.7 ms (40 asked)
- `cand-f66890d0` [`sp-8784d130`]: best 11.1 ms (40 asked)
- `cand-f66890d0` [`sp-d36a6e38`] (expanded space): best 11.0 ms (40 asked)
- `cand-8510db1b` [`sp-72cb5ea2`]: best 15.9 ms (40 asked)
- `cand-8510db1b` [`sp-6c0aca30`] (expanded space): best 14.4 ms (40 asked)
- `cand-47371017` [`sp-e6387567`]: best 9.78 ms (40 asked)
- `cand-47371017` [`sp-ddab161a`] (expanded space): best 9.42 ms (40 asked)
- `cand-faa8862d` [`sp-55435dd7`]: best 15.1 ms (40 asked)
- `cand-faa8862d` [`sp-6ae85ac8`] (expanded space): best 15.1 ms (40 asked)
- `cand-1376c224` [`sp-0461348b`]: best 19.3 ms (40 asked)
- `cand-2b4d5338` [`sp-46d9bbb4`]: best 22.8 ms (40 asked)
- `cand-2b4d5338` [`sp-6496501a`] (expanded space): best 22.1 ms (40 asked)
- `cand-66d76774` [`sp-cf82100a`]: best 15.3 ms (40 asked)
- `cand-66d76774` [`sp-709e9936`] (expanded space): best 15.2 ms (40 asked)
- `cand-0d0dcd49` [`sp-2261d7bd`]: best 9.14 ms (40 asked)
- `cand-0d0dcd49` [`sp-4174d638`] (expanded space): best 8.13 ms (40 asked)
- `cand-d4fe9dfd` [`sp-c1fb16a1`]: best 9.45 ms (40 asked)
- `cand-d4fe9dfd` [`sp-6e3606b1`] (expanded space): best 9.26 ms (40 asked)
- `cand-dd62990b` [`sp-40989c40`]: best 11.4 ms (40 asked)
- `cand-dd62990b` [`sp-5212cd0e`] (expanded space): best 11.0 ms (40 asked)
- `cand-5a3182cb` [`sp-76cd8b17`]: best 10.7 ms (40 asked)
- `cand-5a3182cb` [`sp-8066dd2b`] (expanded space): best 10.7 ms (40 asked)
- `cand-8eda41e5` [`sp-baee86de`]: best 18.6 ms (40 asked)
- `cand-8eda41e5` [`sp-5ec6e681`] (expanded space): best 18.6 ms (40 asked)
- `cand-8bd31d35` [`sp-839766ea`]: best 16.5 ms (40 asked)
- `cand-8bd31d35` [`sp-fd62b949`] (expanded space): best 16.2 ms (40 asked)
- `cand-0a02b061` [`sp-f15b0ba1`]: best 11.3 ms (40 asked)
- `cand-0a02b061` [`sp-108bda23`] (expanded space): best 11.3 ms (40 asked)
- `cand-8e225f0a` [`sp-2eb23159`]: best 10.1 ms (40 asked)
- `cand-8e225f0a` [`sp-063aabda`] (expanded space): best 10.0 ms (40 asked)
- `cand-e6e7c9c5` [`sp-0f8e339b`]: best 6.92 ms (40 asked)
- `cand-e6e7c9c5` [`sp-c27f470d`] (expanded space): best 6.9 ms (40 asked)
- `cand-2dc4fe01` [`sp-ca031cc7`]: best 9.69 ms (40 asked)
- `cand-2dc4fe01` [`sp-72736927`] (expanded space): best 9.58 ms (40 asked)
- `cand-83954796` [`sp-cc1ca366`]: best 8.24 ms (40 asked)

## Bottleneck reports

- `cand-52aaee73`: No tunable parameter has confirmed blocked headroom. All 40 trials completed, and the best profile uses only 38/255 registers per thread (14.9%), 16/101376 B opt-in shared memory (0.016%), no spills, and 4 warps (128 threads versus the 1024-thread limit). PARTIAL_NUM_WARPS and APPLY_NUM_WARPS are ma (suggested: rewrite)
- `cand-6f96a754`: No tuned parameter has demonstrated blocked headroom. BLOCK_SIZE and NUM_WARPS have non-boundary marginal optima, every tested value completed, and the best configuration is far from hardware limits (32/255 registers per thread, 0/49,152 B static shared memory, 0 spills, and 8 warps = 256/1,024 thre (suggested: rewrite)
- `cand-e2cd07de`: TRAIN mode is required. Only APPLY_BLOCK shows weak boundary headroom, but it is not blocked by registers, shared memory, threads, OOM, or compilation: all 40 trials completed, the best uses 30/255 registers per thread, 16/101376 B shared memory, zero spills, and 8 warps (256/1024 threads). REDUCE_B (suggested: rewrite)
  - APPLY_BLOCK wants increase, blocked by arithmetic_throughput
- `cand-e2cd07de`: No tuned parameter has demonstrated blocked headroom. All 80 trials completed, with no OOM, compile, spill, or other failure cluster. Matched comparisons show REDUCE_BLOCK is already flat from 1024 to 2048 (mean higher-minus-lower delta +0.005 ms), while APPLY_BLOCK is flat from 1024 through 4096 (b (suggested: rewrite)
- `cand-61759130`: No trial demonstrates a parameter blocked by a hardware failure. The only credible remaining boundary direction is larger BLOCK_N: aggregate latency improves from 24.6 ms at 64 to 23.7 ms at 128, and a near-matched TF32 comparison improves from 24.4 to 23.2 ms. However, no BLOCK_N >128 trial was att (suggested: rewrite)
  - BLOCK_N wants increase, blocked by none
- `cand-61759130`: One credible blocked tuning direction remains: increasing BLOCK_M beyond 128 at the best TF32 configuration. Controlled trials improve from 23.4 ms at M=32 to 23.2 ms at M=64 and 22.8-22.9 ms at M=128, while registers rise from 80 to 126 to 226 per thread. A conventional M=256 tile would likely exce (suggested: rewrite)
  - BLOCK_M wants increase, blocked by registers
- `cand-80bf3097`: No parameter is demonstrably hardware-blocked by the recorded trials. The 15.6 ms best run uses 152/255 registers (59.6%), 65,536/101,376 B opt-in shared memory (64.6%), 256/1,024 threads, and zero spills. GEMM_BLOCK_N, GEMM_NUM_WARPS, GEMM_NUM_STAGES, FINISH_NUM_STAGES, and APPLY_BLOCK reach search (suggested: rewrite)
- `cand-80bf3097`: TRAIN mode applies current-batch BatchNorm statistics. The clearest blocked direction is increasing GEMM_BLOCK_N beyond the winning 256-wide tile: among focused fp16, K=32, 8-warp, 3-stage trials, N=256 reaches 14.7 ms versus 15.6 ms for N=128, but the winner already uses 255/255 registers per threa (suggested: rewrite)
  - GEMM_BLOCK_N wants increase, blocked by shared_memory
- `cand-f66890d0`: Only GEMM_BLOCK_N has credible blocked headroom. Latency improves monotonically toward N=128 (median 29.1, 20.2, 14.65, 12.9 ms for N=16,32,64,128), while the fastest 128x128 accumulator tile reaches 255 registers/thread with 6 spills. This is not evidence that registers should simply be reduced: th (suggested: rewrite)
  - GEMM_BLOCK_N wants increase, blocked by registers
- `cand-f66890d0`: The only well-supported blocked boundary is GEMM_BLOCK_M increasing beyond 128. The best 128x128x16 fp16 tensor-core tile runs at 11.0 ms with 255 registers/thread and 4 spills, while a near-comparable 64x128x16 tile uses 168 registers with no spills but is slower at 11.8 ms. Thus high register use  (suggested: rewrite)
  - GEMM_BLOCK_M wants increase, blocked by registers
- `cand-8510db1b`: No parameter is empirically blocked by a device/resource limit. The only failures are bf16 correctness mismatches; there are no OOM, compile, register, shared-memory, or thread-limit failures. At the best configuration, registers are 172/255 (67%), shared memory is 8,192/101,376 B opt-in (8%; 17% of (suggested: rewrite)
  - GEMM_BLOCK_N wants increase, blocked by none
  - GEMM_STAGES wants increase, blocked by none
  - REDUCE_BLOCK wants increase, blocked by none
  - APPLY_BLOCK wants increase, blocked by none
- `cand-8510db1b`: TRAIN-mode analysis: only GEMM_BLOCK_N shows credible blocked headroom. Latency improves from N=64 (15.9 ms best) to N=128 (14.4 ms), but the sole N=256 trial takes 161 ms with 7,400 spills. This is a register-live-set problem in that specific N=256/2-warp configuration, not general resource saturat (suggested: tune_more)
  - GEMM_BLOCK_N wants increase, blocked by registers
- `cand-47371017`: Only GEMM_BLOCK_N shows credible blocked headroom. Best observed latency improves from 12.0 ms at N=64 to 9.78 ms at the N=128 search boundary, where the profile reaches 255 registers/thread and incurs 20 spills. This is not evidence that registers should simply be reduced: the fastest lower-registe (suggested: rewrite)
  - GEMM_BLOCK_N wants increase, blocked by registers
- `cand-47371017`: No tunable parameter is convincingly both trending toward a boundary and blocked by a hardware/resource failure. All 15 failed trials are BF16 correctness mismatches; there are no OOM, compile, thread-count, register, or shared-memory failures. The only numeric boundary reported is GEMM_BLOCK_K=16,  (suggested: rewrite)
- `cand-faa8862d`: No parameter is proven to be hardware-blocked. BLOCK_N is the only numeric knob with credible remaining headroom: latency improves monotonically through the tested maximum of 128, but 128 is merely the search-space boundary. The best configuration uses 200/255 registers per thread, 32,768/101,376 B  (suggested: tune_more)
  - BLOCK_N wants increase, blocked by none
- `cand-faa8862d`: No tunable parameter has demonstrated blocked headroom. The apparent BLOCK_M=max signal is a plateau, not a register wall: the best 128x128x16 fp16 configuration is 15.1 ms at 200/255 registers, zero spills, 32,768/101,376 B opt-in shared memory, and 128/1024 threads, while a 64x128x16 fp16 configur (suggested: rewrite)
- `cand-1376c224`: Only STATS_WARPS shows weak residual boundary headroom: latency improves from 19.7 ms at 8 warps to 19.6 ms at 4/2 and 19.55 ms at 1, but Triton cannot launch fewer than one 32-thread warp. FINALIZE_WARPS is also reported at its minimum boundary, but 1, 2, and 4 warps all have the same 19.6 ms media (suggested: rewrite)
  - STATS_WARPS wants decrease, blocked by threads
- `cand-2b4d5338`: No parameter is demonstrated to have hardware-blocked headroom. BLOCK_N improves through the tested maximum, but 256 was never tried and none of the 10 failures was a register/shared-memory/thread/OOM/compile failure: nine were bf16 correctness mismatches and one was an unrelated NUM_STAGES=4 runtim (suggested: tune_more)
  - BLOCK_N wants increase, blocked by none
  - LOGICAL_BLOCK_M wants increase, blocked by none
- `cand-2b4d5338`: The only strong blocked headroom is wider output tiling. BLOCK_N improves monotonically to the tried maximum of 256, but the best 64x256 accumulator already reaches 255 registers/thread; shared memory is only 40.4% of the 101,376 B opt-in limit, there are no spills, and 4 warps use only 128 of 1,024 (suggested: rewrite)
  - BLOCK_N wants increase, blocked by registers
  - BLOCK_K wants decrease, blocked by compile_failure
- `cand-66d76774`: No trial-proven register/shared-memory/thread/OOM blocker exists: all 11 failures are correctness mismatches, not resource failures. The best trial is 15.3 ms with 226/255 registers (88.6%), zero spills, 32,768/101,376 B opt-in shared memory (32.3%), and only 4 warps (128 threads) versus 1,024 threa (suggested: rewrite)
  - PW_BLOCK_N wants increase, blocked by arithmetic_throughput
  - PW_BLOCK_K wants decrease, blocked by compile_failure
- `cand-66d76774`: TRAIN mode applies all three BatchNorm2d layers using current-batch statistics. The only credible blocked boundary is PW_BLOCK_K decreasing below 16: latency improves strongly from K=64 to 32 to 16, but Triton tensor-core tl.dot requires a minimum K tile of 16. The apparent PW_BLOCK_M/PW_BLOCK_N bou (suggested: rewrite)
  - PW_BLOCK_K wants decrease, blocked by compile_failure
- `cand-0d0dcd49`: No tuned parameter meets the strict definition of remaining hardware-blocked headroom. FINAL_BLOCK=1024 is the only marginal boundary optimum, but its two trials bottom out at 10.5 ms, slower than the 9.14 ms global best at FINAL_BLOCK=512, and BLOCK=1024 with one warp does not consume 1024 CUDA thr (suggested: rewrite)
- `cand-0d0dcd49`: No tunable parameter is demonstrated to have blocked headroom. The only marginal boundary optima are GEMM_STAGES=1, FINAL_BLOCK=2048, and FINAL_WARPS=8, but their trends are confounded or non-monotonic and no trial beyond those boundaries failed because of a resource limit. The best actual trial ins (suggested: rewrite)
- `cand-d4fe9dfd`: No tunable parameter is proven to have blocked headroom. The marginal analysis flags EXPAND_WARPS=8 and PROJECT_BLOCK_K=128, but it is confounded by joint parameter changes: the global winner is 9.45 ms with EXPAND_WARPS=4 and PROJECT_BLOCK_K=64, while the best trial at either flagged maximum is 10. (suggested: rewrite)
  - EXPAND_WARPS wants increase, blocked by none
  - PROJECT_BLOCK_K wants increase, blocked by none
- `cand-d4fe9dfd`: No tunable parameter is defensibly hardware-blocked. The only boundary optimum is EXPAND_BLOCK_N=256 in marginal statistics, but the global best uses 128 and is faster (9.26 ms versus the fastest 256 trial at 9.64 ms), so increasing it is not demonstrated headroom. At the winner, 255 registers/threa (suggested: rewrite)
  - EXPAND_BLOCK_N wants increase, blocked by none
- `cand-dd62990b`: The remaining credible blocked headroom is in the coupled 1x1 GEMM tile, not in depthwise warps or BatchNorm tuning. GEMM_BLOCK_N improves monotonically through the maximum tried value (median 28.1 -> 24.5 -> 18.6 -> 17.5 ms for N=16/32/64/128), and GEMM_WARP_GROUPS improves to its maximum tried val (suggested: rewrite)
  - GEMM_BLOCK_N wants increase, blocked by registers
  - GEMM_WARP_GROUPS wants increase, blocked by registers
- `cand-dd62990b`: No parameter is demonstrated to have resource-blocked headroom. GEMM_GROUP_M is flagged at its sampled maximum only by confounded marginal medians (16.35 ms at 64 versus 16.25 ms at 128); the actual fastest trial uses 64 and is faster than the best comparable 128 trials (11.0 versus 11.4 ms), so 255 (suggested: rewrite)
- `cand-5a3182cb`: The only credible blocked tuning direction is a wider 1x1-GEMM output tile. GEMM_BLOCK_N improves strongly through the tested maximum of 128, but the winning 128x128 tile already uses 255 registers/thread and spills 4 registers; simply increasing N would enlarge the accumulator and exceed the regist (suggested: rewrite)
  - GEMM_BLOCK_N wants increase, blocked by registers
- `cand-5a3182cb`: The clearest blocked knob is GEMM_BLOCK_M: latency improves strongly through the maximum tested value, while the best 128x128 tile already reaches 255 registers/thread. This is a productive large-accumulator signature, not a reason to reduce registers: smaller/lower-register GEMM tiles are slower, a (suggested: rewrite)
  - GEMM_BLOCK_M wants increase, blocked by registers
  - GEMM_BLOCK_K wants decrease, blocked by compile_failure
  - DW_STATS_NUM_WARPS wants decrease, blocked by threads
- `cand-8eda41e5`: Two knobs show blocked boundary headroom. PW_BLOCK_M improves toward 128, but a 256x256 accumulator tile would likely exceed the 255-register/thread limit: the current 128x256, 8-warp kernel already uses 210 registers/thread, while shared memory is only 65,536/101,376 B, threads are 256/1,024, and s (suggested: rewrite)
  - PW_BLOCK_M wants increase, blocked by registers
  - PW_BLOCK_K wants decrease, blocked by compile_failure
- `cand-8eda41e5`: The remaining blocked tuning headroom is confined to the pointwise GEMM tile. PW_BLOCK_N improves monotonically through the tested maximum, but extending the best 128x256 tile to N=512 would approximately double its 65,536 B shared-memory footprint beyond the 101,376 B opt-in block limit and would a (suggested: rewrite)
  - PW_BLOCK_N wants increase, blocked by shared_memory
  - PW_BLOCK_K wants decrease, blocked by compile_failure
- `cand-8bd31d35`: Only PW_BLOCK_N shows credible boundary headroom, and it is now running into an arithmetic-throughput/whole-pipeline floor rather than a demonstrated register bottleneck. Latency improves strongly as N grows (median 29.4 ms at 16, 21.9 at 32, 19.05 at 64, 17.7 at 128, 17.1 at 256), but several mater (suggested: rewrite)
  - PW_BLOCK_N wants increase, blocked by arithmetic_throughput
- `cand-8bd31d35`: TRAIN mode was observed, so every BatchNorm must use current-batch statistics and update running state; inference folding is invalid. The tuned floor is about 16.2-16.7 ms. The fast expansion configurations already use fp16 or tf32 tensor-core tl.dot, while IEEE is much slower (best 20.0 ms; median  (suggested: rewrite)
  - PW_BLOCK_K wants decrease, blocked by compile_failure
  - PW_BLOCK_N wants increase, blocked by arithmetic_throughput
- `cand-0a02b061`: The actionable boundary is BLOCK_M: latency improves toward 128, but the 128x128 accumulator already uses 245/255 registers per thread; a direct 256-row tile would exceed the register budget. This is a modest opportunity, not evidence that registers should be reduced: the high register count is the  (suggested: rewrite)
  - BLOCK_M wants increase, blocked by registers
  - BLOCK_K wants decrease, blocked by compile_failure
- `cand-0a02b061`: Only BLOCK_K shows credible blocked boundary headroom. Latency improves as K shrinks (median 17.05 ms at 64, 14.7 ms at 32, 13.45 ms at 16), but 16 is the minimum legal/useful K fragment for this tensor-core tl.dot formulation. The best configuration's 245 registers/thread is not the demonstrated li (suggested: rewrite)
  - BLOCK_K wants decrease, blocked by compile_failure
- `cand-8e225f0a`: No tuned parameter is convincingly both headroom-bearing and hardware-blocked. The boundary signals for EXPAND_BLOCK_M=128, EXPAND_WARPS=8, and PROJECT_BLOCK_M=128 are confounded and flatten at roughly 10.1-10.6 ms across configurations using 92-246 registers and 32-37 KiB shared memory. The best tr (suggested: rewrite)
- `cand-8e225f0a`: No tunable parameter is demonstrably blocked by a hardware/resource limit. The two numeric boundary optima are weak or confounded: DW_WARPS=8 improves the aggregate median only from 12.1 to 12.0 ms, while the only controlled comparison makes 8 warps slower (13.1 ms versus 10.8/11.1 ms at 4); DW_STAG (suggested: rewrite)
  - DW_WARPS wants increase, blocked by none
  - DW_STAGES wants increase, blocked by none
- `cand-e6e7c9c5`: No parameter meets the requested definition of hardware/resource-blocked headroom. The apparent boundary winners are confounded marginal statistics, not demonstrated resource limits: the actual best trial is 6.92 ms at only 144/255 registers, 29,184/101,376 B opt-in shared memory, zero spills, and 4 (suggested: rewrite)
- `cand-e6e7c9c5`: No tuned parameter meets the strict definition of blocked headroom. The 6.90 ms best configuration uses only 144/255 registers per thread (56%), 29,184/101,376 B opt-in shared memory (29%), 128 threads/block, and zero spills. Higher-resource configurations are not faster: examples at 247-255 registe (suggested: rewrite)
- `cand-2dc4fe01`: Two boundary trends are plausibly blocked. GEMM_BLOCK_N wants to increase, but the 64x256 fp16 pointwise tile already reaches 255 registers/thread and spills 8 registers; shared memory is only 16,384 B, 16.2% of the 101,376 B opt-in limit, so register capacity rather than shared memory is the credib (suggested: rewrite)
  - GEMM_BLOCK_N wants increase, blocked by registers
  - GEMM_BLOCK_K wants decrease, blocked by compile_failure
- `cand-2dc4fe01`: The only clear resource-blocked tuning direction is GEMM_BLOCK_N upward. Latency improves monotonically through N=256, where the best configuration reaches the 255-register/thread limit; shared memory is only 16,384 B (16.2% of the 101,376 B opt-in limit) and the launch uses at most 8 warps, so neit (suggested: rewrite)
  - GEMM_BLOCK_N wants increase, blocked by registers
  - GEMM_BLOCK_K wants decrease, blocked by compile_failure
- `cand-83954796`: The only credible blocked tuning directions are EXPAND_BLOCK_M upward and EXPAND_BLOCK_K downward. The best configuration uses 255/255 registers per thread with 8 spills, but only 32,768/101,376 B (32.3%) of opt-in shared memory and at most 8 warps (256/1024 threads), so registers are the relevant c (suggested: rewrite)
  - EXPAND_BLOCK_M wants increase, blocked by registers
  - EXPAND_BLOCK_K wants decrease, blocked by compile_failure

## Convergence decisions

- global `global`: continue
- family `fam-4286a3be`: continue
- family `fam-a4a8353c`: continue
- global `global`: continue
- family `fam-fd92a2d8`: continue
- family `fam-a2688942`: continue
- global `global`: continue
- family `fam-a2688942`: continue
- family `fam-4286a3be`: continue
- global `global`: freeze (budget_exhausted)

## Rejections

- witness_default_failed: [default witness config {'GEMM_BLOCK_M': 128, 'GEMM_BLOCK_N': 128, 'GEMM_BLOCK_K': 16, 'GEMM_NUM_WARPS': 4, 'GEMM_NUM_STAGES': 2, 'COMPUTE_DTYPE': 'fp

## Agent usage

- successful calls: 99; failed (final): 0
- total cost: $0.0000
  - analyst: 42 calls
  - generator: 1 calls
  - parameterizer: 45 calls
  - repair: 1 calls
  - rewriter: 10 calls


_Elapsed: 12.816 h; candidates: 23_
