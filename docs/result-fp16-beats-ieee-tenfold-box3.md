# 低精度契约改动的首个正向实证:box3 L3:48 上 fp16 比 ieee 快 10.5 倍

**日期** 2026-09-11 · box3 (A800) `run-l3-48-20260911-052647`,resume 于 commit `b6315fa` · 24 trial 时的读数,全部来自 on-disk `events.jsonl`

此前对低精度契约改动的两份实证都是**间接**的:

- `docs/result-box2-precorpus-precision.md` —— 改动生效、8/8 候选正确,但真实收益是**补救**(0/48 改动前候选含补偿 dot),不是"拿到低精度"。
- `docs/result-split3-first-production-measurement.md` —— `split3` 1 胜 3 负,支持"做成 knob 而非建议",但那是 dot 模式的账,不是精度的账。

box3 给了第一个**直接**的:同一候选、同一 space、同一台机器,只换 `COMPUTE_DTYPE`。

## 逐精度最优(24 trial)

| `COMPUTE_DTYPE`/`DOT_MODE` | n | best ms |
|---|---|---|
| **fp16/plain** | 3 | **12.2844** |
| ieee/plain | 4 | 129.3809 |
| tf32/split3 | 1 | 308.9393 |
| bf16/split3 | 2 | 458.2641 |

**fp16 比 ieee 快 10.53×。** 两个最快的 trial 都是 fp16(12.28 / 24.94 ms),第三快才是 ieee 的 129.38 ms。

## 与基线的位置

| | median ms |
|---|---|
| eager | 13.9597 |
| eager_tf32 | 13.4349 |
| torch_compile | 8.5832 |
| torch_compile_tf32 | 8.0527 |
| **候选 best(fp16)** | **12.2844** |

已经**胜过 eager**(1.14×),尚未胜过 `torch_compile`(0.70×)。**这是 24 个 trial 时的中途读数,不是结果**;按语料经验,L3 的 best 在 40 trial 预算内还会继续下降,且这条 space 还没经过 K 扩展。

## 为什么这次是"精度的账"而不是别的

三点使归因干净:

1. **同一候选同一 space。** `cand-d2cf7928` / `sp-fc83f2bb`,只有 `COMPUTE_DTYPE` 不同,其余 6 个 knob 由 TPE 采样。fp16 的最优点(BL=64/BN=128/BP=64/NUM_WARPS=4/NUM_STAGES=2)与 ieee 的最优点(同 tile,NUM_WARPS=8/NUM_STAGES=2)几乎重合,所以差异不是 tile 选择带来的。
2. **10.53× 与 A800 实测屋顶比一致。** 本机 `calibration.json`(schema 5)测得:

   | 精度 | cuBLAS TFLOPS | Triton TFLOPS |
   |---|---|---|
   | fp32 | 19.01 | 18.08 |
   | tf32 | 111.46 | 100.99 |
   | fp16 | 226.17 | 193.35 |
   | bf16 | 233.96 | 199.78 |

   **fp32 → fp16 的屋顶比是 226.17 / 19.01 = 11.90×**(Triton 路径 10.70×),观测到的 10.53× 落在其内。所以这是**算力屋顶差**,一个数量级的加速不需要别的解释。

   注意一个我先写错又改掉的说法:A800 **有** TF32 张量核(实测 111.46 TFLOPS,是 fp32 的 5.86 倍),所以"ieee 慢是因为 A800 没有张量核通路"是错的 —— ieee 走的是 `input_precision="ieee"` 的真 fp32 路径,慢的原因就是那条路径的屋顶只有 19 TFLOPS。

3. **tf32 为什么没赢?** tf32 屋顶 111.46 TFLOPS 高于 fp32 的 5.86 倍,但表里 tf32 只有一个 `split3` 样本(308.94ms)、`tf32/plain` 的 2 次尝试全是 `correctness_mismatch`。这与记忆中 "tf32 0-for-172、根因是候选把 tf32 放在未补偿的 else 兜底" 一致 —— **tf32 的失败是候选侧的,不是屋顶侧的**,而它本该是这台机器上第二好的选择。这一条留给实验后统一分析。

4. **这个候选此前因 fp16 被拒过两次**(`witness_default_failed` tf32、`witness_minimal_failed` fp16),repair 第 3 次用指数重标定把它救回来。**如果见证门把这个候选丢掉,这 10.5× 就永远看不到** —— 这正是 `product-order-defeats-a-fallback` 那条修复要保护的东西。

## 与 L3:48 已有记忆的关系

记忆里 L3:48 有 "1.351 GB 固定流量、1.64ms 已达 911 GB/s 屋顶的 90.4%,平台期是物理" 的结论 —— 那是 **4090** 上的。本条是 **A800**,`torch_compile` 基线 8.58ms 与 4090 语料不可直接比,两者不冲突也不互证。

不过 A800 的屋顶这次是**已标定的**:`dram_tbs = 1.686 TB/s`。按 4090 语料测得的同任务 1.351 GB 固定流量,A800 的**纯带宽下界是 1.351 GB / 1.686 TB/s ≈ 0.80 ms**。当前 12.28 ms 距该下界还有 **15.3 倍**,所以这台机器上**远未进入平台期** —— 与 4090 上"只剩 ~10% 空间"是完全不同的处境,不能把那条结论搬过来。

（这个 0.80 ms 是**下界不是目标**:它假设流量与 4090 上相同,而 fp16 的 B/C 张量本就更省字节,真实可达值需要在本机实测,不能由此推断。）
