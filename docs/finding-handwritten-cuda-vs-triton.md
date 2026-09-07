# Step 8 结果:手写 CUDA 没有打败 Triton(同算法、同计时口径)

**日期**:2026-09-07 · **机器**:box 2 (RTX 4090, sm_89) · **脚本**:`linux-server/scripts/probes/probe_cuda_vs_triton.py`

## 问题

我们的 harness 默认生成 Triton。KernelPro 报告 raw-CUDA+CuTe 在一个 MoE kernel 上比手调 Triton 快
1.23x,用户的判断也是 CUDA 的表达上限更高。在花搜索预算去教 agent 生成 CUDA 之前,先在**我们自己的
计时口径**下量一次:同一个算法,手写 CUDA 能不能赢过 Triton。

## 测法(公平性是这次测量的主要成本)

- 同算法、同 tiling 轴、同精度路径:tf32 张量核 + fp32 累加,两边都是。
- **两边都扫配置**:Triton 7 个(BM/BN/BK × num_warps × num_stages),CUDA 10 个模板实例化
  (同样的 tiling 轴 + warp 排布 + 双缓冲 ~ 对应 num_stages)。只调一边就是在制造结论。
- CUDA 侧用了一个称职作者会用的手段:128-bit `float4` 向量化载入、shared memory staging、
  ping-pong 双缓冲(对应 Triton 的软件流水)。写标量载入然后管它叫"CUDA 的上限"只是在量我自己偷懒。
- 两边同一个计时函数 = harness 的口径(CUDA events + 每 trial 冲 L2),100 次取中位数。
- **正确性先判且是硬门**:错的配置直接从对比里剔除,不报它的延迟。

## 结果

| | median | vs cuBLAS |
|---|---|---|
| torch (cuBLAS) | 0.0389 ms | 1.000x |
| Triton 最好 (BM128 BN128 BK32 nw=8 ns=4) | 0.0635 ms | 0.612x |
| 手写 CUDA/wmma 最好 (BM64 BN64 BK32 warps2x2 dbuf) | 0.0727 ms | 0.535x |

**CUDA vs Triton = 0.874x**,即手写 CUDA 比 Triton **慢** 13%。两边相对误差都是 8.07e-04,
完全一致 —— 精度路径确实对齐了,这个数不是精度差异伪装成的后端差异。

配置扫描本身也提供了信息:两边的最优 tile 形状不同(Triton 偏大 tile 128×128,CUDA 偏小 tile
64×64),CUDA 侧最差配置(0.1556)与最好配置(0.0727)差 2.1 倍。所以"扫两边"不是形式主义,
只测一个配置的话结论会随手气翻转。

## 结论的适用范围(必须这样陈述)

一个手写候选测不出"CUDA 后端的上限",只能测出**这一个手写实现的表现**,而手写 kernel 的水平就是
写它的那只手的水平。可以说的是:**在这类 kernel 上,一个称职但非专家的手写 CUDA 实现拿不到超过
Triton 的收益,所以让 agent 生成 CUDA 的预期回报被这次测量压低了** —— 不能说 CUDA 后端更差。

反过来看 KernelPro 的 1.23x 也是一致的:他们的收益来自 CuTe(CUTLASS 的抽象层,自带 swizzle、
async copy、warp specialization),不是裸 CUDA C。我们这次量的是裸 CUDA C + wmma。要复现他们的
数,需要的是 CuTe 而不是"换个后端",而 CuTe 生成不在本期范围(standing 决定)。

## 对实施计划的影响

不改变任何既定项。这次测量的作用是**给"教 agent 写 CUDA"这项工作定价**:在 matmul 这类
compute-bound、cuBLAS 已经很强的 kernel 上,裸 CUDA 相对 Triton 没有可见的头部空间。如果后续要
投入后端工作,应该冲 CuTe/CUTLASS 而不是裸 CUDA C —— 而那是一个独立的、更大的决定。

一个附带的正面结果:`families.structural_signature` 已经把 backend 作为结构轴(prefix `cuda:`),
所以如果哪天真的加了 CUDA 候选,族机制不会把它误判成 duplicate。这条不需要改动。

## 过程中我自己的两个 bug(留档,因为它们正是"不公平对比"的两种形态)

1. 第一版 CUDA kernel **是错的**(max rel err 1.26):8 个 warp 被映射到 2×2 的 64×64 warp tile
   网格上,warp 4-7 写到了 tile 外面。而脚本照样打印了它的延迟(0.3594 ms)并算出 0.276x ——
   违反了脚本自己写下的规则 3。现在正确性是硬门,错的配置根本进不了对比。
2. 第一版只给 Triton 扫了 5 个配置、给 CUDA 一个固定配置。这对 CUDA 不公平,方向恰好和我可能
   希望的结论相反,但仍然是被制造出来的结论。现在两边都扫。
