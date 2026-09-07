# KernelPro 深度借鉴清单(针对当前实施方案)

来源:arXiv:2606.26453v2 "Optimizing CUDA like a Human: Micro-Profiling Tools as Expert Surrogates
for LLM-Based GPU Kernel Optimization"(Jiading Gai 等,AWS,CC BY 4.0)。
本文件是**第二轮**细读的产出 —— 第一轮只看了瓶颈分类方法(见
`research-bottleneck-classification-and-portable-thresholds.md`),这一轮找的是**对我们
0-9 实施方案有直接影响的其余内容**。所有数字都是文中原文。

---

## A. 最高价值:每工具命中率表(Table 9)—— 直接决定我们 Tier 1 的优先级

它在 42 个 KernelBench 任务上量化了**每个诊断工具的实际有效性**。
"Fires" = 该工具建议出现的优化轮次数,"HitRate" = 其中产生 **>1.5x 改进**的比例:

| 工具 | 覆盖任务 | Fires | **HitRate** | 平均增益 |
|---|---|---|---|---|
| **RegisterSpill** | 16/42 | 44 | **18.2%** | 1.44x |
| **OccupancyLimiter** | 35/42 | 241 | **13.3%** | **1.48x** |
| WarpStall | 35/42 | 318 | 11.0% | 1.41x |
| SharedMemTiling | 28/42 | 172 | 9.3% | 1.42x |
| **LaunchOverhead** | 24/42 | 141 | **8.5%** | 1.32x |
| TensorCore/cuBLAS | 42/42 | 835 | 8.1% | 1.31x |
| MemoryCoalescing | 37/42 | 439 | 5.9% | 1.24x |
| Vectorization | 41/42 | 595 | 4.0% | 1.16x |

### 对我们方案的三条直接结论

**1. 我们计划做的 Tier 1 三项恰好是命中率最高的三项之一/之二/之五。**
`RegisterSpill`(18.2%,最高)、`OccupancyLimiter`(13.3%,次高且**平均增益最高 1.48x**)、
`LaunchOverhead`(8.5%)—— 前两项正是步骤 5(SASS + 解析 occupancy)的内容,
第三项是**已实现**的 `cpu_issue_ms`。**这是对我们优先级的独立验证**,不是巧合。

**2. `WarpStall`(11.0%,第三高)我们做不到** —— 它需要 ncu stall 分解。
这量化了我们那个"唯一硬缺口"的代价:**约 11% 命中率、1.41x 平均增益的一类建议我们给不出**。
之前我把 warp 分歧评为"中低价值",**这个评估偏低了**:它排第三。
不过要区分:我评的是"warp 分歧"(divergence),这里的 `WarpStall` 是**stall 原因分解**,
范围更广(mem_dep / short_scoreboard / barrier),所以两者不完全等同 —— 但结论仍是我低估了。

**3. `MemoryCoalescing`(5.9%)和 `Vectorization`(4.0%)命中率最低。**
这两项我们本来也难做(需要 sectors-per-request counter),
**但按它自己的数据,这是最不值得追的两项**。可以安心不做。

---

## B. 反直觉发现:原始 counter 反而有害(统计显著)

Table 8 的成对 Wilcoxon 符号秩检验(单侧):

| 对比 | n | W | p |
|---|---|---|---|
| Ours > **Raw ncu** | 40 | 728 | **<0.0001*** |
| Ours > NoFeedback | 40 | 590 | 0.0078** |
| **NoFeedback > Raw ncu** | 39 | 618 | **0.0007*** |

**第三行是关键**:**不给任何反馈** 显著优于 **给原始 ncu 指标**(p=0.0007)。
也就是说把原始 counter 塞进 prompt **比什么都不给更差**。
文中解释:LLM 缺少把硬件 counter 翻译成优化动作的领域上下文。

微剖工具比原始指标高 **125%** 的加速。

**对我们步骤 7 的意义**:这不是"最好加个翻译层",而是**"不加翻译层会主动变差"**。
所以步骤 7 的 detect-analyze-recommend 结构是**必需**而非优化项。
我们的 `BottleneckVerdict.suggests` 字段设计方向正确,但必须确保
**evidence 里的原始数字不是唯一输出** —— 每个数字都要配一句"因此应该做什么"。

---

## C. 主动 vs 被动工具调用:确定性执行,不让 LLM 自己选

> "reactive invocation would introduce stochastic selection -- the LLM might call 3 tools and
> skip 10, missing critical analyses. Instead, KernelPro **deterministically executes all
> relevant tools** based on the Stage 1 bottleneck classification."

而 "bottleneck-relevant" 不等于 "all tools":Stage-1 分类喂给一个 **tool-affinity 过滤器**,
排除无关分析(memory-bound 跳过张量核检查;compute-bound 跳过 coalescing 分析),
**既防信号稀释,又保证覆盖真实瓶颈**。

**对我们的意义**:我们的架构本来就是 harness 主动采集(不让 agent 自己决定测什么),
**这一点已经对齐**。但可以借鉴 **tool-affinity 过滤**:
按分类结果决定**给 analyst prompt 里放哪些证据**,而不是一律全放 —— 防信号稀释。
这是步骤 7 可以直接采纳的一个细化。

---

## D. 对我们"假 converged"缺陷的独立印证

> "**Profiling prevents correct-but-slow plateaus.** Trajectory analysis reveals that KernelPro
> never produces a correct kernel that fails to exceed baseline (0/62 trajectories stuck at
> 1.0x), while NoFeedback gets stuck in **10% of cases**."

> "profiling does not fix bugs, but it prevents the model from settling on correct-but-slow
> solutions by always providing an **actionable optimization target**."

**与我们刚发现的假 `converged` 直接相关**:我们的 `fam-50ba7c87` 正是一个
"correct-but-slow 却被判收敛"的实例(5.54 → 5.54,从未评测改写)。
KernelPro 的做法是**总是提供可执行的优化目标**,让模型不会停在这里。

**对步骤 9 的补充**:除了抢救产物,还应考虑 —— 当一个族被判 `converged` 但
`rewrite_rounds_used` 对应的改写从未真正评测过时,**这不该算收敛**。
这比"抢救产物"更根本:抢救失败时(产物真的没写出来)也不该记为"没有改进"。

---

## E. MCTS vs 贪心:26% 更高几何均值(与我们"不做早停剪枝"的决定一致)

| 指标 | Greedy | MCTS |
|---|---|---|
| 几何均值加速 | 3.65x | **4.60x** |
| 中位加速 | 4.17x | **6.09x** |
| fast_p(>2x) | 59.5% | **70.3%** |
| fast_p(>5x) | 48.6% | **59.5%** |
| 正面对决胜场 | 8 | **29** |
| Wilcoxon p(单侧) | — | **0.004** |

> "MCTS is most valuable for tasks requiring multi-step optimization chains where greedy
> **converges prematurely to a local optimum**."

NetVLAD 轨迹(Task 46):**43 次连续失败**才出第一个可用解 —— "a search depth that would
exhaust any fixed-budget single-shot approach";但一旦出解,profiling 引导的精修收敛很快。

**对我们的意义**:这**独立支持了用户先前"不做早停/贪心种子选择"的决定**。
我们的 `active_families()` 已经按改进斜率而非绝对延迟排序,并强制每个族至少一轮 ——
方向与此一致。**我们不打算改成 MCTS**(那是搜索策略的重构,不在本轮范围),
但"43 次失败才出解"这个数据点值得记住:**它是我们那个 `frozen_converged` 判据过早触发风险的外部佐证**。

---

## F. 值得记录但本轮不做的三项

**1. search memory(跨迭代记忆)。** 它用一个 append-only 知识库解决三个失败模式:
重复错误、丢失发现(工具调用得到的文件路径/API 模式没传下去)、plateau 行为。
消融结果:**最终加速统计上等价**,但有**更早收敛**的趋势。
**与用户明确否决的"跨候选 report/hypothesis 共享"接近**,所以不做;
但注意它的消融结论是"等价而非更好",这**降低了我们不做它的代价**。

**2. CUTLASS/CuTe 代码生成。** 文中提到它在 VeOmni 的 MoE 训练 kernel 上
"achieves 1.23x over hand-tuned Triton by generating a from-scratch raw-CUDA+CuTe Hopper
WGMMA kernel"。这是**我们步骤 8(手写 CUDA 对比)的一个参考量级**:
1.23x vs 人工调优的 Triton。同时它印证了用户"CUDA 表达力更强 → 上限更高"的判断。

**3. 能量/功耗指标。** 它有一套 `nvmlDeviceGetTotalEnergyConsumption` 的能量测量,
并指出 A100/H100 板载功率传感器只采样约 25% 的运行时间,
**不校正 duty-cycling 会让能量测量平均差 35%(最高 65%)**。
我们不测能量,但这条方法论警告(板载传感器采样率陷阱)值得记住。

---

## G. 汇总:对 0-9 方案的具体修改建议

| 步骤 | KernelPro 带来的修改 |
|---|---|
| **5(Tier 1)** | **优先级已被独立验证**:先做 spill(18.2%)+ occupancy(13.3%,增益最高 1.48x),再做张量核检测。**不必追 coalescing/vectorization**(命中率最低 5.9%/4.0%) |
| **6(分类器)** | 已纳入双方法交叉验证 + 冲突采信解析值 |
| **7(接线)** | ① detect-analyze-recommend 是**必需**(不加会主动变差,p=0.0007);② 新增 **tool-affinity 过滤**:按分类结果筛选放进 prompt 的证据,防信号稀释 |
| **9(产物抢救)** | **加强**:不只抢救产物,还要区分"改写评测后没改进"与"改写从未评测" —— 后者不该记为收敛(对应 KernelPro 的 correct-but-slow plateau 预防) |
| **8(CUDA 对比)** | 参考量级:它报告 raw-CUDA+CuTe 比人工调优 Triton 快 **1.23x** |
| 诚实记录 | **我们无法复现 `WarpStall`(11.0% 命中率、1.41x)** —— 这是 counters 缺失的量化代价,比我先前评的"中低价值"要高 |
