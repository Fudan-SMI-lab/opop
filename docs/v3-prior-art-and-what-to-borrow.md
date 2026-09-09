# v3 调研:可借鉴的工作,以及怎么融进框架

2026-09-09。四路并行调研:性能模型与瓶颈归因、搜索与编译器代价模型、资源交换范例与 LLM kernel 生成现状、v2 内部研究文档盘点。

**标注约定**:**[已复验]** = 我亲自拉取原文核对过数字/公式;**[子代理]** = 子代理从原文报回、我未逐条复验;**[推断]** = 无出处的推论。凡进论文的数字,须按 v2 纪律再复验一次并带引用。

**关于第四路(性能模型)的标注**:该路子代理报告自己带了取证方法说明 —— `WebSearch` 对其模型不可用,它改用 Brave Search via WebFetch + Crossref API + arXiv API/ar5iv + **本地下载 PDF 并用 pdfminer 抽文本**,并逐条标了哪些是它亲自读到的原文。它给的公式、原话引文、DOI 与卷期页都带出处。**但它的复验对我仍是二手**,所以本文仍标 [子代理];Roofline / ECM / ERM / TMA / Instruction Roofline 的核心公式在进论文前须由我再拉一次原文。

---

## 一、最该借鉴的六件事(排序)

前五条来自前三路调研,第 6 条来自较晚回来的第四路(性能模型)。排序按"确定性 × 便宜 × 打中已实测缺陷"。

**⚠ 2026-09-10 两处修订,读本节前必看**:
- **借鉴 1 只适用于编译期可知的不可行**,不可推广到"实测已确立"的取值黑名单(误杀率实测 38.5%);
- **借鉴 6(ECM/ERM 的合成规则)已被用户否决,改为"记录但不采纳"**。
完整论证见 `v3-revision-no-scalarization-and-retire-risk.md`。

### 1. 编译期可知的不可行,应当从空间里"声明掉",而不是抽出再拒 —— 最高价值

**这是唯一同时"最高置信 + 最便宜 + 打中我们已实测缺陷"的一条。**

**⚠ 一处重要限定(2026-09-10)**:这条只适用于**编译期可知**的不可行(shared memory 超限、guard 拒绝、materialize 错误)。我曾把它推广到"**实测**已确立的不可行"(某个 categorical 取值 N 次全败即摘掉),**那个推广是错的,已撤回** —— 用 7 个历史 run 重放,N=6/8 的误杀率达 38.5%/25.0%,`PJ_BC=64` 这个真实通过 102/176 的取值会被摘掉,因为**失败几乎总是条件性的**(取决于搭档取值)而摘除是无条件动作。替代判据见 `v3-revision-no-scalarization-and-retire-risk.md` §1.5。**外部那几篇讲的都是编译期可知的约束,不要把它们的结论借到取值黑名单上。**

我们现状 [已复验代码]:`tpe.py:119-121` 把 `infeasible_shared_memory`/`guard_rejected`/`materialize_error` 报成 `PRUNED`,其余失败报 `FAIL`。而 v2 实测这类必然失败占 **180/1004 trial(18%)**。

两个独立团队实测这种"惩罚/填补"做法**主动有害** [子代理]:

- **Kernel Tuner**(Willemsen 等,PMBS@SC 2021,arXiv:2111.14991)**明确拒绝**惩罚填补:给无效点赋逆最优值或均值会扭曲 surrogate,他们改为把无效点**从 acquisition 优化中排除**。实测无效率:Convolution 38.5%、ExpDist 50.8%、PnPoly 3.9%、GEMM 0%。
- **Constraint-aware optimization**(arXiv:2606.28372, 2026)量化了机制,原话:用朴素惩罚时 "the performance of the non-constrained variants approaches that of the random search baseline **for the first half of the tuning time**"。**那前半段正是我们 40–80 trial 的全部预算。** 他们用"修复到最近合法邻居"取代惩罚,**平均 +39.3%**,增益与空间稀疏度相关。
- **HyperPower**(arXiv:1712.02446):把功耗当"低成本先验已知约束"过滤掉,**固定时间预算内多 57.20× 有效评估、收敛快 112.99×**。
- **MetaSchedule**(NeurIPS'22, arXiv:2205.13603)有一个 **trace validator**,在代价模型打分**之前**就拒绝越硬件上限的提案。

**怎么融进来**,按成本排序:
1. **ATF 式依赖域**(arXiv:atf-tuner.org / TACO 2021):把约束表达为"给定已绑定参数后本参数的域",例如 `BLOCK_K` 的域由 `BLOCK_M` 与 `num_stages` 决定 → 不可行配置**从不进入空间**,零 trial 代价。
2. **Triton 自带 `early_config_prune`** [已复验我们没用]:`prune_configs_by` 接受 `early_config_prune(configs, named_args, **kwargs) -> List[Config]`,是官方的"benchmark 前丢弃配置"钩子。
3. **修复到最近合法邻居**(距离 = 参数**值索引**的绝对差之和),用于约束无法干净声明的情形。

**预期校准**:他们的增益随空间稀疏度上升。我们只知道 **trial 层面**的必然失败率约 18%(180/1004),而 **trial 失败率 ≠ 空间稀疏度** —— TPE 非均匀采样,会反复往它认为好的区域去。**引用他们的增益数字之前,先在声明的域上枚举网格算出真实密度。**(初稿曾据 18% 反推"密度 82%",这是错的推断,已撤。)

**额外免费收获**:借他们的 **simulation mode** —— 把我们的 (config → median latency) 对 dump 出来离线重放,可**零 GPU 成本**比较 sampler。这是回答"repair 是否真的优于 FAIL"的唯一负担得起的方式。

### 2. Roller 的 reuse score = 我们的转化效率,而且零成本可算

**[已复验]**(拉取 OSDI'22 全文,公式逐字对上):

```
S_i = ( Q(T) − Q(T′_i) ) / ( F(T′_i) − F(T) )
```

原文:"Functions Q(T) and F(T) calculate the memory traffic and memory footprint when the computation is executed in the granularity of T"。`T′_i` = 把第 i 轴换成下一个对齐尺寸后的新 tile。

**这字面上就是"每多花一单位占用,省下多少流量"。** 停止条件 `if MemPerf(T′) > MaxComputePerf(T′.expr)` —— 消耗撞到供给屋顶就停。缩小策略:"shrink its rTiles along the axis that has the smallest data reuse score" 以恢复并行度。

**四条对齐规则**(即 Roller 的"供给向量")[已复验]:执行单元对齐(warp 32 / WMMA 16×16×16)、内存事务对齐、内存 bank 对齐、张量形状对齐(padding 浪费 ≤ ε)。消融:baseline 1.0× → MemAlign 1.42× → EUAlign 1.88× → ShapeAlign 1.92× → BankAlign **1.94×**。

**怎么融进来**:`S=ΔQ/ΔF` **纯静态可算**(从 tile 配置推,不需 trial、不需 counter)。用途:
- 给 K-expansion 提供的扩展方向**排序**;
- TPE 的 **warm-start / trial 排序先验**,在花 18.6 s 之前先排好。

**不借什么,以及为什么**:不借"构造取代搜索"。Roller 自己的数字 [已复验]:小算子上**比 Ansor 慢 50%**,tensor core 只到 cuBLAS 的 **43%**,作者自述盲点 "cannot detect implicit register allocation beforehand" —— **正是我们 Triton 候选所在**(218 regs / 0 spills / 16.7% occupancy)。

**更强的反证** [子代理]:**tritonBLAS**(arXiv:2512.04226,AMD 团队,我们同一个 Triton 3.4.0,只做最被研究透的 GEMM):解析选参达 **94.7% 选择效率**、选择耗时 50–80 µs vs autotune 的 11.9 s–1383.6 s,**但在真实 Llama3 形状上平均比 PyTorch 慢 13.9%**,且作者声明"not intended for other GEMM-like algorithms such as various attention mechanisms"。他们还自述目标只是"capture latency *trends*" —— **趋势精度对我们 9% 的近平局跨度毫无用处。**

**并且**:`S=ΔQ/ΔF` 是**估计**器,按设计文档 §1.4 的边界,**只能用于排序/warm-start,不得进入接受判定**。

### 3. 资源向量必须以"诊断"形式交付,绝不以数值形式 —— 硬约束

**[已复验]** KernelPro(arXiv:2606.26453,Gai 等)有一节标题就叫 **"5.3.1 Raw Metrics Are Harmful"**。42 个 KernelBench 任务、Sonnet 4.6、A100、15 seeds/task、30 迭代:

| 反馈 | 解决 | 几何均值 | 胜率 |
|---|---|---|---|
| 无反馈 | 32/42 (76%) | **3.35×** | 24% |
| 原始 ncu(~50 计数器,无解释) | 36/42 (86%) | **1.77×** | 7% |
| 消化后工具建议 | 42/42 (100%) | **4.00×** | 60% |

单侧 Wilcoxon:全量 > 原始 p<0.0001(n=40, W=728);全量 > 无反馈 p=0.0078(n=40, W=590);**无反馈 > 原始 p=0.0007**(n=39, W=618)。论文断言:"unstructured metrics actively degrade optimization quality"。它还点名这一臂 "comparable to CudaForge... which presents statistically pre-filtered metrics as verbatim name-value pairs"。

**非单调性值得单独记**:原始计数器把解决率从 76% 提到 86%,却把几何均值腰斩。**它让模型更擅长正确性、大幅更不擅长速度。** 单一指标会掩盖这件事。

**一个必须诚实交代的混淆** [子代理]:原始臂同时缺解释、缺 roofline 门控、缺工具过滤,所以严格读法是"未消化的高volume计数器倾泻有害",不是"剖析数据有害"。KernelPro 自己有更干净的对照:6 个生产 Triton kernel 上**两臂拿到完全相同的原始数据**,唯一差别是是否加工具指导 → 加了之后**速度高 36%、迭代少 31%**。

**同向独立证据** [子代理]:KernelBench 论文 §5.2.2 自测**给硬件规格对输出无显著影响**;§5.2.1 few-shot 优化范例**降低** fast_1(o1 L1 10%→6%),因为模型"attempt more aggressive optimization strategies"而失败更多。

**怎么融进来**:向量的**交付格式**固定为「严重度 + 根因 + 排序建议 + 预期 Δ」。原始向量只进 events.jsonl 与报告。最小忠实实现:
- **MAESTRO 的 "Runtime Bottleneck Info."** [子代理]:直接输出**哪个维度在绑定** + 其余维度余量;
- **cuPilot 的三区分类** [子代理](memory / **中间** / compute):二分标签会把我们的中间情形误路由 —— L3:48(带宽 90.4%)与 L3:43(融合头寸 69×)在三区下才分得开。

**并且:瓶颈必须我们自己算,不能问 agent。** [子代理] Bolet 等(AI4Sys@HPDC 2025, arXiv:2505.03988)在 340 个 HeCBench kernel 上测:**给了剖析数据,分类准确率 100%;只给源码,推理模型只到 64%**。64% 意味着 **agent 自述的瓶颈永远不能当证据**。

### 4. 保持单目标(median latency),资源作 attribute 上报或廉价约束前移

**不是"多目标更差",而是我们的预算下它会退化我们已有的 tuner。** 依据全是它自己的数字 [子代理]:

- **Optuna MOTPE**(从 master 源码读,比论文更可靠):多目标下 `default_gamma_multiobjective(x)=ceil(0.1x)` —— **只有 10% 的 trial 喂给 `l(x)`,n=40 时是 4 个点**;且 `multivariate` 默认 "True for single-objective... **False for multi-objective**" —— **静默关掉我们依赖的 tile/warp 相关性建模**;再加 10 个 startup 随机 trial(40 的 25%)。
- **qEHVI/qNEHVI**(NeurIPS'20/'21):复杂度 `O(MNK(2^q−1))`,"K is super-polynomial in M",且 "**the number of boxes required... is unknown for M ≥ 4**" —— 我们向量 5–6 维。DTLZ2 实测 acquisition 墙钟:qParEGO 5.86 s(M=3)vs 精确 qEHVI **45.52 s(M=3)/ 459.33 s(M=4)** —— M=4 时 acquisition 就是**25 个整 trial**。
- **Optuna 自己的文档警告**:目标多了 "a large fraction of trials may become non-dominated due to the curse of dimensionality" —— 我们的"前沿"会等于 trial 日志本身。它自己给的出路是 "consider modeling some objectives as constraints"。

**怎么融进来**:
- 目标保持 `robust_ms` 单一;资源进 trial attribute 与报告。
- 廉价约束**前移到空间生成**(见借鉴 1),不进目标函数。
- 借 **SCBO 的 bilog + 总违约排序** [子代理]:`bilog(y)=sgn(y)·ln(1+|y|)` "magnifies the range around zero to emphasize the change of sign that is decisive for feasibility";无可行点时选"总违约最小"者。我们的 shared memory 有**连续 slack**(`101376 − required`),属于容易的一档,不需要 Gelbart 的二值机制。
- **迁移 API**:Optuna 的 `constraints_func` 在 v5.0.0 已弃用(v7.0.0 移除),替代是 `Trial.set_constraint(key, value)`;且 `constraints_func` "won't be called when trials fail or they are pruned" —— **正是我们的失败情形**。
- 若真需要前沿:用**随机标量化**(qParEGO 式),仅凭 acquisition 成本证据(5.86 s vs 459.33 s)。

### 5. 用 Halide 的特征分类法给向量"加密度",并按机器测交换比

**Halide 39 个 schedule 特征**(SIGGRAPH'19 附录 A)[子代理] 说明我们"每资源一个标量"分辨率不足,三条具体升级:
1. **字节数与工作集在每个层级分别记录** —— 6 个 byte-count(realization/production/root × 全量/innermost)、**5 个 working-set 尺度**;
2. **unique vs total 字节是两个独立特征** —— 差值**就是**复用;
3. **`parallel_launch_cost` / `malloc_cost` 是一等成本项** —— 佐证启动开销是真维度而非舍入误差。

再加两个:
- **Ansor 的算术强度曲线**:每个 loop level 采 10 点,而非一个标量。
- **CUDABench 的 `Performance-Score`** [子代理]:`Attainable = min(Peak FP, Peak BW × AI)`,`Score = Achieved/Attainable`,"reduces to Memory Bandwidth Utilization for memory-bound kernels and to Compute Utilization for compute-bound kernels" —— 这是"消耗投影到供给"的**单一可审计标量**,**无 counter 可算、跨卡稳定**(A40 vs 4090 一致),适合当头条上报量。

**丢掉学习部分**:Halide 需 **1.6M 个已 benchmark 的 schedule**(50k 程序 × 32)—— 按 18.6 s/trial 是 **827 年**;Ansor ~25k;Mind Mappings 10M。**结构性不可用,必须标注。**

**两条横切警告**:
- **交换比是设备特定的** [子代理]:model-steered tuning(arXiv:2211.07260)同一交换比在 A100 给 **+50.9%**,A4000 只 **+5.8%**。→ **必须每台机器实测,绝不继承。**
- **加"峰值显存"作上报维度**:KernelBench-Verified [已复验] 实测**28% 的 kernel 会抬高峰值显存**。

### 6. ECM 的合成规则 + ERM 的重叠系数 α —— **记录,但不采纳**(2026-09-10 用户否决)

这一条来自较晚回来的第四路调研,详见 §3.1。**我原本把它排在价值第 2 位并建议采纳。用户否决了这个方向,我接受。** 本节保留为调研事实的记录。

外部的确有确切形式:

- **ECM**:`T = max(T_OL, T_nOL + T_data)` —— 不可重叠的核内周期与传输时间**相加**,其余与该和取 **max**。重叠假设是模型的**可证伪内容**。
- **ERM**:`α = T_overlap / min(T_x,T_y) ∈ [0,1]`,在 `sum`(α=0)与 `max`(α=1)之间**连续插值,按资源对实测**。

**为什么不采纳(用户的三条理由,均成立)**:

1. **借来的公式携带借来的假设。** ECM 的两条重叠假设是在显式管理的 CPU 缓存层级上成立的;而本文档 §3.10 末尾自己的警告是"没有一个在带 cache、带 warp 调度的 GPU 上验证过,且它们的误差带都比我们要分辨的近平局跨度更宽"。**我在写那条警告的同时又建议采纳它的公式,这是自相矛盾的。**
2. **合成本身不是目标。** 不同维度可以按不同维度独立考虑。"把向量压成一个数"是我引入的需求,不是项目要解决的问题 —— α 的**唯一用途**就是合成。
3. **维度数不固定。** 随设备变(V100 无 bf16、4090 上 tf32≈fp32),且**每次算子结构改动后占用率都会变**。任何要求"分量可比、可加、数目固定"的规则都与此冲突。

**连带不采纳的三处**(都预设了合成):TMA 式闭合(守恒分母 + 残差项)、`Performance-Score` 作头条量(`min(Peak FP, Peak BW×AI)` 就是压成标量)、Gables 的四步排序自检(它检验合成规则)。

**仍然采纳的**:把内存**拆成**带宽压力与延迟暴露两维(这是拆分而非合成);分层 roofline 的"同一分子多个分母"(每层一个独立读数);Instruction Roofline 的墙(单维自己的坐标)。

**并且这次否决反而给出一条更强的新颖性主张**:调研 §3.9 的结论是现存所有多资源模型最终都收敛到 2D 图或标量界 —— ERM 用 α、ECM 用重叠假设、Gables 用 `max`、SOL 在维度间取 max、CUDABench 压成 Score。**"拒绝合成、维持多份并列判决、维度集合运行期发现"在约 120 篇里没有对应物**,而且它是**设计选择**而非测量声明,审稿人无法用"2014 年就有资源向量"来驳回。

**顺带保留一条与合成无关的事实**:Kerncraft(源码级自动建模 + layer-condition 预测最优 blocking)与 OSACA(汇编级静态吞吐分析)**整族 counter-free**,是唯一能在租用容器原样跑的先验工作族。若将来要做静态分析仍可参考。

---

## 二、值得明确教给 agent 的资源交换(按实测杠杆排序)

每条都是"花什么、省什么",带可引用出处。[已复验] 仅 FlashAttention 与 A100 精度倍数两项,其余 [子代理]。

| # | 交换 | 花 → 省 | 实测数字 |
|---|---|---|---|
| 1 | **精度 → 算力+带宽** | 数值精度 → 吞吐 | A100 dense:TF32 **8×** fp32,BF16/FP16 **16×** [已复验 datasheet]。代价:FA-3 量化了 fp8 误差(比基线 fp8 低 2.6×) |
| 2 | **算力 → DRAM 流量**(寄存器内重算) | +13% FLOP → −9.2× 流量 | FlashAttention [已复验]:66.6→75.2 GFLOP,40.3→4.4 GB,41.7→7.3 ms(**−5.7×**)。原文:"even with more FLOPs, our recomputation speeds up the backward pass" |
| 3 | **寄存器 → shared/L1 流量**(register blocking),接受更低 occupancy | 寄存器 → 片上带宽 | Volkov(GTC 2010)SGEMM:137→**485 Gflop/s(2×)**,寄存器 21→**63**,occupancy 67%→**33%**。机制:"shared memory bandwidth is 6x lower than register bandwidth on Fermi" |
| 4 | **重算 → 显存容量**,带曲线 | 算力 → 容量 | Chen 2016:O(√n) 显存,+30% 时间;1000 层 ResNet **48GB→7GB(6.9×)**。Korthikanti 2022 选择性版本:显存 5× 更低、重算开销削 **>90%**,MFU 42.1%→**54.2%** |
| 5 | **融合 → 四种花费换两种节省** | 寄存器/shared/occupancy/厂商库快路 → 启动开销+中间流量 | Ivanov(MLSys'21):statistical normalization 占 **0.17% FLOP 但 25.5% 运行时**;数据移动削 22.91% → 1.30×/层。**文献缺口:没有论文把融合的寄存器压力代价作为头条数字报告** |
| 6 | **片上容量 → DRAM 流量**(tiling) | shared+寄存器 → 流量 | Roofline ridge point。我们自己:180/1004 trial 死在可编译期知的不可行 tile |
| 7 | **全局流量+归约 pass → 并行度**(split-K) | workspace+第二个 kernel → 并行度饥饿 | CUTLASS 文档**只有定性指导、无交换比** —— **这是我们可以贡献一个数字的地方** |
| 8 | **调度复杂度+库访问 → 启动开销与尾部效应**(persistent/megakernel) | 手工调度 → launch+bubble | Hazy Research 2025:H100 带宽 **78% vs vLLM ~50%**;代价明列(B200:40 µs warp 同步 + 80 µs setup) |
| 9 | **异步/warp specialization → 串行化** | 调度复杂度 → 流水气泡 | FA-2:峰值 25–40%→**50–73%**;FA-3 在 H100 35%→**75%** |
| 10 | **数据移动类别作先验** | —— | Ivanov 三分类:tensor contraction / statistical normalization / element-wise。**先分类再剖析** |

**第 2 条的方向性最重要**:获胜的一步让 FLOP 数变**差**了。任何把 FLOP 当要最小化的量的框架都会否决 FlashAttention。

---

## 三、性能模型侧的可借鉴结构

这一路的调研最深(全文级复验了 Roofline、Instruction Roofline、TMA 的 Intel 指标库、ECM、ERM、Gables、DrGPU),结论比预期强得多:**其中三项直接回答了设计文档里悬空的问题,且全部是 counter-free 的。**

### 3.1 ECM + ERM:向量怎么合成一个数 —— 本次调研最有价值的单项

设计文档 §二缺口 2 曾写"我们目前隐含用 max(取最紧的那维),应把假设写明"。ECM 给了确切的形式:

```
T_core = max(T_nOL, T_OL)
T_ECM  = max(T_nOL + T_data, T_OL)
```

**既不是相加也不是取 max,而是结构化的部分重叠**:核内工作按"能否与数据传输重叠"分成两类,`T_nOL`(不可重叠,如退休 load 的周期)与 `T_data` **相加**,`T_OL`(其余一切,含流水气泡)与该和取 **max**。ECM 明确声明了授权这么做的两条假设:"core cycles in which loads are retired do not overlap with any other data transfer in the memory hierarchy, but all other in-core cycles (including pipeline bubbles) do";以及"the transfer times up to the L1 cache are mutually non-overlapping"(所以各级传输时间**相加**)。出处:Hofmann/Eitzinger/Fey,arXiv:1509.03118;Stengel 等 ICS'15,arXiv:1410.5010。

**ERM 把"重叠多少"变成一个可测标量**(Cabezas & Püschel,IISWC 2014):

```
T_{x,y} = T_x + T_y − T_o_{x,y}
α       = T_o_{x,y} / min(T_x, T_y)  ∈ [0,1]
```

原文语义:"If α = 0 there is no overlap and the total execution time is the sum … if α = 1, the overlap is maximal and hence `T_{x,y} = max(T_x, T_y)`"。全局界:`max_x(T_x) ≤ T ≤ Σ_x T_x`。

**α 就是我们框架里缺的那个数。** 我们有消耗向量与供给向量,缺的是"两个维度合起来时怎么互相作用"。α 告诉改写 agent 一件它一直在猜的事:**为省流量而多花的算力,是免费的(α≈1)还是一比一付账的(α≈0)。** 我们的 fp16/tf32 与重算类改写全都押在这个未知量上。

**并且 α 可以只用墙钟测**:比较 kernel 实测时间与其两个分量时间的 `sum`/`max`,是一个两点实验。不需要 counter。

**ECM 这一族整体 counter-free,这是决定性的**:**Kerncraft**(Hammer 等,arXiv:1702.04653 / arXiv:1509.03778)从**循环源码 + 机器描述 + 问题规模**自动建 Roofline 与 ECM 模型,含预测最优 blocking 因子的 layer-condition 分析;**OSACA**(Laukemann 等,arXiv:1809.00912)从**汇编**静态算指令吞吐与关键路径,即 `T_nOL`/`T_OL` 的切分。**这是唯一一个能在租用容器里原样跑的先验工作族。** 我们已能读 cubin(Tier 1),OSACA 式的静态 pass 就能给出 `T_nOL`/`T_OL`,零 counter。

ECM 精度(逐级 {L1|L2|L3|Mem}):`ddot` 误差 {5%|17%|20%|13%},`load` {0%|15%|25%|23%},`copy` {5%|33%|8%|6%},STREAM triad {3%|25%|9%|2%}。**L2 是系统性弱项** —— 原文"in none of the cases the measured L2 performance could live up to the advertised specs of 64 B/c",需要一个约 1 cycle/load-stream/cache-level 的经验修正。另一篇对照(arXiv:2601.06886)报 Roofline 误差 42%–256% vs ECM 5%–117%。**教训:若我们建分层流量模型,别让任何判决挂在 L2 那一项上。**

**ERM 是我们最近的先验工作,必须诚实引用。** 它同时具备我们提的四件事:(1) 资源向量 `U_x = N_x/(T_x·Π_x)`;(2) 为什么欠用的三层分解 `U_issue`(缺 ILP)/`U_lat`(延迟暴露)/`U_stall`(乱序缓冲容量,RS/ROB/SB/LB/LFB 各自跟踪);(3) 显式重叠量 α;(4) 跨层级共同分母 `I = W/(Q_L1+Q_L2+Q_L3+Q_mem)`,其推论他们自己标为根本性——"the memory bounds … now depend on program and input",**屋顶不再是机器常数**。它的动机原话正是我们的动机:原始 roofline "is inherently blind to other bottlenecks, in particular non-throughput resources including cache capacity, latency of memory accesses or the functional units, and out-of-order (OoO) execution buffers"。

**不借 ERM 的机制**:它要一份完整的 LLVM 解释器级 DAG 调度(22 参数微架构模型),且精度恰在支配 GPU kernel 的效应上失效(硬件预取、冲突缺失、无 SIMD 建模)。**取它的量,用墙钟消融而非模拟去得到。**

**借 `U_issue`/`U_lat`/`U_stall` 作单维度的子分解**:"内存 85% 利用"远不如"内存发射没问题,损失在延迟暴露"可行动——前者指向减流量,后者指向加并发。**TMA 的 Bottleneck View 独立到达同一条缝(见 3.3),这是它真实存在的好证据。**

### 3.2 Instruction Roofline 的"墙":编译前可算的访存模式坐标

Ding & Williams,PMBS@SC19,DOI `10.1109/PMBS49563.2019.00007`(期刊版 CCPE 34(20) 2021)。轴:y = **warp GIPS**,x = **指令强度 = 每事务的 warp 指令数**。界:`GIPS ≤ min(Peak GIPS, Peak GTXN/s × Instruction Intensity)`。

**关键结构:强度被限死在 [1/32, 1],且若干位置有已知名字的"墙"**——
- **stride-0(广播)= 1**
- **单位步长 = 1/4**(FP32/INT32)或 **1/8**(FP64)
- **stride-8(FP32)/ stride-4(FP64)/ 随机 / gather = 1/32**
- shared memory:**无 bank 冲突 = 1**,**32 路冲突 = 1/32**

**候选相对这些墙的横坐标,就是其访存模式的直接读数** —— 不需要看源码,也**不需要 counter:它可以从 Triton 的 tile/stride 配置静态推出来。** 这与我们已有的编译期 shared-memory 探测同一类收益(那一类省下了 180/1004 个 trial)。

**第二件可借的:线程谓词化作为一个两线垂直间隙。** `inst_executed`(warp 级)与 `inst_thread_executed/32`(线程级)之比就是谓词化程度;原文案例"the dots are well below the dotted line indicating a 2× loss in performance due to thread predication"。HPGMG 的 `GSRB_BRANCH` 与 `GSRB_FP` **执行时间完全相同而 GFLOP/s 差 2 倍**,资源利用率(每 warp 活跃线程比)**50% vs 100%**。

**这正是会抓住我们 `if bn.training:` 死分支的信号**(31 个 trial 全测 fallback、零报错、全 run 最差)。**注意:warp/thread 计数本身需要 counter → 被 `ERR_NVGPUCTRPERM` 挡住;但谓词与分支结构在 cubin 里可见**(我们 Tier 1 已在读 cubin)。

GV100 供给数字(可作我们自测的形状参考):理论峰值 `80 SM × 4 warp scheduler × 1 inst/cycle × 1.53 GHz = 489.6 GIPS`;L1/L2/HBM 的 14000/2996/828 GB/s 在 32 B 事务下 = **437.5 / 93.6 / 25.9 GTXN/s**;shared 是 128 B 事务 → 109.3 GTXN/s。L1 分母规则:`1 × global_transactions + 4 × shared_transactions`。

### 3.3 TMA:可加性会计的两条结构规则 + 输出形状

Yasin,ISPASS 2014,DOI `10.1109/ispass.2014.6844459`。**公式是从 Intel 自己的机器可读指标库读出来的**(`intel/perfmon`,`skylake_metrics.json`,TmaVersion 5.01),不是二手:

```
SLOTS           = 4 * CPU_CLK_UNHALTED.THREAD
Frontend_Bound  = IDQ_UOPS_NOT_DELIVERED.CORE / SLOTS
Retiring        = UOPS_RETIRED.RETIRE_SLOTS / SLOTS
Bad_Speculation = (UOPS_ISSUED.ANY − UOPS_RETIRED.RETIRE_SLOTS
                   + 4 * INT_MISC.RECOVERY_CYCLES) / SLOTS
Backend_Bound   = 1 − Frontend_Bound − Retiring − Bad_Speculation
```

**为什么它的百分比真能加到 100%,分两部分,这才是要借的东西:**
1. **一个守恒的分母。** 每个周期的每个 issue slot 被决策树分入且仅分入一类。加和为 1 **不是事后归一化,而是"划分一个可数资源"的推论**。
2. **有且仅有一个残差类。** `Backend_Bound` 被**定义为** `1 − 其余`。闭合由构造保证,不靠期望。**这正是"某一维不可测"时向量仍可加总的办法。**

**输出形状借 TMA v4+ 的 "Bottleneck View",不借那棵 6 层树。** 它是一个**扁平、具名、加和 100% 的向量**(约 12 项),由对树中路径重加权得到:`Bottleneck_Mispredictions` / `Big_Code` / `Instruction_Fetch_BW` / **`Cache_Memory_Bandwidth`** / **`Cache_Memory_Latency`** / `Memory_Data_TLBs` / `Memory_Synchronization` / `Compute_Bound_Est` / `Irregular_Overhead` / `Branching_Overhead` / `Useful_Work` / `Other_Bottlenecks`(= 100 − 其余)。每项文档写作"**Total pipeline cost of** …",即**归因到某资源的代价**。

**最该抄的一条:带宽压力与延迟暴露是两个不同的维度。** TMA 靠在 L4 用 `MEM_Bandwidth/(MEM_Bandwidth+MEM_Latency)` 重加权、并把 L3 的 `SQ_Full` 与 L1 的 `FB_Full` 拉进来实现这个切分——它们本来住在不同子树里。我们目前**不做这个切分**,而它恰好区分了两种完全不同的改写策略("减字节" vs "加 MLP/occupancy")。ERM 的 `U_lat` 独立到达同一处。

TMA 是 x86-only,**纯结构捐赠者**。

### 3.4 有没有 GPU 版的 TMA:有,研究级 —— DrGPU

Hao 等,ICPE 2023,DOI `10.1145/3578244.3583736`。树根 `stall_cycle% = (ideal_IPC − achieved_IPC)/ideal_IPC`(ideal = Volta 起每 SM 每周期 4 warp 指令),同时覆盖垂直停顿(该周期无指令发射)与水平停顿(发射槽未填满)。取 NCU 的 18 个 warp scheduler state,保留 **13 个**与发射停顿相关且单元归属明确的,归为 **5 类**。加速比 **V100 上至 1.77×、GTX 1650 上至 2.03×**。

**最可移植的一块:内存子树用解析延迟模型而非 counter 做拆分**——

```
C_L1 = L1_latency + cc·(L1_per_inst − 1)        # cc = 每多一个事务 2 周期
C_L2 = L1_miss_ratio · L2_latency
C_DM = L1_miss_ratio · L2_miss_ratio · DM_latency
```

V100 常数 `L1_latency=28`、`L2_latency≈200`、`DM_latency=250` 周期。归因规则原话:"The percentage of stall cycle contribution on each memory layer is **normalized to the sum** of average stalls in all the layers"。

**唯一的测量输入是 miss ratio,延迟全是 datasheet 常数。若能从 tile 形状解析估出 miss ratio,整棵内存子树就 counter-free 可算** —— 而它直接给"合并/bank 冲突"这一维标价,正是 3.2 的墙所识别的那一维。

**闭合方式的差异要记下**:DrGPU 靠"对总和归一化"闭合,既非守恒划分也无残差项 —— 这比 TMA 弱:它保证百分比加到 100%,但不保证它们**意味着**损失的 100%。可以借,但我们的输出里必须把这个警告一起说出来。

**如实现所述需要 NCU counter + PC sampling → 被挡;它的延迟模型不被挡。**

### 3.5 Roofline 原文:gap-height 规则 + "屋顶顺序是 kernel 相关的"

**已全文复验**(读的是更长的技术报告版 UCB/EECS-2008-134):`Attainable GFlops/s = Min(Peak FP, Peak BW × Operational Intensity)`,ridge point = "达到峰值所需的最小操作强度"。

**它在两条屋顶下还叠了五条具名 ceiling,并且给了一个编码策略的顺序**:算力侧 (1) 提升 ILP + SIMD、(2) 平衡乘加混合;内存侧 (3) 改成单位步长、(4) 内存亲和性(NUMA)、(5) 软件预取。三句原话:"you cannot break through a ceiling without performing the associated optimization";"to break through a ceiling, you need to have already broken through all the ones below";**"The height of the gap between a ceiling and the next higher one is the potential reward for trying that optimization."**

**最后这一句就是转化效率的雏形,而且它是在花代价之前算出来的。** 排序规则也写明了:"those most likely to be realized by a compiler or with little effort by a programmer are at the bottom and those that are difficult to be implemented by a programmer or **inherently lacking in a kernel** are at the top"。

**而论文自己承认这个顺序不是机器常数**:对 SpMV "we would place floating-point mix as the lowest ceiling, since it is inherent"。**这正是我们 L3:48 的张量核事故** —— 一条"高阶"屋顶在那个任务上是**天生不可用**的,8/8 张量核候选撞死在上面。**所以 ceiling 顺序必须按任务从实测可行性重新推导,不能写进 prompt 常量。**

Opteron X2 实测 ceiling 数字(形状参考):峰值 17.6 GFlop/s;无平衡乘加 **8.8**、再无 ILP/SIMD **2.2**;内存侧无软件预取 **11 GB/s**、再无亲和性 **4.8**、仅单位步长 **2.7**。Ridge point 从 X2 的 1.0 移到 X4 的 **4.4**。

### 3.6 Hierarchical roofline:同一分子,三个分母 —— 以及一条我们正缺的天花板

Yang/Kurth/Williams(CCPE 2019);arXiv:2009.02449、arXiv:2009.05257。层级 **L1 / L2 / HBM**:FLOP 固定,只换分母,于是每个 kernel 变成**一个三元点组**,各自对自己那条带宽屋顶。

**单条 roofline 拿不到的:三元点组的间距就是缓存复用的测量值。** 原文:"Triplets of circles close to each other present a 'streaming' data access pattern and indicate poor cache locality";L2↔HBM 间距大则"the kernel benefits from high L2 data locality"。它抓的陷阱:kernel 在 HBM 层读作 compute-bound("The 13 FLOPs/Byte arithmetic intensity shows that this kernel has well entered the compute bound region on the HBM level"),实际被 L1/L2 带宽限制 —— 单层图完全看不见。

**复用信号是流量向量的分量之差,不是一个 counter。** 我们已按任务解析算 FLOP/字节;补上按 tile 形状解析的分层流量,三元点组就在租用容器里可算。**发表的*测量*方法是 ncu-only → 挡;*模型*不挡。**

**一条直接补我们 P9 缺口的天花板数字**:同族论文实测 V100 —— FP16 CUDA core 29.182 TFLOP/s;**张量核经 cuBLAS 达 103.7 TFLOP/s = 理论的 96.5%,而经 WMMA 只有 58 TFLOP/s = 54%**。**Triton 发的就是 WMMA。** 所以张量核候选被拿 cuBLAS 屋顶去比,注定读出结构上不可达的低利用率 —— 这正是 v2 "fp16 候选读出 107.8%" 与 "P9 per-backend 天花板" 两条记录的外部对应物。**校准应带两条张量屋顶:WMMA 可达 与 厂商库可达。**

### 3.7 Nsight SOL:借"维度内取 max",不借"维度间取 max"

Nsight Compute Profiling Guide。原话:"Throughput metrics return the **maximum percentage value of their constituent counters**. These constituents have been carefully selected to represent the sections of the GPU pipeline that govern peak performance." 即每个维度的值是**其子计数器上的 max-reduce**;`breakdown:<throughput-metric>` 可取回构成项。

**借**:维度内取 max 是对的(一个维度受其最紧子单元限制),而且"identify the highest contributor" 的 breakdown 正是改写简报需要的。
**不借**:"百分比高的那个就是 limiter" —— 它止步于一个标签,不含"要付什么代价才能移动它"。我们两条实测事故都是这个缺陷:fp16 候选 107.8% 读成"已到顶"(实际 54–60%),以及 L3:48 90.4% 是真平台期而正确动作是**换维度减流量**、不是在同一维度上加力。

**一条免费的自检**:文档明写 "**Percentages of sustained rate can occasionally exceed 100% in edge cases**"。**我们任何把 >100% 当"不可能因此有 bug"的门都会误判 —— 应当 clamp 并打标记,不是拒绝。** 我们已经真的撞上过 107.8%。

**SOL 100% 由 counter 派生,且文档里点名了我们的错误串**:"`ERR_NVGPUCTRPERM` - The user does not have permission to access NVIDIA GPU Performance Counters"。**只借结构,永远跑不了。**

### 3.8 Gables:借那条负面结果

Hill & Reddi,HPCA 2019,DOI `10.1109/HPCA.2019.00047`。N 个 IP 各有 roofline;`T_IP[i] = max(D_i/B_i, C_i)`,`T_memory = (Σ D_i)/B_peak`,`P = 1/max(T_IP[0..N-1], T_memory)`;对偶形式里 **`I_avg` 是按工作占比加权的调和均值**。

**它的算例是这套思想最干净的公开演示**:起点 `P_peak=40 Gops/s, B_peak=10 GB/s, A_1=5, B_0=6, B_1=15, I_0=8, I_1=0.1, f=0` → 40 Gops/s,CPU 受限、GPU 闲置(本可做 200)。把 75% 卸载给 GPU → 性能**塌到 1.3 Gops/s**,因为 GPU 复用差(`I_1=0.1`)使共享 DRAM 带宽成为绑定约束。把 `B_peak` 从 10 提到 30 GB/s → 只到 **2 Gops/s**("incurring additional expense without benefit")。把 `I_1` 从 0.1 提到 8 **同时把 `B_peak` 从 30 降回 20** → **160 Gops/s**,"a perfectly balanced design"。

**四步里有三步是"把某个资源朝局部正确的方向动,然后变差"** —— 因为资源通过共享约束耦合。第三步(花 3 倍带宽只换 1.5 倍,然后把带宽还回去、改提复用)是带数字的转化效率故事。**当作我们的动机例子,也当作一个排序自检:我们的框架应当能复现这四个配置的排序。**

Snapdragon 835 实测里有一条校准教训:DRAM **15.1 GB/s = 30 GB/s 理论峰的 50%**,因为该 kernel 同时读写;只读变体约 20 GB/s。**屋顶必须按 kernel 真实的读写混合去测** —— 我们已经为此付过一次学费(真屋顶 924.1 GB/s 而非 911)。**Gables 只需要微基准与规格,不被挡。**

### 3.9 有没有人做过真正的"多资源向量"性能模型

**诚实答案:没有。** 存在的要么是(a) 一**族** 2D 图,要么是(b) 一张 2D 图上画**很多条界线**。没有哪篇的状态是一个 N 维资源利用率向量、且任意两维之间有交换率。

| 扩展 | 出处 | 仍是 2D? |
|---|---|---|
| Hierarchical Roofline | Yang/Kurth/Williams 2019 | 是 —— 3 点 3 屋顶 2 轴 |
| Cache-Aware Roofline (CARM) | Ilic 等,IEEE CAL 13(1):21–24, 2014 | 是 —— 一点多屋顶 |
| Instruction Roofline | Ding & Williams, PMBS 2019 | 是 —— 换的是**单位**,不是维数 |
| Ridgeline(加网络轴) | arXiv:2209.01368 | 是 —— **标题自己写着 "A 2D Roofline Model for Distributed Systems"** |
| Gables(N 个加速器) | Hill & Reddi, HPCA 2019 | 是 —— N 条 roofline 一张图,用 `max` 合成 |
| **ERM** | Cabezas & Püschel, IISWC 2014 | 2D 只为可视化,**底层模型是真正的资源向量** —— 现存最接近的 |

**所以新颖性要说得精确**:不要claim "第一个多资源性能模型"(ERM 2014 有 >20 参数资源模型,Gables 2019 有 N 个单元)。要 claim 那个更窄且为真的:已有多资源工作要么(i) 预测一个界就停,要么(ii) 需要 cycle 级 DAG 调度或硬件 counter;**没有人把资源维度间的替代率当作一个按任务实测、并用于引导搜索的量。**

### 3.10 其他(各一行)

| 工作 | 机制 | 怎么融 / 为何不融 |
|---|---|---|
| **Kerncraft** arXiv:1702.04653 | 从循环源码自动建 Roofline+ECM,含 layer-condition **预测最优 blocking 因子** | counter-free;最接近"我们要建的东西的现成实现" |
| **Campaign Diagrams** arXiv:2607.15225 | 相位级利用率随时间,明确反对"把 workload 聚合成一个点"的 roofline | 若我们的 kernel 有异质相位(attention:memory-bound softmax + compute-bound GEMM)则相关 |
| **CoSA**(ISCA'21) | 唯一带显式加权和:`Ô = −w_U·Util + w_C·Comp + w_T·Traf`,容量作 log-linear MIP 约束一次解出 | 借它的**约束/目标切分**:容量作硬约束在评测前解决。GPU 上 vs TVM:**2500× 更短 time-to-solution 换 1.10× 加速**——比值本身是警告 |
| **Timeloop / MAESTRO / Interstellar** | 多资源 mapping 代价 | 见借鉴 3(MAESTRO 的 bottleneck info)与下方 Interstellar 两条观察 |
| **Mind Mappings**(ASPLOS'21) | 需 10M 训练样本 | **结构性不可用**(≈6 年)。但一条发现可借:**预测 12 维分解代价向量而非标量 EDP,使对真值 EDP 的 MSE 低 32.8×** —— 支持"向量优于标量"本身 |
| **Opal** arXiv:2510.00932 / **KernelPro** arXiv:2606.26453 | "roofline → stall events → 优化决策"(1640 次实验平均 19–52% 增益)/ roofline 瓶颈分类 | **我们这个细分方向的直接竞争者**,两者都用 counter |
| **Interval analysis / CPI stack**(Eyerman 等,IEEE Micro 27(1) 2007;TOCS 27(2) 2009)[仅摘要复验] | 按 miss 事件切区间,建**周期可加**的 CPI 栈 | 与 TMA 的对比值得记:CPI 栈**周期可加**、TMA **槽划分**;超标量下可加周期惩罚会重复计数,这正是 TMA 存在的理由 |
| **Balanced job bounds**(Zahorjan & Sevcik)/ 操作分析(Denning & Buzen, CSUR 1978,Roofline 自己引的 [20]) | 吞吐的**上界与下界** | **借"下界"这个想法**:本报告所有模型都只出上界,而我们反复把噪声与平台期读成信号。双侧界能让我们说"这一族里没有配置能超过 X"然后停 |
| **Dong & Pai** arXiv:2503.17893 | shared-memory atomic 的排队模型;动机是"models like Roofline are not very useful"于数据相关程序 | 唯一把 GPU 资源建成队列而非屋顶的实例;瓶颈迁移造成至 30% 差异 |

### 3.11 LP 对偶 / 影子价格:框架开放,但只作局部使用

**调研结论(带正对照)**:LP 对偶/影子价格在优化领域成熟,**但没找到任何把它用于代码或 kernel 优化作瓶颈归因的工作**。三种查法都只返回通用 LP/经济学解释。**这是搜索的负面结果、不是不存在的证明** —— 按我们自己的纪律(`probe needs a positive control`),正对照是:同一套检索方法**正确地**找到了 ERM、DrGPU、Gables、TMA,所以工具是工作的。

形式:`max c'x s.t. Ax ≤ b` 的最优对偶变量 `y*_i` = 每单位 `b_i` 带来的目标边际改善;非绑定约束 ⇒ `y*_i = 0`(互补松弛):**不是瓶颈的资源价格为零**。`y*_i/y*_j` 恰是资源 i 与 j 的边际替代率。已知地雷:**退化时影子价格不唯一**(左右导数不等),存在"退化 LP 的真影子价格"这一支文献 —— 对我们要紧,因为**近平局是我们的常态**(一个 run 里 33 个近平局)。

**怎么用**:
- **借词汇与两条定理。** (1) **互补松弛作可证伪自检**:若模型判定维度 j 松弛,则在 j 上花资源应买到**零**延迟改善 —— 这是对我们自己瓶颈归因的廉价证伪实验(一次消融)。(2) **对偶之比 = 替代率**:把转化效率定义为 `Δlatency / Δresource_j`,两维之间的交换率取二者实测效率之比。这让"转化效率"成为**有量纲、可比较**的量而非语感。
- **不建 LP。** kernel 性能对资源向量非线性(ERM 的 α、ECM 的重叠结构、我们 L3:48 的 90% 平台期都是非线性),而有意思的区间恰是线性化失效处。诚实版本是**局部经验梯度**:影子价格是导数,而扰动一个 knob 观察 Δlatency 本来就在我们 Optuna trial 里发生 —— 只要我们把每 trial 的资源向量与时间一起记下来。
- **预期退化并处理**:33 个近平局 + 每 trial std 占均值 16% 就是教科书式的对偶退化(多个资源同时接近绑定,各自的"价格"不可定)。文献的答案(双侧导数、扰动法真影子价格)提示我们应**每维报一个价格区间而非点值,区间重叠时拒绝行动**。**单这一条规则就能阻止把噪声内的"增益"当排序。**

**Interstellar(ASPLOS'20)两条可直接用的** [子代理]:
- **Observation 1**:"many different dataflows are able to achieve similar and close-to-optimal energy efficiency, **as long as proper loop blocking and replication are used**";而 blocking 只有 **30%** 的方案落在最小能耗 1.25× 内。**只优化资源分配(固定吞吐)就有 4.2×(MobileNet)。**
- **Observation 2**:"The total energy of an efficient system **should not be dominated by any individual level** in the memory hierarchy" → **可直接当验收诊断**。配套先验:**相邻层级片上存储尺寸比应在 4–16**。

**一条不舒服但值得记的推论** [推断]:Interstellar 说结构(dataflow)在最优附近大体可互换,而 blocking + 资源尺寸承载了方差和那个 4.2×。**我们把昂贵高方差的 LLM agent 花在结构上、把便宜的 TPE 花在 tile 上** —— 这个分配值得重新审视,尤其我们在 tile 侧已有浪费证据(18% 不可行 trial、一个按 fp16 选的 tile 在 tf32 下根本跑不起来)。

**一条覆盖全类的警告** [子代理]:上述代价模型全部针对**显式管理的 scratchpad + 固定 dataflow + 无 cache 竞争**。它们 2–8% 能耗、95% 平均性能精度是**在那个 regime 内**达到的;没有一个在带 cache、带 warp 调度的 GPU 上验证过,而**它们的误差带都比我们 tuner 要分辨的近平局跨度更宽**。

---

## 四、LLM kernel 生成的现状(定位我们自己)

**按"多少资源信息到达模型"排序** [子代理,除标注外]:

| 系统 | 反馈给模型的东西 | 数字 |
|---|---|---|
| KernelLLM(Meta 8B) | 无在环资源反馈 | 他人复测:L1 fast_1 21 / geo 0.80;**去掉作弊后 3 / 0.51** |
| AutoTriton | 规则奖励(语法)+ 执行奖励 | L1 84% 正确 / fast_1 21;L2 94% / 58 |
| Kevin-32B | 正确性 + 加速比奖励,纯执行 | 正确 56→82%,均值加速 **0.53→1.10×** |
| CudaForge | **24 个 NCU 指标逐字插入**("Nsight Compute metrics (verbatim)") | 97.6% 正确,1.68× 均值。**即 KernelPro 点名的那一臂** |
| Kernel Foundry | **结构化诊断 → 自然语言提示**,三分类(memory/latency/instruction-bound),提示带触发条件+置信度、按实测效果强化或剪除 | 消融:L1 正确 86→95%,fast_1 **17→35** |
| NVIDIA SOL agent | **Speed-of-Light 余量**引导搜索、给接近 SOL 的 kernel 降优先级 | DSL −0.40× → 1.27×;**+SOL → 1.56×**;省 19–43% token |
| **KernelPro** | 15 个微剖析工具把 ~50 个计数器翻译成严重度+根因+排序建议,由 roofline 分类门控 | **SOTA:L1/L2/L3 2.42/4.69/5.30×** |
| SOLAR | 解析 SOL 上界(unfused/fused/cache-aware),多保真度余量分析 | 零 SOL 违反 |
| Compiler-Grounded Hierarchical Diagnosis | 逐级升级:模式分诊 → 剖析 → IR 归因 → 编译器级分析 | Ascend NPU 37 任务:**4.35× geomean** |
| Autocomp | plan-then-implement;**优化菜单**(GPU/L40S 65 条)+ 每轮 70–80% dropout;插入 scratchpad/accumulator 利用率 | Trainium **1.9× geomean 超专家手写**;**L40S 2.05× 超 PyTorch、3.8× 超 TVM MetaSchedule(1000 trial)** |
| cuPilot | "Roofline Prophet" agent 解析分区,**三个区** | 两个 epoch 后**延迟降 44.2%**;3.09× geomean |

**基准完整性侧** [已复验 KernelBench-Verified]:最好的模型在 **TF32 基线 + 四分布隐藏测试**下从 **1.43× → 0.88×**(比 PyTorch 慢);模型会"hardcoding bypasses for specific tensor values";**28% 的 kernel 抬高峰值显存**。同向:The Correctness Illusion(arXiv:2606.20128)攻击固定形状 allclose;Hardening Agent Benchmarks 发现 16% 的终端 agent 任务可被 hack。

**Sakana 的更正** [子代理,部分未能复验]:successor 论文(arXiv:2509.14279)引入 `robust-kbench`,承认 "existing kernel generation benchmarks suffer from exploitable loopholes";TechCrunch 报道原声称 "up to 100×"、用户实测 **3× 变慢**,Sakana 承认系统 "found exploits in the evaluation code"。**原始声明与具体 exploit 机制未能从一手源复验,引用请用 successor 论文。**

**METR 相关工作:未能找到,不要引用。**

---

## 五、我们真正还站得住的地面(与已被 claim 掉的)

**已被 claim 掉(必须直说)**:
- 把瓶颈诊断/roofline 分类反馈给 LLM —— KernelPro、NVIDIA SOL、Kernel Foundry、SOLAR、cuPilot、HIERA 等都在做。
- "消化优于原始计数器" —— KernelPro 以 p<0.0001 claim,且带 p=0.0007 的"原始不如没有"。
- "余量作停止规则" —— NVIDIA SOL,省 19–43% token。
- **"多资源向量作性能模型的状态" —— ERM(2014)已有 22 参数资源模型 + 三层利用率分解 + 显式重叠系数;Gables(2019)有 N 个单元。**(见 §3.9;这一条是第四路调研新增的,初稿低估了先验工作。)
- **"用 gap 高度作优化的预期回报" —— Roofline 原文(2009)就写了**:"The height of the gap between a ceiling and the next higher one is the potential reward for trying that optimization."

**调研找不到已发表对应物的**:
1. **把"交换"作为优化对象,用转化效率作被评分量。** 所有系统诊断"什么在限制"并开处方;没有一个把优化表示为**资源交换向量**或按"每单位资源换多少延迟"排序候选。**精确表述(§3.9 校正后)**:已有多资源工作要么预测一个界就停,要么需要 cycle 级 DAG 调度或 counter;**没人把维度间替代率当作按任务实测、并用于引导搜索的量。**
2. **闭合"预期 vs 实现"的环。** KernelPro 的建议带"expected improvement estimates"(如 3–5×)但**从不回头验证**。记录**已实现**的交换比并反馈,约 120 篇中未见。
3. **按设备/精度/后端实测的天花板作分母。** 所有人用规格书或理论 roofline。**外部同向证据(§3.6)**:同一篇 hierarchical roofline 实测 V100 张量核经 cuBLAS 达理论 96.5%、经 **WMMA 仅 54%** —— 而 Triton 发的就是 WMMA。**这不再是 [推断]:分母错配是有实测的真实失效模式,只是没人把"每后端实测天花板"做成机制。**
4. **测量"哪个反馈信号真的导致了改善"。** KernelPro 的 per-tool 命中率表最接近,但**没人核查获胜配置是否真用了该信号建议的值**。
5. **噪声底感知的正确性门。** 领域已把 eval 完整性立为研究问题,而**没有按任务实测的噪声底**。**可能是比"交换"框架更强更好守的贡献。**
6. **没人把资源向量给 tuner**,只给代码生成器。[推断]
7. **把 LP 对偶/影子价格用于代码或 kernel 优化的瓶颈归因** —— 三种检索方式零命中,且检索方法有工作的正对照(§3.11)。**借词汇与两条定理,不建 LP**:(a) 互补松弛作可证伪自检;(b) **每维各自的价格区间**。**2026-09-10 收窄**:"两维交换率取二者之比"这半句撤回 —— 那是聚合。
8. **拒绝合成:维持多份并列的按维度判决,且维度集合本身是运行期发现的。**(2026-09-10 新增,来自用户对设计方向的否决)§3.9 已证:现存所有多资源模型最终都收敛到 2D 图或标量界 —— ERM 用 α 合成、ECM 用重叠假设合成、Gables 用 `max`、Nsight SOL 在维度间取 max、CUDABench 压成 `Performance-Score`。**"拒绝合成、维持多份并列判决、维度集合运行期发现"在约 120 篇里没有对应物,而且它是设计选择而非测量声明** —— 审稿人无法用"2014 年就有资源向量"驳回它(那恰是我们承认的先验工作),要驳回必须论证合成是必要的,而我们有三条理由说它不是。

**方法论标准**:KernelPro 的消融(42 任务 × 15 种子、单侧 Wilcoxon、报非平局 n、六阈值 fast_p)就是我们会被要求达到的。我们目前 19 个 run 只有 1 个跑到墙钟、Loop D 零执行、调参在噪声底内 —— **差距在统计功效,不在想法。**

---

## 六、明确不借的东西,及理由

| 不借 | 理由(带数字) |
|---|---|
| 学习型代价模型(Halide/Ansor/TVM/Mind Mappings) | 需 1.6M / 25k / 10M 次实测;按 18.6 s/trial 分别是 827 年 / 5.2 天 / ~6 年。**结构性不可用** |
| 构造取代搜索(Roller 全套) | Roller 小算子慢 50%、TC 只到 cuBLAS 43%、自述盲点是寄存器分配;tritonBLAS 在真实形状慢 13.9% |
| 多目标 TPE / qEHVI | n=40 时 MOTPE 只有 4 个好点、静默关掉 multivariate;qEHVI 在 M=4 时 acquisition 459 s = 25 个 trial |
| 任何依赖 ncu counter 的方法 | `ERR_NVGPUCTRPERM` 在租用容器**永久不可用**(root 也不行:`lsmod` 零 nvidia 模块、`/etc/modprobe.d` 不存在)。**被挡的具体清单**:Nsight SOL、hierarchical roofline 的*测量*、Instruction Roofline 的*测量*、DrGPU 的实现、warp-vs-thread 谓词化计数 |
| 原始计数器/规格数值直接进 prompt | p=0.0007 比什么都不给更糟;KernelBench 自测硬件规格无显著影响 |
| Hyperband / successive halving 作早停 | [推断] 需要单调学习曲线,而 kernel benchmark 的部分结果只是同一个数的更噪估计。**正确替代:自适应重复次数**(CI 排除 incumbent 就停) |
| 把预测增益用于分配 | v2 自己实测:中位偏差 −5.0%,21 次里 8 次符号错 |
| **ERM 的 DAG 调度机制** | 要 LLVM 解释器级 22 参数微架构模拟,且其精度恰在支配 GPU 的效应上失效(硬件预取、冲突缺失、无 SIMD)。**取它的量(α、三层利用率),用墙钟消融而非模拟去得到** |
| **TMA 的 6 层树** | 树的存在理由是 Intel 有上千 PMU 事件需要下钻;我们零 counter,且消费者是必须据此行动的 LLM agent。**借扁平的 ~12 项 Bottleneck View 即可** |
| **固定的全局 ceiling 顺序** | Roofline 原文自己在 SpMV 上推翻了自己的顺序("we would place floating-point mix as the lowest ceiling, since it is inherent")。**这就是我们 L3:48 的 8/8 张量核事故** —— 顺序必须按任务从实测可行性重新推导 |
| **把 >100% 利用率当 bug 拒绝** | Nsight 文档明写 "Percentages of sustained rate can occasionally exceed 100% in edge cases"。我们已真的撞上 107.8%。**应 clamp 并打标记** |
| **让判决挂在 L2 那一项上** | ECM 自己最差的一级恒是 L2(误差 17–33%),原文"in none of the cases the measured L2 performance could live up to the advertised specs of 64 B/c" |

---

## 七、待复验清单(进论文前必须做)

1. ~~Roofline 原文全文~~ —— 第四路已读技术报告版 UCB/EECS-2008-134 并给出逐字引文与 Opteron X2 的五条 ceiling 数字。**仍需我本人拉一次**(CACM 52(4):65–76,DOI 10.1145/1498765.1498785)。
2. **ECM 的合成公式与逐级误差表**(arXiv:1509.03118 / arXiv:1410.5010)—— α 与 `max(T_nOL+T_data, T_OL)` 是要写进设计的东西,必须一手。
3. **ERM 的 α 定义与三层利用率**(Cabezas & Püschel, IISWC 2014, spiral.ece.cmu.edu 有 PDF)—— 这是我们最近的先验工作,引用错会被抓。
4. **TMA 的 SLOTS 公式与 Bottleneck View 项名** —— 第四路是从 `intel/perfmon` 的 `skylake_metrics.json` 读的,这个来源比论文更可靠;**复验路径就是直接读那个 JSON**。
5. **Instruction Roofline 的墙位置**(1、1/4、1/8、1/32)与 GV100 的 GTXN/s 换算(PMBS 2019)。
6. **Hierarchical roofline 的 V100 张量核 96.5%(cuBLAS)vs 54%(WMMA)** —— 这条直接支撑 P9,是本次调研对我们最有用的单个天花板数字,必须一手。
7. Halide 39 个特征的完整清单(附录 A,[子代理])。
8. Optuna MOTPE 的源码断言(gamma 0.1、multivariate 静默 False)—— 应在我们锁定的 Optuna 版本上直接读源码复验。
9. Volkov SGEMM 表、Chen 2016 / Korthikanti 2022 的数字。
10. Nsight 文档的 "can occasionally exceed 100%" 原句(要用它给 clamp 规则作依据)。
11. A100 datasheet 精度倍数 [已复验] —— 但 **A100 上的实际交换比必须在我们自己的 A100 上实测**(见借鉴 5 的设备特定性警告)。
12. 所有引用 v2 "已修复"的断言,须在当前 checkout 复验并带 commit。
13. **Eyerman 等的 CPI stack**(IEEE Micro 2007 / TOCS 2009)—— 第四路只复验了摘要,"周期可加 vs 槽划分"的对比是**它的推断**,标为未验证。
14. **Sakana 的原始 100× 声明与 exploit 机制** —— 第四路与前几路都未能从一手源复验,引用只用 successor 论文(arXiv:2509.14279)。