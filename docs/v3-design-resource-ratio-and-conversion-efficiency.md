# v3 设计:以资源比例与转化效率为核心的算子优化框架

**状态**:设计稿,2026-09-09(经调研修订),**2026-09-10 经用户否决两处后修订**。v3 分支从 v2 的 `8772831` 继承全部代码,`D:\Pyhon_projects\opop\v3`(git worktree,同仓库分支 v3)。v2 分支继续跑完当前两个实验,box 1 已于 2026-09-10 00:15 空闲、转为 v3 的实验平台。

本文档定**整体思路、项目逻辑与设计约束**。实施细节见 `v3-implementation-plan.md`。调研出处与借鉴清单见 `v3-prior-art-and-what-to-borrow.md`。**2026-09-10 的两处方向性修订(拒绝向量合成、撤回取值黑名单)见 `v3-revision-no-scalarization-and-retire-risk.md`,该文档的结论优先于本文档任何相冲突的早期措辞。**

每条判据都写明"用什么证据算通过",这是 v2 立下的规矩:改动必须泛化、必须有会失败的反向对照。**外部结论凡未亲自复验的,一律标注。**

---

## 零、调研带来的五条修订(先读这一节)

写下初稿后做了四路调研(三路外部文献、一路 v2 内部文档盘点)。五条结论改变了设计,必须放在最前面。前三条来自前三路,后两条来自较晚回来的性能模型那一路。

### 修订 1:"把瓶颈诊断反馈给 LLM"在 2026 年已不新颖——但"原始数值有害"是硬约束

**已亲自复验**(拉取 arXiv:2606.26453 全文表格):KernelPro 有一节标题就叫 **"5.3.1 Raw Metrics Are Harmful"**,42 个 KernelBench 任务三臂对照:

| 反馈方式 | 解决 | 几何均值加速 | 胜率 |
|---|---|---|---|
| 无反馈 | 32/42 (76%) | **3.35×** | 24% |
| 原始 ncu 计数器(~50 个) | 36/42 (86%) | **1.77×** | 7% |
| 消化后的工具建议 | 42/42 (100%) | **4.00×** | 60% |

单侧 Wilcoxon:全量 > 原始 p<0.0001;全量 > 无反馈 p=0.0078;**无反馈 > 原始 p=0.0007**(n=39, W=618)。论文原话:"unstructured metrics actively degrade optimization quality"。

注意**非单调性**:原始计数器把解决率从 76% 提到 86%,却把几何均值**腰斩**。它让模型更擅长正确性、大幅更不擅长速度。

**对 v3 的硬约束(取代初稿的说法)**:初稿写"给 agent 的 prompt 主体改成向量"——**这是错的,会精确复现那个 1.77× 的臂**。资源向量必须先经**消化层**:每个数字配"因此该做什么"(严重度 + 根因 + 排序建议 + 预期 Δ)。**原始向量只进报告与事件日志,不进 prompt。**

同向的独立证据:KernelBench 论文自己测出**给硬件规格对输出无显著影响**(§5.2.2),few-shot 优化范例反而**降低** fast_1。

拥挤程度也要如实记:KernelPro、NVIDIA SOL agent、Kernel Foundry、SOLAR、cuPilot、CudaForge 都在做某版本的"诊断反馈"。**建立在"没人做过"上的论文框架过不了审稿。**

### 修订 2:转化效率的公式已经存在,而且可算——Roller 的 reuse score

**已亲自复验**(拉取 OSDI'22 论文全文,逐字对上):Roller 的 rTile 放大策略按**数据复用分数**选轴:

```
S_i = ( Q(T) − Q(T′_i) ) / ( F(T′_i) − F(T) )
```

其中 `Q` = **memory traffic**(内存流量),`F` = **memory footprint**(内存占用),`T′_i` 是把第 i 轴换成下一个对齐尺寸后的新 tile。原文:"Functions Q(T) and F(T) calculate the memory traffic and memory footprint when the computation is executed in the granularity of T"。

**这字面上就是转化效率:每多花一单位占用,省下多少流量。** 停止条件同样对上我们的思路:`if MemPerf(T′) > MaxComputePerf(T′.expr)` —— 即**消耗向量撞到供给屋顶就停**。缩小策略是"沿复用分数最小的轴收缩"以恢复并行度。

**对 v3 的意义**:这个量**零 trial、零 counter 可算**(纯静态,从 tile 配置推)。它给了我们一个**不依赖候选真实流量测量**的转化效率估计器——正好绕开下面§二的最大缺口。借它的**比值**,不借它的构造式流水线(理由见修订 3)。

### 修订 3:不要用模型取代搜索——供应商团队在我们这个栈上只到 94.7%

**tritonBLAS**(arXiv:2512.04226,AMD 团队,Triton 3.4.0,纯 GEMM):解析模型选参,**94.7% 选择效率** vs 穷举,选择耗时 50–80 µs vs Triton autotune 的 11.9 s–1383.6 s。但**在真实 Llama3 形状上平均比 PyTorch 慢 13.9%**,且作者明确声明不适用于 attention 等非 GEMM。

Roller 自己的限制也在同一方向 [2026-09-10 拉取全文重核]:小算子上 "slower than Ansor, e.g., **by 50% on average, on small operators**";tensor core "within a **43% performance gap** to cuBLAS"(即约 57% of cuBLAS —— **本文原写"只到 43%"是误读,已更正**)。

**而 Roller 对寄存器的处理,是"不预测、改编译"这条设计决定最有力的外部依据**(原文两句):
- "We notice that the nvcc compiler will implicitly declare more registers (for loop variables or other purposes). Given that this behaviour is hard to predict, **we reduce the register limit empirically to only 96 registers** for both V100 and K80 GPUs per thread"
- "ROLLER **cannot detect implicit register allocation beforehand**, hence it is difficult to estimate and decide the precise register usage."

**96/255 = 37.6%,一个 2.66 倍的硬编码安全系数,不是模型、不随 kernel 变。** 一篇主张"解析构造优于搜索"的论文,在拿到完整张量表达式与硬件规格后,结论是**寄存器用量无法从源码预测,于是丢掉 62% 的寄存器文件**。而我们的 agent 一直被要求做 Roller 明确放弃的事 —— 这正是"手写约束中位只有真值 32%"的成因。**我们自己的实测独立证实了同一条缝**:全新进程里 `warmup()` 对每个 tile 都给出 shared memory,而 `n_regs` 每一行都是 `None`,只有真正启动后才出现。

**结论**:解析模型作 **warm-start 与 trial 排序先验**,绝不取代 TPE。

### 修订 4:"多资源向量"本身不新颖 —— ERM(2014)已经有了,新颖点必须重新表述

初稿隐含地把"用向量取代标签"当作贡献之一。**性能模型那一路的调研推翻了这个假设。**

**ERM**(Caparrós Cabezas & Püschel,IISWC 2014,CMU SPIRAL)已经同时具备我们提的四件事:每资源利用率向量 `U_x = N_x/(T_x·Π_x)`;为什么欠用的三层分解(`U_issue` 缺 ILP / `U_lat` 延迟暴露 / `U_stall` 乱序缓冲容量);**显式的重叠系数**(见修订 5);跨层级共同分母 `I = W/(Q_L1+Q_L2+Q_L3+Q_mem)`。它的动机原话就是我们的动机 —— 原始 roofline "is inherently blind to other bottlenecks, in particular non-throughput resources including cache capacity, latency of memory accesses or the functional units, and out-of-order (OoO) execution buffers"。**Gables**(HPCA 2019)另有 N 个单元的版本。

**并且 Roofline 原文(2009)自己就写了 gap-height 规则** —— "The height of the gap between a ceiling and the next higher one is the potential reward for trying that optimization" —— 那是转化效率的雏形,且在花代价之前算出。

**对 v3 的修订**:新颖性主张必须收窄成那个**更窄且为真**的版本(已改写进 §四):已有多资源工作要么预测一个界就停,要么需要 cycle 级 DAG 调度或硬件 counter;**没有人把资源维度间的替代率当作一个按任务实测、并用于引导搜索的量。** 不要 claim "第一个多资源模型"。

**同时借它两条可直接用的**:
- **单维度内部再分解**:"内存 85% 利用"远不如"内存发射没问题,损失在延迟暴露"可行动 —— 前者指向减流量,后者指向加并发。TMA 的 Bottleneck View 独立到达同一条缝(它把 `Cache_Memory_Bandwidth` 与 `Cache_Memory_Latency` 列为两个维度),两个独立来源撞到同一处,是这条缝真实存在的好证据。
- **屋顶不是机器常数**:ERM 自己标为根本性的推论是 "the memory bounds … now depend on program and input"。这与我们已立的"天花板必须实测"同向,但更进一步:**同一台卡上,屋顶还随任务变**。

### 修订 5(**已被用户否决,保留作调研记录**):外部有"向量合成一个数"的成熟公式——但我们不采纳

**2026-09-10 更新:本节原本主张采纳 ECM/ERM 的合成规则。用户否决了这个方向,理由见 `v3-revision-no-scalarization-and-retire-risk.md`。本节降级为调研事实的记录,不再是设计要采纳的东西。**

外部的确有确切形式。ECM(Hofmann/Eitzinger/Fey,arXiv:1509.03118;Stengel 等 ICS'15):

```
T_ECM = max( T_nOL + T_data , T_OL )
```

既不相加也不取 max,而是结构化的部分重叠。ERM(Cabezas & Püschel,IISWC 2014)把重叠程度做成标量 `α = T_overlap/min(T_x,T_y) ∈ [0,1]`,0 = 相加、1 = 取 max。

**为什么不采纳(用户的三条理由,我接受)**:

1. **借来的公式携带借来的假设。** ECM 的两条重叠假设是在显式管理的 CPU 缓存层级上成立的;我们的 GPU 有 cache 竞争与 warp 调度,而调研自己的结论是"没有一个在带 cache、带 warp 调度的 GPU 上验证过,且它们的误差带都比我们要分辨的近平局跨度更宽"。
2. **合成本身不是目标。** 不同维度可以按不同维度独立考虑。"把向量压成一个数"是我引入的需求,不是项目要解决的问题。
3. **维度数不固定。** 随设备变(V100 无 bf16、4090 上 tf32≈fp32),且**每次算子结构改动后占用率都会变**。任何要求"分量可比、可加、数目固定"的合成规则都与这一点冲突。

**因此连带撤回三处**(都预设了"要合成一个数"):§二缺口 2 原本要借的 TMA 式闭合(守恒分母 + 残差项)、原路线的 S5b(α 作可测标量)、S3 里 `Performance-Score` 作头条上报量的地位。替代设计见 §二缺口 2 改写后的内容与那份修订文档的 §2.3。

**保留的**:ECM/ERM 作为**最近的先验工作**必须在论文里诚实引用(见 §四),这一点不变。

---

## 一、核心思想

### 1.1 一句话

一个算子(或一个模型)在每个资源维度上有自己的**消耗向量**;一台硬件在每个维度上有自己的**供给向量**。两者方向不一致时,某个维度先撞顶而其余维度闲置——这就是瓶颈。**算子优化的本质是在资源维度之间做交换,把消耗向量的方向转向供给向量的方向。**

例:算子融合 = 用存储/内存占用,换掉启动开销、传输开销、通信开销。降精度 = 用数值精度,换算力吞吐与带宽。两者在 v2 里是"两种互不相干的技巧",在 v3 的模型里是**同一类操作的两个实例**——都是资源维度间的重新分配。

**文献佐证这个抽象是对的**:FlashAttention(arXiv:2205.14135)在一张表里同时印出账本两侧——GPT-2 medium / seq 1024 / A100:标准实现 66.6 GFLOP、40.3 GB HBM 读写、41.7 ms;FlashAttention **75.2 GFLOP(+13%)、4.4 GB(−9.2×)、7.3 ms(−5.7×)**。它**多花 13% 算力换来 9.2× 流量削减**。原文:"even with more FLOPs, our recomputation speeds up the backward pass due to reduced HBM accesses"。

**这个例子同时是一条警告**:一个把 FLOP 当作要最小化的量、或把"算得更多"当退步的框架,**会否决 FlashAttention**。所以评分必须是"每单位资源换来的延迟",不是资源最小化。

### 1.2 但"利用率均衡"不是目标

严格讲最优是**延迟最小**,不是利用率均衡。一个打满带宽、算力空转的 kernel,若已达该任务的物理下界,它的"比例失衡"是**物理性质而非缺陷**——继续"均衡"只会更慢。

已有实测反例(必须内建进设计):

- **L3:48 是带宽墙**:固定 1.351 GB 流量,1.64 ms 已达实测 911 GB/s 屋顶的 **90.4%**。平台期是物理不是噪声。
- v2 曾因缺 fp16 天花板,把候选读成 **107.8% 已到顶**,指令反转(该候选实际约 54–60%)。

**所以 v3 必须以"每个维度的下界"为锚。** `task_cost.py` 的 `compulsory_bytes` 已是流量维度的下界雏形。

**文献同向**:Interstellar(ASPLOS'20)的 Observation 2 原文——"The total energy of an efficient system should not be dominated by any individual level in the memory hierarchy"——可直接当**验收诊断**:任何被单一层级主导的候选,是 tile 尺寸没配好的指纹,不是结构不好。它还给了一条可直接测试的先验:**相邻层级片上存储尺寸比应在 4–16 之间**。

**并且屋顶的可达性是按任务而变的,这是我们付过学费的一条。** Roofline 原文在两条主屋顶下叠了五条具名 ceiling,并按"编译器易得 → 程序员难得"排序;**但它自己在 SpMV 上推翻了这个顺序** —— "we would place floating-point mix as the lowest ceiling, since it is **inherent**"。它对高位 ceiling 的定义里就包含"**inherently lacking in a kernel**"。

**这正是我们 L3:48 的张量核事故**:一条"高阶"屋顶在那个任务上是天生不可用的,而框架仍拿它当目标,8/8 张量核候选撞死在上面(7/7 标量候选被收下,2.09 ms / 8.90x 的成绩完全没用上张量核)。**所以:屋顶顺序与可达性必须按任务从实测可行性重新推导,不得作为常量写进 prompt。**

**一条免费的门限修正**:Nsight 文档明写利用率百分比 "can occasionally exceed 100% in edge cases"。我们已真的撞上 107.8%。**任何把 >100% 当"不可能因此有 bug"的判定都会误判 —— 应 clamp 并打标记,不是拒绝。**

### 1.3 资源不是可自由兑换的:软资源 vs 硬墙

- **软资源**(可交换、有成本):DRAM 流量、算力、启动次数、寄存器压力、occupancy、溢出流量、峰值显存、(未来)通信量。
- **硬墙**(越界即不可运行,不是"效率低"):shared memory > 101376 B、寄存器 > 255/thread。v2 实测有 62 个 trial 直接死在 shared memory 上。

模型里必须区分这两类,**不能都当成成本项加权求和**——硬墙是可行性判定,软资源才进交换计算。

**关键修订(来自调研,这是最高价值的一条)**:v2 目前把编译期可知的不可行报成 Optuna 的 `FAIL`/`PRUNED`(`tpe.py:119-121`)。两个独立团队实测这种"惩罚/填补"做法**主动有害**,且机制已被命名:Willemsen 等(arXiv:2606.28372)原文——用朴素惩罚时"the performance of the non-constrained variants approaches that of the random search baseline **for the first half of the tuning time**"。**那前半段正好是我们 40–80 trial 的全部预算。** Kernel Tuner 明确拒绝惩罚填补,改为把无效点**从 acquisition 优化中排除**;HyperPower 报告过滤已知约束带来**固定预算内 57.20× 更多有效评估**。

**所以:编译期可知的不可行,应当从空间里"声明掉",而不是抽出来再拒。** 三条实现路径按成本排序见实施方案。

### 1.4 资源转化效率:v3 的核心,但必须是"已实现"而非"预测"

- **资源转化效率**:降低 1 单位瓶颈资源,付出了多少其他维度的开销。
- **性能转化效率**:增加的开销换来了多少实际性能提升,即 `Δ性能 / Δ开销`。

**v2 已经用实测否决过"预测"版本,这是 v3 最重要的内部约束。** `measurement-predicted-gain-overshoots.md`(已亲自复验原文):v2 唯一一次做增益数值的尝试 `predicted_gain_pct`,n=21,**中位预测 5.0%、中位实际 1.8%、中位带符号误差 −5.0%**,**21 次里 8 次实际为负**(预测 +16.4% → 实际 −16.5%)。文档第 3 条建议原文:"**Do not use it for allocation.** … Any future ranking that weights families by predicted gain would be weighting a quantity with a −5.0% median bias and an 8-in-21 sign error rate."

**边界必须写死**:v2 失败的是**预测**;v3 要做的是**事后测量同一次改写实际花了什么、换到了什么**。这是两件不同的事。凡是进入选择/排序/分配的数字,必须是**已实现的实测量**。Roller 的 `S=ΔQ/ΔF` 是静态**估计**器,因此只能用于 trial 排序与 warm-start,**不得进入接受判定**。

同向外部证据:KernelPro 的工具建议携带"expected improvement estimates",但**从不回头验证它是否实现**——这正是下面 §四认定的新颖点之一。

### 1.5 效率指标的噪声纪律(踩过坑换来的)

比值的分母小时,比值本身就是噪声。已有实测:**L3:48 每 trial 的 std 占均值 16%,而 33 个近平局的跨度只有 9%**。另有 `measurement-retune-repeatability.md`:**重跑同一个空间可摆动 2.1%**。

**硬规则:效应量未显著超过噪声底时,效率一律报 `unknown`,不报数字。**

**并且必须带因果检查**(借 v2 已有纪律):"扩展后 best 变好"不等于扩展造成了它——重新调参本身就能移动数字(实测移动 6.0%)。判据是**比对空间版本、确认获胜配置真的用了新增的值**。

### 1.6 转化效率是设备特定的,不可跨卡继承

**已复验的外部实测**:model-steered tuning(arXiv:2211.07260)在 A100 上同一交换比给 **+50.9%**,在 A4000 上相同减速只换来 **+5.8%**。

**所以任何交换比都必须在本机测,绝不继承。** 这与 v2 已立的"天花板必须实测"是同一条原则的推广。

---

## 二、v2 与该思想的差距(全部在代码里复验过)

差距不在"有没有做优化",而在**框架根本不度量"资源转化"这件事**。

### 缺口 1(最致命):候选实际消耗的资源从未被测量,只有任务的下界

`orchestrator.py:1421` 把 `byte_count=cost.compulsory_bytes` 传给分类器——**任务的强制流量,对所有候选是同一个常数**。后果:

- 框架**无法知道任何候选实际移动了多少字节**。融合掉一半中间张量的候选与不融合的,在分类器眼里流量完全相同。
- 现有 `pct_of_dram_peak` 实际是"**任务级分子 ÷ 候选级分母**",不是候选真实带宽利用率;真实流量更大 → 真实利用率更高 → 真实可优化空间更小。**偏差在融合差的候选上最大**,恰在最需要准确判断处最不准。

**而三条现成路都不通(已逐一在代码里复验,这推翻了初稿的乐观设想):**

1. **SASS 指令计数是静态的**——`statics.py:63-74` 数的是 LDG/STG 在汇编**文本里出现几次**,不含循环执行次数。K 维循环 100 次的 GEMM,其 LDG 可能只出现两三条。"指令数 × grid"得不到动态字节数。
2. **`TorchDispatchMode` 是 aten 级的**(`worker_main.py:1336-1380`)——它能测出参考实现的 `reference_bytes` 是因为参考由许多 aten op 组成;而**融合良好的候选只有一个自定义 Triton op**,dispatch 模式在它内部什么也看不见。这个仪器在候选上会**系统性低报,且融合越好越低报**,方向完全错。
3. **`ProfileRecord` 里没有 grid 尺寸/启动次数**(已复验字段表),连静态→动态的换算乘数都缺。

**这是 v3 最难的一项,必须当作探索项而非确定项处理。** 但§零修订 2 给了一条绕路:Roller 的 `S=ΔQ/ΔF` 是**静态可算**的,不需要测真实流量。

### 缺口 2:瓶颈判决是单标签,不是资源向量

`classify()` 返回单个 `kind`,8 种取值之一。"每个维度的消耗比例"被**压缩成一个词**。

好消息:分量大多已算出来了。`evidence` 里已有 `pct_of_dram_peak`、`pct_of_compute_peak`、`occupancy`(带 `occupancy_limiter`)、`n_regs`、`n_spills`、`shared_bytes`、`arithmetic_intensity`、`ridge_flop_per_byte`、`uses_tensor_cores`。**向量已经存在,只是被 if-else 链塌缩成标签。**

实证代价:**L3:21 的 20 份报告里 19 份是 `resource_limited`**,几乎没有区分度。

**并且四个 SASS 计数器已在采集、从未被读取**(已复验:`shared_load`/`shared_store`/`vec_64` 在 statics.py 之外 0 次引用,`barrier` 1 次)。**它们已经躺在每个 run 的磁盘上**——共享内存维度与同步维度的免费数据。

**文献给了三条形状上的改进**:
- **Halide 的 39 个特征**(SIGGRAPH'19 附录 A)说明我们"每资源一个标量"分辨率不足:字节数与工作集在**每个层级分别记录**(6 个 byte-count、5 个 working-set 尺度);**unique vs total 字节是两个特征**——差值**就是**复用;**launch cost 是一等成本项**。
- **Ansor 的算术强度曲线**(每个 loop level 采 10 点)而非单一标量。
- **MAESTRO 的 "Runtime Bottleneck Info."**:直接输出**哪个维度在绑定**,这是最小的忠实实现。
- **cuPilot 用三个 roofline 区**(memory / 中间 / compute),二分标签会把我们的中间情形误路由。

**性能模型那一路又给了三条 counter-free 的改进 —— 但第一条已于 2026-09-10 撤回**:

- ~~**向量的闭合性靠 TMA 的守恒分母 + 显式残差**~~ —— **已撤回(用户否决"合成一个数"这个方向)**。把 `total_ms` 划分成具名分量并要求加和守恒,**本质就是合成**:它假定分量可比、可加、**数目固定**,而这与"维度数随设备变、且每次结构改动后占用率都变"直接冲突。详见 `v3-revision-no-scalarization-and-retire-risk.md`。

  **替代设计:按维度独立,维度集合按设备/任务判定适用性。**(原文写作"维度集合运行期发现",**2026-09-10 已更正为实际能做到的形态** —— 见下方 `applicable` 一条:我们能做的是**预置维度集合 + 逐维判定是否适用**,而**不是**自动发现一个未预置的新维度。那个机制从未设计,把它说成"运行期发现"是把可扩展性说成了自动性。)每个维度是一条独立记录:
  `{dimension_id, measured, ceiling, ceiling_provenance, verdict, confidence, applicable}`
  - **`applicable` 是一等字段**:V100 无 bf16 → 相关维度**不存在**,而不是"值为 0";4090 上 tf32≈fp32 → 精度维的交换价值结构性地小。**`applicable=false` 与 `measured=0` 必须永远可区分** —— 这正是 `task_cost.py` 已用 `notes` 解决过的同一个问题("`flop_count = 0` 对 maxpool 是合法结果、对 matmul 是失败结果,只有这个字段能区分"),沿用那个已验证的做法。
  - **判决保持多份,不投票**:每维输出自己的 verdict(binding / near-binding / slack / not-applicable / unknown)。多维同时 binding 时,**输出就是"多个维度同时 binding"** —— 不选赢家、不加权、不取 max。这恰好修掉 v2 已实测的缺陷:20 份报告 19 份 `resource_limited`,**问题不是标签选错,而是"必须选一个"这个要求本身**。
  - **资源占用是结构特定的**:改写后必须重测,不得沿用父候选的向量。代码后果:资源记录须按 `(候选, 结构签名)` 键入而非按族键入,否则 rewrite 会继承父代数字 —— S2 实现时验证。
  - **维度间关系记录成对观测,不聚合**:`(维度 i 的变化, 维度 j 的变化, 延迟的变化)`,来自同一次改动的前后对比;**不归约成交换率标量、不跨候选平均、不作选择依据**。用途是给消化层一句可证伪的话。
  - **留下的空缺(必须记)**:TMA 式闭合本可用"分量和 ≤ total"抓算错,撤回它就失去了这个自检。**替代自检待定** —— 候选做法是逐维度量纲/范围检查 + `applicable`/`measured` 一致性检查(`applicable=false` 却带 measured 值 = 缺陷)。

  **仍然保留的一条**:把内存拆成"带宽压力"与"延迟暴露"**两个独立维度**(TMA 与 ERM 独立到达的同一条缝)。这不是合成 —— 它是把一个维度**拆成两个**,方向与按维度独立一致,且恰好区分"减字节"与"加 MLP/occupancy"两种相反的改写策略。

- **访存模式坐标可在编译前算出(Instruction Roofline 的"墙")。** Ding & Williams(PMBS@SC19)的指令强度被限死在 `[1/32, 1]`,且若干位置有已知名字的墙:**stride-0 广播 = 1;单位步长 = 1/4(FP32)或 1/8(FP64);stride-8 / 随机 / gather = 1/32;shared 无 bank 冲突 = 1,32 路冲突 = 1/32**。候选相对这些墙的横坐标就是其访存模式的直接读数,**而它可以从 Triton 的 tile/stride 配置静态推出来,不需要 counter、不需要跑**。这与我们已有的编译期 shared-memory 探测同一类收益(那一类省下 180/1004 个 trial)。**这是某一维自己的坐标而非合成量,不受上面的撤回影响。**
  它还给了一个我们本该有的维度:**线程谓词化**。`inst_executed`(warp)与 `inst_thread_executed/32`(线程)之比就是谓词化程度;原文案例里 HPGMG 两个变体**执行时间完全相同而 GFLOP/s 差 2 倍**,每 warp 活跃线程比 **50% vs 100%**。**这正是会抓住我们 `if bn.training:` 死分支的信号**(31 个 trial 全测 fallback、零报错、全 run 最差)。计数本身要 counter → 被挡;但**谓词与分支结构在 cubin 里可见**,我们已在读 cubin。

- **分层化不需要 counter,而复用是分量之差(hierarchical roofline)。** FLOP 固定、只换分母,每个 kernel 变成 L1/L2/HBM 的**三元点组**;**点组的间距就是缓存复用的测量值**(原文:点挨在一起 = streaming 访存 + 差的局部性;L2↔HBM 间距大 = 高 L2 局部性)。它抓的陷阱正是我们会犯的:kernel 在 HBM 层读作 compute-bound,实际被 L1/L2 带宽限制。**复用是流量向量的分量之差,不是一个 counter** —— 我们已按任务解析算 FLOP/字节,补上按 tile 形状解析的分层流量即可。**这也不是合成**:它是"同一个量在多个层级各有一个独立读数",与按维度独立一致。
  **一条警告**:ECM 的实测误差里 **L2 恒是最差的一级**(17–33%),原文"in none of the cases the measured L2 performance could live up to the advertised specs"。**别让任何判决挂在 L2 那一项上。**

- **访存模式坐标可在编译前算出(Instruction Roofline 的"墙")。** Ding & Williams(PMBS@SC19)的指令强度被限死在 `[1/32, 1]`,且若干位置有已知名字的墙:**stride-0 广播 = 1;单位步长 = 1/4(FP32)或 1/8(FP64);stride-8 / 随机 / gather = 1/32;shared 无 bank 冲突 = 1,32 路冲突 = 1/32**。候选相对这些墙的横坐标就是其访存模式的直接读数,**而它可以从 Triton 的 tile/stride 配置静态推出来,不需要 counter、不需要跑**。这与我们已有的编译期 shared-memory 探测同一类收益(那一类省下 180/1004 个 trial)。
  它还给了一个我们本该有的维度:**线程谓词化**。`inst_executed`(warp)与 `inst_thread_executed/32`(线程)之比就是谓词化程度;原文案例里 HPGMG 两个变体**执行时间完全相同而 GFLOP/s 差 2 倍**,每 warp 活跃线程比 **50% vs 100%**。**这正是会抓住我们 `if bn.training:` 死分支的信号**(31 个 trial 全测 fallback、零报错、全 run 最差)。计数本身要 counter → 被挡;但**谓词与分支结构在 cubin 里可见**,我们已在读 cubin。

- **分层化不需要 counter,而复用是分量之差(hierarchical roofline)。** FLOP 固定、只换分母,每个 kernel 变成 L1/L2/HBM 的**三元点组**;**点组的间距就是缓存复用的测量值**(原文:点挨在一起 = streaming 访存 + 差的局部性;L2↔HBM 间距大 = 高 L2 局部性)。它抓的陷阱正是我们会犯的:kernel 在 HBM 层读作 compute-bound,实际被 L1/L2 带宽限制。**复用是流量向量的分量之差,不是一个 counter** —— 我们已按任务解析算 FLOP/字节,补上按 tile 形状解析的分层流量即可。
  **一条警告**:ECM 的实测误差里 **L2 恒是最差的一级**(17–33%),原文"in none of the cases the measured L2 performance could live up to the advertised specs"。**别让任何判决挂在 L2 那一项上。**

### 缺口 3:资源维度覆盖面窄

| 维度 | v2 现状 |
|---|---|
| 算力 | ✅ 实测四档屋顶。**但缺 per-backend**:Triton 的 ieee `tl.dot` 被拿 cuBLAS 屋顶比,读出虚低的 18% |
| 内存带宽 | ⚠️ 已测,但分子是下界(缺口 1) |
| 存储/容量 | ⚠️ shared/regs 有硬限检查,但**仅作布尔** |
| 启动开销 | ⚠️ 测了地板,但 `launch_bound` 判决 **848/848 trial 从未触发** |
| 峰值显存 | ❌ 不存在。**KernelBench-Verified 实测 28% 的 kernel 会抬高峰值显存**——这是个必须报告的维度 |
| 通信 | ❌ 完全不存在 |
| 多卡 | ❌ 不存在 |

### 缺口 4:"效率"从未被计算,搜索目标是单一延迟

- 目标函数是 `robust_ms`(`tpe.py:100`)。**Δ性能/Δ开销 在任何地方都不被计算。**
- 改写推进判据 `min_improvement_pct: 2.0` 只看性能变化,不看代价。框架**无法区分**"用 2 倍 shared memory 换 2% 提升"与"零额外开销换 2% 提升"。
- 中间步骤效率无记录,**事后也无法重建**。

**但"改成多目标"是错的解法**(调研否决,依据是它自己的数字):Optuna 的 MOTPE 只把 **10% 的 trial 喂给 `l(x)`——n=40 时是 4 个点**,且 `multivariate` 在多目标下**静默变成 False**(丢掉我们依赖的 tile/warp 相关性建模),再加 10 个 startup trial。qEHVI 在 M=4 时 acquisition 耗时 **459 s**(是一整个 trial 的 25×),而 box decomposition 在 M≥4 时是未解问题——我们的向量有 5–6 维。Optuna 自己的文档警告:目标多了"a large fraction of trials may become non-dominated",我们的"前沿"就会等于 trial 日志本身。

**所以:保持单目标(median latency),资源作为 attribute 上报 / 作为廉价约束前移。**

---

## 三、v3 改造路线(概要;细节见实施方案)

按依赖与确定性排序。**第 1、2 阶段是"确定 + 便宜 + 判据明确",第 3 阶段起是探索项。**

- **阶段 1(最高价值,最便宜)**:把编译期可知的不可行**从空间里声明掉**,而不是抽出再拒。判据:`infeasible_shared_memory` 的 record 数应大幅下降,且早期搜索质量改善(依据 §1.3 的两个外部实测)。**注意"实测已确立的不可行"那一半(取值级别)已于 2026-09-10 降级** —— 原提案"N 次全败即摘掉取值"经历史重放证明在 N=6/8 下误杀率 38.5%/25.0%,替代判据见 `v3-revision-no-scalarization-and-retire-risk.md` §1.5,且**必须先通过 7 个历史 run 的零误杀重放才允许实现**。
- **阶段 2(原料齐备)**:判决从标签改为**按维度独立的多份判决 + 消化层**。向量按 Halide 的形状加密度(分层字节/工作集、unique vs total)、加上四个已在采集但无人读取的 SASS 计数器、加峰值显存;**把内存拆成带宽压力与延迟暴露两维**;**每维一条独立记录并带 `applicable` 字段,多维同时 binding 时就输出多份,不投票不聚合**(§二缺口 2)。**不做 TMA 式闭合、不做残差项** —— 那是合成,已撤回。**给 agent 的必须是消化后的诊断,不是数值**(§零修订 1)。
- **阶段 2b(与阶段 2 同源,静态可算)**:**访存模式坐标**——按 Instruction Roofline 的墙(1 / 1/4 / 1/8 / 1/32)从 tile+stride 配置**静态**定位候选,把 gather 墙上的候选在编译前标出。与阶段 1 同一类收益,同一批代码路径。
- **阶段 3**:每维**下界**作为"还剩多少空间"的锚。**`Performance-Score` 降级**:它是把两维压成一个标量,**不作头条上报量、不作判据**,只在报告里作参考数字。**同时按 §1.2 的 ceiling-order 教训:屋顶的可达性必须按任务从实测推导,不得作为常量写进 prompt。**
- **阶段 4(转化效率)**:Roller 的 `S=ΔQ/ΔF` 作**静态估计器**用于 trial 排序;**已实现**的成对观测 `(Δ维度 i, Δ维度 j, Δ延迟)` 作报告与 prompt 信号。**四条硬约束**:不作 TPE 目标;噪声底以下报 unknown;不作筛选剪枝(v2 实测反例:L2:37 上**最慢**的种子所属族改善最多 −30.5%,按效率剪枝会杀掉赢家);**不把成对观测归约成交换率标量、不跨候选平均**(这是 2026-09-10 新增的一条,与"不合成"同源)。
- **阶段 5**:候选真实流量测量(**探索项**,缺口 1)。
- ~~**阶段 5b**:重叠系数 α~~ —— **整条取消**(2026-09-10)。它的前提是"要把向量合成一个数",而那个需求已被否决。
- **阶段 6**:通信维度,为多卡/异构预留。

~~**一条排序自检(借 Gables 的算例)**~~ —— **降级为动机例子,不作验收判据**(2026-09-10):它检验的是"合成规则能否复现某个排序",而我们不再有合成规则。Gables 的四步(40 → 1.3 → 2 → 160 Gops/s,三步都是"把某个资源朝局部正确的方向动然后变差")仍是这套思想最干净的公开演示,保留作动机。

---

## 四、新颖性的诚实评估

**已不新颖(必须直说)**:把瓶颈诊断/roofline 分类反馈给 LLM。KernelPro、NVIDIA SOL agent、Kernel Foundry、SOLAR、cuPilot、HIERA、Compiler-Grounded Hierarchical Diagnosis 都在做,多个还做了有利消融。**"消化优于原始计数器"也已被 KernelPro 以 p<0.0001 claim 掉**;"用余量作停止规则"已被 NVIDIA SOL claim(省 19–43% token)。

**再加两条(性能模型那一路新增,初稿低估了先验工作)**:
- **"多资源向量作性能模型的状态"不新颖** —— ERM(IISWC 2014)已有 22 参数资源模型 + 三层利用率分解 + 显式重叠系数 α;Gables(HPCA 2019)有 N 个单元的版本。
- **"用 gap 高度作优化的预期回报"不新颖** —— Roofline 原文(2009)就写了"The height of the gap between a ceiling and the next higher one is the potential reward for trying that optimization",并把五条 ceiling 按"编译器易得 → 程序员难得"排序。

**调研找不到已发表对应物的(可守的地面)**:

1. **把"交换"作为优化对象,用转化效率作为被评分的量。** 精确表述(按修订 4 收窄):已有多资源工作要么**预测一个界就停**,要么**需要 cycle 级 DAG 调度或硬件 counter**;**没有人把资源维度间的替代率当作一个按任务实测、并用于引导搜索的量。** 不要 claim "第一个多资源模型"——那会被 ERM 直接驳回。
2. **闭合"预期 vs 实现"的环。** KernelPro 给出 3–5× 的预期改善却**从不回头验证**。记录**已实现**的每单位资源延迟增益、并把**经验交换比**反馈回去,在约 120 篇调研中未见占据。
3. **按设备、按精度、按后端实测的天花板作分母。** 所有人用规格书或理论 roofline。**外部实测支撑(这条从"推断"升级为"有证据")**:hierarchical roofline 那一族实测 V100 张量核**经 cuBLAS 达理论峰的 96.5%,经 WMMA 只有 54%** —— 而 **Triton 发的就是 WMMA**。所以分母错配是一个有实测的真实失效模式,只是没人把"每后端实测天花板"做成机制。我们自己的对应实证是 fp16 候选读出 107.8%(实际约 54–60%)。
4. **测量"哪个反馈信号真的导致了改善"。** KernelPro 的 per-tool 命中率表最接近,但**没人核查获胜配置是否真的用了该信号建议的值**。
5. **噪声底感知的正确性门。** KernelBench-Verified / The Correctness Illusion / Hardening Agent Benchmarks 已把 eval 完整性立为研究问题,而**领域目前没有按任务实测的噪声底**。**这可能是比"交换"框架更强、更好守的贡献。**
6. **把 LP 对偶/影子价格用于代码或 kernel 优化的瓶颈归因** —— 三种检索方式零命中,且该次检索有工作的正对照(同一方法正确找到了 ERM/DrGPU/Gables/TMA)。**借词汇与两条定理,不建 LP**:(a) **互补松弛作可证伪自检** —— 若模型判定某维松弛,则在该维花资源应买到**零**改善,这是一次廉价消融就能做的证伪;(b) **对偶之比 = 替代率** —— 把转化效率定义为 `Δlatency/Δresource_j`,使它成为有量纲的量。**并预期退化**:33 个近平局 + 每 trial std 占均值 16% 是教科书式的对偶退化,文献的答案提示我们应**每维报价格区间而非点值,区间重叠时拒绝行动**。
   **2026-09-10 收窄**:"两维之间的交换率取二者之比"这半句撤回 —— 那是聚合。保留的是**每维各自的价格区间**与**成对观测**,不构造跨维度的比值标量。

7. **拒绝合成:维持多份并列的按维度判决,且维度集合按设备与结构逐维判定适用性。**(2026-09-10 新增,来自用户对设计方向的否决;**同日更正**:原文此处写"维度集合本身是运行期发现的",实际能做到的是**预置集合 + 判定 applicable**,新维度的自动发现机制不存在)

   调研 §3.9 的结论是:**现存所有多资源模型最终都收敛到一张 2D 图或一个标量界** —— ERM 用 α 合成、ECM 用重叠假设合成、Gables 用 `max` 合成、Nsight SOL 在维度间取 max、CUDABench 的 `Performance-Score` 把两维压成一个标量。**"拒绝合成、维持多份并列判决、并让每个维度自带 applicable 与 ceiling_provenance"在约 120 篇调研里没有对应物。**(此处原写"把维度集合当作运行期产物",按上面的更正收窄 ——**新颖性主张不能建立在一个我们没有实现机制的能力上**。)

   **这条比第 1 条更好守**,因为它是一个**设计选择**而非测量声明:审稿人无法用"某篇 2014 年的论文也有资源向量"来驳回它(那恰恰是我们承认的先验工作)。要驳回它必须论证合成是必要的,而我们有三条理由说它不是 —— 借来的公式携带借来的假设、各维度可独立考虑、维度数不固定。

**一条方法论警告**:KernelPro 的消融标准(42 任务 × 15 种子、单侧 Wilcoxon、报非平局 n、六个阈值下的 fast_p)就是我们会被要求达到的。而我们 19 个 run 只有 1 个跑到墙钟、Loop D 在 18 个 run 里零执行、调参发生在噪声底以内——**差距在统计功效,不在想法。**

**一条基线警告**:KernelBench-Verified 已复验——最好的模型在 TF32 基线 + 四分布隐藏测试下从 **1.43× 掉到 0.88×**(比 PyTorch 慢)。**任何对 eager 基线宣称的加速比现在都可被攻击。** 我们已在用 `torch_compile_tf32` 作同精度基线,应在论文里明确前置。

---

## 五、异构显卡与该思想的关系

已实测的关键事实:**4090(Ada)上 tf32 ≈ fp32 速率**(实测 88.1 vs 54.8 TFLOP/s,1.61×)。这压缩了精度维在 4090 上的交换价值。

**外部数据佐证精度是最大单旋钮杠杆**(NVIDIA A100 datasheet,dense):TF32 **8×** fp32,BF16/FP16 **16×**。这正是 v3 思想在 A100 上会真正起作用的原因。

| 卡 | 架构 | 对 v3 的价值 |
|---|---|---|
| **A100-40GB** | Ampere sm_80 | **验证"精度作为算力维度交换"最佳**:tf32 8×、bf16 16×,精度从"可有可无"变第一杠杆。全功能卡,四档精度都真实存在 |
| **H20-96GB** | Hopper sm_90 | **这套资源比例思想最好的试验台**:算力阉割 + 带宽极高(4TB/s)= **极端不平衡卡**,比例失配最严重。唯一有 TMA/wgmma(CuTe 主场)。注意结果不可外推到 H100 |
| **V100-32GB** | Volta sm_70 | 鲁棒性破坏测试:**无 tf32、无 bf16**,顶破框架隐含假设,检验校准能否优雅降级 |
| **RTX 5090** | Blackwell sm_120 | **信息量最低**:与已测的 5080/4090 同族同故事 |

**CUTLASS/CuTe 推迟到 H 卡**:worker 未安装;实测**手写 CUDA 比 Triton 慢 0.874×**,KernelPro 的 1.23× 来自 **CuTe on Hopper** 而非裸 CUDA。在 4090 上写它**测不出收益、无判据**。地基已完成:后端中立 profiling、`structural_signature` 含后端前缀。

---

## 六、代价、风险与纪律

- **阶段 5(真实流量)是探索项**,三条现成路都不通(§二缺口 1)。不得把它当确定项排进关键路径。
- **阶段 1、2 性价比最高**(原料齐备 + 外部实测支持),先做。
- 全部改动**泛化**(不针对任何 task/candidate),符合既定修复标准。
- **阶段 1、2 改变 agent 看到的信息 → 改动前后 run 不可直接比较。** 必须在两箱空窗时统一上线并记下分界线。
- **凡引用 v2 "已修复"的断言,进论文前须在当前 checkout 复验并带 commit**(memory 已两次因跨机断言出错)。

### 从 v2 继承且不重做的(已验证承重,勿动)

fp64 相对门;cosine 溢出修复(float64+归一化);四档精度实测天花板 + 校准 schema 版本;F5 编译期 shared-memory 判定(已前移进 `guard_ok`);median 作为调参目标(排序正确率 93.2% vs mean 64.8%——与 AutoTVM 发现 rank loss 优于回归同向);厂商库许可;artifact rescue;后端由源码裁决;**自校准阈值链**(实测四个已知瓶颈的标尺反推门限,两个猜测常数已被实测否决);**Tier 0–3 能力分级**。

### v2 遗留的开放项

D1 死值升级(bf16 在 L3:21 两轮 0-for-100,但在 L3:43 是**获胜精度**);P6-C per-precision 最优上报;P9 per-backend 天花板(已复验未落地);P2 `launch_bound` 不可达;固定 0.99 门 vs 三任务噪声底(0.9554/0.9767/0.9778)全低于它;Loop D 在 L3 上因墙钟从未被调度。

---

## 七、实验平台安排

- **v2 分支继续**跑完当前两个 run:box 1 的 L3:48 r2、box 2 的 L3:21+ceilings(D4 的 ceilings 开/关对照)。
- **box 1 任务完成后转为 v3 的实验平台**。
- v3 的第一个实验应当是**阶段 1 的判据验证**,而不是新任务。
