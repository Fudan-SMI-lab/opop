# v3 实施方案

**状态**:实施方案,2026-09-09。配套 `v3-design-resource-ratio-and-conversion-efficiency.md`(思路与设计约束)与 `v3-prior-art-and-what-to-borrow.md`(调研出处)。

本文档的职责是把设计文档 §三 的路线展开成**可执行、可验收、可否证**的阶段。每个阶段写四件事:**改什么、判据是什么、反向对照是什么、失败时怎么退**。

---

## 零、贯穿全程的六条纪律(违反即回退)

这些不是建议,是 v2 用实测事故换来的门槛。每个阶段的验收都要过这六条。

| # | 纪律 | 来源事故 |
|---|---|---|
| D-1 | **泛化,禁止针对单个 task/candidate 的特判,尤其禁止硬编码** | 用户既定标准 |
| D-2 | **进入选择/排序/分配的数字必须是已实现的实测量,不得是预测** | `predicted_gain_pct` n=21,中位偏差 −5.0%,8/21 符号错 |
| D-3 | **效应量未显著超过噪声底时报 `unknown`,不报数字** | L3:48 每 trial std 占均值 16% > 33 个近平局的 9% 跨度 |
| D-4 | **每个"改善"断言都要带因果检查**:比对空间版本 + 确认获胜配置真的用了新增值 | 10 次扩展 4 改善,但 4/4 由原域取值拿下、仅 1/8 可归因 |
| D-5 | **每个反向对照必须能大声失败**;测试必须驱动真实函数,不得在测试体内复刻被测逻辑 | 6 个测试全绿而循环 2.05M 次空转;断言源码字符串让测试在坏代码上通过 |
| D-6 | **改动作用域按 worker/driver 分**:worker 侧修复立即影响运行中实验,driver 侧不可能。且"已修复"的作用域是**那台机器那个 checkout** | 曾据 commit 时间冤枉判定;曾 scp 覆盖 Linux 变体 |

**另一条只对 v3 有效的**:阶段 1、2 改变 agent 看到的信息与搜索空间 → **改动前后的 run 不可直接比较**。必须在两箱空窗时统一上线,并在 `docs/` 记下分界 commit 与分界时间。

---

## 一、阶段总览与排期

| 阶段 | 内容 | 确定性 | 依赖 | 预估 |
|---|---|---|---|---|
| **S1** | 编译期可知的不可行从空间里声明掉 | **确定** | 无 | 2–3 天 + 1 个对照 run |
| **S2** | 判决从标签改为向量 + 消化层 | **确定**(原料齐备) | 无(可与 S1 并行开发,但要串行上线) | 3–4 天 + 1 个对照 run |
| **S2b** | 访存模式墙(静态坐标) | 中(公式确定,映射到 Triton 需验证) | S2 的向量骨架 | 2 天 |
| **S3** | 每维下界 + `Performance-Score` 头条量 | 中高 | S2 | 2 天 |
| **S4** | 转化效率(静态估计器 + 已实现测量) | 中 | S2、S3 | 3 天 |
| **S5** | 候选真实流量测量 | **探索** | — | 未定,先做可行性探针 |
| **S5b** | 重叠系数 α | **探索** | S5 | 未定 |
| **S6** | 通信维度 | 未开始 | 多卡环境 | 未定 |

**排期原则**:S1 与 S2 各自需要一个对照 run 才能宣称生效,而对照 run 要 12 h × 2(开/关)。**所以 S1、S2 的验收是这个计划的关键路径,不是编码。**

---

## 二、S1:编译期可知的不可行,从空间里声明掉

### 2.1 现状(已在 v3 checkout 复验)

- `tuning/tpe.py:119-121`:`hard = failure_kind in ("infeasible_shared_memory","guard_rejected","materialize_error")` → 报 `PRUNED`,其余 → 报 `FAIL`。
- `control/orchestrator.py:1028-1029`:`guard_ok=lambda p: (check_config(space, p, self.cfg.device) is None and self._shared_memory_ok(crun, p))` —— 已有的 F5/P5 前移做的是**采样时拒绝并重问**,不是**从空间里去掉**。
- 全仓库 **0 处** `early_config_prune` / `prune_configs_by`。
- v2 实测:此类必然失败占 **180/1004 trial(18%)**;box 1 当前 run 的 `CONFIG_SCREENED_INFEASIBLE` = 22,box 2 = **96**。

**"抽出再拒"与"声明掉"的差别**是这一阶段的全部内容:前者仍让 TPE 在一个含大量死点的空间上建模并消耗 ask 预算;后者让死点不存在。

### 2.2 三条实现路径中只有两条有作用面(B 已实测排除)

**路径 A(最便宜,先做):依赖域(ATF 式)**

把约束表达为"给定已绑定参数后,本参数的合法域"。例:`BLOCK_K` 的域由 `BLOCK_M`、`NUM_STAGES`、dtype 字节宽共同决定,因为三者一起决定 shared 需求。

- 落点:`paramspace/` 新增一层,在 `guard.py` 的**上游**;`tpe.py` 的 `ask()` 从**收缩后的域**取值。
- 关键约束:**收缩必须由已有的 `_shared_memory_ok` 那条真值判据驱动**,不得引入新的手写公式。理由是实测记录 `handwritten shared-memory constraints are vacuous`:agent 手写约束中位只有真值的 32%,L3:43 那条 36/36 恒真、从不拒绝。**真值来自编译期探测,不来自公式。**
- 因此域收缩的实现方式是:对该参数的每个 choice,用现成的编译期判据问一次"在当前已绑定值下这个 choice 可行吗",不可行的移出本次 ask 的域。**这是纯查表 + 已有判据,没有新模型。**

**路径 B(已实测排除,不做):Triton 官方钩子 `early_config_prune`**

`prune_configs_by` 接受 `early_config_prune(configs, named_args, **kwargs) -> List[Config]`,是官方的"benchmark 前丢弃配置"入口。它管的是候选**内部自带的** `triton.autotune`(若候选写了),与路径 A 管我们自己的 Optuna 空间互补。

**先测再定,已经测了:box 1 的全部 L3 run 里共 3162 个候选 `.py` 文件,使用 `triton.autotune` 的是 0 个**(`grep -rl 'triton.autotune' */candidates/`,2026-09-09 于 box 1 实测)。我们的 generator/rewriter prompt 要求把可调项走 `PARAMS`,所以候选把调参交给我们的 Optuna 而不是 Triton 自己的 autotuner。

**结论:路径 B 没有作用面,不实现。** 若未来 prompt 改变、候选开始自带 autotune,这条要重新评估 —— 届时的注入必须是 materializer 之外的独立步骤,否则违反"materialize 只改 PARAMS 字节 span"的契约。

**路径 C(仅当 A 不足时):修复到最近合法邻居**

距离 = 参数**值索引**的绝对差之和。用于约束无法干净声明为域的情形。外部实测平均 +39.3%,增益与空间稀疏度相关。

- **我们的预期要下调,但下调多少目前说不准。** 那份工作的增益随稀疏度上升。我们只知道**trial 层面**的必然失败率约 18%(180/1004),而这**不等于空间稀疏度** —— TPE 非均匀采样,它会反复往它认为好的区域去,所以 trial 失败率既可能高于也可能低于空间中不可行点的占比。**先算真实的空间密度**(在声明的域上枚举网格,用编译期判据数可行点占比),再引用外部增益数字。在算出来之前,不要写"我们空间密度 82%"这类推断。
- **风险**:修复引入采样偏置(把概率质量挤到合法域边界)。**这是 C 比 A 差的地方**,也是建议先做 A 的理由。

### 2.3 判据(必须全过)

| 判据 | 怎么测 | 反向对照 |
|---|---|---|
| J1-1 死点确实消失 | 同任务同预算,`infeasible_shared_memory` + `guard_rejected` 的 record 数应从约 18% 降到接近 0 | **若降到 0 但 trial 总数不变**,说明只是改了标签没省预算 → 不算通过 |
| J1-2 有效评估数上升 | 同墙钟内 `status=complete` 的 trial 数 | 若持平 → S1 没起作用,别宣称 |
| J1-3 早期搜索质量改善 | 前 20 trial 的 best,新旧对比 | **这是外部证据预测的地方**(惩罚法"前半段退化成随机搜索")。若前 20 trial 无差别而只有后期有差别,则我们的机制解释错了 |
| J1-4 终局不变差 | final_reeval_ms 新 ≤ 旧 × (1 + 噪声底) | 若变差 → 域收缩误删了合法点,回退 |

### 2.4 离线重放:零 GPU 成本回答 "A 是否真优于现状"

借 Kernel Tuner 的 **simulation mode**:把历史 run 的 (config → median latency) 对从 events.jsonl 抽出来做成查表,离线重放 sampler。

- **能回答**:同一批 trial 预算下,"域收缩" vs "抽出再拒" 谁的 best 更好、收敛更快。
- **不能回答**:域收缩后 TPE 会去采**历史上没测过**的点 —— 查表里没有。**所以重放只能给下界式证据,不能替代对照 run。** 必须在文档和论文里如实说。
- 实现:`scripts/replay_sampler.py`,输入一个或多个 run 目录,输出两条 best-so-far 曲线。**这是唯一负担得起的 sampler 比较手段**,值得先做。

### 2.5 失败退路

`v3.search.declare_infeasible_out_of_space: false` 一键回到现状。**必须是配置开关而非代码分支删除**,因为 S1 的对照 run 需要开/关两次。

---

## 三、S2:判决从单标签改为向量 + 消化层

### 3.1 现状(已复验)

- `evaluation/bottleneck.py` 的 `classify()` 返回单个 `kind`(8 种取值之一)。实证代价:**L3:21 的 20 份报告里 19 份是 `resource_limited`**,几乎无区分度。
- 但**分量大多已算出来了**:`evidence` 已有 `pct_of_dram_peak`、`pct_of_compute_peak`、`occupancy`(带 `occupancy_limiter`)、`n_regs`、`n_spills`、`shared_bytes`、`arithmetic_intensity`、`ridge_flop_per_byte`、`uses_tensor_cores`、`at_limit`。
- **四个 SASS 计数器已在采集、从未被读取**(已复验:`statics.py` 之外 `shared_load` 0 次、`shared_store` 0 次、`vec_64` 0 次引用;`barrier` 1 次)。**它们已经躺在每个 run 的磁盘上。**

### 3.2 向量的形状

**分三层,理由是它们的可信度不同,不能混在一起给 agent。**

**层 1 — 直接实测量(高可信)**
`occupancy` + `limiter`、`n_regs`、`n_spills`、`shared_bytes`、`num_warps`、`compile_s`、`peak_memory_bytes`(**新增维度**,依据 KernelBench-Verified 实测 28% 的 kernel 抬高峰值显存)。

**层 2 — 实测量 ÷ 实测天花板(中可信,分母是已知弱点)**
`pct_of_dram_peak`、`pct_of_compute_peak`(**必须按精度**,fp16 分母缺失曾让候选读出 107.8%)、`Performance-Score`(见 S3)。
**两条门限修正,现在就写进代码**:
- **>100% 时 clamp 并打标记,绝不当作 bug 拒绝**(Nsight 文档明写 "can occasionally exceed 100% in edge cases",我们已真的撞上 107.8%);
- **每个比值必须带分母的出处**(哪次校准、什么精度、什么后端),否则读者无法判断它可信到什么程度。

**层 3 — 静态推导量(低可信,只作排序先验)**
SASS 派生的 `shared_load/store` 强度、`barrier` 密度、`vectorized_frac`;`S=ΔQ/ΔF`(S4);访存模式墙坐标(S2b)。
**层 3 的任何数字都不得进入接受判定**(D-2)。

### 3.3 闭合性:借 TMA 的两条结构规则

**这是 S2 里唯一有真正设计含量的部分。**

1. **分母守恒**:选一个可数、可划分的量作分母。GPU 侧现实可行的是 `total_ms`,划分为具名分量。
2. **显式残差**:定义 `unexplained_ms ≡ total_ms − Σ(已归因分量)`,**而不是**指望分量加满。

**为什么必须这样做**:我们目前的向量分量不满足任何守恒律,任何一处算错都不会被发现。有了残差项,一致性检查变成可执行的:`Σ分量 ≤ total` 且 `unexplained ≥ 0`。**残差大就是"我们不懂这个 kernel"的诚实信号,而不是把损失强行摊到已知维度上。**

**必须一起交代的弱点**:GPU 侧最接近的实现 DrGPU 靠"对各层平均停顿之和归一化"闭合,既非守恒划分也无残差项 —— 它保证百分比加到 100%,但不保证它们**意味着**损失的 100%。**若我们只能做到 DrGPU 那一档,输出里必须把这个警告一起说出来。**

### 3.4 内存拆成两维:带宽压力 vs 延迟暴露

TMA 把 `Cache_Memory_Bandwidth` 与 `Cache_Memory_Latency` 列为两个维度;ERM 用 `U_issue`/`U_lat`/`U_stall` 独立到达同一条缝。**两个来源独立撞到同一处,是这条缝真实存在的好证据。**

**为什么对我们要紧**:这两维指向**完全不同的改写策略** —— 带宽压力 → 减字节;延迟暴露 → 加 MLP/occupancy。我们目前不做这个切分,所以"内存受限"这个标签同时指两件相反的事。

**可执行的最小版本**:用 `occupancy` + `occupancy_limiter` 作延迟暴露的代理(占用低 → 并发不足 → 延迟暴露),用 `pct_of_dram_peak` 作带宽压力。**这是代理不是测量,必须标注为层 2。**

### 3.5 消化层:硬约束,不可绕过

**这是本阶段风险最高的部分,因为做错的后果有实测量化:复现 KernelPro 那个 1.77× 的臂(比不给反馈的 3.35× 更差,p=0.0007)。**

**交付格式固定为四元组**,每个维度一条:

```
{ dimension, severity, root_cause, ranked_recommendation, expected_delta }
```

- `severity`:三档(binding / near-binding / slack)。**用三档不用二档**:二分标签会把我们的中间情形误路由 —— L3:48(带宽 90.4%)与 L3:43(融合头寸 69×)在三区下才分得开。
- `root_cause`:一句话,指向**可改的 knob 或结构**,不是现象重述。
- `ranked_recommendation`:按预期收益排序;**排序依据必须写出来**。
- `expected_delta`:**允许缺失**(报 unknown),不允许编造。按 D-2,它只进 prompt 作方向提示,不进任何选择逻辑。

**原始向量的唯一出口是 events.jsonl 与 report.md。** 代码层面加一道门:构造 prompt 的函数不得访问原始向量对象,只能访问消化后的四元组列表。**这是可测试的**(见 3.7 的 N2)。

**并且瓶颈必须我们自己算,不能问 agent。** 外部实测:给了剖析数据分类准确率 100%,只给源码推理模型只到 64%。**64% 意味着 agent 自述的瓶颈永远不能当证据** —— 它可以进 evidence,不能进判决。

### 3.6 屋顶顺序按任务推导,不作常量

Roofline 原文自己在 SpMV 上推翻了自己的 ceiling 顺序("we would place floating-point mix as the lowest ceiling, since it is **inherent**"),而它对高位 ceiling 的定义里就含"**inherently lacking in a kernel**"。

**这正是我们 L3:48 的事故**:张量核这条"高阶"屋顶在那个任务上天生不可用,框架仍拿它当目标,**8/8 张量核候选被拒、7/7 标量候选被收**,最终 2.09 ms / 8.90x 完全没用上张量核。

**可执行规则**:任何屋顶在进入 prompt 前,先问"这个任务在这张卡上能达到它吗";答否的屋顶**降为背景信息并注明不可达原因**,不进 recommendation。判据见 3.7 的 J2-4。

### 3.7 判据与反向对照

| 判据 | 怎么测 | 反向对照 |
|---|---|---|
| J2-1 区分度上升 | 同任务的 N 份报告里,`severity` 三元组的不同取值组合数。基线:L3:21 的 19/20 同标签 | **若组合数上升但都是噪声抖动造成的**(同一候选重跑给出不同 severity)→ 不算通过。需重跑一致性检查 |
| J2-2 闭合性成立 | 每份报告 `Σ分量 ≤ total_ms` 且 `unexplained ≥ 0`,100% 满足 | **N1(必须会失败的测试)**:人为把一个分量放大 10×,一致性检查必须报错。若不报错,检查是装饰 |
| J2-3 原始向量不进 prompt | **N2**:在 prompt 构造函数上加禁止访问断言;写一个测试**故意**把原始向量传进去,断言它抛错 | 按 D-5,测试必须驱动真实的 prompt 构造路径,不得在测试体内复刻 |
| J2-4 不可达屋顶被标出 | 在 L3:48 上重放:张量核屋顶必须被标为不可达并附原因 | **若它仍出现在 recommendation 里**,S2 没解决 L3:48 那类事故 |
| J2-5 终局不变差 | final_reeval_ms 新 ≤ 旧 × (1 + 噪声底) | **这是 S2 最大的风险点**:改反馈内容可能变差(外部有 few-shot 范例降低 fast_1 的先例)。变差就回退消化层措辞,不回退向量 |

### 3.8 失败退路

`v3.diagnosis.mode: label | vector` 开关。**向量与标签共存一段时间**:向量进 events.jsonl(零风险,只是多记数据),`mode` 决定 prompt 用哪个。这让 J2-5 的对照 run 可以只切一个开关。

---

## 四、S2b:访存模式墙(静态坐标)

### 4.1 内容

Instruction Roofline 的指令强度被限死在 `[1/32, 1]`,若干位置有已知名字的墙:

| 模式 | 强度 |
|---|---|
| stride-0(广播) | 1 |
| 单位步长 FP32/INT32 | 1/4 |
| 单位步长 FP64 | 1/8 |
| stride-8(FP32)/ stride-4(FP64)/ 随机 / gather | 1/32 |
| shared 无 bank 冲突 | 1 |
| shared 32 路冲突 | 1/32 |

**候选相对这些墙的横坐标就是其访存模式的直接读数,而它可以从 Triton 的 tile/stride 配置静态推出来** —— 不需要 counter、不需要跑。与 S1 同一类收益(编译前排除坏点),同一批代码路径。

### 4.2 判据

| 判据 | 怎么测 | 反向对照 |
|---|---|---|
| J2b-1 映射正确 | 手工构造已知模式的 Triton kernel(单位步长 / gather / bank 冲突各一个),推导坐标须落在预期墙上 | **正对照必须存在**(纪律 `probe needs a positive control`):至少一个 kernel 必须落在 1/32 墙上,否则说明推导器恒返回好结果 |
| J2b-2 与实测相关 | 在历史 run 的候选上算坐标,与其 `pct_of_dram_peak` 求相关 | **若无相关**,坐标是装饰 → 只进 events.jsonl,不进 recommendation |

**明确的能力边界**:warp-vs-thread 谓词化间隙(会抓住我们 `if bn.training:` 死分支那类事故的信号)**需要 counter,被 `ERR_NVGPUCTRPERM` 挡住**。cubin 里能看到谓词与分支结构,但"多少线程实际活跃"看不到。**不要把 S2b 宣称成能抓死分支。**

---

## 五、S3:每维下界 + `Performance-Score` 头条量

### 5.1 每维下界

`evaluation/task_cost.py` 的 `compulsory_bytes` 已是流量维度的下界(且文档已写明它是**上界式**的 `reference_bytes` 与**下界式**的 `compulsory_bytes` 之差承载信号)。要补的是其余维度的"还剩多少空间"锚点。

**注意 `task_cost.py` 自己的设计声明**:这些量是**在参考上测的、任务级的**,"a per-candidate count would instead measure what that candidate happens to do, which is exactly the quantity being optimized and therefore useless as a yardstick"。**这是对的,不要改。** S5 要补的是**另一个**量(候选实际流量),两者并存,不是替换。

### 5.2 `Performance-Score`

```
Attainable = min(Peak FP, Peak BW × AI)
Score      = Achieved / Attainable
```

原文语义:"reduces to Memory Bandwidth Utilization for memory-bound kernels and to Compute Utilization for compute-bound kernels"。

**为什么它适合当头条量**:单一可审计标量、**无 counter 可算**、跨卡稳定(外部报 A40 vs 4090 一致)。它就是"消耗投影到供给"的最简形式。

**判据**:在历史 run 上重算,`Score` 必须与我们已知的三个判决一致 —— L3:48 应给出高分(90.4% 带宽),fp16 那个曾读出 107.8% 的候选应在 clamp 后给出约 0.54–0.60,L3:21 的 `resource_limited` 群应分散开而非全挤在一处。**第三条是真正的检验**:若仍全挤在一处,`Score` 没有比标签多给信息。

---

## 六、S4:转化效率

### 6.1 两个量,可信度不同,用途不同

**(a) 静态估计器 `S = ΔQ/ΔF`(Roller 式)**

`Q` = 内存流量,`F` = 内存占用,`T′_i` = 把第 i 轴换成下一个对齐尺寸后的新 tile。**纯静态可算,零 trial、零 counter。**

- **用途**:K-expansion 的扩展方向排序;TPE 的 warm-start / trial 排序先验 —— 在花 18.6 s/trial 之前先排好。
- **硬边界**:它是**估计**器 → 按 D-2 **不得进入接受判定**。
- **不借的部分**:不用它取代搜索。Roller 自己小算子比 Ansor 慢 50%、tensor core 只到 cuBLAS 43%、自述盲点"cannot detect implicit register allocation beforehand" —— **正是我们 Triton 候选所在**(218 regs / 0 spills / 16.7% occupancy)。更强的反证是 tritonBLAS:我们同一个 Triton 栈、只做 GEMM、94.7% 选择效率,**但真实 Llama3 形状上比 PyTorch 慢 13.9%**,且作者只声称"capture latency *trends*" —— **趋势精度对我们 9% 的近平局跨度毫无用处。**

**(b) 已实现的 `Δlatency / Δresource`**

事后测量同一次改写实际花了什么、换到了什么。**这是 v3 的核心量。**

- **用途**:报告 + prompt 信号(经消化层)。
- **四条硬约束**:
  1. **不作 TPE 目标**(多目标会退化我们的 tuner:MOTPE 只把 10% trial 喂给 `l(x)`,n=40 时是 4 个点,且 `multivariate` 静默变 False;qEHVI 在 M=4 时 acquisition 459 s = 25 个整 trial);
  2. **噪声底以下报 unknown**(D-3);
  3. **不作筛选剪枝** —— v2 实测反例:L2:37 上**最慢**的种子所属族改善最多(−30.5%),按效率剪枝会杀掉赢家。同向另一例:L3:21 领跑族的 seed 是四个里最差的;
  4. **每维报价格区间而非点值,区间重叠时拒绝行动** —— 这是对偶退化的标准处理(33 个近平局 + 每 trial std 占均值 16% 就是教科书式退化),也正好防住把噪声内的"增益"当排序。

### 6.2 互补松弛作可证伪自检

LP 对偶给的两条可用定理之一:**非绑定约束的影子价格为零**。翻译成我们的实验:**若向量判定维度 j 松弛,则在 j 上花资源应买到零延迟改善。**

**这是一次廉价消融就能做的证伪。** 值得做,因为它检验的是我们自己的归因,而不是候选的质量。**若松弛维度上花资源反而买到了改善,我们的向量是错的** —— 这个反向对照必须能大声失败(D-5)。

### 6.3 排序自检(不需要 GPU)

Gables 的四步算例:`40 → 1.3 → 2 → 160` Gops/s。四步里三步是"把某个资源朝局部正确的方向动然后变差",第三步是花 3 倍带宽只换 1.5 倍、然后把带宽还回去改提复用。

**我们的合成规则应当能复现这四个配置的排序。做不到就说明规则错了。** 纯算术,零 GPU 成本,应当写成单元测试。

---

## 七、S5 / S5b:探索项

### 7.1 S5:候选真实流量 —— 三条现成路已逐一在代码里复验为不通

| 路 | 为什么不通 |
|---|---|
| SASS 指令计数 × grid | `statics.py:63-74` 数的是 LDG/STG 在汇编**文本里出现几次**,是**静态**计数,不含循环执行次数。K 维循环 100 次的 GEMM 其 LDG 可能只出现两三条 |
| `TorchDispatchMode`(`worker_main.py:1336-1380`) | **aten 级**。它能测参考的 `reference_bytes` 是因为参考由许多 aten op 组成;**融合良好的候选只有一个自定义 Triton op**,dispatch 在它内部什么也看不见。**系统性低报,且融合越好越低报** —— 方向完全错 |
| 静态→动态换算 | `ProfileRecord` 里**没有 grid 尺寸/启动次数**(已复验字段表),连乘数都缺 |

**所以 S5 的第一步不是实现,是可行性探针。** 三个候选方向,各写一个会大声失败的探针:

1. **给 `ProfileRecord` 补 grid/launch 计数** —— 这是最小的、独立有用的改动(不依赖 S5 成功),先做。
2. **在 Triton kernel 外层包一个计数装饰**,统计实际 launch 次数与每次的 grid。**风险**:改变候选源码 → 与 materializer 契约冲突,需独立步骤。
3. **解析 Triton IR 而非 SASS** —— IR 里循环边界可能可见。**未验证,优先级最低。**

**S5 不进关键路径。** 但方向 1 单独就有价值,应当排进 S3 之后。

### 7.2 S5b:重叠系数 α

ECM 给了合成规则 `T = max(T_nOL + T_data, T_OL)`,ERM 给了可测标量 `α = T_overlap/min(T_x,T_y) ∈ [0,1]`(0 = 相加,1 = 取 max)。**α 正是改写 agent 一直在猜的数:为省流量而多花的算力,是免费的还是一比一付账的。**

**为什么仍是探索项**:α 的定义需要"该维度单独的分量时间"。ECM/ERM 在 CPU 上从 OSACA 静态汇编分析与分层传输模型取这些分量;**我们在 GPU 上取哪两个分量、怎么取,还没有可执行方案** —— 与 S5 是同一个瓶颈。

**一条不采信的说法**:调研子代理称"两点墙钟实验即可测 α"。**我不采信,因为它没有说这两点具体是什么。** 若要做,第一步是把这两点写清楚并给出正对照。

**这一族的可用性事实值得单独记**:Kerncraft(从循环源码 + 机器描述自动建 Roofline/ECM,含预测最优 blocking 因子的 layer-condition 分析)与 OSACA(从汇编静态算 `T_nOL`/`T_OL` 切分)**整族 counter-free** —— 这是唯一能在租用容器原样跑的先验工作族,而我们已在读 cubin。技术上路是通的,缺的是映射方案。

---

## 八、S6:通信维度

未开始。**它的价值绑定在多卡/异构环境上**,而按设计文档 §五 的卡选择分析,第一张要测的是 **A100-40GB**(验证"精度作为算力维度交换":tf32 8×、bf16 16×,精度从可有可无变第一杠杆)。

**先决条件**:S3 的天花板链在第二张卡上跑通。**自校准阈值链目前只在 4090 上跑过** —— 这是 H 卡/A 卡测试的真实前置项,也是当前最值得先做的"对后续异构测试有价值的工作"。

---

## 九、验收顺序与实验平台

### 9.1 关键路径

```
S1 编码 → 离线重放(零 GPU)→ S1 对照 run(开/关 × 12h)→ 判据 J1-1..4
                                    ↓
S2 编码(可与 S1 并行)→ S2 对照 run(mode=label/vector × 12h)→ 判据 J2-1..5
                                    ↓
S2b / S3 → S4 → (S5 探针独立进行)
```

**S1 与 S2 的对照 run 不可合并。** 两者都改变 agent/tuner 看到的东西,合并会让归因不可能(D-4)。

### 9.2 平台

- **v2 分支继续**跑完当前两个 run:box 1 的 L3:48 r2(已至 714.7/720 min,分钟级内结束)、box 2 的 L3:21+ceilings(485/720 min,best tuned 3.7315)。
- **box 1 任务完成后转为 v3 的实验平台。**
- **v3 的第一个实验是 S1 的判据验证,不是新任务。** 理由:S1 是唯一"确定 + 便宜 + 打中已实测缺陷(18% 浪费)+ 有外部实测支撑"的一项,应当最先拿到自己的证据。

### 9.3 每个阶段收尾时要写的东西

一份 `docs/result-s<N>-<判据结论>.md`,含:分界 commit、两个 run 的 id、逐条判据的实测数字、**反向对照是否真的失败过**、以及"这一阶段没能验证什么"。**最后一项是硬要求** —— v2 的教训是未验证项被沉默地当成已验证(`L1 冒烟通过 ≠ 可跑 L3`)。

---

## 十、与论文的接口

**方法论标准是外部定的,不是我们定的。** KernelPro 的消融规格(42 任务 × 15 种子、单侧 Wilcoxon、报非平局 n、六阈值 fast_p)就是我们会被要求达到的。

**我们当前的差距在统计功效,不在想法**:19 个 run 只有 1 个跑到墙钟、Loop D 在 18 个 run 里零执行、调参发生在噪声底以内。

**所以实施方案里最该保护的资源是墙钟。** 这直接支持 S1 的排序 —— 它把 18% 的 trial 预算还给我们,是唯一能同时改善"结果"与"统计功效"的一项。

**一条基线警告要前置进论文**:KernelBench-Verified 已复验 —— 最好的模型在 TF32 基线 + 四分布隐藏测试下从 **1.43× 掉到 0.88×**(比 PyTorch 慢)。**任何对 eager 基线宣称的加速比现在都可被攻击。** 我们已在用 `torch_compile_tf32` 作同精度基线,应在论文里明确前置,而不是作为附录里的额外一栏。
