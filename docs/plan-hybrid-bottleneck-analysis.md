# 实施计划:瓶颈分类的"硬编码解析 + agent 分析"混合架构

状态:**分析与设计,尚未实施**。用户指示"先不要做任何实施,先分析并确认清楚,等待冒烟实验结束后再统一实施"。
本文档回答用户的 5 个问题,并给出待批准的实施方案。

---

## 问题 1:步骤 1 的阈值测量具体做什么,测出来的阈值有什么用

### 做什么

`linux-server/scripts/probes/probe_bottleneck_signals.py`(已入 git)在目标机器上做三件事:

**(a) 实测这台卡的两个天花板** —— 不用规格书数字:

| 天花板 | 怎么测 | 为什么必须实测 |
|---|---|---|
| DRAM 带宽 | 512MB stream copy 的中位时间 | 容器里时钟常被限频;规格书的 1008 GB/s 在实际容器里可能只有 60% 可达 |
| fp32 算力 | 8192³ matmul,**关掉 tf32** | 同上;而且 tf32 开关会让同一张卡的"峰值"差 8 倍 |

两者相除得到 **roofline 拐点**(ridge point,单位 FLOP/byte):低于它的 kernel 在内存侧,高于它在计算侧。

**(b) 在四个"标准答案已知"的负载上测每个候选信号**:

| 负载 | 解析真值 | 应该被判成 |
|---|---|---|
| COMPUTE 4096³ matmul | 137 GFLOP / 201 MB → ~50 FLOP/byte | compute_bound |
| MEMORY 256MB copy | 0 FLOP / 512 MB → ~0 FLOP/byte | memory_bound |
| LAUNCH 40 个 tiny op | 计算量微不足道,40 次发射 | launch_bound |
| MIXED level2:37 参考 | 6 个 op @ 128×512×1024 | 混合(真实任务对照) |

**(c) DISCRIMINATION CHECK** —— 这是整个探针的真正产出:

```
=== DISCRIMINATION CHECK: does each signal separate the four workloads?
  arithmetic intensity     [...]  spread    XXX.Xx  DISCRIMINATES / does NOT discriminate
  % of DRAM peak           [...]  spread    XXX.Xx  ...
  % of fp32 peak           [...]  spread    XXX.Xx  ...
  CPU-issue / GPU ratio    [...]  spread    XXX.Xx  ...
```

对每个候选信号,报告它在四个负载上的**取值跨度**(max/min)。

### 阈值有什么用 —— 两个层面,缺一不可

**层面一:决定"采集什么"。** 一个在四个负载上取值几乎相同的信号,**无论多便宜都不该采集** ——
它进了 `ProfileRecord` 只会让 agent 以为自己有证据。这是先测量再定字段的唯一理由。

**层面二:决定分类边界。** 目前 `bottleneck.py` 里的四个常数是我**猜的**:

```python
SATURATED_FRAC = 0.60          # >= 60% DRAM 峰值算饱和
COMPUTE_SATURATED_FRAC = 0.50  # >= 50% fp32 峰值算饱和
IDLE_FRAC = 0.25
LAUNCH_BOUND_CPU_RATIO = 1.0
```

猜的常数会直接翻转判决,这不是假设而是已经复现的:**4096³ fp32 matmul 在 30 TFLOPS 天花板下
落在 45.8% 峰值**,恰好跌到 50% 门限的错误一侧,被判成 `mixed` 而不是 `compute_bound`。
探针会给出"一个已知就是 compute_bound 的负载实际达到百分之几",门限就定在那个观测值之下。

**同时也决定 `overhead_floor` 的绝对值**:空 kernel 发射底噪(这台卡上一次最小发射要多少微秒)。
没有这个数,`overhead_floor` 类无法判断。

### 成本与时机

单次运行约 2–4 分钟,**必须独占 GPU**(它测的就是天花板,旁边有人抢卡会把每个天花板都压低,
并把偏低的值永久写进阈值)。已挂监控:冒烟结束 → 确认显存 <500MiB 且无 worker 进程 → 自动执行并回报。

---

## 问题 2:分类器是纯硬编码还是硬编码 + agent 结合

### 现状(诚实回答)

**目前是纯硬编码**,而且比这更糟:`bottleneck.py` **零调用者**,是个孤立模块。
它算出的 `BottleneckVerdict` 现在不去任何地方。

### 用户的意见与决定

> "我更偏向后者, 不能仅依靠纯硬编码或是纯agent, 而是两者结合"

**采纳。** 而且现有架构已经为此留好了位置 —— 现在缺的是把两者接起来,而不是新建机制:

```
                硬编码层(确定性、可复算、agent 不能编造)
                ├─ 实测天花板:DRAM / fp32 / 发射底噪         [步骤 1 产出]
                ├─ 实测本 kernel:cpu_issue_ms / gpu_ms / wall_ms   [已实现 47504ba]
                ├─ 实测资源:regs / spills / shared / 每 kernel 时间 [已实现 6efa850]
                ├─ 实测规模:flop_count / byte_count            [问题 3]
                └─ classify() → BottleneckVerdict{kind, evidence, suggests}
                                        │
                                        ▼ 作为"证据 + 初步判断"喂给
                agent 层(解释、归因、跨层推理)
                └─ AnalystAgent → BottleneckReport{hypotheses, suggested_action}
                                        │
                                        ▼ 纯建议
                harness 决策层(正确性/计时/预算/收敛/晋升 —— 绝不由 agent 决定)
```

**分工的原则(以及为什么这样切)**:

| 层 | 负责 | 不负责 | 理由 |
|---|---|---|---|
| 硬编码 | 测量、算比值、给出**可复算**的初步类别 + 完整 evidence | 解释"为什么"、跨算子归因 | 数字必须不可编造。agent 说"我觉得是内存瓶颈"和实测 88.9% DRAM 峰值不是一回事 |
| agent | 读 evidence + 初步类别 + 参考实现结构,给出**假设与改写方向** | 改判正确性、改判延迟 | 六类之外的东西(bank conflict、warp 分歧、L2)硬编码看不见,而 agent 能从**代码结构**推断 |
| harness | 一切门控 | — | 既有原则,不变 |

**关键设计点:分类结论必须可被 agent 反驳。** `BottleneckVerdict.evidence` 里带着每一个数字,
prompt 里明确写"这是基于可测量信号的初步判断,**counters 不可用**,以下现象它看不见:
bank conflict、warp 分歧、L2 命中率;如果你从代码结构看出别的限制因素,说出来并给出理由"。
一个 agent 不能审计的分类比没有分类更糟 —— 它会恰好在错的时候被信任。

**已有实证支持这个分工**:`latency_bound` 是**残差类**(排除法),诚实读法是"我们能测的都没饱和",
不是"已证明是延迟"。这种类别本身就必须交给 agent 去解释。

---

## 问题 3:为什么需要 flop_count / byte_count,六大类分别怎么判

### 为什么需要

roofline 的两个坐标轴就是它们:

- **`byte_count`** → 实际达到带宽 = `byte_count / gpu_ms` → 占实测 DRAM 天花板的百分比
- **`flop_count`** → 实际达到算力 = `flop_count / gpu_ms` → 占实测 fp32 天花板的百分比
- 两者相除 = **arithmetic intensity**,与 ridge point 比较决定 kernel 在内存侧还是计算侧

没有它们,"这个 kernel 是不是已经把带宽跑满了"**无法回答** —— 而这正是决定"还有没有优化空间"的问题。
一个跑到 89% 带宽的 kernel 再怎么调算术都没用;一个跑到 12% 的则相反。

### 用户的约束

> "整个测量和分析的过程不应该需要外部的任何元数据输入, 完全由框架自己完成"

**已实测确认可以做到,零人工元数据、零源码解析**(在 box 2 上验证):

| 量 | 来源 | 实测结果 |
|---|---|---|
| `byte_count` | worker **已经**建好模型和真实输入 → `numel × element_size` 遍历输入/输出/参数 | L1:42 = 4286.6 MB(in 2147.5 + out 2139.1) |
| `flop_count` | `torch.utils.flop_counter.FlopCounterMode`(PyTorch 自带) | matmul 4096³ 与 conv2d 对解析真值 **ratio = 1.0000** |

**必须说清的限制**:`FlopCounterMode` 只统计 matmul/conv 类算子,maxpool 返回 **0**(实测)。
这不是缺陷而是符合 roofline 语义 —— `flop_count=0` 的算子本就该由内存侧判据裁决。所以:

- GEMM/conv 类任务(异构显卡实验的重点):FLOP 精确 + 字节精确 → 六类全可判
- 纯数据搬运类任务(maxpool 等):FLOP=0 + 字节精确 → 正确落到 memory_bound 一侧

### 泛化性验证(补做,不能只靠一个 case 就写进计划)

我最初只用单输入 maxpool 验证过 `byte_count`,这不足以支撑"框架自测"的结论。已在 box 2 上补测:

| 情形 | 结果 | 说明 |
|---|---|---|
| 含可学习参数的 conv2d | in 205.5 + **params 0.3** + out 411.0 MB | 参数被正确计入(它们也要被读) |
| 多输入(matmul) | in 8.4 / out 4.2 MB | 多个输入张量正确求和 |
| 多输出(`torch.topk` 返回 named tuple) | 0.0614 MB,类型 `topk` | 递归遍历到 tuple 内部,不漏 |
| 非 fp32(fp16 1024²) | 2.10 MB(fp32 同尺寸 4.19) | `element_size()` 让 dtype 自动正确 |

`FlopCounterMode` 也在**我们真实的 L3 任务形状**上验证,而非只在教科书 matmul 上:

| 任务类型 | counter vs 解析真值 | ratio |
|---|---|---|
| L3:43 类 causal attention(含 softmax / masked_fill) | 2,147,483,648 vs 2,147,483,648 | **1.0000** |
| L3:21 类 depthwise conv + BatchNorm | 57,802,752 vs 57,802,752 | **1.0000** |

注意第一行的读法:softmax 与 masked_fill **不计入**(它们没有 mul-add),两个 matmul 精确。
这正是 roofline 想要的语义 —— 这些算子的成本在字节侧,不在 FLOP 侧。

**在参考实现上测,不在候选上测**:`flop_count`/`byte_count` 是**任务的属性**(要做多少数学、
搬多少数据),不是某个实现的属性。在参考实现上测一次,所有候选共用同一个分母 ——
这样"候选 A 达到 60% 带宽、候选 B 达到 30%"才是可比的。

### 六大类各自怎么判(判据顺序 = 补救顺序,不是量级顺序)

| # | 类别 | 判据(全部无需 counters) | 需要什么输入 | 现在能判吗 |
|---|---|---|---|---|
| 1 | `overhead_floor` | `gpu_ms <= 空发射底噪 × 1.15` | 步骤 1 的底噪 | ❌ 缺底噪 |
| 2 | `launch_bound` | `cpu_issue_ms / gpu_ms >= 1.0` | 已实现 ✅ | ✅ |
| 3 | `memory_bound` | 达到带宽 `>= SATURATED_FRAC ×` 实测 DRAM 天花板 | byte_count + 天花板 | ❌ 两者都缺 |
| 4 | `compute_bound` | 达到算力 `>= COMPUTE_SATURATED_FRAC ×` 实测 fp32 天花板 | flop_count + 天花板 | ❌ 两者都缺 |
| 5 | `resource_limited` | 有 spills,或 regs/shared `>= 80%` 设备上限,**且** 3、4 都不成立 | 已实现 ✅ | ⚠️ 需 3/4 才能排除 |
| 6 | `latency_bound` | 两个百分比都 `< IDLE_FRAC`(**残差类**) | 同 3+4 | ❌ |

**顺序为什么是这个 —— 每一步都有理由**:

1. `overhead_floor` 最先:若 GPU 时间已在空发射底噪上,kernel 本体不再是成本所在,后面所有判据都不该说话。
2. `launch_bound` 在吞吐判据**之前**:否则一个 launch-bound 的 kernel 会报"什么都没饱和",
   把 agent 送去找它并不需要的并行度。
3/4. 饱和判据:对着**实测**天花板,绝不用规格书数字。
5. `resource_limited` 在饱和**之后**:一个已经饱和的 kernel 即使寄存器用得多,也不是寄存器受限,而是已经到头了。
6. `latency_bound` 最后,作为残差。

**当前实际能力:六类里只有 2 类能判**(`launch_bound`、部分 `resource_limited`)。
这就是为什么问题 3 是接线之前的前置条件。

### 用户提出的补充要求

> "需要综合考量当前硬件的资源(比如内存大小, flop, 带宽等等一切和kernel任务有关的硬件资源约束),
> 以及当前任务对不同资源的使用程度"

超出当前六类判据的部分,拟增加(全部框架自测):

| 资源 | 采集方式 | 判据用途 |
|---|---|---|
| VRAM 容量与本任务占用峰值 | `torch.cuda.max_memory_allocated` / `mem_get_info` | 解释 OOM 类失败;判断"能否再放大 tile" —— 已知 24GiB 卡上 34/78 个 level1 任务无法评测 |
| L2 容量 vs 工作集 | `device_properties.l2_cache_size` vs byte_count | 工作集远大于 L2 → 重用改写无效,该判 memory_bound |
| SM 数 vs 网格规模 | `multi_processor_count` vs 实际 grid | 波次量化(wave quantization):grid 略大于 SM 整数倍会浪费半个波次 |
| 每 kernel 时间分解 | `torch.profiler` 已可得(步骤 2 已用) | 多 kernel 候选里**哪个** kernel 是瓶颈 |
| 参考实现结构 | 交给 **agent**:算子序列、可融合点、精度路径 | 硬编码看不出"这两个算子本可融合" |

最后一行是问题 2 的分工落点:**前四行硬编码测,最后一行 agent 分析**,两者一起进 `BottleneckReport`。

---

## 问题 4:步骤 5(手写 CUDA 候选)—— 快则直接做

> "如果需要花费很长时间可以延后, 如果可以快速测验并得出结论可以直接处理"

**评估:约 30–50 分钟 GPU 时间,可以直接做,但必须排在步骤 1 之后**(两者都要独占卡)。

已有的东西让成本很低:
- `linux-server/scripts/probes/measure_external_candidate.py` 已经能编译 `.cu`、做双精度正确性、
  对 fp64 真值比对、并在两种计时口径下计时 —— 外部候选验证就是用它跑的
- box 2 已确认 nvcc + ninja 都在 PATH 上(`load_inline` 的硬前提)

要测的问题很明确:**在我们自己的计时口径下,同一算法的 CUDA 实现比 Triton 快多少**(如果快)。
这直接检验你先前的判断"cuda 表达力更强 → 理论上限应更高"。

风险:一个手写候选测不出"CUDA 后端的上限",只能测出"这一个手写实现的表现"。
结论必须这样陈述,不能外推成后端优劣。

---

## 问题 5:两个清理项 —— 已核实风险

> "如果删除没有风险的话可以删除"

**E1:`D:\ClaudeCode\tmp\*.ps1` 五个脚本 —— 可以删。**

核实结论:
- 无任何计划任务/启动项引用它们(`Get-ScheduledTask` 过滤 powershell 动作,零命中)
- 当前无进程在跑它们(我第一次查询时匹配到 3 个进程,**是我自己的查询命令匹配到了自己的 pattern**,
  已复查确认无真实链式进程)
- 它们不会自启;风险仅在于**被手工执行会重启实验**
- 其中 `run_chain_dry.ps1` 是 dry-run 版本,只打印不执行

**E2:`v2-glm/_proxytest/` —— 可以删,而且删除是安全收益。**

核实结论(纠正我先前的描述):三个 `.log` 各只有 **119 字节**,60MB 全在 `.opencode/node_modules`。
框架代码/配置/测试**均无引用**。

**更重要的发现**:`_proxytest/.opencode/opencode.jsonc` 里有**一个明文 API key**
(`apiKey: "d8b9ea02ca..."`)。所以删除这个目录不只是省 60MB,而是**减少一处明文密钥暴露面**。
这也再次说明交付时轮换密钥是必须的(该密钥现已在两台共享机器上出现)。

---

## 待批准的实施顺序(冒烟结束后统一执行)

| # | 内容 | 依赖 | GPU | 预估 |
|---|---|---|---|---|
| 0 | 删除 E1 五个 .ps1 + E2 `_proxytest/` | 无 | 否 | 分钟级 |
| 1 | 跑 `probe_bottleneck_signals.py`,读 DISCRIMINATION CHECK | 独占 GPU | **是** | 2–4 min(已挂自动) |
| 2 | 空发射底噪测量(补 `overhead_floor` 的缺口) | 独占 GPU | **是** | 数分钟 |
| 3 | `flop_count`/`byte_count` 运行时采集(FlopCounterMode + numel×itemsize),在**参考实现**上测一次 | 无 | 否 | 半天 |
| 4 | 天花板进 doctor:每台机器启动时实测并记录 DRAM/fp32/底噪 | 1 | 是(轻) | 半天 |
| 5 | 按步骤 1 的观测值**替换四个猜测常数**;补充 VRAM/L2/SM 判据 | 1,3,4 | 否 | 半天 |
| 6 | **接线(问题 2 的核心)**:verdict + evidence → analyst prompt + report;prompt 里写明可反驳与盲区 | 3,5 | 否 | 半天 |
| 7 | 步骤 5:手写 CUDA 候选对比 | 1(让开卡) | **是** | 30–50 min |

**不做的事**(沿用既有决定):不改计时方法本身;不做任务分类→后端排序的搜索机制;不启用 CUTLASS/CuTe 生成;
分类结论**永不参与门控**。

---

## 冒烟实验现状(供实施起点参考)

- 2.66h/3h,预期约 22:35 因墙钟结束
- **Loop D 首次真正产出新族**:novelty 种子 `cand-37c8e342` 被接受,族数 2→3,整轮 7.8 分钟零拒绝
  (前 18 个 run 中 Loop D 执行次数为 **0**)
- 第一个族三次 rewriter 尝试烧掉 1.96h 零产出(agent 自跑 GPU sweep);第二个族的同类调用 **109 秒**完成并产出
  —— 时间预算是错误工具的最直接证据
- 判读报告时的注意:若出现 `frozen_converged`,必须交叉检查该族是否有 `AGENT_CALL_FAILED final=true`,
  否则会把"调不通"读成"优化到头了"

---

# 修订版实施方案(2026-09-07 晚,合并调研结论)

本节取代前文第 "待批准的实施顺序" 表。三份依据:
- `docs/research-bottleneck-classification-and-portable-thresholds.md`(阈值可移植性 + KernelPro 调研)
- `docs/research-counter-free-profiling-capability-tiers.md`(counters 不可用的挽回方案,全部 box 2 实测)
- 本文档前半(混合架构分工、flop/byte 可自测性验证)

## 三条被调研改掉的设计

**(1) 门限不再硬编码,改为自校准。** 原方案"用步骤 1 观测值替换四个常数"会把一张 4090
的数字烧进代码。修正为:天花板每机自测 + 门限由四个解析真值已知的标尺负载**自动导出**,
代码里只留无量纲 margin。KernelPro 的门限(DRAM>80%、occupancy<50%、registers>96)
也是硬编码的,所以这是我们相对该领域现状的一个改进点,而非补课。

**(2) 新增双方法交叉验证 + 冲突仲裁。** 借用 KernelPro:解析 roofline 与实测吞吐两条独立判定,
一致则高置信,**冲突时采信解析值**(它反映 kernel 固有上限,而非当前可能未优化的实现)。
对我们更重要,因为我们的实测侧没有 ncu。

**(3) 新增 profiling 能力分层 Tier 0-3。** 关键实测发现:**SASS 静态分析(`nvdisasm`)无需任何
权限即可用**,能挽回张量核检测(`HMMA` 计数)、spill 交叉验证、共享内存流量、屏障密度;
理论 occupancy 可解析计算。唯一真正不可得的是 **stall 原因分解**。

## 需要修正的既有代码陈述

`src/kernel_optimizer/evaluation/bottleneck.py` 的模块 docstring 现在写着
"bank conflicts, warp divergence, instruction-cache pressure, L2 hit rate — 那些需要 counters"。
**这个陈述过于悲观,实测后需要改**:张量核使用**可见**(SASS)、occupancy **可算**、
L2 可用"工作集 vs 容量"解析判断、bank conflict 的**流量**可见(只有冲突倍数不可见)。
唯一该保留为"不可见"的是 stall 分解。

## 实施顺序(修订)

| # | 内容 | 依赖 | GPU | 预估 |
|---|---|---|---|---|
| 0 | 清理 E1 五个 .ps1 + E2 `_proxytest/`(后者含明文 key,删除是安全收益) | 无 | 否 | 分钟 |
| 1 | 跑 `probe_bottleneck_signals.py`,读 DISCRIMINATION CHECK | 独占 GPU | **是** | 2-4 min(已挂自动) |
| 2 | **校准器**:四个标尺负载 + 空发射底噪 → 自动导出门限;结果缓存,可 `--recalibrate` | 1 | **是** | 1 天 |
| 3 | `flop_count`/`byte_count` 运行时采集(在参考实现上测一次;已验证 ratio=1.0000) | 无 | 否 | 半天 |
| 4 | 天花板 + 校准结果 + **tier 探测**进 doctor,写入 run 元数据 | 2 | 是(轻) | 半天 |
| 5 | **Tier 1 采集器**:SASS 指令统计(HMMA/STL/LDL/LDS/STS/BAR.SYNC)+ 解析 occupancy | 无 | 否 | 1 天 |
| 6 | 分类器改造:吃校准门限而非常数;加双方法交叉验证 + 冲突采信解析值;修正 docstring | 2,3,5 | 否 | 半天 |
| 7 | **接线(问题 2 核心)**:verdict + evidence + tier 可用性 → analyst prompt + report,必须是 **detect-analyze-recommend** 结构(KernelPro 消融证明:原始 counter 直接给 LLM 会让性能**变差**),并明确写出"本机不可测"项 | 3,6 | 否 | 半天 |
| 8 | 步骤 5:手写 CUDA 候选对比(排在 1 之后让开卡) | 1 | **是** | 30-50 min |

## 不做(沿用既有决定)

不改计时方法本身;不做任务分类→后端排序的搜索机制;不启用 CUTLASS/CuTe 生成;
分类结论**永不参与门控**;Tier 3(ncu)只留接口,不在租用容器上尝试启用。
