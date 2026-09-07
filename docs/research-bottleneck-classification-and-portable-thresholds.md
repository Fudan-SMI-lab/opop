# 调研:领域内如何做 kernel 瓶颈分类 + 阈值可移植性设计修正

状态:**调研与设计,尚未实施**。回答用户的两个追问。

---

## 问题 1:阈值不能定死在一张卡上 —— 这是我计划里的真实设计缺陷

### 用户的批评

> "如果测量阈值是为了决定分类边界, 不应该仅基于当前该卡的环境(比如说就不应该依靠硬编码定死的常数来决定,
> 因为后续会在不同的设备上和环境上进行测试, 不能每到一个新环境都要手动的重写测量和计算一遍)"

**批评成立。** 我原计划的第 5 步写的是"按步骤 1 的观测值**替换**四个猜测常数",这等于把
一张 4090 上的观测值烧死进代码 —— 换到 H100/A100/5080 上要么不准,要么得人工重测重写。
这跟我自己反复主张的"必须实测、不用规格书数字"是同一类错误的镜像:我把"实测"做成了
"在一台机器上实测一次然后当常数"。

### 修正后的设计:两层分离

关键区分:**天花板是设备属性(必须每机自测),判据形状是物理属性(跨设备不变)**。

| 层 | 内容 | 跨设备行为 |
|---|---|---|
| **设备天花板** | DRAM 带宽、fp32/tf32 算力、空发射底噪、L2 容量、SM 数、VRAM | **每台机器自动实测一次**,写进 run 元数据 |
| **判据形状** | "达到带宽占天花板的比例 ≥ X" 这个**形式** | 跨设备不变 |
| **X 的取值** | 由**自校准**得出,不写死 | 每台机器自动导出 |

### X 怎么来:自校准(calibration),不是硬编码

思路是用**四个解析真值已知的负载**当"标尺",在每台新机器上自动跑一次,让机器自己回答
"在**这台**卡上,一个**已知就是** compute_bound 的 kernel 能达到峰值的百分之几":

```
校准(每台机器自动执行一次,约 2-4 分钟,结果缓存进 doctor 记录):
  COMPUTE 4096³ matmul  → 实测达到 fp32 峰值的 p_c%     (已知 = compute_bound)
  MEMORY  512MB copy    → 实测达到 DRAM 峰值的 p_m%     (已知 = memory_bound)
  LAUNCH  40 tiny ops   → 实测 cpu_issue/gpu 比 = r_l    (已知 = launch_bound)
  EMPTY   空 kernel     → 实测发射底噪 t_0

导出门限(不是我猜的常数):
  COMPUTE_SATURATED_FRAC = p_c × margin      # margin 取 0.8 之类的安全系数
  SATURATED_FRAC         = p_m × margin
  LAUNCH_BOUND_CPU_RATIO = min(1.0, r_l × margin)
  overhead_floor 门限     = t_0 × 1.15
```

这样"45.8% 的 matmul 被 50% 门限误判"这个问题**自动消失**:门限不再是我猜的 50%,
而是"这台卡上一个真 compute_bound 负载实测能到多少"再留余量。换到 H100 上,`p_c` 自己变。

**代码里保留的常数只有 margin(安全系数)和 1.15 这类形状参数**,它们是无量纲的、
不随设备变化;所有带物理量纲的数字都是实测来的。

**失败时的行为**:校准跑不起来(无 GPU、被抢占、驱动异常)时,`DevicePeaks = None`,
分类器只输出它在无天花板情况下仍能判的类(`launch_bound` / `overhead_floor` /
`resource_limited`),其余返回 `unknown` —— **不用近似值凑**,与我们既有的
"测不到的界限不该拿近似值凑"原则一致(见 `_memory_pressure` 的同款处理)。

---

## 问题 2:领域调研 —— 别人怎么做

### 主要参考:arXiv:2606.26453(用户给的那篇)

**"Optimizing CUDA like a Human: Micro-Profiling Tools as Expert Surrogates for LLM-Based
GPU Kernel Optimization"**(KernelPro,Jiading Gai 等,Amazon 系作者,CC BY 4.0)。
我把 PDF 抽文本读了方法部分,以下都是文中原话的转述。

#### 它的瓶颈分类

**四类**:`compute-bound` / `memory-bound` / `latency-bound` / `mixed`,
由 Stage-1 的 **BenchmarkingAgent** 在优化开始前对**参考实现**跑一次(one-time)。

**这一点和我们的设计一致**:也是在参考实现上定类型、供下游用,不是每个候选重算。

#### 它怎么判:两条独立方法 + 冲突时的仲裁规则(重要)

> "Stage 1 classifies the bottleneck using two independent methods for robustness: a theoretical
> roofline bound (arithmetic intensity vs. the hardware ridge point) and the measured
> compute-vs-memory throughput from ncu. Agreement between the two gives high confidence; on
> disagreement, KernelPro defers to the theoretical bound, since it reflects the kernel's
> **inherent ceiling** rather than the current (possibly unoptimized) implementation."

- 方法 A:解析 roofline —— arithmetic intensity vs. 硬件 ridge point
- 方法 B:实测吞吐 —— 来自 ncu 的 compute vs. memory throughput
- **两者一致 → 高置信;不一致 → 采信解析值**,理由是解析值反映 kernel 的**固有上限**,
  而实测值反映的是"当前这个(可能没优化好的)实现"

这条仲裁规则值得我们直接借用,而且它对我们尤其重要:我们**没有 ncu**,
方法 B 只能用 `torch.profiler` + 实测天花板的弱版本,所以"冲突时采信解析值"更该是默认。

**arithmetic intensity 来源**:它**解析参考代码**识别数学算子(GEMM、Softmax、convolution 等)
再算 CI。我们的方案(`FlopCounterMode` + `numel×element_size` 运行时实测)**比它更稳** ——
不依赖源码解析的正确性,已验证在 matmul/conv/attention/depthwise 上 ratio = 1.0000。

**硬件峰值来源**:文中写的是 "from compute intensity and **hardware specs**" ——
即**规格书数字**,不是实测。这正是我们要偏离它的地方:容器限频会让规格书值偏高,
而我们的 `probe_bottleneck_signals.py` 实测天花板。**在这一点上我们的做法更保守也更正确。**

#### 它的具体阈值(表格原文,我们可以对照)

| 工具 | 信号源 | 类别 | 触发条件 |
|---|---|---|---|
| MemoryCoalescing | ncu | MEMORY | L1 hit < 30%, stalls > 40% |
| HighDRAMThroughput | ncu | MEMORY | **DRAM throughput > 80%** |
| OccupancyLimiter | ncu | OCCUPANCY | **Occupancy < 50%** |
| HighRegisterUsage | ncu | OCCUPANCY | **Registers > 96** |
| WarpStall | ncu | LATENCY | Dominant stall > 40%(60% 升级为 critical) |
| BarrierStall | ncu | LATENCY | Barrier stall > 30% |
| **LaunchOverhead** | **nsys** | **LAUNCH** | **Launch overhead > 20%** |
| Uncoalesced | ncu | — | sectors-per-request > 1.2× ideal |
| RegisterSpill(SASS) | SASS | LATENCY | STL+LDL > 0 **且**满足三种情况之一 |

RegisterSpill 的三种情况写得很细,值得注意它的**复合判据**形式:

> "Case A (Latency-bound): long-scoreboard stall > 15% **and** compute throughput < 50%
> **and** memory throughput < 50% (spills add latency while neither roofline is saturated).
> Case B (Memory-bound by spills): memory throughput > 60% and local-memory …"

**对我们的三点启示**:

1. **它也承认"launch overhead"是独立一类**(LAUNCH_OVERHEAD),用 nsys 测,门限 20%。
   这印证了我们把计时盲区单独立类是对的方向 —— 而我们用 `cpu_issue_ms/gpu_ms` 达到同样目的,
   **不需要 nsys**。
2. **它的门限也是硬编码的常数**(80%/50%/96/40%/30%/20%),而且是在 A100/H100 上定的。
   所以用户的批评不只对我们成立 —— **这是该领域现状的普遍弱点**,我们的自校准设计
   在这一点上比这篇更强。可以作为论文的一个小贡献点。
3. **`Registers > 96`** 这类绝对阈值明显不可移植(不同架构每 SM 寄存器文件大小不同),
   我们现有的做法(regs 相对于 `MAX_REGS_PER_THREAD` 的百分比)本来就更好,保留。

#### 它最有价值的发现(与我们问题 2 的"硬编码+agent 结合"直接相关)

> "Raw metrics **actively degrade** performance because the LLM lacks the domain context to
> translate hardware counters into optimization actions; only the micro-profiling tools'
> detect-analyze-recommend pattern bridges this gap reliably."

**把原始 counter 直接塞给 LLM 会让性能变差**,不是变好。只有"检测→分析→建议"的结构化翻译才可靠。
文中把这一步称为 "semantic feedback operator",作用是把
"47% memory dependency stalls"(一个数)翻译成
"add shared memory tiling with `__syncthreads()` between load and compute phases"(一个动作)。

**这正是用户在问题 2 里要求的分工**,而且有消融实验支持:
它还报告 "Profiling prevents correct-but-slow plateaus" —— 有 profiling 反馈时
0/62 条轨迹卡在 1.0×,无反馈时 10% 卡住。

**结论:我们的架构方向与该领域最新工作一致,且在两点上更强**
(实测天花板 vs 规格书;自校准门限 vs 硬编码门限),在一点上更弱
(无 counters,看不见 bank conflict / warp 分歧 / L2 命中率 / stall 原因 —— 已在
`bottleneck.py` 文档里如实声明)。

### 相邻工作(从该文引用列表提取,尚未逐篇细读)

同一方向的 LLM×kernel 工作,按文中描述的定位:

| 系统 | 引用键 | 文中定位 |
|---|---|---|
| **cuPilot** | chen2025cupilot | "adds roofline-based bottleneck classification but **stops at a label** (memory-bound vs. compute-bound) without generating actionable directives from the underlying counters" |
| KernelBench | ouyang2024kernelbench | 我们用的基准 |
| KernelBlaster | dong2026kernelblaster | 同类 LLM 优化系统 |
| StitchCUDA | li2026stitchcuda | 同上 |
| CUDA-Forge | zhang2025cudaforge | 同上 |
| CUDA-Master / CudaAgent | zhang2026cudamaster / zhang2026cudaagent | 同上 |
| **KernelFoundry** | wiedemann2026kernelfoundry | 我们已借鉴(见 memory) |
| GPU Kernel Scientist | lange2025gpukernelscientist | 同上 |
| AVO | chen2026avo | 同上 |
| KernelEvolve (Meta) | nvidia2025kernelevolvemeta | 同上 |
| GPA | zhou2021gpa | 非 LLM 的性能归因工作 |

**cuPilot 的定位对我们特别有用**:它做了 roofline 分类但"止于一个标签"。
我们的设计(verdict + **完整 evidence** + 可被 agent 反驳 + `suggests` 动作建议)
正是针对这个不足 —— 而 KernelPro 的消融实验证明了"标签/原始数字不够,必须给动作"。

### NVIDIA 官方(Nsight Compute Profiling Guide)

查证结论:官方文档给的是**概念框架**(Speed-of-Light 百分比、Memory Workload Analysis 的
三种受限机制 Mem Busy / Max Bandwidth / Mem Pipes Busy、Warp State Statistics),
**没有给出固定的数值分类门限**。它只说"取 compute 与 memory 子指标里最高的那个"来指示受限资源,
并提醒 "only focus on stall reasons if the schedulers fail to issue every cycle"。

**这条提醒对应我们判据顺序里的第 2 条**(launch_bound 要排在吞吐判据之前),
即"在调度器还没喂满的时候,不要去看 stall 原因" —— 我们的顺序设计与之一致。

---

## 对实施计划的修改(相对 `plan-hybrid-bottleneck-analysis.md`)

| 原步骤 | 修改后 |
|---|---|
| 5. 按步骤 1 观测值**替换四个猜测常数** | 5. **实现自校准**:四个标尺负载自动跑 → 自动导出门限;代码里只留无量纲 margin |
| 4. 天花板进 doctor | 4. 天花板 **+ 校准结果**进 doctor,**每台机器自动、缓存、可 `--recalibrate` 强制重跑** |
| (无) | 新增:**双方法交叉验证 + 冲突采信解析值**(借用 KernelPro 的仲裁规则) |
| 6. 接线到 analyst | 6. 接线时**必须是 detect-analyze-recommend 结构**,不能只给标签或原始数字(KernelPro 消融证明后者会让性能变差) |

其余不变。仍待用户批准后实施。
