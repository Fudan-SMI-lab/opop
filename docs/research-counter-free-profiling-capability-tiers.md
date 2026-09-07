# Counters 不可用:能挽回多少,值多少钱

状态:**调研已完成(全部在 box 2 实测),实施方案见文末**。回答用户问题 2。

---

## 先确认前提:counters 到底能不能开

重新实测(不是复述先前结论):

```
$ /usr/local/cuda/bin/ncu --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed ... python -c "..."
==PROF== Connected to process 37197
==ERROR== ERR_NVGPUCTRPERM - The user does not have permission to access NVIDIA GPU
          Performance Counters on the target device 0.
==PROF== Disconnected from process 37197
```

`ncu` 装了(`/usr/local/cuda/bin/ncu`、`/opt/nvidia/nsight-compute/2025.1.1/ncu`),但被权限拒。
**能不能自己开?实测不能**:

| 检查 | 结果 | 含义 |
|---|---|---|
| `id -u` | `0`(root) | 权限不是用户级问题 |
| `lsmod \| grep -c nvidia` | `0` | 容器内看不到宿主的 nvidia 内核模块 |
| `/etc/modprobe.d` | 不存在 | 无法写模块参数 |
| `/sys/module/nvidia/parameters/NVreg_RestrictProfilingToAdminUsers` | 不可读 | 参数未暴露给容器 |
| `/proc/driver/nvidia/params` | 不可读 | 同上 |

**结论:在租用容器里,即便是 root 也无法开启 counters** —— 这需要宿主机重载
nvidia 内核模块并带 `NVreg_RestrictProfilingToAdminUsers=0`。
这不是"我们没配好",是租用环境的结构性限制。**`nsys` 也未安装**(找不到二进制),
所以 KernelPro 用 nsys 做的那部分我们也没有现成工具。

**这不是永久判决**:换到自有机器/裸金属(或云厂商愿意改内核参数)时 counters 就能用。
所以设计上必须是"**有则用、无则降级**",不能二选一写死。

---

## 关键发现:没有 counters ≠ 丢失全部信息,SASS 静态分析可挽回一大半

**实测(box 2,`nvdisasm` 在 `/usr/local/cuda/bin/nvdisasm`,无需任何权限)**:

对一个 `tl.dot(..., input_precision='tf32')` 的 Triton kernel 反汇编,指令统计:

```
40  LDS        共享内存读
32  HMMA       张量核指令  <= 直接证明张量核被用上了
16  STS        共享内存写
 8  LDSM       矩阵加载(张量核数据通路)
 5  BAR.SYNC   屏障同步
```

对一个故意造寄存器压力的 kernel:`regs=30, spills=0`,SASS 里无 `STL`/`LDL` —— **判定正确**。

这直接回答"能否解决"的一部分:**KernelPro 的两个 SASS 类工具我们可以完整复现**
(它的 RegisterSpillDetector 数 `STL`/`LDL`,TensorCore 检测数 `HMMA`/`WGMMA`),
因为它们本来就不用 counters。

---

## 逐项评估:每类信息的价值,以及有无替代

按用户列的四项 + 补充项。**"价值"栏是对我们这个框架的实际决策价值**,不是一般性重要程度。

| 信息 | counters 缺失的影响 | 无 counters 的替代方案 | 替代质量 | 价值 |
|---|---|---|---|---|
| **张量核是否被用** | 原本要 `sm__inst_executed_pipe_tensor` | **SASS 数 `HMMA`/`IMMA`/`WGMMA` 指令数**(已实测可行) | **等价** | **极高**。我们有实证:L3:48 拒了 8/8 张量核候选、2.09ms 的成绩完全没用上张量核;fp32 IEEE 路径是主要性能天花板 |
| **寄存器 spill** | 原本要 `local` 内存 counter | **SASS 数 `STL`/`LDL`**;且 Triton 直接给 `n_spills`(已在用) | **等价** | **高**。spill 是我们已在判的 `resource_limited` 主判据 |
| **共享内存 bank conflict** | 看不见冲突**次数** | **SASS 数 `LDS`/`STS` 指令数**可知共享内存**流量**,但 N-way 冲突倍数不可知;可用**参数扫描间接暴露**(改 padding/swizzle 若显著变快 → 原本有冲突) | **部分**(流量可见,冲突倍数不可见) | **中**。它是"改写方向"而非"是否该改写";TPE 的参数扫描本身就在间接优化它 |
| **warp 分歧** | 看不见分支效率 | **SASS 数分支/predication 指令**(`@!P` 谓词、`BSSY`/`BSYNC`);源码层 agent 可读出数据相关分支 | **部分** | **中低**。我们的任务(matmul/conv/pool/attention)分歧本就少;agent 从源码看分支比 counter 更直接可行动 |
| **L2 命中率** | 看不见命中率 | **工作集 vs `l2_cache_size`**(`device_properties` 可读)做解析判断:工作集 >> L2 → 必然低命中 | **弱但可用** | **中**。决定"重用类改写是否有意义";我们的 byte_count 已能算工作集 |
| **stall 原因分解** | 完全看不见(mem_dep / short_scoreboard / barrier …) | **无直接替代**。只能由 `latency_bound` 残差类 + agent 从代码结构推断 | **无** | **中高但不可得**。KernelPro 的 WarpStall 工具(dominant stall > 40%)我们无法复现 |
| **occupancy** | 原本 `sm__warps_active` | **解析计算**:regs/thread、shared/block、block size 已知 + 设备上限已知 → **理论 occupancy 可算**(与 CUDA Occupancy Calculator 同法) | **近似**(理论 vs 实测) | **高**。KernelPro 的关键案例就是 occupancy 诊断(18.8% → 换小 footprint 配置)驱动的 |
| **per-kernel 时间/发射数** | — | `torch.profiler`(CUPTI **tracing**,不需 counters)。实测可得 per-kernel `self_device_time_total`、`count`、`device_memory_usage`、`flops` 字段 | **等价** | **高**。多 kernel 候选里定位瓶颈 kernel |
| **达到带宽/算力占峰值** | 原本 SOL 百分比 | **byte_count/flop_count ÷ 实测天花板**(我们的方案) | **等价甚至更好**(实测天花板 vs 规格书) | **极高**。roofline 两轴 |

### 汇总:六类判据的可达性(修正先前的说法)

我先前在 `bottleneck.py` 里写"bank conflict、warp 分歧、L2 命中率不可见,会被误归入
`resource_limited`/`latency_bound`"。**这个陈述过于悲观,需要修正**:

| 原判断 | 修正后 |
|---|---|
| 张量核使用不可见 | **可见**(SASS `HMMA` 计数,已实测) |
| bank conflict 完全不可见 | 流量可见(`LDS`/`STS`),冲突**倍数**不可见 |
| warp 分歧完全不可见 | 分支/谓词指令可见,**分歧率**不可见 |
| L2 完全不可见 | 命中率不可见,但**工作集 vs L2 容量**可解析判断 |
| occupancy 不可见 | **理论 occupancy 可解析计算** |
| stall 原因不可见 | **确实不可见,无替代** ← 唯一真正的硬缺口 |

**唯一真正丢失的是 stall 原因分解**。其余都有等价或部分替代。

### 这些信息值多少 —— 用我们自己的数据回答

不是抽象判断,而是查我们 19 个 run 的实证:

1. **张量核(极高价值,现已可得)**:L3:48 拒了 **8/8** 张量核候选、收了 7/7 标量候选,
   最终 2.09ms 完全没用上张量核。如果当时能看见"这个候选真的发出了 HMMA 指令",
   `resource_limited` 与"精度路径错误"就能区分开。**这一项的价值高于其余所有项之和。**
2. **occupancy(高价值,可解析算)**:我们的 `at_boundary` 机制已经在猜"参数想再往前但被顶住",
   理论 occupancy 能把"被什么顶住"从猜变成算。
3. **stall 原因(不可得)**:代价是 `latency_bound` 只能是残差类。
   已在文档里如实标注"这意味着'我们能测的都没饱和',不是'已证明是延迟'"。
4. **bank conflict / warp 分歧(中低)**:对我们**当前**的任务集(pool/matmul/conv/attention)
   不是主要限制;而且 TPE 的参数扫描(padding/swizzle/block 形状)本来就在间接优化它们。

---

## 实施方案(并入主计划)

### 新增:profiling 能力分层,自动探测、自动降级

```
Tier 0(总是可用,无任何权限要求)
  ├─ CUDA events                 → gpu_ms
  ├─ CPU 计时循环                 → cpu_issue_ms / wall_ms      [已实现 47504ba]
  ├─ FlopCounterMode + numel     → flop_count / byte_count      [待实施]
  ├─ device_properties           → L2 容量 / SM 数 / VRAM / 寄存器上限
  ├─ Triton metadata             → n_regs / n_spills / shared    [已实现]
  └─ cuobjdump -res-usage        → 同上,CUDA/CUTLASS/CuTe        [已实现 6efa850]

Tier 1(需要 nvdisasm,实测 box 2 可用,无权限要求)         ← 本次新增
  ├─ SASS: HMMA/IMMA/WGMMA 计数  → 张量核是否真的被用
  ├─ SASS: STL/LDL 计数          → spill 交叉验证
  ├─ SASS: LDS/STS 计数          → 共享内存流量
  ├─ SASS: BAR.SYNC 计数         → 屏障密度
  └─ 解析 occupancy               → regs/shared/blocksize + 设备上限

Tier 2(需要 CUPTI tracing,实测可用)
  └─ torch.profiler              → per-kernel 时间 / 发射数      [部分已实现]

Tier 3(需要 counters,box 2 不可用,自有机器可用)
  ├─ ncu SOL 百分比               → 交叉验证 Tier 0 的 roofline
  ├─ ncu stall 分解               → 唯一无替代的一项
  └─ ncu bank conflict / L2 命中率 → 冲突倍数、真实命中率
```

**设计原则**:doctor 启动时探测每个 tier 是否可用并记录进 run 元数据;
分类器与 analyst prompt 按**实际可用的 tier** 组织,缺失项**明确写成"本机不可测"**
而不是静默省略 —— 否则 agent 会把"没有这项证据"读成"这项没问题"。

### 与 KernelPro 的对照(用于论文)

| 能力 | KernelPro | 我们 | 说明 |
|---|---|---|---|
| 张量核检测 | SASS | **SASS(同)** | 等价 |
| spill 检测 | SASS `STL`/`LDL` | **SASS + Triton n_spills(双源)** | 我们多一个交叉验证源 |
| roofline 天花板 | **规格书** | **实测微基准** | 我们更保守正确(容器限频) |
| 分类门限 | **硬编码常数**(A100/H100 上定) | **自校准导出** | 我们可移植 |
| stall 分解 | ncu | **不可得** | 他们强 |
| bank conflict 倍数 | ncu | 流量可见、倍数不可得 | 他们强 |
| occupancy | ncu 实测 | **解析计算** | 近似 |
| launch overhead | nsys(> 20%) | **cpu_issue_ms/gpu_ms** | 等价,且不需 nsys |

**诚实结论**:在**有 counters 的机器上** KernelPro 的 profiling 深度高于我们;
在**租用容器(我们的实验环境,也是多数人的环境)上他们的方案跑不起来**,而我们的能跑。
Tier 3 留好接口,自有机器上可补齐。

---

## 落地前的字段核实(实测,修正本文两处错误)

写方案时我按印象写了属性名和依赖,核实后有两处要改。

### 错误 1:属性名是 `L2_cache_size`,不是 `l2_cache_size`

`torch 2.13.0+cu129` 上 `device_properties` 的**全部**可读属性:

```
L2_cache_size, clock_rate, gcnArchName, is_integrated, is_multi_gpu_board, major,
max_threads_per_block, max_threads_per_multi_processor, memory_bus_width,
memory_clock_rate, minor, multi_processor_count, name, pci_bus_id, pci_device_id,
pci_domain_id, regs_per_multiprocessor, shared_memory_per_block,
shared_memory_per_block_optin, shared_memory_per_multiprocessor, total_memory, uuid,
warp_size
```

我先前在计划里写的 `device_properties.l2_cache_size`(小写 l2)**取不到**,会静默返回缺失 ——
正是"没有这项证据被读成这项没问题"的那类 bug。实测值:RTX 4090 **L2 = 75.5 MB**。

同时确认 `regs_per_block` **不存在**(只有 `regs_per_multiprocessor`),
所以 occupancy 的寄存器约束必须按 per-SM 寄存器文件算,不能按 per-block。

### 错误 2:"理论 occupancy 可解析计算"—— 结论成立,但要说清用哪些字段

已实测跑通,三个约束取最小,并且**能报出 limiter 是谁**:

```
BLOCK= 256 nw=4 regs=26 shmem=0 -> 理论 occupancy 100.0%  (limiter: warps/threads)
BLOCK=1024 nw=8 regs=39 shmem=0 -> 理论 occupancy 100.0%  (limiter: warps/threads)
```

公式(全部输入我们**已经在采集**):

```
threads   = num_warps × warp_size
by_warp   = max_threads_per_multi_processor // threads
by_reg    = regs_per_multiprocessor // (n_regs × threads)
by_shmem  = shared_memory_per_multiprocessor // shared_bytes
blocks    = min(by_warp, by_reg, by_shmem)
occupancy = blocks × threads / max_threads_per_multi_processor
limiter   = 三者中取到最小值的那一个        ← 这是给 agent 的可行动信息
```

`limiter` 字段的价值在于:它把我们现有的 `at_boundary`(参数想再往前但被顶住)从**猜**变成**算** ——
直接说出"被寄存器顶住"还是"被共享内存顶住",这正是 KernelPro 那个关键案例
(occupancy 18.8%,limited by 168 registers/thread → 换小 footprint 配置)所依赖的信息。

### 顺带确认:规格带宽可推导,但仍只作交叉核对

`2 × memory_clock_rate × memory_bus_width / 8` 得 RTX 4090 = **1.008 TB/s**,与官方规格一致。
**判据仍用实测值**(容器限频),这个推导值只用于"实测值是否明显低于规格"的健康检查 ——
若实测只有规格的 50%,说明卡被限频或被抢占,校准结果不可信,应重跑。
这给了校准器一个廉价的自检:**规格值是我们唯一不需要 GPU 就能算出的参照**。
