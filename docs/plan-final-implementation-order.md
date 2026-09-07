# 实施方案与顺序(最终版,待批准执行)

状态:**已记录,尚未实施**。本文件是**唯一权威的实施顺序**,取代
`plan-hybrid-bottleneck-analysis.md` 里的两份旧表(那两份的编号已过时:旧表第 8 项标题写着
"步骤 5",沿用的是用户早期提问时的编号,容易混淆。本文件统一用 0-9)。

依据文档:
- `plan-hybrid-bottleneck-analysis.md` —— 混合架构分工、flop/byte 自测性验证
- `research-bottleneck-classification-and-portable-thresholds.md` —— 阈值可移植性、KernelPro 分类法
- `research-counter-free-profiling-capability-tiers.md` —— Tier 0-3、counters 缺失的挽回
- `research-kernelpro-borrowable-details.md` —— **每工具命中率、原始 counter 有害的统计证据**

---

## 步骤 1 已完成:实测结果与它推翻的两个门限

RTX 4090 / box 2,`bottleneck_signals.json` 已存盘:

| 项 | 实测 | 自检 |
|---|---|---|
| DRAM 天花板 | **0.910 TB/s** | 规格 1.008 → **90.3% (OK)**,未限频 |
| fp32 天花板(关 tf32) | **54.60 TFLOP/s** | — |
| roofline 拐点 | **60.0 FLOP/byte** | — |
| L2 / clock | 75.5 MB / 2.520 GHz | — |

**DISCRIMINATION CHECK:四个信号全部 DISCRIMINATES**,无一需剔除:

| 信号 | 四负载取值 | 跨度 |
|---|---|---|
| arithmetic intensity | 682.67 / 0.12 / 0.25 / 34.13 | 5461x |
| % of DRAM peak | 8.57 / **99.47** / 2.86 / 3.74 | 34.7x |
| % of fp32 peak | **97.54** / 0.21 / 0.01 / 2.13 | 8170x |
| CPU-issue / GPU | 0.00 / 0.01 / **0.97** / 0.71 | 195x |

**两个我猜的门限被实测推翻**:

| 常数 | 我猜的 | 标尺实测 | 应改为 |
|---|---|---|---|
| `COMPUTE_SATURATED_FRAC` | 0.50 | matmul 达 **97.5%** | 0.75-0.85 |
| `LAUNCH_BOUND_CPU_RATIO` | 1.0 | 无争议 launch 负载仅 **0.97** | 0.5-0.6 |

第二条是**判反**而非偏差:按 1.0 门限,一个故意造的、无可争议的 launch-bound 负载会被判成
`latency_bound`。这正是自校准存在的理由。

**同时更正我先前的说法**:我曾说"45.8% 的 matmul 会被 50% 门限误判"。那个 45.8% 是我用
**假想的** 30 TFLOPS 天花板算的;真天花板 54.6 TFLOPS 下 matmul 落在 97.5%。
**误判风险的方向和我说的相反**(门限太低,不是太高)。

**新发现:MIXED(level2:37)落在内存侧但两个都没饱和。** AI = 34.13 < 拐点 60.0(内存侧),
但达到带宽仅 **3.7%**、达到算力仅 **2.1%**,CPU/GPU = **0.71**。
真实画像是 **launch/latency 受限**。说明六类里的残差类对真实任务是**常态而非边缘**。

---

## 十个步骤,各是什么

| # | 名称 | 内容 | GPU | 预估 |
|---|---|---|---|---|
| 0 | 清理遗留 | 删 5 个 `.ps1` 链式脚本 + `v2-glm/_proxytest/`(**含明文 API key**,删除是安全收益)。已核实:无计划任务、无进程、框架无引用 | 否 | 分钟 |
| 1 | ✅ **已完成** 信号测量 | 见上 | 是 | 已用 |
| 2 | **校准器** | 四标尺负载 + **空发射底噪** → 自动导出门限;代码只留无量纲 margin;缓存 + `--recalibrate`;实测低于规格 60% 标 SUSPECT。**核实后补记:`bottleneck_signals.json` 里没有底噪,`overhead_floor` 仍无输入,必须在此补测** | **是** | 1 天 |
| 3 | flop/byte 采集 | `FlopCounterMode` + `numel×element_size`,在**参考实现**上测一次(任务属性,非实现属性,各候选共用分母)。已验证 ratio=1.0000 且泛化性已测 | 否 | 半天 |
| 4 | doctor 集成 | 天花板 + 校准结果 + **tier 探测**写入 run 元数据。这是"换机器不用人工"的落点 | 轻 | 半天 |
| 5 | **Tier 1 采集器** | SASS 指令统计(`HMMA`/`STL`/`LDL`/`LDS`/`STS`/`BAR.SYNC`)+ 解析 occupancy(含 **limiter** 字段)。`nvdisasm` 无需权限,已实测可行 | 否 | 1 天 |
| 6 | 分类器改造 | 吃校准门限而非常数;双方法交叉验证 + **冲突采信解析值**;修正 `bottleneck.py` 过于悲观的 docstring | 否 | 半天 |
| 7 | **接线** | verdict + evidence + tier 可用性 → analyst prompt + report。必须 **detect-analyze-recommend**;新增 **tool-affinity 过滤**;缺失项明确写"本机不可测" | 否 | 半天 |
| 8 | CUDA 候选对比 | 同一算法的 CUDA 实现在我们计时口径下比 Triton 快多少。结论不可外推成"后端优劣" | **是** | 30-50 min |
| 9 | **产物抢救 + 真伪收敛** | ① transport 失败后先检查沙箱是否已有通过 `check_output`+`_triton_lint_check` 的产物(落点:`base.py:158` 的 `break` 之前);② **区分"改写评测后无改进"与"改写从未评测"**,后者不得记为 `converged` | 否 | 半天 |

---

## 推荐实施顺序(最终)

| 次序 | 步骤 | 为什么在这个位置 |
|---|---|---|
| **1** | **9. 产物抢救 + 真伪收敛** | **最前**。它是当前唯一在**污染实验结论**的缺陷:L1:42 两个 `frozen_converged` 里一真一假,7152 字节已写完的改写被丢弃。无 GPU、零依赖、半天 |
| **2** | **8. CUDA 对比** | **趁卡空着**。需 GPU 的只有 2/4/8;后面 5 个步骤全程不碰卡 |
| **3** | 2. 校准器(含补测底噪) | 需 GPU,与 8 挨着做,一次性用完卡 |
| **4** | 0. 清理 | 顺手,含密钥暴露面 |
| **5** | 3. flop/byte | 无 GPU;6 的依赖 |
| **6** | 5. Tier 1 采集器 | 无 GPU;6 的依赖。**按命中率排序:先 spill(18.2%)+ occupancy(13.3%,增益最高),后张量核** |
| **7** | 6. 分类器改造 | 依赖 2+3+5 |
| **8** | 4. doctor 集成 | 依赖 2;放 6 后可一并接入 tier 状态 |
| **9** | 7. 接线 | **最后**。唯一消费全部上游的一步,提前做会返工 |

**三处调整的理由**:
1. **9 提到最前** —— 不是新功能,是修正一个已在产生错误结论的缺陷。不修则后续 L3 的收敛判定不可信。
2. **8 提前** —— 把两个需 GPU 的步骤挨着做,避免二次等卡。
3. **7 留最后** —— 消费全部上游产出。

---

## KernelPro 第二轮细读带来的方案修改

详见 `research-kernelpro-borrowable-details.md`,对本方案的影响:

**(1) 步骤 5 的优先级被独立验证。** 它在 42 个任务上的每工具命中率:
`RegisterSpill` **18.2%**(最高)、`OccupancyLimiter` **13.3%**(次高,**平均增益最高 1.48x**)、
`LaunchOverhead` 8.5%。前两项正是步骤 5 的内容,第三项已实现。
而 `MemoryCoalescing` 5.9% / `Vectorization` 4.0% **命中率最低** —— 我们本来难做,
按它的数据也**最不值得追**,可安心不做。

**(2) 步骤 7 的翻译层是必需,不是优化项。** 成对 Wilcoxon:
**不给任何反馈显著优于给原始 ncu 指标(p=0.0007)**。即原始 counter 塞进 prompt
**比什么都不给更差**。微剖工具比原始指标高 125% 加速。
所以 `evidence` 里的原始数字**不能是唯一输出**,每个数字都要配"因此该做什么"。

**(3) 步骤 7 新增 tool-affinity 过滤。** 按分类结果筛选放进 prompt 的证据
(memory-bound 跳过张量核检查等),防信号稀释 —— 而非一律全放。

**(4) 步骤 9 加强。** KernelPro 的 "profiling prevents correct-but-slow plateaus"
(0/62 卡在 1.0x,NoFeedback 10% 卡住)与我们的假 `converged` 是同一问题的两面。
所以步骤 9 不只抢救产物,**还要在产物真的没写出来时也不记为"无改进"**。

**(5) 诚实记录一项我低估的代价。** `WarpStall` 命中率 **11.0%(第三高)、平均增益 1.41x**,
需要 ncu stall 分解,**我们无法复现**。我先前把它评为"中低价值",**偏低了**。
(需区分:我评的是 warp divergence,它的 `WarpStall` 是 stall 原因分解,范围更广。)

**(6) 步骤 8 的参考量级。** 文中报告 raw-CUDA+CuTe 比**人工调优的 Triton** 快 **1.23x**
(VeOmni MoE 训练 kernel)。这给我们的 CUDA 对比一个预期量级。

**(7) 独立支持"不做早停剪枝"。** MCTS vs 贪心:几何均值 3.65x → **4.60x**、
中位 4.17x → **6.09x**、p=0.004;NetVLAD 任务 **43 次连续失败**才出第一个可用解。
我们不改成 MCTS(超出本轮范围),但这是 `frozen_converged` 过早触发风险的外部佐证。

---

## 不做(沿用既有决定)

不改计时方法本身;不做任务分类→后端排序的搜索机制;不启用 CUTLASS/CuTe 生成;
分类结论**永不参与门控**;Tier 3(ncu)只留接口,不在租用容器上尝试启用;
不做 search memory(接近用户已否决的跨候选共享;且其消融结论是"统计等价",代价低)。


---

# 实施完成记录(2026-09-08)

九项全部完成,推荐顺序 **9 → 8 → 2 → 0 → 3 → 5 → 6 → 4 → 7** 依次执行。测试从 276 增到 321。
所有测量都在 box 2 (RTX 4090, sm_89) 上真机跑过,关键数字全部落盘留档。

| 步骤 | commit | 结论 |
|---|---|---|
| 9 抢救产物 + 真假收敛 | `8ea5858` | 传输失败后已写完的产物被救回;未评测的轮次不再伪造 `converged` |
| 8 手写 CUDA vs Triton | `0d418c1` | **0.874x —— 手写 CUDA 慢 13%**;给"教 agent 写 CUDA"定价,不是授权 |
| 2 自校准 | `0d418c1` | DRAM 0.9102 TB/s、fp32 54.94、tf32 88.88、launch floor 17.4us;两个常数被实测推翻 |
| 0 清理 | `2e530ba` | 5 个 .ps1 + `_proxytest/`(含明文 key)已删 |
| 3 任务代价 | `2e530ba` | 三处解析式交叉验证 ratio 1.0000;**L3:43 融合头寸 69.09x** |
| 5 Tier 1 静态分析 | `2f3fc9f` | SASS 指令混合 + 解析式 occupancy;实测 winner 33.3% occupancy / 零向量化 |
| 6 分类器重做 | `0cf471c` | 四个基准工况全部复原自身 ground truth;推翻的常数会让它们错判 |
| 4 doctor 集成 | `03da61c` | tier 可用性随校准落盘;ncu 存在 ≠ counters 可用 |
| 7 接线 | `0cf471c` | detect→analyze→recommend;tool-affinity 过滤;分类器不再是死代码 |

## 实测推翻的三个我自己的判断

1. **`COMPUTE_SATURATED_FRAC = 0.50`** —— 真正 compute-bound 的 matmul 达到 fp32 天花板的
   **94.9%**。0.50 的线会把还有 2x 余量的 kernel 判成"已饱和、别动了"。派生值 0.807。
2. **`LAUNCH_BOUND_CPU_RATIO = 1.0`** —— 无可争议的 launch-bound 工况实测 cpu/gpu = **0.963**,
   **在线以下**,所以这个判据方向是反的、永远不会触发。派生值 0.866。
3. **"没有 counters 就看不到张量核/occupancy/spill"** —— 步骤 5 用实测推翻:nvdisasm 反汇编
   cubin 就能拿到 HMMA/STL/LDL/LDS/STS,occupancy 可以纯解析算出来还能给出 limiter。
   真正看不到的只剩:bank conflict、warp divergence、i-cache、L1/L2 命中率、stall 分解、
   **achieved**(而非理论)occupancy。

## 过程中我自己犯并修正的错(留档,因为它们都是"测量自己坏掉"的形态)

- **第一版手写 CUDA kernel 是错的**(max rel err 1.26:8 个 warp 映射到 2x2 warp-tile 网格,
  warp 4-7 写到 tile 外),而脚本照样打印了它的延迟并算出 0.276x —— 违反了脚本自己写下的规则。
  现在正确性是硬门,错的配置进不了对比。
- **第一版只给 Triton 扫配置、给 CUDA 一个固定配置**。方向恰好和我可能希望的结论相反,
  但仍然是被制造出来的结论。现在两边都扫。
- **我说 nvdisasm"不需要权限"** —— 对,但无关:box 2 上它根本不在 login shell 的 PATH 上
  (在 `/usr/local/cuda/bin/`)。只查 PATH 会把一个能用的工具报成缺失,和 opencode 的 PATH bug
  同一类。现在 `find_cuda_tool` 查 PATH + CUDA_HOME + 版本化目录。
- **两个我专门写来 spill 的 kernel 都没 spill**,而 Triton 自己的 `n_spills` 两次都和 SASS
  一致报 0 —— 所以是探测器对、我的预期错。由此得到的发现:**Triton 的寄存器分配器宁愿把寄存器
  顶到上限、丢掉 occupancy,也不 spill**(128x128 fp32 累加器 → 218 regs / 0 spills /
  16.7% occupancy)。所以在 Triton 上 occupancy 才是会触发的信号。
- **第一版 wiring 测试直接调 `orch._calibrate()`**,所以把 `_run` 里的调用点删掉它照样过 ——
  它测的是方法,不是接线。改成驱动真实 `_run` 并断言顺序后,删调用点会失败。
- **渲染真实 L3:43 数据时发现 108.5% 的 fp32 天花板占用率**,而指令混合说没用张量核。
  超过 100% 物理上不可能,而裸阈值判据会把它变成"已饱和、停止优化" —— 最贵的错误答案。
  现在会标记,并点名两个可能原因(用了更快的算术路径 / 候选比参考做了**更少**的算术)。

## 遗留(不在本轮范围)

- `WarpStall`(KernelPro 命中率第三高 11.0%、平均增益 1.41x)需要 ncu stall 分解,
  租用容器上不可复现。我先前把它评为"中低价值"**偏低了**。
- CuTe/CUTLASS 生成:步骤 8 的结论指向"若要投后端工作应冲 CuTe 而非裸 CUDA C",
  但那是独立的、更大的决定。
