# v2 Harness 效果分析与改进方向

> 日期:2026-09-02。作者:Claude Code(基于 events.jsonl 磁盘核实 + kernelfoundry 源码调研)。
> 数据来源:`runs/run-l3-21-20260902-113144`、`runs/run-l3-43-20260902-140823`、
> `runs/run-l3-43-20260902-213608` 的 events.jsonl;KernelBench 与 kernelfoundry 源码。
> **纪律**:所有结论以磁盘 events.jsonl 为准,不采信任何 session 叙述噪音。

---

## TL;DR

- **正确性通过率低不是 eval 阈值太严,是生成质量问题**:generator 反复踩两个 Triton 坑——
  设备代码里用 `tl.next_power_of_2`(编译失败)、默认 `tf32` 点积精度(数值超差),
  repair 的 2 次预算不足以彻底修复。
- **调参环节本身有效**(真实探索、找到最优区);**L3:21 无提升的真因是"结构改写产出的候选性能全部等价"**,
  都撞在同一寄存器/共享内存瓶颈上,LLM 的 Triton 实现跑不赢 cuDNN 融合卷积。
- **novelty(Loop D)在最该触发时反而不触发**:失败/dropped 的族占满 `max_families_total` 名额。
- **两个 resume 健壮性 bug 已修复**(见 `HANDOFF.md`);修复后代码可信,但 agent 生成本身有随机性,
  一次干净重跑仍可能因 4/4 seed 全灭而空手(已实测,见 §2.3)。
- **最高收益改进**:借鉴 kernelfoundry 的**双精度见证正确性门** + **prompt 内置 Triton 反模式清单**。

---

## 1. 实验结果总览(磁盘核实)

| Run | 任务 | 结果 | best | vs eager | vs compile | 可信度 |
|---|---|---|---|---|---|---|
| run-l3-21-113144 | level3:21 EfficientNet MBConv | 负向 | cand-ef4785e1 @25.2ms | 0.87× | 0.65× | ✅ 完全可信(无 resume 边界,3 轮真实改写) |
| run-l3-43-140823 | level3:43 MinGPT Attention | 正向* | cand-e4096974 @29.2ms | 1.42× | 1.21× | ⚠️ 延迟数字可信,但 Loop C trace 被已修复的 resume bug 污染 |
| run-l3-43-213608 | level3:43(修复后干净重跑) | 空 | None(4/4 seed 全灭) | — | — | ✅ 流程可信,但本批 agent 生成质量差 |

\* 基线口径注:v2 基线用 ieee fp32 参考;kernelfoundry 用 tf32 参考,speedup 不可跨项目直接比。

---

## 2. 五个核心问题

### 2.1 为什么正确性通过率这么低?——生成质量问题,非阈值问题

**正确性判定机制**:v2 完全委托 KernelBench 的 `eval_kernel_against_ref`,自身不设阈值。
- fp32 容差 = `torch.allclose(output, output_new, atol=1e-4, rtol=1e-4)`
  (`KernelBench/src/kernelbench/eval.py:83` `get_tolerance_for_precision` + `:804`)。
- **整张量任一元素超差即整体判 fail**,跑 `num_correct_trials` 次(quick=3 / full=5),每次固定 seed=42。
- 这个阈值来自 torchbench 标准,**合理,不算严**。

**level3:43 四个 seed 全灭的真实失败模式**(两次 run 高度一致):
1. **编译失败(runtime_error)**:候选在 Triton kernel 里调用 `tl.next_power_of_2(D)` 作为
   设备代码 / `tl.arange` 上界 → JIT 编译报错。这是 Triton 新手陷阱。
2. **精度不达标(correctness_mismatch)**:默认 `DOT_PRECISION="tf32"`(TF32 仅 10 位尾数 ≈3 位十进制)
   在 attention 的长 reduction(QK^T 点积 + softmax 归一 + 乘 V)上累积误差 >1e-4。

repair agent **诊断正确**(明确指出 tf32 截断尾数),但在 2 次 repair 预算内未能把所有点积
彻底改为 fp32 累加。**结论:通过率低 = generator 反复踩同两坑 + repair 预算太浅。**

### 2.2 "cand-4edfa030 从 25.2ms 改进到 19.6ms" 是什么?——磁盘上不存在

核对 run-l3-21 事件流:cand-4edfa030 的真实 `TUNING_DONE` 记录是 **25.5ms / improved=False**。
**19.6ms 这个数字在 events.jsonl 中从未出现**。所谓"改进到 19.6ms 但 JSON 损坏"是上一个 session
的叙述错误 / 前后不一致,不是磁盘事实。因此:既没有 19.6ms 的改进,也不存在对应的"JSON 损坏"
磁盘证据。（这条正是"只信磁盘、不信叙述"纪律拦下的一次误报。）

### 2.3 level3:21 历史数据 / 为什么调参没提升?

**调参本身有效,在真实探索**。以 seed cand-ef4785e1 为例(空间 = BLOCK_P × NUM_WARPS × NUM_STAGES):
- 38 个 trial 全部完成(0 失败),延迟真实分布在 **25.2 – 28.1ms**,TPE 找到了最优区。

**问题在结构改写**。4 个候选(1 seed + 3 rewrite)调参最优值:

| 候选 | 类型 | 调参最优 | 入族最优? |
|---|---|---|---|
| cand-ef4785e1 | seed | 25.2ms | ✓ 首次 |
| cand-4edfa030 | rewrite 1 | 25.5ms | ✗ |
| cand-aac4d608 | rewrite 2 | 25.4ms | ✗ |
| cand-deb5eea7 | rewrite 3 | 25.2ms | ✗ |

`best_history = [25.2, 25.2, 25.2]` → `budget_exhausted` 冻结。

**关键洞察**:3 个"新结构"的最优值全部聚在 ~25.2ms,说明 LLM 结构改写产出的候选在性能上
**与种子本质等价**——都撞在同一寄存器/共享内存瓶颈上(bottleneck report 也如此判定)。
而 25.2ms > eager 21.8ms,是真实负向结果:LLM 的 Triton 实现跑不赢 cuDNN 融合卷积。
**"参数调优没提升"的表象,实际是"改写没带来结构性差异",不是调参失效。**

### 2.4 kernelfoundry 可借鉴什么?

见 `docs/research/kernelfoundry-findings.md` 完整报告。最高收益的 3 点:
1. **双精度见证正确性门**:参考跑 tf32 + ieee 两遍,候选匹配任一即过(直解 §2.1 的 tf32 失败)。
2. **宽松但双判据的容差**:相对误差 <1% 的元素占比 >99% + 余弦相似度 ≥0.99985。
3. **prompt 内置 Triton 反模式 BAD/GOOD 代码对**(直解 `next_power_of_2` 类编译失败)。

### 2.5 框架各流程是否真有用?

**验证过有用**:materializer(字节级 PARAMS 替换)、双见证正确性门(有效拦坏候选)、
TPE 调参(真实探索)、事件溯源 + resume(修完两 bug 后可信)、convergence 判定。

**有缺陷 / 形同虚设**:
- **novelty(Loop D / 需求 8)在最该触发时不触发**:`_novelty_round` 首行
  `if len(families) >= max_families_total(=3): reject`。seed 全灭时 4 个失败族占满名额,
  novelty 一个新族都产不出 → 直接 budget_exhausted。**失败/dropped 族不腾名额是真实设计缺陷。**
- **repair 预算太浅(2 次)**:面对"编译错 + 精度错"双层问题不够。
- **generator prompt 缺 Triton 陷阱防护**:反复踩 `next_power_of_2` 和 tf32。

---

## 3. 已修复的健壮性缺陷(详见 HANDOFF.md)

两个同一类("resume 丢失内存态控制决策")bug,均已用事件溯源修复 + 回归测试:
1. **httpx.ReadTimeout 未捕获崩溃**:挂起的 agent 调用杀死整个 run。
2. **resume 后 Loop C 静默禁用**:`crun.report` / `best_history` / `rewrite_rounds_used`
   是内存态,resume 不恢复 → 伪 budget_exhausted。

全套 82 测试绿。

---

## 4. 改进路线(按收益/成本排序)

### 第一梯队 — 直接提通过率,成本低
1. **加双精度见证正确性门**(收益最高):v2 双见证 gate 里参考跑 tf32 + ieee 两遍,匹配任一即过。
   改动小(每任务多一次参考前向),直接救活 level3:43 那类 tf32 候选。
2. **generator/repair prompt 加 Triton 反模式 BAD/GOOD 对**:见 §2.1 两坑。详见 §5 的辨析。
3. **GPU 评估前加零成本 regex 静态门**:拦掉设备代码里的 `tl.next_power_of_2`、
   非 constexpr 的 `tl.arange` 上界等已知致命模式,省 GPU 配额。

### 第二梯队 — 提搜索有效性
4. **修 novelty 名额逻辑**:dropped 族不占 `max_families_total`。
5. **提高 repair 预算 + 失败分级回传**:让 repair 针对数值错 / 编译错分别施策。
6. **给 generator 参考 kernel 库**(FlashAttention 等高质量 Triton 模板)供改写而非从零写——
   针对 L3:21 那种"改写不出结构差异"的问题。

### 需要决策的口径问题
- **基线精度口径**:若采纳双精度见证门,speedup 应明确对哪个精度的参考计算,并在报告注明,
  避免与 kernelfoundry 等 tf32-基线项目产生不可比的数字。

---

## 5. 待办 / 遗留

- `v2/D:/Git/mnt/...` 目录是 WSL 路径翻译遗留的误创建 triton 缓存(应在 `.triton-cache-wsl`),
  属垃圾可清理(破坏性操作,待用户确认)。
- **交付时提醒**:`.opencode/opencode.jsonc`、`opencode_backup.jsonc`、`kimi-provider.yaml`、
  全局 opencode.jsonc 有明文 API key,建议用户轮换(用户侧任务)。
