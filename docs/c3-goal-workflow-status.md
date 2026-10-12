# C3：最终要做成什么，现在做到哪里

## 1. 最终目标不是“写出一个更快的小 kernel”

**最终要做成一个通用优化框架：输入模型和固定性能目标，由 agent 生成真实 Triton/CUDA 计算算子替换，经过局部调参、正确性与完整模型评测，交付可验证的模型级优化结果。**
当前 C3 是这一目标在 Qwen3-4B 上的 **C3-B：人工提供任务、评测器与规则**。从自然语言自动构建任务/评测器的 C3-A 尚未做；现有局部证明路线主要支持 Triton，不能宣称任意 CUDA 实现都已接通。

优化对象是**替换算子后的完整模型任务指标**，不是只降低 kernel 的局部微秒数，也不是换三种 prompt、生成三份 Python wrapper 就算成功。
最终应交付三类东西：
1. 可执行、可复核的框架闭环：目标驱动生成 → 设备证明 → 局部调参 → 模型质量/性能 → 选择或回退。
2. 同一冻结框架下，三个目标各自新生成并选定的产物或明确 baseline fallback，附源码、参数、质量与设备执行证据。
3. 原 baseline 加三个目标产物的**三指标交叉结果表**及完整测量记录，说明收益、代价、失败和适用范围。

## 2. 三个目标到底测什么

固定 Qwen3-4B、BF16/eager/SDPA、输入输出长度及质量规则，不为获得好数字改变题目。

| 目标 | 固定任务 | 完整模型指标与方向 |
|---|---|---|
| TTFT：首 token 延迟 | 4096 输入 → 1 输出 token | 请求开始到首 token 就绪的 wall time，含 prefill；毫秒，越低越好 |
| single：单流解码速度 | 256 输入 → 128 输出 tokens | `127 / (last_ready - first_ready)`，分母以秒计；tokens/s，越高越好 |
| multi：多请求吞吐 | 8 个请求，每个 512 输入 → 128 输出 tokens | 同步 batch/wave 的 `1024 / shared_wall`，分母以秒计且含 prefill；tokens/s，越高越好 |

Single 仍执行 prefill，只是解码速率分母不含首 token 延迟；multi 是有限的同步请求波次，**不是在线服务吞吐/SLO**，也不是单次 forward 乘以 8。
研究假设是：目标不同可能促使框架选择不同算子、配置或性能取舍。**不保证三个 kernel 不同，也不保证每行在自己的目标列胜出。**

## 3. 完整流程应该怎样走

以下是目标流程，不是已全部验收的现状：

```text
固定模型、合同、开发数据与独立 heldout
  → GPU profile 选择窄算子，提供 reference/ABI、真实 shape/dtype/cast 语义
  → agent 根据完整模型目标编写实际计算 kernel
  → 真实输入 fixture：编译、CUDA trace 确认 launch、输出参与、局部正确、配对微测量
  → 有界局部参数调优：比较 reference A/A 噪声，不把局部 us 当模型得分
  → 完整模型质量 + 匹配 baseline + 该目标的真实模型指标
  → 按冻结规则接受并确认，或保留 baseline
  → 冻结每个目标的最终产物/回退身份
  → 新 heldout 上每个产物测全部三个指标，形成交叉表和报告
```

这里有两个不同的环，不能混在一起：
* **开发环 F0/F1/F2**：调试框架。每轮框架固定，agent 从原 baseline 新生成；父级只修框架，不人工修候选算法。开发产物不是正式 winner。
* **正式环 F\***：先达到 DEV_GO，再冻结框架和选择规则；TTFT/single/multi 各从同一原 baseline、新 session 重新生成，不能把开发最好产物直接搬过去。
* **DEV_GO** 要真实设备证明、局部数值正确、可信实测 local headroom 和完整模型质量/有效测量。局部收益须结合 A/A 噪声，狭窄可恢复配置路径也须实测支持；不保证端到端改善不等于豁免这些条件。
* **正式接受**还需匹配 baseline 的目标改善及确认；否则 fallback。Final 不再调参或根据 heldout 重选 winner。

## 4. 现在的位置：还在 DEV_GO 前，不是只差最后测表

**已完成部分基础能力与诊断 →【当前：真实 agent 开发被阻塞，NO_DEV_GO】→ DEV_GO → 冻结 F* → 正式三目标新生成 → 新 heldout 交叉表。**

| 阶段 | 已有能力 | 验证状态 | 还缺什么 |
|---|---|---|---|
| 专用 agent 与 C2 隔离 | 显式 C3 device profile、C2 compatibility profile，保留 generic/default | **已实现、CPU 验证**；有限合同比较通过 | 后续框架修改继续保护 C2，不把 CPU 兼容当优化成功 |
| IR 归属与 setup control | 编译后 IR 输入/输出归属修复、设备证明与模型检查 | **人工 control 已 native 通过**；wrong/skip 拒绝有效 | control 只证明评测路线，不是 agent 结果 |
| 局部 helper | fixture 编译、数值/输出检查、CUDA 采集与微测量 | **已实现；真实 generation 仍阻塞** | fresh agent 生命周期中 CUDA trace 为空的根因未明 |
| Owner-thread 路由 | agent transport 与 resident owner callback 分离 | **CPU 通过、原样诊断重放 native 通过** | fresh F1 仍失败，不能称完整修复 |
| 预算与开发入口 | 持久 ledger、develop/probe、停止及 review/resume | **已实现、CPU 验证、两轮真实开发已使用** | 没有任何一轮达到 DEV_GO；旧窗口已关闭 |
| 局部 TPE 与模型 adapter | 有界配置搜索、模型质量与 native 指标接口 | **已实现、CPU 验证；模型 adapter 有 control 验证** | 未完成真实 agent 的端到端 GO；计划 headroom 条件不能仅凭程序状态替代 |
| 正式三目标编排 | 有 freeze/formal helpers 与阶段规则 | **端到端实现/验收不完整，且 NOT_RUN** | 最小 driver、冻结身份、三目标新生成/选择及终止链须完整接通 |
| 新 heldout 终测 | 已冻结新数据和 4×3×3 规则 | **新 final 验收链不完整，且 NOT_RUN** | final reference/证明账本、36 块测量及报告连接；旧 CLI 不能静默套新合同 |

“正式链不完整”是现有技术交接明确记录的缺口，不因 helper 函数存在就视为完成。
当前 TPE 的超噪声收益检查用于调优准入，`prove_candidate` 不单独检查 local headroom；计划要求和实现检查的差别仍需保留。

两次 fresh probes（F0、F1）都有编译、launch counter、局部数值、IR 和输出参与的部分证据，但都 **`cuda_names=[]`，且 480s transport timeout**；没有完成返回的合格产物或官方模型得分 J。
原 F0 不变产物在 owner-thread RPC 诊断中通过，不代表 fresh F1 通过。F1 已在 owner thread 采集仍失败；**profiler 生命周期/采集残余根因、timeout 原因及两者因果关系均未知**。
所以目前没有 DEV_GO、F*、formal S、正式三目标产物或新矩阵。最终 review 只批准如实交付失败，方法效力仍未建立。

## 5. 最终结果表必须长这样

下表是**本轮真设备 kernel 实验的预期交付模板**，不是已经产生的结果。“— 待测”表示 NOT_RUN，不是零分，也不表示已选出对应产物。

| 正式冻结行 | TTFT ms ↓ | single TPS ↑ | multi TPS ↑ |
|---|---|---|---|
| Baseline | — 待测 | — 待测 | — 待测 |
| TTFT 目标优化产物 | — 待测 | — 待测 | — 待测 |
| 单流目标优化产物 | — 待测 | — 待测 | — 待测 |
| 多请求目标优化产物 | — 待测 | — 待测 | — 待测 |

**每行都测三列，不能只测对角线。** 每格 3 个 blocks，共 4 行 × 3 列 × 3 = **36 blocks**，在同一 A0 物理 GPU 串行、固定质量规则下独立测量，保留 block 原始值与汇总口径。
每列使用开发前冻结的新 heldout；开发/正式搜索使用 16 条允许记录，新 heldout 为 42 条。Sealed 容器合计 58 条是 16+42，不是 58 条新 heldout。

每列相对同列 baseline 的收益统一约定“正数为改善”：
* TTFT：`100% × (baseline_ms - row_ms) / baseline_ms`。
* single：`100% × (row_single_TPS / baseline_single_TPS - 1)`。
* multi：`100% × (row_multi_TPS / baseline_multi_TPS - 1)`。

每行还必须附 bundle/source SHA、选中 target、最终参数、设备证明、模型质量结果和是否 fallback。重复 winner 或 fallback 行也重新测，不能复制旧分数；无效测量标无效，不填 0。
同一 winner、baseline fallback、跨目标退化或 heldout 负收益都可如实报告。框架达标需要真实 kernel 与质量证据；**研究上的正效果必须由实际分数证明**，不能由愿望、局部加速或不同源码代替。

历史边界：旧 C3 有一份 **Python decoder composition** 的模型级部分结果，single 约 **+12.88%**、multi 约 **+9.09%**，并有其自己的恢复矩阵。它不是本轮真设备 kernel 三目标结果，不能拿来填上述“待测”；本轮 NOT_RUN 也不抹掉旧数据。

## 6. 接下来只需抓住四步

1. 定位**真实 agent/helper 生命周期**的空 CUDA trace 与 timeout，区分采集为空和名称关联失败，不用成功诊断重放替代真实路径。
2. 最小修框架，让 fresh agent 通过设备证明、local headroom 与完整模型质量/测量，达到可信 DEV_GO；不人工改候选迎合验收。
3. 完成最小正式 driver 和新 heldout 验收链；在合格开发证据上冻结 F*、三目标任务及规则，不能借旧 manual/heldout CLI 绕过缺口。
4. 同一 F* 下三目标从原 baseline 新生成、选择/回退，再做 36 块交叉终测，交付表格、产物和诚实的效果报告。

优先真实 GPU 快反馈、修框架而非候选、保留 C2 与已有工作树；不扩建测试平台或铺开无关回归。
原 90 分钟 D 为 **2026-09-20 10:30:17.689003Z–12:00:17.689003Z，已过期**。上述是后续方向，不触发运行或新增授权；未来实验须另获明确预算与授权，旧状态不重置。

## 7. 详细材料

* [技术交接与运行说明](handoff-c3-fast-kernel-iteration.md)：命令、输入、环境、准入差别与证据路径。
* [论文数据资产总目录](paper-data-inventory.md)：哪些历史数据已经存在、能支持什么主张。
* [本轮结果报告](result-c3-fast-kernel-iteration.md)：NO_DEV_GO、两次 fresh 失败及科学结论边界。
