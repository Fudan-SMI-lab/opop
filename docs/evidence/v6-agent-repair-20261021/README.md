# Agent 修复、绑定成本与新入口验证

box4-2 双 RTX 4090；真实 OpenCode/provider 会话和真实 GPU 执行。
结论与责任边界见 [实验报告](../../v6-agent-repair-and-binding-cost.md)。
该目录保留失败与中断记录，不只收录成功候选。

## 如何读取结果

- [汇总](final-summary.json) 是从终态文件提取的索引；原始 trial/result、worker receipt 和逐 rank 输出是依据。
- `summary.json` 为大 trial/result 的字段摘录，包含来源说明；原文件哈希在 manifest 中。
- [manifest](manifest.json) 列出原字节导出的 SHA-256，以及超过导出大小阈值的服务器原文件及其哈希。
- `generation-snapshots` 与去重后的 `snapshot-artifacts` 保存生成前 prompt/输入；generated、candidate-code
  和 Agent receipt 保存真实生成产物及会话定位信息。大输入附件可按 manifest 从 box4-2 获取。
- 性能结果使用原质量检查、预热、重复测量和 AB/BA 验收；diagnostic profiler 数值不替代干净分数。
  路径标签是候选自报，不能单独证明原生实现执行。

## 诊断辅助修复与独立验证

四组 control/treatment 均使用冻结输入与独立真实会话；treatment 增加一项经核实的诊断事实。
这些不是无人工诊断的原始入口实验。框架冻结为 `95280f2`；独立复测使用 `4936eb2`。

| 任务 | 结果入口 | 实际结论 |
|---|---|---|
| Qwen TTFT | [A/B](qwen-ttft/result.json) | 两组均通过；Graph treatment 原任务约降低 18.7%，独立复测约降低 17.6%；新形状无有效收益 |
| Qwen throughput | [A/B](qwen-throughput/result.json) | 缓存接口和数值问题修复；Graph 自检失败后内部 eager 回退，两组均拒绝 |
| DDP step | [A/B](ddp-step/result.json) | 两组均实际运行 reduce-scatter/all-gather；完整步退化，两组均拒绝 |
| DDP communication | [A/B](ddp-communication/result.json) | treatment 自主改选 module 边界并满足整步约束；通信收益未越过噪声界 |

[独立复测与 held-out 记录](independent-validation-result.json)；
[Qwen Graph 原生证据](qwen-ttft-native-graph-evidence.json)；
[DDP 通信实现原生证据](ddp-step-native-path-evidence.json)。
独立复测由开发者启动，保留候选代码和参数，不重新生成、不发布到其他任务。

## 原始入口自主运行

输入只有原项目、命令和目标；未提供 adapter、分组、候选或人工诊断。每个任务运行两轮。
Qwen/ResNet 冻结为 `88b9651`；修正可用性接口后另行重启的 DDP 冻结为 `e16e80f`。

- ResNet：第一轮 BN folding，第二轮选择 12 组、28 个 module 站点做联合融合并替换第一轮 bundle。
  最终完整版本独立 AB/BA 为 4.225842 → 3.4788785 ms，约降低 17.7%，质量与覆盖检查通过。
  [原生融合证据](resnet-native-path-evidence.json) 包含 9 次 cudnn_convolution_relu 与 8 次 Triton epilogue kernel。
- Qwen：第二轮虽被原计时规则接受，但两个调用均内部回退；最终完整版本独立复测被拒绝。
  不能计为有效融合优化。[仅打印原比较误差的诊断](fresh-qwen-guard-probe/result.json) 表明，
  prefill logits 仍有 15,511 个元素超出正式 0.02 容差。该诊断保持原阈值、计算和回退逻辑。
- DDP：第一轮三个 NCCL 参数候选整步约 254–295 ms，第二轮默认 group 加 CPU 到达对齐
  仍为 249.80 ms，均超过 144.80 ms 的整步限制；质量通过，但没有可接受候选。
  两轮仍使用 functional 站点，不能将退化全部归因于新通信算法。
  第二轮后 Agent 自主选择并执行 binding_cost 等五项诊断；纯绑定增量约 92–101 ms。

原始两轮结束后，开发者又启动一轮隔离续跑，使用完全相同的原始框架反馈，没有加入人工诊断、
分组或候选。见 [续跑启动记录](ddp-automatic-continuation-launch.json)；它单独统计，
不与原始两轮混称为一次无人触发续跑的三轮新入口运行。
该轮已完成：[终态结果](ddp-automatic-continuation-result.json)。Agent 自主改选整个 project
调用边界，实际使用 `direct`，纯绑定对照不再出现约 100 ms 增量。Graph capture 因 CPU→CUDA
拷贝的 CPU tensor 未固定内存而失败，随后回退父路径；两个 rank 的原异常均由现有诊断自动捕获。
正式通信指标 66.547669 → 73.144745 ms，一组整步超过原限制，最终拒绝。
后续 Agent 自主选择的六项诊断均完成；没有已接受候选，未启动独立版本复测。

[Qwen/ResNet 最终完整版本独立结果](fresh-final-independent-result.json)；
[重启 DDP 的独立结果](fresh-fixed-independent-result.json)。
首次 DDP 新入口因已测量字段未被 available_fields 暴露而失败，保留失败记录；
[原请求和 adapter 不变的双卡复测](ddp-observation-contract-probe/result.json) 验证修正。

Adapter 核验见 [第一次新入口](fresh-source-audit-result.json)、
[重启 DDP](fresh-fixed-source-audit-result.json)。后者包含完整参数、梯度、loss、额外 micro-loss
及 replica 梯度一致性；未通过删减 observer 输出或放宽阈值获得通过。

## 绑定成本与信息重放

- [职责审计](binding-ownership-audit.json)：实际 functional dispatch 在框架 BindingSession 安装并激活。
- [DDP functional 透传](binding-cost-probe/result.json)：保留完整父版本、仅省略新增候选 build/计算，仍有大额开销。
- [Qwen module 透传](module-cost-probe/result.json)、[DDP 实际 module 边界透传](ddp-module-binding-cost/result.json)：
  未重现约 100 ms 的全局 functional dispatch 增量；噪声差值不能作为固定或负成本。
- [自动采集规划实际执行](feedback-plan-probe/summary.json)：Agent 自主选择五项诊断，包括 binding_cost，均实际运行。
- [干净基线输入重放](feedback-plan-baseline-ab/result.json)：只增加原始干净基线，纠正 intrusive discovery 基准混用。
- [拒绝候选的路径摘要重放](qwen-plan-routes-ab/result.json)、
  [形式通过但回退候选的路径摘要重放](qwen-accepted-routes-ab/result.json)：只补回既有路径记录，
  让规划转向真实 fallback/执行路径问题。它们只比较规划生成，不代表新的性能收益。
- [原始基线 bundle 身份重放](ddp-plan-binding-revision-ab/result.json)：只增加真实 revision bundle 列表，
  Agent 不再排除绑定成本，选择 binding_cost 等诊断；对应代码为 `ce10994`，相关 9 项测试通过。
  [先行探索的来源复核](ddp-plan-binding-reference-ab-review.json) 说明为何不以缺失 setup 字段的
  空值归一化支持改动结论。探索会话及其产物保留，实际 API 费用计入统计。

## 中断、成本与恢复

最初四组对照和第一批新入口因 ENOSPC 中断。恢复 control 复用已经生成的原代码，重新执行完整 GPU
验证；恢复 treatment 从同一生成前快照重放。中断运行不算完成的端到端结果。

冷历史目录及大 reference/trace 文件先转存本机并核对 SHA-256，再从服务器展开副本中移除。
本目录各 `*archive*.json` 保存原路径、归档哈希和恢复说明；私有完整归档位于本机 `.scratch`，不上传 GitHub。
Provider 配置和 reference tensor 本体也不在公开导出中。

[会话统计](usage-snapshot.json) 包含中断和探索性会话，仅观察费用和 token，不限制真实 API 使用。
数值是 provider 报告估算，排除外层 Codex 对话，不等同于账单。
最终快照含 63 个真实框架会话、1,615 条 assistant message，`reported_cost=52.54925979724`；
其中 2 条历史中断 message 没有完成标记，保留在统计中，不代表本轮仍有实验运行。
