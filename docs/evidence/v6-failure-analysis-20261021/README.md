# Qwen / DDP 失败诊断证据

分析结论、方法和限制见 [失败原因核验](../../v6-qwen-ddp-failure-analysis.md)。

- `trial-*/result.json`：四个原 Qwen 失败候选的逐步复现。
- `ttft_*`、`decode_*`、`graph_preserve_prefill_mask`、`static_live_attention_window`：八个开发诊断变体，均非新的自主优化结果。
- `qwen-math/result.json`：第一层中间值、真实 SDPA 输入差异及单算子诊断。
- `ddp-ablation/summary.json`：14 个质量通过的独立性能端点。
- `ddp-cpu/`：五组双 rank CPU profile；插桩时间不用于性能接受。
- `ddp-reduceop-audit.json`：boxed ReduceOp 的 schema、返回类型及原候选守卫行为。
- `ddp-source-jobs.json`：复用原参数筛查实例的源码哈希、参数及参考路径。
- `ddp-communication-repeat-traces.json`：48 个原通信复测 trace 的首核／其余核分解与文件哈希。
- `runtime-source-audit.json`：实际安装的 Transformers 分支实现与版本信息。
- `scripts/`：收尾保存的诊断脚本（分析过程中逐步扩展）；各变体候选字节独立保存，路径指向 box4-2 原冻结实验。
- `manifest.json`：导出的原始字节及远端文件索引。大 trace、worker request/environment 留在 box4-2。

原始自主流程和参考保持不变。此目录保留诊断脚本早期失败日志；完成状态以各 `result.json`、
`completed.json` 和最终 `runtime-completion.json` 为准。
