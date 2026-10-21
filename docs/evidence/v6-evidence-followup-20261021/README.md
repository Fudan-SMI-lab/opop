# 失败定位与后续验证原始记录

远端目录：`/root/autodl-tmp/dev/v6-work/evidence-followup-20261021`。
本目录按时间点导出原始记录；当前检查点包含已结束的诊断与冻结索引对照，以及尚在运行的后续会话快照。
后续会话没有终态记录时，不得当作完成或成功实验。

- `qwen-boundaries`：原融合候选的实际 GPU 中间边界，纯读数。
- `qwen-rotary-isolation`：第一层同输入 shadow 隔离。
- `qwen-fp-fusion-probe`：只关闭两个 launch 的 FP fusion，完整候选仍回退。
- `qwen-boundaries-no-fusion`、`qwen-third-layer-isolation`：编译选项干预后的边界及第三层 shadow；同时阅读 `intervention-scope.json`。
- `ddp-capture-trace`：原 capture 异常处的 print-only traceback。
- `framework-traceback-probe`：新框架在未修改原 DDP 候选上自动捕获同一库内异常。
- `qwen-index-ab`：冻结候选生成的已有证据索引 A/B；两个候选均被拒绝，均回退。
- `ddp-stack-repair`、`qwen-numerical-repair`：开发者诊断辅助的真实 Agent 修复，与自主原入口运行分开。
- `ddp-feedback-continuation`、`qwen-feedback-continuation`：诊断辅助分支的一轮续跑，仅使用上一轮框架产生的实际反馈。
- `ddp-steady-replay-validation`：同一 DDP 候选额外四次连续更新，完整输出逐项一致；只在独立验证请求中增加步数，未改正式合同。
- `ddp-feedback-independent-fixed`：同一 DDP 候选仅一次独立完整版本 AB/BA；整步收益复现，通信收益仍未越过噪声门槛。
- `fresh`：同一 `a6f2b6e` 框架下的原项目／命令／目标入口，未注入上述人工诊断。
- `checkpoint-independent`：第三轮等待 provider 时，对固定第二轮完整版本的独立 AB/BA；不是三轮全部完成后的最终验证。
- `source-audit`、`source-audit-summary.json`：三个新 adapter 的原项目独立 GPU 核验，均通过。
- `native-path-summary.json`：从对应候选的原始 Nsight SQLite 汇总 Graph API、graph node 和 kernel 数量，附原件哈希。
- `provider-wait-status.json`：三个会话的实际 429 错误及恢复时间；会话保留，未人为限制 API 使用。

当前阶段：诊断辅助 DDP 已越过 mask 和 stale stream 两个障碍，Graph 与后续 replay 正确，
独立复测整步约降低 28.0%，但通信目标连续两次未通过噪声门槛。新入口 ResNet 第二轮完整版本
独立验证降低 18.9%；新 Qwen 第二轮计时被接受，但实际全回退，独立复测未通过；新 DDP 前两轮
分别因 capture 回退和高昂 functional 绑定开销未通过整步约束。

三个自主任务均已完成前两轮，第三轮遇到 provider 的七天使用上限（HTTP 429），服务端给出
`2026-10-23 17:38:14 +08:00` 恢复时间。三轮新入口实验和完整最终版本独立验证尚未结束。
`usage-snapshot.json` 是仅统计的时点费用记录，等待中的会话不会当作终态费用。

`manifest.json` 中 `exports` 记录逐字节导出原件的路径、大小和 SHA-256；`derived_exports`
是明确标记的字段摘要。大 trace／反馈文件只索引哈希，原件保留在 box4-2。
模型、reference tensor 和 provider 配置不公开；生成前快照及其可公开附件位于各运行目录和
`snapshot-artifacts`。`local-verification.json` 记录下载后逐项哈希复核。

CUDA hooks／shadow／Python tracing 属于侵入式诊断，不能用其耗时解释优化收益。
正确性检查通过若对应内部父路径回退，也不能证明优化实现正确或生效。
