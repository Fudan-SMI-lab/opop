# 失败定位与后续验证原始记录

远端目录：`/root/autodl-tmp/dev/v6-work/evidence-followup-20261021`。
本目录保留诊断、冻结索引对照、开发者辅助修复，以及三个原入口任务完整三轮的原始记录。
当前检查点已完成新凭据恢复后的原会话续跑、最终完整版本独立验证和原生执行路径核对。

| 自主原入口任务 | 最终结果 |
|---|---|
| ResNet batch-32 延迟 | 第三轮退化被拒绝，保留第二轮完整版本；独立 AB/BA `4.2171135 → 3.402626 ms`，降低约 19.3%，原生 Graph／融合 kernel 确认 |
| Qwen3-4B TTFT | 第三轮 RMSNorm 融合的完整 logits 不通过；保留版本独立 AB/BA `24.094085 → 24.5556405 ms`，拒绝，仍回退且无 Graph launch |
| GPT-2 DDP 通信 | 第三轮直接项目绑定质量通过，但整步 `144.138116 > 138.6598437 ms`，筛查拒绝；三轮均无接受版本 |

`final-cohort-summary.json` 汇总九轮、源结果哈希、最终独立检查和原生证据。
恢复时没有向原入口 Agent 注入人工诊断或算法建议；后续独立复测由开发者启动，分别记录。

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
- `provider-wait-status.json`：旧凭据下三个会话的实际 429 错误及恢复时间，作为历史保留。
- `replacement-key-check.json`、`credential-rotation-resume.json`：新凭据真实请求验证和原会话恢复记录，不包含凭据。
- `resume-input-integrity.json`：恢复前后全部声明输入、模型、公开工具设置与框架源码指纹一致。
- `fresh-independent`、`fresh-independent-result.json`：三轮结束后的最终完整版本独立 AB/BA；DDP 明确记为无接受版本。
- `final-native-result.json`：精确对应最终独立版本的原生 trace；ResNet 为 1 次 Graph launch／59 个 graph-node kernel，Qwen 均为 0。
- `qwen-third-native-summary.json`：被拒绝的第三轮候选确实执行了 145 次原生融合归一化 kernel，不能与最终保留版本的回退混淆。
- `resnet-third-diagnostic-summary.json`：被拒绝的新卷积候选主要设备开销位于 stem kernel，诊断计时不是干净性能分数。
- `ddp-contract-scope-audit.json`：本次冻结 bucket 参数及通信 hook 的限制来自实际生成合同，尚未调整。

诊断辅助 DDP 已越过 mask 和 stale stream 两个障碍，Graph 与后续 replay 正确，
独立复测整步约降低 28.0%，但通信目标连续两次未通过噪声门槛。该结果不属于本次自主原入口通信成功。
上午固定第二轮的 ResNet 独立结果为降低 18.9%；下午三轮终态再次独立复测为降低 19.3%，
两个结果分别保留。Qwen 第二轮原计时接受记录也保留，未因后续独立拒绝而改写。

DDP 第三轮直接绑定的透传对照测得增量 `13.129280 / 3.832763 ms`；同期父路径
`139.129207 / 138.533927 ms` 已接近冻结约束，需要进一步区分候选开销与基线漂移。
不能把通信筛查值 `64.149113 ms` 与旧基线直接相减当作配对验收收益。

旧 provider 阻塞在新凭据真实请求验证后解除。`usage-snapshot-before-key-rotation.json`
保留旧统计，最终 `usage-snapshot.json` 为 55 个会话、1,426 条 assistant message、
0 条未完成消息，provider `reported_cost` 约 48.851815；这是估计值，不是账单，不含外层 Codex。
项目没有新增 API／token／费用限额。

`manifest.json` 中 `exports` 记录逐字节导出原件的路径、大小和 SHA-256；`derived_exports`
是明确标记的字段摘要。大 trace／反馈文件只索引哈希，原件保留在 box4-2。
模型、reference tensor 和 provider 配置不公开；生成前快照及其可公开附件位于各运行目录和
`snapshot-artifacts`。`local-verification.json` 记录下载后逐项哈希复核。

CUDA hooks／shadow／Python tracing 属于侵入式诊断，不能用其耗时解释优化收益。
正确性检查通过若对应内部父路径回退，也不能证明优化实现正确或生效。
