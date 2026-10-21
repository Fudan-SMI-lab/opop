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
- `fresh`：同一 `a6f2b6e` 框架下的原项目／命令／目标入口，未注入上述人工诊断。

`manifest.json` 中 `exports` 记录逐字节导出原件的路径、大小和 SHA-256；`derived_exports`
是明确标记的字段摘要。大 trace／反馈文件只索引哈希，原件保留在 box4-2。
模型、reference tensor 和 provider 配置不公开；生成前快照及其可公开附件位于各运行目录和
`snapshot-artifacts`。`local-verification.json` 记录下载后逐项哈希复核。

CUDA hooks／shadow／Python tracing 属于侵入式诊断，不能用其耗时解释优化收益。
正确性检查通过若对应内部父路径回退，也不能证明优化实现正确或生效。
