# 固定现场隔离、真实 Agent 对照与修复验证

远端原件：`/root/autodl-tmp/dev/v6-work/targeted-isolation-20261021`。
详细解释见 [实验报告](../../v6-targeted-isolation-20261021.md)。

本轮 GPU 实验使用冻结框架 `a6f2b6e88ec31f169acd888601fcbb32cc2c1f9c` 和 box4-2。
这是开发者发起的固定候选隔离、角色提示对照及诊断辅助修复，不属于重新完成原入口
三任务自主流程。新的通用提示和目标参考聚合字段另有针对性测试，不能混用版本结论。

| 范围 | 已完成的事实 |
|---|---|
| Qwen 数值隔离 | 首个 RMSNorm 隔离出 BF16 舍入位置差异；仅改舍入仍不能通过完整模型 |
| Qwen 修复 A/B | 两个真实 Agent 候选均完整零误差，均有新 kernel；control 性能退化，treatment 首轮配对接受 |
| Qwen 独立验证 | 首次因开发者 noexec 临时目录设置全部回退；恢复原环境后新路径执行且完整零误差，`24.522982 → 24.305078 ms`，两对收益均低于噪声界，拒绝 |
| Qwen 绑定对照 | 145 个直接绑定的透传 TTFT 增量约 2.51 / 3.33 ms；不能作为可扣除常数 |
| DDP 三臂隔离 | 六端点质量通过；父路径步时明显变化；通信配对第二对收益未超过噪声界 |
| DDP 合同 A/B | 通用提示改善不变量与实现选择区分；两个新 adapter 均通过双 rank 连续更新和 reset 回放的完整零误差核验 |
| 采集规划 | 逐项补充真实工具能力、直接绑定适用条件和原参考目标聚合值，真实回复纠正了对应误读；仍存在其他推理过度 |

主要记录：

- `final-targeted-summary.json`：最终判定、生成输入核验、14 个零误差端点及有效独立版本的新 kernel 原生证据。
- `protocol.json`、`next_*.py`、各 `protocol.json`：预定边界、控制变量和实验脚本。
- `qwen-norm-shadow`、`qwen-rounding-intervention`：开发者数值隔离，不能用侵入式耗时作性能分数。
- `qwen-repair-ab`、`qwen-repair-ab-result.json`：真实两臂生成、原参数筛查和正式 AB/BA。
- `qwen-repair-ab-evidence.json`：仅 treatment 获得的实测摘要。
- `qwen-repair-input-integrity.json`：声明输入仅一个证据字段不同，增加 4,691 字节；prompt、模型、公开工具设置及框架相同。
- `qwen-repair-native-profile.json`：两种候选各自的新平方／归一化 kernel 均执行 145 次。
- `qwen-independent`、`qwen-independent-result.json`：第一次整版本原始记录；质量通过对应全部回退，不可作为新 norm 实现的性能验证。
- `qwen-independent-tempdir-audit.json`：`/dev/shm` 的 noexec 属性与精确扩展导入失败；原 `/tmp` 缓存可加载。
- `qwen-independent-matched`、`qwen-independent-matched-result.json`：恢复原启动环境后的独立整版本 AB/BA 及精确版本原生核验，不发布候选。
- `ddp-triplet`：固定 P/T/C/C/T/P 的六端点记录，无重试到通过。
- `contract-origin-audit.json`、`ddp-contract-ab`：源要求与 evaluator 合同来源审计、冻结生成前输入对照。
- `ddp-adapter-fidelity`、`ddp-adapter-fidelity-result.json`：人工观察字段映射后的双卡完整独立核验。
- `qwen-planner-ab`、`qwen-capability-probe`、`planner-clarity-check`、`ddp-aggregate-probe`：先后发生的小范围真实提示实验；后续探针沿用先前对照，不宣称统计显著性。
- `qwen-binding-eligibility`：原候选实际可执行的直接绑定成本采集。
- `targeted-tests.json`／`targeted-tests.log`：4 个相关测试模块共 16 项通过，含实际测试源码哈希。
- `usage-snapshot.json`：仅统计本目录真实会话，12 个 session、178 条 assistant message、0 条未完成，provider reported cost 约 5.264858；不是账单，不含外层 Codex，无调用限制。

`manifest.json` 的 `exports` 保留逐字节原件大小和 SHA-256，`derived_exports` 是明确标记的
字段摘要。大 trace／反馈保留原路径并索引哈希；模型、参考张量和 provider 配置不公开。
`local-verification.json` 记录下载后逐项验证，`export-credential-check.json` 记录已知凭据扫描，
不含凭据值。生成前快照附件位于 `snapshot-artifacts`。

`completed-jobs-storage.json`、`qwen-profiles-storage.json` 记录为缓解服务器磁盘压力而迁移的
已完成大文件：原路径通过符号链接保持可读，另一持久磁盘的副本逐项校验，压缩归档另存
本机 `E:/opop-evidence/20261021` 并匹配 SHA-256。没有删除原始测量内容。
