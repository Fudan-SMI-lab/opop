# 同版本三项目批次证据

协议：[同版本自主验证](../../v6-autonomous-cohort-20261027.md)。远端目录 `/root/autodl-tmp/dev/v6-work/autonomous-cohort-20261027`，冻结提交 `4cdd076376de`。

- `launch.json`、`followup-launch.json`：同一个控制器的两个兼容启动记录，包含精确代码和命令，不代表运行了两个实验控制器。
- `protocol.json`：角色对照、三个原入口项目及独立检查的固定顺序、目标、轮数和输入源码哈希。
- `collection-ab/protocol.json`、已出现的初始 `generation-snapshots/`：角色重放所改变的唯一能力描述及真实会话输入捕获。它们不代表三项目已完成，也不代表 A/B 已成功。
- `wakeup-setup.json`：本机普通完成守候和真实 SSH 状态验证；不属于远端逐字导出清单。
- `manifest.json`：已导出远端文件的路径、长度和 SHA-256。

结果随真实终态更新。原始实验和新批次使用独立目录；私有 provider 配置、模型权重和完整大张量不进入 GitHub。

## 终态归档

`controller-result.json` 与两个 completed 文件是冻结原始结果，保留 ResNet/DDP 初始 source_fidelity_valid=false；实际原因和后续独立修正见主报告，不覆盖原始结果。

- `manual-audit/cohort-audit.json`：按真实会话顺序整理的九轮选择、提案、反馈快照摘要、配对数据、完整版本复测、revision 事件和原生 kernel 计数；含原始大结果路径/SHA。它是派生审计，原始 trial JSON、sqlite 和大张量仍留远端。
- `manual-audit/collection-input-comparison.json`：真实 planner A/B 的输入、模型、代码与 prompt 差异核对。
- `manual-audit/result.json`：后续案例映射审计，ResNet/DDP 的完整检查；路径属于另建的 `cohort-schema-audit-20261027`。
- 三项目 `sandboxes/`、request、prepared、events、intake result：真实生成物及过程证据，没有带入模型/provider 私有配置。
- `manual-audit/reflink-*.json`、`artifact-duplicates.json`：磁盘调查和单文件 CoW 试验，配额空间未增加，未批量清理。
- `wakeup-delivery.json`：本次实际终态触发的新轮次确认。

旧固定 Qwen 候选的拒绝/诊断记录属于前两批，未被本次新候选结果替代。
