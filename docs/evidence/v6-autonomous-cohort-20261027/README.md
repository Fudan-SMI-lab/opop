# 同版本三项目批次证据

协议：[同版本自主验证](../../v6-autonomous-cohort-20261027.md)。远端目录 `/root/autodl-tmp/dev/v6-work/autonomous-cohort-20261027`，冻结提交 `4cdd076376de`。

- `launch.json`、`followup-launch.json`：同一个控制器的两个兼容启动记录，包含精确代码和命令，不代表运行了两个实验控制器。
- `protocol.json`：角色对照、三个原入口项目及独立检查的固定顺序、目标、轮数和输入源码哈希。
- `collection-ab/protocol.json`、已出现的初始 `generation-snapshots/`：角色重放所改变的唯一能力描述及真实会话输入捕获。它们不代表三项目已完成，也不代表 A/B 已成功。
- `wakeup-setup.json`：本机普通完成守候和真实 SSH 状态验证；不属于远端逐字导出清单。
- `manifest.json`：已导出远端文件的路径、长度和 SHA-256。

结果随真实终态更新。原始实验和新批次使用独立目录；私有 provider 配置、模型权重和完整大张量不进入 GitHub。
