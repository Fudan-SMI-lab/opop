# 目标切换与 TP 准备证据

协议见[执行文档](../../v6-goal-switch-20261027.md)。冻结提交 `1d6f3a5bd822`；远端根目录 `/root/autodl-tmp/dev/v6-work/goal-switch-20261027`。

- `launch.json`、`followup-launch.json` 是同一控制器的兼容启动记录，并非两个实验进程。
- `protocol.json` 固定实际顺序、工作量、目标、源哈希与空间情况。
- `intake-ab/protocol.json` 和实际生成前快照记录唯一新增段落及模型/输入身份；结果完成后另行审计，不预判 treatment 成功。
- `wakeup-setup.json` 记录普通后台守候、真实 SSH probe 和一次终态投递配置；不属于远端逐字导出文件。
- `manifest.json` 列出逐字导出文件的远端路径、长度和 SHA。

无私有配置/API key、模型权重或原始大 trace。旧批次保持独立，不覆盖其失败或独立接受结论。

终态已归档：controller/followup 记录保留原 TP 依赖失败、ResNet 初始字段不匹配；`manual-audit/` 来自另建的 `goal-switch-audit-20261027`，含两个 bundle 的 trial/反馈/完整独立结果摘要、完整源码入口与数值审计、单行持有 images 的手工因果对照、依赖 overlay 身份及真实 TP 重跑。TP 重跑执行成功但 logits 质量失败；ResNet 原冻结接受不等于源资源状态一致。所有区别见主报告，不以派生结果覆盖冻结文件。

`wakeup-delivery.json` 确认本次真实自动续行。大 trial、trace、tensor 和依赖 wheel 留在远端，导出文件的路径/长度/SHA 见 manifest。审计生成脚本单独归档供复现，不属于优化流程。
