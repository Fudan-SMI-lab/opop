# 首批终态后的最小实验

协议与解释：[最小诊断与 evaluator 对照](../../v6-post-batch-audit-20261027.md)。远端目录 `/root/autodl-tmp/dev/v6-work/post-batch-audit-20261027`，冻结代码 `96e08fc59aba`。

- `launch.json`、`followup-launch.json`：原负例与后续控制器的启动命令、源码快照和时间。
- `probe-protocol.json`、`probe-result.json`、`controller-completed.json`：负例预定方法及真实双卡完整结果。成功执行且确认漏检；不是优化候选。
- `qwen-sampling/sampling-protocol.json`：固定 B/C/C/B、仅 repeats 5→20 的开发诊断协议。原任务和旧独立失败保持不变。
- `qwen-sampling/result.json`、`qwen-sampling-summary.json`：已完成的四端点、全部 80 个原始样本及派生摘要；四端点全量质量均零误差，不作为正式接受分数。后者为本地派生文件。
- `evaluator-ab/protocol.json`、已出现的 `generation-snapshots/`：真实角色对照协议与初始会话捕获；生成前状态不等于生成物与 GPU 检查已成功。
- `wakeup-setup.json`：本机普通守候部署及真实远端状态检查，无模型轮询。此文件不属于远端逐字导出清单。
- `manifest.json`：当前已导出远端文件的路径、长度和 SHA-256。后续结果随真实终态补充，未完成时不能声称 evaluator 改进成功。

原 adapter、参考张量和原批次保留在服务器。导出只含必要证据，不包含私有 provider 配置或认证信息。
