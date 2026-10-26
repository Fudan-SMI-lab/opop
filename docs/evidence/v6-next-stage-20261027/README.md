# 下一阶段首批实验

方案：[工作计划](../../plan-v6-next-stage-20261027.md)。解释与进展：[实验报告](../../v6-next-stage-results-20261027.md)。

远端：`/root/autodl-tmp/dev/v6-work/next-stage-20261027`。框架与控制器快照来自提交 `01fcb47`。实验控制器为仓库 `scripts/experiments/v6_box43_next_stage_20261027.py`，不进入核心框架的任务路由。

- `launch.json`：源码快照、真实模型与启动命令。
- `qwen-fixed/protocol.json`、`progress.json`、`result.json`：预定四臂顺序、全部 16 个端点及精确结果。
- `qwen-fixed/python-P/T/C.json`：单独的完整生成 Python 诊断，不能作为 clean 分数。
- `qwen-fixed/native-summary.json`：精确完整版本的原生 kernel 核验与 SQLite 哈希。
- `qwen-summary.json`：由原记录导出的紧凑摘要；保留该实验是开发者固定候选诊断的身份。
- `ddp-protocol.json`：独立的三轮原项目入口自主实验协议，三轮已完成且均未接受。
- `ddp-raw/request.json`、`prepared.json`、`agents/`：冻结请求、双 rank controls，以及全部 evaluator/selection/candidate 输出；终态以 ddp-result 和 ddp-audit 为准。
- `followup-protocol.json`：自主批次之后的 Qwen 标准独立 AB/BA 和 DDP 原源码一致性检查。
- `metric-order-audit.json`、`metric-order-tests.json`：原始源定义差异及 22 项受影响检查。
- `metric-order-ab/`：独立真实 intake 对照；`input-comparison.json` 确认 prompt、源码、模型、工具设置一致，仅 schema 文件不同；原输入和生成结果均保留。
- `wakeup-setup.json`：普通完成守候进程的部署快照、六项真实远端状态检查和两项队列替身检查；真实自动续行仍以终态触发后的回执为准。该文件汇总本机与远端记录，不属于原文件逐字导出清单。
- `wakeup-live-delivery.json`：用户授权的一次真实 `codex queue` 投递回执，以及测试消息自动开启新轮次后写入的确认。证明当前会话可以自动续行；这是较早测试记录；实际批次投递见 wakeup-batch-delivery.json。该文件为汇总记录。
- `manifest.json`：原始导出文件路径、大小与 SHA-256。模型、完整参考张量、私有 provider 配置和大原生 trace 留在服务器。

- `qwen-independent-summary.json`：标准独立 AB/BA 的全部样本、质量和拒绝决定。
- `ddp-audit.json`、`ddp-result.json`：三轮结果、完整观察摘要、绑定对照、聚合顺序重算、provider 统计；原始大结果的路径和哈希保留。
- `controller-completed.json`、`followup-completed.json`、`followup-result.json`：控制器正常结束、Qwen 独立性能失败、DDP 原源码一致性通过。
- `wakeup-batch-delivery.json`：本批真实终态触发投递及新轮次确认，属于本地/远端汇总证据。
