# 下一阶段首批实验

方案：[工作计划](../../plan-v6-next-stage-20261027.md)。解释与进展：[实验报告](../../v6-next-stage-results-20261027.md)。

远端：`/root/autodl-tmp/dev/v6-work/next-stage-20261027`。框架与控制器快照来自提交 `01fcb47`。实验控制器为仓库 `scripts/experiments/v6_box43_next_stage_20261027.py`，不进入核心框架的任务路由。

- `launch.json`：源码快照、真实模型与启动命令。
- `qwen-fixed/protocol.json`、`progress.json`、`result.json`：预定四臂顺序、全部 16 个端点及精确结果。
- `qwen-fixed/python-P/T/C.json`：单独的完整生成 Python 诊断，不能作为 clean 分数。
- `qwen-fixed/native-summary.json`：精确完整版本的原生 kernel 核验与 SQLite 哈希。
- `qwen-summary.json`：由原记录导出的紧凑摘要；保留该实验是开发者固定候选诊断的身份。
- `ddp-protocol.json`：独立的三轮原项目入口自主实验协议，结果在后续更新。
- `manifest.json`：原始导出文件路径、大小与 SHA-256。模型、完整参考张量、私有 provider 配置和大原生 trace 留在服务器。
