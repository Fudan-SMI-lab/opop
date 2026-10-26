# 实验完成后自动续行

用户希望只在实验结束后恢复 Codex 核查和开发，不使用 Goal 持续生成轮次来检查进度。当前采用普通本机 Python 后台进程，通过 SSH 等待远端终态，再向本会话提交一次后续指令。

## 当前配置

- 运行脚本：`scripts/experiments/wake_codex_on_completion.py`。
- 远端目录：`/root/autodl-tmp/dev/v6-work/next-stage-20261027`，box4-3。
- 等待对象：`followup-completed.json`，即 DDP 自主批次及已排定的 Qwen 独立复测、DDP 原源码检查全部结束。失败结果同样需要核查；不把 JSON 文件存在解释为优化成功。
- 检查周期：30 秒，由普通 Python/SSH 执行，不调用模型。
- 目标会话：`01a154d9-d2c0-7d80-9492-7c5ff93d4f66`（当前项目会话）。
- 触发命令：本机 Codex 0.164.0 的 `codex queue --thread <id> --message <continuation>`。保持当前会话的模型和权限，不创建新 Agent 或启用 Goal。
- 本机状态：`.scratch/v6-next-stage-wakeup.json`；日志：`.scratch/v6-next-stage-wakeup.log`；后续指令：`.scratch/v6-next-stage-wake-message.txt`。

等待期间只覆盖一个小状态文件，不持续累积大量日志，也不读取大 trace。当前项目本身正在运行的候选生成等 Agent 请求仍正常计费；这里的零模型调用仅指新增等待机制。

脚本看到完整可解析的终态记录后排入一次续行，保存 CLI 回执并退出。如果控制器已退出且没有写终态，也会触发核查，避免无人处理失败。临时 SSH 不通只重试连接，不唤醒模型。终态文件尚未写完时继续等待。

若同一会话仍在处理用户消息，续行通过 queue 等待当前轮次结束。已经提交的记录不重复发送；CLI 超时等无法确认投递结果的情况记录为 `delivery_uncertain`，需核查回执与会话队列，避免盲目重复提交。

## 核查与后续行为

续行消息要求 Codex 读取真实终态、分析 Qwen/DDP 结果、按已批准的方案继续验证和修改，并在阶段完成后推送 v6。后续长实验同样由 Codex 配置一个普通完成守候进程，再结束当前轮次；不会用模型反复询问实验进度。

本机必须保持开机和联网，脚本及 Codex 服务可用。此配置不是关闭本机后仍能运行的云调度器。要取消尚未触发的等待，可创建 `.scratch/v6-next-stage-wakeup.stop`；脚本下一次检查即退出。

## 验证范围

- 已通过真实 SSH 查询当前两个控制器，确认仍在运行。
- 已在远端真实 Linux 进程/临时文件上验证六种状态：缺少启动记录、仍在运行、JSON 写入中、包含失败结果的正常终态、控制器异常退出、主控制器完成但后续检查仍在进行。
- 队列适配器的两项本地替身检查验证单次提交和不确定回执记录；没有为测试而额外唤醒模型。
- 完整的真实队列投递与自动续行要等本批到达终态后才能确认。部署时不提前宣称已经完成端到端唤醒验证。

状态阶段为 `waiting` → `queueing` → `queued`，异常投递则为 `delivery_uncertain`。证据见 [部署记录](evidence/v6-next-stage-20261027/wakeup-setup.json)。

官方 CLI 接口：[codex queue](https://learn.chatgpt.com/docs/cli/reference#codex-queue)。
