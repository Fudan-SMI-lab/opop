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
- 部署时先用队列适配器的两项本地替身检查验证单次提交和不确定回执记录，当时没有额外唤醒模型。
- 随后用户明确要求真实投递验证，已使用守候脚本同一个 `queue_continuation()` 函数实际调用 `codex queue`。CLI 退出码为 0，返回消息 ID `01a179dd-f023-7fd3-be76-1718bfceb6b7`，目标为上述当前项目会话。
- 上一轮回复结束后，这条消息已自动进入当前会话的新一轮；新轮次于 `2026-10-26T20:20:06.434749+00:00` 写入确认记录，`source=actual queued Codex turn`。这证明真实消息投递和自动续行均可行，不只是命令行帮助或模拟测试。
- 本次是用户授权的一次测试唤醒，没有启用 Goal、再次排入测试消息或重启实验。实际实验守候进程仍为 `waiting`；当前整批远端完成事件本身尚未发生，不能把该测试当成实验已经完成。

状态阶段为 `waiting` → `queueing` → `queued`，异常投递则为 `delivery_uncertain`。证据见 [部署记录](evidence/v6-next-stage-20261027/wakeup-setup.json)和[真实投递与新轮次确认](evidence/v6-next-stage-20261027/wakeup-live-delivery.json)。部署记录中的“尚未真实投递”是部署当时的状态，由后者补充后续验证结果。

官方 CLI 接口：[codex queue](https://learn.chatgpt.com/docs/cli/reference#codex-queue)。

## 实际实验批次投递已完成

远端 followup 于 `2026-10-26T21:15:11.300364Z` 写入终态。守候进程在同秒成功 queue，消息 ID 为 `01a17a11-3a95-7871-b54e-a1e6c0d3d941`；该消息实际开启了本次核查轮次。已读取 Qwen 独立复测和 DDP 自主运行/源码一致性结果，区分流程结束与优化失败。此次验证了实际批次的完整“结束 → 投递 → 当前会话继续工作”，未启用 Goal，也未靠模型轮询。原守候状态为 `queued`，不复用它等待下一批。

见[真实批次投递证据](evidence/v6-next-stage-20261027/wakeup-batch-delivery.json)。上文部署和一次测试记录描述的是各自当时状态。

## 下一批守候已配置

正在等待 `/root/autodl-tmp/dev/v6-work/post-batch-audit-20261027/followup-completed.json`。该批先完成 Qwen 固定连续采样，再完成真实 evaluator 对照。新状态为 `.scratch/v6-post-batch-wakeup.json`，取消文件为 `.scratch/v6-post-batch-wakeup.stop`，启动 PID `25492`。已真实确认负例控制器完成、后续控制器运行；使用同一已验证的队列发送流程，新的终态只投递一次。见[部署记录](evidence/v6-post-batch-audit-20261027/wakeup-setup.json)。

后续更新：上述批次于 `2026-10-26T21:52:14.871494Z` 完成，守候成功提交消息 `01a17a33-51d1-7bc1-a277-e3bbb293fc0a` 并退出；消息实际启动了下一轮核查，当前结果已审计。该状态文件现为 `queued`，不再用于等待。见[本批真实投递证据](evidence/v6-post-batch-audit-20261027/wakeup-delivery.json)。

## 当前等待：同版本三项目批次

现在由新的普通守候等待 `/root/autodl-tmp/dev/v6-work/autonomous-cohort-20261027/followup-completed.json`，范围含独立 planner 对照、三个项目自主运行和其后检查。单个远端控制器 PID `63346`，本机守候 PID `22328`；两个启动记录明确指向同一个远端控制器。新状态文件 `.scratch/v6-cohort-wakeup.json`，取消文件 `.scratch/v6-cohort-wakeup.stop`。已真实验证远端运行，仍使用同一已实证的单次投递逻辑，未启用 Goal。见[部署证据](evidence/v6-autonomous-cohort-20261027/wakeup-setup.json)。

后续更新：该批于 `2026-10-27T04:41:53.122740Z` 完成，守候于 04:42:09Z 成功提交 `01a17baa-6cd1-7f90-9010-1ea59fd8c22e`，并实际启动本次审计轮次。已经读取原始结果、日志、生成物、快照和独立测量，补做有明确源码依据的 observation 映射审计。原守候状态为 queued，不再等待；未启用 Goal 或额外投递测试。见[实际续行证据](evidence/v6-autonomous-cohort-20261027/wakeup-delivery.json)。

## 当前等待：目标切换与 TP 准备

新的普通守候 PID `65792` 等待 `/root/autodl-tmp/dev/v6-work/goal-switch-20261027/followup-completed.json`；远端单控制器 PID `179345`。状态 `.scratch/v6-goal-switch-wakeup.json`，取消文件 `.scratch/v6-goal-switch-wakeup.stop`。真实 SSH probe 已确认 running，范围包含 intake A/B、TP 基线和 ResNet 目标切换后的完整检查。仍使用已实证的终态单次 queue，不使用 Goal 或模型轮询。见[部署证据](evidence/v6-goal-switch-20261027/wakeup-setup.json)。

后续更新：该批于 `2026-10-27T07:09:12.612180Z` 结束，已真实唤起本轮审计。守候状态 queued，未复用；本轮已区分 TP 依赖错误/后续质量失败、ResNet 冻结接受/源资源状态偏差，并完成真实追加诊断。见[投递回执](evidence/v6-goal-switch-20261027/wakeup-delivery.json)。

## 当前等待：语义与生命周期批次

新普通守候 PID `49404` 等待 `/root/autodl-tmp/dev/v6-work/semantics-lifetime-20261027/followup-completed.json`；远端单控制器 PID `207298`。状态 `.scratch/v6-semantics-lifetime-wakeup.json`，取消文件 `.scratch/v6-semantics-lifetime-wakeup.stop`。真实 SSH probe 确认 running，初始真实 Agent 会话已保存。使用同一终态单次投递机制，未启用 Goal、额外测试消息或模型轮询。见[部署回执](evidence/v6-semantics-lifetime-20261027/wakeup-setup.json)。

后续更新：该批10:31:21 UTC结束，守候10:31:46 UTC提交消息 `01a17cea-8819-7422-ae5a-746c84681cc2`，实际开启本轮审计。已完成角色输入/输出、源数值、目标变体和独立拒绝核查；状态 queued，不复用。见[投递证据](evidence/v6-semantics-lifetime-20261027/wakeup-delivery.json)。

## 当前等待：DDP 原入口语义验证

新的单控制器 PID `240963` 负责 `/root/autodl-tmp/dev/v6-work/ddp-semantics-20261027`，包括新 DDP 三轮、独立源码/关系负例/整版复测及 TP 小型诊断。本机普通守候 PID `20504`，状态 `.scratch/v6-ddp-semantics-wakeup.json`，取消文件 `.scratch/v6-ddp-semantics-wakeup.stop`。已确认 controller running 和真实 intake 会话/初始快照；结束或异常退出后只 queue 一次。等待不调用模型，未启用 Goal 或重复投递测试。见[部署证据](evidence/v6-ddp-semantics-20261027/wakeup-setup.json)。

2026-10-27 用户补充要求：本批结束后做大规模空间清理。已更新同一守候在投递时读取的 prompt 文件，要求先审计本批，再按[清理方案](v6-storage-cleanup-20261027.md)核对本地完整归档/依赖并清理冗余服务器副本，然后继续新实验。更新时仍 waiting，未重启守候、增加 Goal 或提交额外消息；见[追加指令记录](evidence/v6-storage-cleanup-20261027/deferred-cleanup.json)。
