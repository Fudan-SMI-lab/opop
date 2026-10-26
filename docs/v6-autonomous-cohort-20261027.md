# 同版本三项目原入口自主验证

本批执行[阶段 D](plan-v6-next-stage-20261027.md)，前提是关系检查提示已通过真实 evaluator A/B、双卡负例和原源码一致性验证。脚本为 `scripts/experiments/v6_autonomous_cohort_20261027.py`，目录 `/root/autodl-tmp/dev/v6-work/autonomous-cohort-20261027`；启动记录提供精确提交和源码摘要。

## 固定任务与顺序

| 任务 | 原项目入口 | 冻结工作量与目标 | 轮数 |
|---|---|---|---:|
| Qwen3-4B | qwen_inference.py 与既有本地权重 | interactive：batch 1、256 输入 token、完整生成 32 token；最小 TTFT | 3 |
| ResNet-18 | resnet_inference.py 与既有真实图片/预训练权重 | offline：batch 32；最小完整推理延迟 | 3 |
| GPT-2 DDP | 原 torchrun 双 rank 命令 | 原累积训练步；最小诊断 NCCL kernel 总时长，保留整步约束 | 3 |

从原项目目录、原启动命令、目标和当前设备开始，全部重新执行 intake、evaluator、发现/分组、选择、生成、回装、评测和反馈。不给现成 adapter、算子组、候选或旧人工诊断。三轮用于覆盖两次后续反馈机会，不是 Agent/API 次数、费用或 token 限制。不强制 Agent 换组或采用某类实现；失败与无可接受候选均是实验结果。

全部任务和后续检查串行使用共享 GPU 租约；每个任务分别记录输入、冻结代码、终态和空间变化。启动前数据盘约 25 GiB 可用；模型资产不复制，独立版本检查复用不可变参考，保留已有冻结实验。只在数据盘新增本批结果。

## 单独的 collector planner 真实对照

自主 cohort 前，重放旧 DDP 第三轮 post-evaluation collection planner 的初始快照 `generation-40fef4adc280.json`，按 control/treatment 各一次真实角色会话。仅将 candidate_diagnostics 的能力描述改为与实现一致：候选文件当前线程的异常与 False 返回；不提供正常调用、其他标量返回或由空事件证明成功分支。输入文件、模型和其他提示不变；保留已有自动契约修复。

旧实际回复曾请求不存在的 `candidate_diagnostics.executed_calls`、`candidate_diagnostics.scalar_returns`，并将其视为路径证明，因此修正有真实问题依据。此对照只检查描述与后续计划，不再执行旧候选采集；它的输出、选组和反馈不进入三个自主任务，也不与上一批 evaluator 对照混为一项实验。

## 独立核验与解释范围

三个自主任务结束后，分别检查新 baseline adapter 的原源码一致性。Qwen/ResNet 使用既有源项目完整参考 logits 和精确 token/class；任何新增观察列出但不因对比子集而被声称已经源验证。输出命名不同导致检查不可用时明确记录并人工核查，不将其等同数值错误。DDP 使用原源码 serial global-batch 两次连续更新和完整梯度副本比较，再执行同一关系负例检验新 evaluator。

若项目当前版本包含接受的优化 bundle，再对整个完整版本运行一次新的标准独立 AB/BA，与该任务原基线比较；不发布这个独立复测，不追加到分数通过为止。无接受候选则记录未执行的原因。

报告区分原入口自主链、角色对照、源一致性与独立性能验证；检查多组共存、接受后重新发现与反馈使用，但不预判这些行为必然出现。Qwen 旧固定候选的 80 次采样仍只属于开发诊断，原标准失败保留。当前仍缺少可信通信优化正例，阶段 E 的目标切换和新通信项目待本批证据继续推进。

## 完成后续行

本批由一个控制器串行管理全部阶段；`launch.json` 与 `followup-launch.json` 是同一进程的启动记录，终态写入 `controller-completed.json` 和 `followup-completed.json`，兼容现有普通守候协议。错误也会写入结果；进程异常退出且无终态时由守候触发核查。

配置独立本机状态 `.scratch/v6-cohort-wakeup.json`，每 30 秒通过普通 SSH 检查终态；结束后向当前会话排入一次核查与继续指令。不启用 Goal，不用模型轮询，不添加投递测试。

## 已启动

冻结提交 **`4cdd076376de`**，于 `2026-10-26T22:02:24.751310Z` 启动控制器，PID `63346`。collector planner control 的真实初始会话已建立并保存生成前快照，三个项目按上述顺序排在角色对照之后。

本机守候 PID `22328` 已启动，真实 SSH 检查确认控制器运行，状态为 `waiting`。模型调用只发生在框架实际工作或批次结束后续行，等待由普通脚本完成。启动时数据盘可用 26,295,762,944 bytes（约 24.49 GiB）。[部署与输入证据](evidence/v6-autonomous-cohort-20261027/README.md)已归档；当前不预判自主优化或独立验证结果。
