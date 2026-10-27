# DDP整步目标变体与证据索引对照

依据[allocator审计](v6-memory-snapshot-audit-20261028.md)及主计划C/E开展。当前生产evaluator、提示、GC、计时及接受规则均不改。快照改动不采纳；不再重测旧固定候选以寻求通过。

## 固定实验协议

1. 从`memory-snapshot-validation-20261028`的真实`measurement_review`生成前快照`generation-1697496b45a9`做两个新会话。control输入不变；treatment只修改`inputs/ddp-index.json`，由4,551增至7,751 bytes，增加已经存在的八个GC旁文件的路径、SHA和格式说明。八端点全部原始数值保留，其他输入、prompt、model/tool设置和核心源码相同；不提供人工GC结论。计时格式说明包含warmup索引、metrics入口未测量延迟、同机时钟核对及观察器影响。对照仅检验证据发现/使用，不能自动变成生产提示。
2. 以`ddp-semantics-20261027/ddp-communication`已审计baseline adapter创建明确的整步目标变体。只改变goal/objective、对应目标说明、当前设备放置0/1；原源码、质量、world_size=2、每rank microbatch=1、seq128、累积4、总8样本/1024tokens、FP32/TF32关闭、两连续步和测量生命周期保持。主目标为对齐rank取max后再取样本median的`step_ms`；仍保留原`step_ms <= 1.05×baseline`约束，通信kernel sum继续单独采集。基线在新任务内重测，不用历史分数。
3. 使用现有`--reuse-evaluator`路径，通过真实`variant_contract`生成新目标合同和全新controls，然后按原定三轮真实Agent搜索。复用的是baseline revision，不导入旧通信候选、开发诊断或review回答。本实验是同adapter目标变体，不冒称从原入口重新生成evaluator的cohort。
4. 核对adapter SHA仍为`a4056384…`，运行此前人工核对的独立源映射：原项目serial global batch的两步完整梯度/参数/loss与精确rank关系。做真实容差内梯度差异负例。若有搜索接受版本，再做完整独立AB/BA；检查质量、覆盖、约束与噪声，`publish=False`。没有接受版本则如实记录；不增加直到通过的重复。
5. 对新/旧独立完整版本结果做整步和通信两指标的交叉诊断。各自保留其测量时期和基线，不覆盖原判定，不跨合同导入bundle。NCCL kernel sum不解释为wire time或暴露的关键路径。

CPU预检已通过：新请求模型有效，仅四个预定字段改变；所有原端点数值逐项保留，GC旁文件分布为0/0/2/2/2/2/0/0。空列表表示该端点没启用这个观察器，不是没有GC。GPU结果、真实审阅质量和目标选择差异仍待实测。

## 执行与核查

脚本`v6_ddp_step_goal_20261028.py`，工作目录`/root/autodl-tmp/dev/v6-work/ddp-step-goal-20261028`。先串行完成两次review，再执行新目标三轮和独立检查；review资料不传给搜索。每个stage单独记录错误，终态completion不等于优化成功。初始实际容器剩约47.53 GiB，足以开始本小批；不启动大型Qwen trace，原始数据留服务器，只归档代码、索引和小型摘要。没有API预算或限额。

旧DDP独立拒绝、TP完整32token失败、Qwen/ResNet拒绝和所有诊断样本、另一memory adapter的36.68%冻结接受及missing images事实不变。只启用一个新的普通终态监听，不开Goal，不做投递测试或模型轮询；完成后核对两次review实际快照、合同保留的完整语义、三轮反馈/实际候选路径、源一致性、关系负例和完整独立结果。

## 实际启动记录

2026-10-27 19:27:44 UTC冻结`c886e5455283db7f26ea098f5d0f83d39f7ea9f7`启动，controller PID323353；核心源码SHA仍为`cb4ea28c…`，没有采纳快照evaluator。容器df可用51,024,904,192 bytes（47.52 GiB）。[launch](evidence/v6-ddp-step-goal-20261028/launch.json)、[总协议](evidence/v6-ddp-step-goal-20261028/protocol.json)、[角色对照协议](evidence/v6-ddp-step-goal-20261028/review-ab-protocol.json)已留档。

启动检查时处于`review-ab`，control真实会话`ses_e812ab2caffeXDBYydAYLpJavC`，生成快照`generation-9eff10fb641a`，初始输入仍16,725 bytes。此时尚不能报告审阅结果或新目标收益。新普通监听PID63980、状态waiting，两份launch均指向同一存活controller；[监听设置](evidence/v6-ddp-step-goal-20261028/wakeup-setup.json)已留档。旧快照验证监听已完成本轮投递；没有重启旧实验或旧转存监听。
