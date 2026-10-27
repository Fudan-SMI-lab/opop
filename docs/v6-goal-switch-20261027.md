# 目标切换与通信案例准备

依据[上一批审计](v6-autonomous-cohort-20261027.md)，Qwen TTFT 与 ResNet 延迟的新自主版本通过独立复测，DDP 仍因整步约束失败。执行[阶段 C/E](plan-v6-next-stage-20261027.md)，新目录为 `/root/autodl-tmp/dev/v6-work/goal-switch-20261027`，全部保留原批次。

## 固定顺序

1. **DDP intake 真实单段对照。** 使用上一批 `generation-4b715484e927.json` 初始快照，各一次 control/treatment 新会话，只有新增通用语义/实现区分段落。输入文件、模型、框架和工具配置不变，自动契约修复仍可用。检查生成 request 是否仍冻结 bucket/no_sync 等实现选择，同时没有遗失 effective batch、完整优化器更新、质量与指标聚合定义。本轮不把 treatment 自动合入生产提示，也不将其输出交给下一项任务。
2. **官方 Qwen TP=2 就绪检查。** 复用原 Qwen3-4B 本地权重、interactive 的 B1/S256/完整 32 token、BF16/SDPA 和原 `infer()`；使用 Transformers 5.16.1 官方 `tp_plan=auto`，不改框架绑定或候选后端。先单卡，再双卡；每种两次完整质量检查、两次 warmup、五次 clean sample、另一次完整 profile。逐 rank 对比既有原源码完整 logits 和精确 token；不放宽容差。记录实际 TP plan、NCCL 数量/总 kernel 时长与 clean TTFT/decode/完整延迟。诊断 NCCL 时间仍含等待，不能解释成链路流量或 critical path；这些是开发者基线，不是自主优化或正式加速接受。若源一致性失败，保留失败，不进入该案例优化。
3. **ResNet 原入口显存目标。** 与前批相同离线 batch 32、真实图片/预训练权重、FP32 和完整输出，目标改为源项目 `peak_allocated_bytes` 最小，吞吐不低于原 baseline 的 95%。重新 intake/evaluator/发现/选择/生成/回装/反馈，三轮；不给旧 adapter、选组、候选或本轮开发诊断。检查源一致性；若有接受版本，再做一次完整版本独立 AB/BA，不追加到通过为止。
4. **交叉指标评价。** 对旧延迟版本和新显存版本各自已固定的独立端点，按框架统一 elapsed/peak-allocated 指标以及原吞吐约束重新计算交叉目标，所有样本保留。同时列出 lifecycle allocated/reserved，防止把移入 graph setup 的分配描述成一般物理显存节约。这是回顾性诊断，不更改旧接受结果，也不把不同 evaluator/anchor 的 bundle 强行合并；实际质量、边界及指标可比性仍需完成后审计。

## 执行边界

脚本为 `scripts/experiments/v6_goal_switch_20261027.py` 与 `v6_tp_qwen_readiness_20261027.py`。本批不修改生产提示，不增加基于失败类型的调度规则。三轮是实验协议，不限制 Agent 会话请求、tokens 或费用。GPU 任务使用已有租约串行执行。

数据盘余量约 11 GiB；本批先做较小 ResNet 和一次 TP 基线，不再同时新建多个大 Qwen 搜索。新冻结快照只存运行所需源码、scripts、examples 和项目配置，Git 提交与核心 SHA 仍完整记录；不复制已有文档证据历史。旧 trace、参考、实验目录不删除或迁移。

完成后普通守候只向当前会话投递一次真实核查/继续指令，覆盖正常结束或控制器异常；等待不调用模型，未启用 Goal，不新增投递测试。具体启动/守候回执随启动记录归档。
