# 上游语义与资源生命周期的最小对照

依据[目标切换审计](v6-goal-switch-20261027.md)，不采用尚未解决问题的 intake 提示，不把新 ResNet adapter 内的接受当作源项目显存收益，也不将质量失败的 TP 基线用于优化。新批目录 `/root/autodl-tmp/dev/v6-work/semantics-lifetime-20261027`，既有结果全部保留。

## 固定执行顺序

1. **Intake 第二版 A/B。** 从同一旧 DDP 初始生成前快照，各一次 control/treatment 真实新会话，唯一差异为一段通用说明：后续角色接收 description，而非顶层 assumptions；在实际传递的 notes/invocation 中区分 baseline 实现与必要语义，明确计时内外边界。没有 DDP 特定术语规则，没有转发全部 assumptions。逐字保存输入和输出，保留自动契约修复；检查完整工作量、质量、关系检查、聚合顺序，以及是否仍误冻实现。
2. **Evaluator 生命周期 A/B。** 从新 ResNet memory evaluator 的真实初始快照，各一次新会话，只追加“在 state_protocol 给出源计时对象到 adapter 所有者的简短映射，并检查对象确实活过 prepare/run”的通用段落。两臂分别执行真实 controls，再另开 worker 记录返回 state 的嵌套张量形状/字节数。该 inventory 不遍历任意对象或 globals，不能因缺少记录就判所有权不存在；完成后仍要阅读生成代码、核实 observation 映射和原源码数值。此试验不向自主搜索提供诊断。
3. **TP prefill/权重诊断。** 同一源 interactive 输入和 BF16/SDPA；单卡记录 embedding、每个 decoder block、final norm、lm_head 的完整 prefill 输出，双卡比较相同边界（shape 不同则标记不可直接比较）。另对双卡实际参数与 checkpoint 的完整/兼容连续 rank 分片逐元素比较。隐藏层误差只描述数值变化，不把 logits 容差套作隐藏层质量门槛；无法匹配的分片布局不是自动判定权重错误。仅一次 prefill，不冒充完整 32 token 质量验证或性能实验。使用上批隔离依赖 overlay，原 venv 不变。
4. **冻结 evaluator 的 ResNet 目标变体。** 从前批已审计的 resnet-latency baseline evaluator 开始，保留代码、source、工作量、资产与质量；改为 peak_allocated_bytes 最小、吞吐 ≥0.95x 原基线，并替换旧 notes 中冲突的目标说明。使用已有 `--reuse-evaluator`，真实 Agent 只适配合同，重新 controls 和基线，再三轮自主选择/生成/评测/反馈。不给旧优化 bundle 或开发诊断。该实验明确是 goal variant，不称为重新从原入口生成 evaluator 的 cohort。
5. 有接受版本则仅一次标准完整 AB/BA；核实 adapter SHA 未变后，复用已人工检查的 predicted_top1 源映射做全量质量审计。交叉指标保留原样本及 lifecycle memory，不跨合同强行导入 bundle，不覆盖旧接受/拒绝。

生产 intake/evaluator 提示均保持不变，是否采用新段落由真实结果决定。脚本 `scripts/experiments/v6_semantics_lifetime_20261027.py`、`v6_tp_prefill_audit_20261027.py`。固定三轮是实验协议，不限制 Agent 请求、tokens 或费用；GPU 工作串行共享租约。新冻结只复制运行源码，数据盘余量在启动/完成时记录；不删除冻结证据。

## 验证与续行

已完成 Python 编译、远端真实模块导入、原快照 prompt 锚点与核心源码一致性检查；GPU 和真实角色验证属于本批实际运行。完成后由一个普通守候排入当前会话的一次核查/继续消息，不使用 Goal、模型轮询或额外投递测试。启动回执另行归档，不预判任何一臂或候选成功。

## 已启动

冻结提交 `6cd5c80f637b`，核心 SHA 仍为 `ed1f8a63296fa915e6c18737293e94d9fa3d82fd3918958400e5033754156ebd`。2026-10-27 07:40:21 UTC 启动远端控制器 PID `207298`，启动时数据盘可用 11,010,150,400 bytes。Intake control 的真实会话为 `ses_e83b2606effeaawsFiIi3QsUv8`，生成前快照已保存。

本机普通守候 PID `49404`，状态 `.scratch/v6-semantics-lifetime-wakeup.json`；已真实验证控制器运行，覆盖完整串行批次。新实验没有改动旧冻结目录或生产提示，见[启动证据](evidence/v6-semantics-lifetime-20261027/README.md)。
