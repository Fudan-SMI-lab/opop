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

## 已启动

冻结提交 `1d6f3a5bd822`，核心源码 SHA 与上一批相同（`ed1f8a63296fa915e6c18737293e94d9fa3d82fd3918958400e5033754156ebd`）；新增实验脚本在冻结快照中，不修改运行中的生产角色提示。2026-10-27 05:09:37 UTC 启动，远端控制器 PID `179345`。DDP intake control 已建立真实会话 `ses_e843c6085ffeSwMIVWdY1hMkug`，初始生成快照已保存。

本机普通守候 PID `65792`，状态 `.scratch/v6-goal-switch-wakeup.json`；真实 SSH probe 已确认运行，终态投递对象仍是当前会话。启动时数据盘可用 11,881,414,656 bytes。见[启动及输入证据](evidence/v6-goal-switch-20261027/README.md)。

启动前完成脚本编译、远端实际模块导入、原始快照锚点/核心代码一致性检查，并用旧 ResNet 固定独立端点实际执行交叉指标计算。旧延迟版本的 warm-run allocated peak 为 281,975,808 → 132,312,064 bytes；这只是原端点的交叉诊断，需结合 lifecycle/retained memory 才能解释，未改变原目标或发布新结果。新显存目标自主搜索尚未完成，不能据此声称已经验证目标驱动差异。

## 终态审计

批次于 `2026-10-27T07:09:12.612180Z` 结束并触发当前会话。intake、ResNet 搜索及复测执行完毕；TP 阶段为 completed_with_errors，不能统称全部验证成功。生产 intake/evaluator 提示在本批仍未改动。

### Intake A/B：有改善，但不采用当前段落

实际两臂快照的 framework、model、tool settings 和全部输入文件相同，prompt 仅增加协议中的通用段落；control/treatment 均 ready、零契约修复，真实 session 分别为 `ses_e843c6085ffeSwMIVWdY1hMkug`、`ses_e8439eff8ffeYHQCrMtqbrleAK`。

两臂均保留 world_size=2、micro_batch=1、sequence_length=128、accumulation_steps=4、有效 batch=8、完整 SGD 更新、原梯度/参数容差、精确跨 rank 梯度要求、整步 1.05x 约束；通信目标及整步约束均使用 ranks_then_samples。treatment 改用了源指标名并映射至同一 collector field，补入源模型配置，communication 为 required，未改变测量定义。

但 treatment 的改善主要位于顶层 assumptions：将 bucket、foreach 等称为可替换的 baseline implementation。`resolve_request()` 只把 description.notes 和 invocation 送入 request，顶层 assumptions 不进入 evaluator 输入。实际冻结 notes 仍有 baseline DDP 配置，invocation 仍明确指定前三次 no_sync/最后一次 allreduce；另有 reset 是否属于计时单位的矛盾。后续角色不能可靠获得 treatment 的可替换说明。因此不将此段落直接合入生产，也不声称已经解除上游错误约束。下一次只改说明，明确后续接收字段及在这些字段内区分 baseline 描述与任务要求，不转发全部上下文。

### ResNet：两个 bundle 共存，但新增显存基线缺陷

三轮真实顺序为：

| 轮次 | 范围/执行 | 搜索结果 |
|---|---|---|
| `trial-e9279012c6` | stem region，保持完整 32 行，按 8 行切块执行原子调用 | 280,169,472 → 203,099,136 bytes，接受 |
| `trial-fd9bdc62a2` | stage-0 region，保留 stem bundle，再加入第二个 bundle | 203,099,136 → 177,409,024 bytes，接受 |
| `trial-2ffc01d1ba` | 尝试替换 stem bundle，stage-0 保留；最终参数仍为 rows=8 | 177,409,024 → 177,409,024 bytes，无收益，拒绝 |

第二、三轮快照含实际反馈，proposal 引用了前轮接受、内层参数筛选的吞吐失败及剩余峰值假设。最终完整版本确有两个 active bundle，独立 AB/BA 中两者的身份、覆盖和质量均通过。原生 trace 157 个 kernel、无 CUDA Graph；代码逐块执行全部输入，无输入去重或缓存答案。此项补上了“多个独立 bundle 同时共存并完成独立验证”的流程证据。

冻结 adapter 内的独立完整版本测量为 280,169,472 → 177,409,024 bytes（下降 **36.68%**），两端点吞吐 8571.85 / 7482.32 images/s，均高于最低要求 7158.88；该版本判定 accepted，publication=exploration。此记录保持原样。

**不能直接认定为原项目的正式显存收益。** 初始源码审计因 `top1` 与实际 `pred_classes` 命名不同返回 false。人工核对 observe() 后，两次完整 32×1000 logits 和 32 类别与原源码参考零误差，证明数值一致。但进一步检查发现，新 prepare() 只保存 pixels，没有保存原脚本在计时期间仍持有的 images（3×3×224×224 FP32，1,806,336 bytes）。旧延迟 adapter 正确保留了该对象。

已另建 `goal-switch-audit-20261027` 执行两项独立诊断：原脚本未经修改的完整入口运行成功，offline peak 为 282,273,280 bytes；只在新 adapter 返回前增加 `state['images']=images` 的因果对照通过完整 controls，peak 从 280,169,472 升至 **281,975,808 bytes**，差值恰为 1,806,336。完整原脚本还先执行 online 行，其分配轨迹与只运行 offline 的 adapter 不同，剩余差值不据此强行归因。手工补丁仅用于诊断，未替换自主 adapter、原判定或冻结结果。

另外新 understanding 将“参数值不变”作为 invariant，selection 据此排除了 BN folding，而旧延迟搜索采用了这种等价变换。因此两条原入口运行体现了不同目标下的搜索行为，但并非只改变目标的严格对照，不能把策略差异全部归因于目标。下一步以已经核实保留源图片张量的旧 adapter 做显式 goal variant，同时对本次 evaluator 的真实生成前快照做一项最小生命周期说明对照。

### 交叉指标与显存解释

| 独立版本 | warm-run allocated peak | 生命周期 allocated peak | 生命周期 reserved peak | 交叉延迟判断 |
|---|---:|---:|---:|---|
| 旧延迟版本 | 281,975,808 → 132,312,064 | 282,392,064 → 313,063,424 | 398,458,880 → 572,522,496 | 原延迟独立接受保留 |
| 新显存版本 | 280,169,472 → 177,409,024 | 280,425,472 → 280,297,472 / 280,553,472 | 396,361,728 → 406,847,488 | 4.243588 → 4.004949 ms，但一对无改善，不能接受为延迟加速 |

旧 graph 版本的运行 allocated counter 更低，但生命周期峰值和 reserved 增加；新版本未迁移分配到 graph setup，主要降低运行中临时张量峰值，生命周期峰值没有实质下降。均不能笼统称为物理显存下降。交叉计算保留所有样本、原目标判定和独立合同；数值及测量生命周期问题另行列明。

### TP：依赖错误已区分，真实双卡仍未通过质量

首批单卡成功：两次完整 logits/token 与旧参考零误差，完整 32 token 延迟中位数 699.336 ms、TTFT 24.085 ms；profile 中无 NCCL。`_tp_plan` 属性在单卡也有默认定义，不能凭此证明 TP 执行。双卡最初在加载模型前因缺少 accelerate 退出，未产生质量/性能观测。

此次在实验独立目录安装 accelerate 1.15.0、psutil 7.2.2，保存 wheel SHA，只通过新 worker 的 PYTHONPATH 使用；未升级或修改原 venv。用同一冻结 worker、同一工作量/容差重新执行 TP=2，完整两 rank 均成功运行，但**质量失败**：两次全部 32 token 相同，全量 logits 有 1,567,446 / 4,861,952 个元素（32.2%）不满足原容差，最大绝对差 3.0。不得放宽容差或把 token 一致代替完整质量。

该失败运行的完整延迟中位数约 1.622 s、TTFT 约 53.6 ms；每 rank 的完整 profile 有 2,368 个 NCCL kernel（2,336 个 CPU allreduce event，另有 AllGather），原生名称为 RING_LL；rank0/1 的 summed NCCL kernel duration 分别约 625.79 / 1707.22 ms。这证明实际分布式通信存在，不证明线速、传输字节或暴露 critical path，且两个 rank 分别受到 profiling 扰动，不能与 clean 时间直接作比例分解。它仍是未通过质量的开发基线，尚未成为可优化项目或通信正例。后续先定位首个已观测数值差异及权重分片一致性，不预判是 BF16 归约顺序还是实现问题。

本批 ResNet 14 个真实角色会话，provider 报告成本 6.02190807764（原 provider 单位；不含 intake A/B），仅作统计。原有 Qwen 独立拒绝和全部 80 个诊断样本未覆盖；磁盘仍约 10 GiB 可用，无删除冻结证据。
