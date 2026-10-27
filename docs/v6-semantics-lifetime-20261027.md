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

## 终态与审计结论

批次于 2026-10-27 10:31:21 UTC 完成，并由普通守候实际唤起本轮。六个阶段均执行完成；这不等于六项验证均成功。以下结论来自实际快照、生成源码、controls、全量数值和完整版本测量，证据见[目录](evidence/v6-semantics-lifetime-20261027/README.md)。

### DDP intake：采用第二版单段说明

两臂均为真实新会话，没有修复。control `ses_e83b2606effeaawsFiIi3QsUv8`，treatment `ses_e83aed317ffeIi7pwYPAuG76zF`。输入比较确认框架、模型、工具设置一致，文件输入差异为空，唯一变化是 `INTAKE_NOTE` 一段提示。

control 仍在 invocation 要求前三次 no_sync、第四次 all-reduce。treatment 将 no_sync、bucket_cap_mb=25、gradient_as_bucket_view、broadcast_buffers 明确标为 baseline 实现，允许语义等价同步方式，而且说明确实进入 resolved request，不再只留于 assumptions。两者均保留 world=2、每 rank microbatch=1、seq=128、累积4次、全局8样本/1024 tokens、loss 除4、完整 SGD 更新、源全局 batch 的全量梯度/参数比较、精确 rank 副本一致、有限 loss 及负例要求。reset 在计时外；质量 atol=1e-6/rtol=1e-4，通信目标与整步上限均使用 max ranks_then_samples，整步≤原基线1.05倍。

treatment 新增的 source_metric_key/source_definition/profiled_invocations 描述未改变已安装采集器定义。其保留的其他源实现描述仍须在下游核查，不能把这一对照推广为所有 intake 问题已解决。

据此采用与对照逐字相同的段落到生产 intake prompt；不改 schema、路由或旧 request 输入路径。随后从原入口重新做 DDP 三轮自主实验，检验下游合同和实际候选，而不把角色级对照视为端到端成功。

### 生命周期：两臂均正确，暂不增加生产提示

control `ses_e83ab87b7ffeJRP0vLFID0zEZC`，treatment `ses_e83a5dc02ffelBQQDTfnOImFrv`，输入同样只有一段提示差异、无修复。两臂真实 controls 均通过，peak_allocated_bytes 均为 **281,975,808**。

代码审查确认 prepare 返回的 state 都持有 GPU images `[3,3,224,224]`（1,806,336 bytes）和 pixels `[32,3,224,224]`（19,267,584 bytes），并活过 run。返回状态 inventory 只能提供这些正面证据，不能排除任意对象/global 内的其他所有者。两臂 logits/predicted_classes 对原项目完整 reference 的两个连续步骤均为零误差，每步32,032元素；额外状态标志还依赖代码核查。

control 在 run 内额外同步，treatment 没有；框架本身也在边界同步。这是实际输出差异，但不能据此宣称生命周期提示更稳定。原生产提示已包含生命周期要求，本次 control 也保留了 images，因此暂不采用新增段落。

### ResNet 同 adapter 目标变体：搜索接受，独立拒绝

这是明确的 goal variant，非新的原入口 evaluator cohort。复用 latency baseline 的 adapter SHA `3ca44406fa2b8bba7d0308de68366d37182c2aca54cc6c45f7ad9706abeab50d`，保留 images；真实合同适配会话 `ses_e83a04020ffe6U56Ua5yhqCDFT`。工作量、源文件、资产、质量保持一致，目标改为峰值显存并要求吞吐≥原基线0.95倍，重新 controls 和基线。

通用源码审计最初因 top1/predicted_top1 字段不同返回需人工核查；验证 adapter SHA 未变后，用先前已审查的映射比较，全部数值零误差。原始字段失败没有被覆盖。

| 轮次 | 自主范围和回装 | 搜索峰值 bytes | 搜索判定 |
|---|---|---:|---|
| 1 | stem 分块，单 bundle | 281,975,808→204,905,472 | 接受 |
| 2 | stage0/1 分块，与第一轮 bundle 共存 | 204,905,472→155,229,184 | 接受 |
| 3 | 外层 model.resnet 区域流式处理，替换前两 bundle | 155,229,184→121,412,608 | 接受 |

完整32行输入均保留，反馈快照和后续提案明确消费先前结果；接受后重新发现与两 bundle 共存有证据。最终候选调用原 aten 运算，无新增 Triton/CUDA graph；原生 trace 有241个 kernel、0个 graph kernel。最终搜索端点延迟约4.273/4.276 ms，吞吐7488.5/7483.6，满足当时约束。

随后标准完整版本 AB/BA 质量通过、覆盖无错误，显存仍为281,975,808→121,412,608（计数下降56.94%），但候选吞吐 **6619.16/6266.94 < 7120.31** 下限，被 `original_baseline_constraint_failed` 拒绝。候选端点4.834/5.106 ms，原版4.279/4.250 ms；是同一候选、参数和实际路径。生命周期 allocated 为282,392,064→122,245,120，reserved 为398,458,880→161,480,704；没有 graph pool 转移，但也不能忽略吞吐失败而声称目标达成。

本次13个真实角色会话（合同1、采集规划6、选择3、候选3），provider 报告费用8.06780447852，仅统计，未限额。

追加一次预先固定 B/C/C/B 诊断，每端点20次，完整80样本与质量全部保留。只在诊断请求内将 repeats 从5改20，不改冻结 request 或候选，不重新发布。四个端点按每5样本分组的延迟中位数（ms）：

| 端点 | 第1组 | 第2组 | 第3组 | 第4组 |
|---|---:|---:|---:|---:|
| B | 4.236 | 4.243 | 4.236 | 4.249 |
| C | 4.349 | 4.366 | 4.319 | 4.321 |
| C | 4.871 | 4.860 | 4.225 | 4.222 |
| B | 4.246 | 4.249 | 4.258 | 4.242 |

候选存在进程内时序变化；端点前后 GPU/CPU 环境快照不能定位发生变化时的原因。当前不能断言是频率、调度、算法缓存或绑定开销，也不据此推翻独立拒绝。后续如继续诊断，应先取得能与逐次调用对齐的一项观测，而不是反复测到通过或向 Agent 倾倒更多环境信息。

上一版不同 memory adapter 的36.68%冻结接受及1,806,336-byte缺失 images 事实各自保留，本批没有覆盖它们。

### TP：权重匹配，数值差异出现在 embedding 之后

单 GPU prefill 对原始第一步 logits 为零误差。TP=2 每 rank 的398个命名参数均找到与 checkpoint 逐元素相等的复制或连续 rank 分片；embedding 完全相同，首个采样到差异的边界是 layer0 输出（max_abs=.0625，RMS约.00104）。这不是宣称 layer0 输出就是首次执行分歧点，只是现有采样边界。

随后误差传播，layer35 max_abs=32，final norm=5.5；首步 logits max_abs=.4375，29,513/151,936元素不满足原容差，首 token 仍相同。中间误差仅为描述，未套用 logits 门槛。分片检查排除了此次测试布局下的字节不一致，不把无法匹配的布局假设当错权重。

先前完整32-token TP=2 的 logits max_abs=3.0 失败仍成立。本批 prefill 不能建立完整工作量验收，也没有优化吞吐/通信的正例。下一步只检查第一 block 的内部边界及同输入投影的分片舍入影响，不改变生产精度或质量约束。

### 验证与空间

box4-3 的 intake 相关6项检查通过，并核对生产段落与真实 treatment 逐字相同。本机先缺 PYTHONPATH，补上后仍缺 optuna，故在已有依赖的主开发环境完成检查。旧 request 路径未变，未运行无关旧回归。

数据盘实际 `df -B1` 可用10,875,371,520 bytes（约10.13 GiB）。对部分嵌套实验路径，`shutil.disk_usage` 却返回底层文件系统约703GB；对挂载根又返回正确额度，因此不能单凭该 API 判断容器剩余额度。旧协议里的值保留为原始记录。后续批次同时记录路径、df 与 statvfs，按 df 安排一次 DDP 和小型 TP 诊断，不删除冻结证据、不并发大型 trace。
