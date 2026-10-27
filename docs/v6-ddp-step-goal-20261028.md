# DDP整步目标变体与证据索引对照

## 终态审计（2026-10-28）

21:33:15 UTC正常完成并自动唤醒。核查时无GPU计算进程，实际150GiB容器数据盘剩45,704,740,864 bytes，约42.57GiB。审计脚本`v6_audit_step_goal_20261028.py`重算独立配对判据、核对真实快照与provider工具记录；[终态证据](evidence/v6-ddp-step-goal-20261028/terminal-audit.json)约123KB，原始模型/trace/参考仍在服务器。

**本批整步目标的完整版本独立评测通过，耗时降低55.73%。** 这属于复用已审计baseline adapter的目标变体；不是新原入口evaluator cohort，也不是已证明的通信链路加速。

|三轮搜索|结果|实际反馈与执行|
|---|---|---|
|trial-0ee027f2e6|拒绝：无有效参数候选|CUDA graph捕获失败，退到更慢的整步实现；3个screening配置均未满足整步上限|
|trial-592873c70d|拒绝：约束/噪声/收益|尝试修复mask与fallback，最终仍有捕获失败；独立搜索配对116.05/119.05ms，对应parent114.12/101.29ms|
|trial-08c5dfa5f6|搜索接受|弃用graph/额外communicator/flat buffer，把本rank当前4份输入沿batch维合并，逐原微批次计算loss并除4后求和，一次backward/DDP reducer和一次原SGD更新|

最终候选代码读取每次调用的当前输入，没有读取参考文件、改TF32/精度、跳过层或按measure/quality选择不同计算。zero_grad、cat、forward、loss、backward、SGD及CUDA同步均在替换的整步内。保留4个微批次的样本及权重、每rank分片、总8样本/1024tokens；计算执行被批量化，不是换一个更小workload。静态代码中异常后的parent fallback不保证任意部分更新可回滚，这是一项候选局限；本次质量端点未触发fallback，仍需独立trace确认实际运行路径，不能只凭候选自报route。

原合同中的工作量、FP32/noTF32/eager、两连续步、完整梯度/参数、精确rank关系、invocation生命周期、rank先max再median及1.05倍baseline约束均保留。baseline adapter实际SHA为`a4056384…`，新controls全部通过。真实`variant_contract`只适配目标合同，未生成adapter。最终只有一个新bundle，无旧通信候选导入；三轮结束在首次接受处，本批没有接受后重新发现或多bundle共存证据。

完整版本独立AB/BA：B为113.926/114.807ms，C为51.540/49.715ms，均值参考114.3668→50.6274ms；两对收益62.386/65.093ms，均大于噪声界4.089/1.838ms；候选均低于新baseline锚定上限117.6254ms。全部四端点、每rank五个样本保留。两连续步全量检查每rank约2.489亿元素通过；候选对baseline参考最大误差约1.19e-7/1.68e-7。覆盖无fallback、reference调用计数为空，实际声明`batched_eager_step`每rank两次；这仍是质量阶段遥测，不等于原生执行路径证明。

独立原项目serial global-batch两步轨迹与baseline controls的完整梯度/参数/loss比较通过，精确rank梯度/参数关系通过。该源审计没有直接将新候选输出对独立serial模型再比较；新候选由冻结evaluator与已审计baseline参考作全量比较，两个层次分别保留。实际rank1约2.500019e-7的容差内梯度扰动被observe精确关系断言拒绝，rank0无扰动；不是因普通容差比较失败而拒绝。

### 真实审阅对照

初始快照逐字核对：control不变，treatment仅`inputs/ddp-index.json`由4,551增至7,751 bytes，其他文件、prompt/model/tools/core相同，原端点值全部保留。control后续有一次输出schema修复（缺少顶层findings），treatment无修复，因此不能把最终回复差异全部归因于唯一初始改动。[control](evidence/v6-ddp-step-goal-20261028/review-control.json)和[treatment](evidence/v6-ddp-step-goal-20261028/review-treatment.json)保留，估算费用分别0.23084993464/0.17263070314，仅统计。

只读核查两个真实会话的21/15次工具记录：两者都未读取GC旁文件；treatment读到了新索引，但只读1-S receipt，并未进一步核对P/Q事件。它认识到空列表不等于没有GC，且承认计时外分配可能改变GC调度；然而仍错误建议高置信采纳、把非零内存值当成native backend证明，未用已有P/Q证据审查实际影响。control还误称该固定顺序“没有交错”、断言计时外改动不可能产生长尾。**单独增加路径索引没有解决主动取证问题；不采纳任何回答，不修改生产prompt或allocator逻辑，也不继续重复审阅直到通过。**

### 信息与诊断边界

轮2/3的选择Agent实际获得此前反馈文件1.72MB/30.46MB，最大部分是resources.groups；三轮理由明确引用前次失败、实际route和新采集证据，说明确有反馈消费。文件大小不等于全部进入模型context，当前没有证据支持一次性重构或扩充上下文；这一体积问题单独记录。大文件未下载到本地。

通信交叉评分在旧通信版本上仍保留原拒绝；新整步版本的通信评分为unavailable。原因已定位：evaluator的summarize仅对`objective.source`执行独立metric采集，整步目标没有source；`required_observations`负责发现阶段能力要求，并不会自动给每个evaluate端点增加数值。因而新请求/合同写了保留通信诊断，实际正式端点中却没有该字段。不得填0或从旧trace推算。下一步用现有显式collector接口单独补采冻结版本，保留原整步接受结果；再决定是否需要通用多指标采集接口。

后续按主计划D/E：补充固定B/C原生路径与B/C/C/B通信诊断，然后从原项目入口做全新DDP整步目标三轮，不提供现成adapter、候选或开发诊断。新adapter需独立源一致性核对，字段命名变化单列待人工审查，不能自动当数值失败或通过。旧DDP独立拒绝、TP完整32token失败和所有历史样本/结果不变。

---

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
