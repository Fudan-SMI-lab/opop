# DDP 长尾与审阅输入呈现的最小对照

承接[固定DDP/TP批次审计](v6-ddp-tp-diagnostics-20261028.md)。上一批完整执行，未做新的正式接受；C/N共7个约445–519 ms长尾集中在第8/13次附近，关闭Gloo barrier没有消除它们。GC与逐次CPU限流尚无直接证据，因此先做本批诊断，不修改生产执行逻辑。

## 固定 DDP GC 对照

所有端点使用同一个C版本 `75e28d3d2ccd5fab06175ef1b827a325ed2159d011758c7b2deccda6173b9e2d`、相同adapter、参考、工作量、参数、质量与整步约束。顺序 **U/O/D/D/O/U**，每端点20样本、两次warmup。U调用原evaluator；O增加GC回调及计时外CPU计数读取；D与O只差在measure阶段关闭循环GC，离开该阶段恢复。这里measure包括warmup、reset和测量样本，D会改变对象生命期/资源使用，明确是开发干预，不能作为候选收益或从旧分数扣除。

GC回调记录start/stop、generation、线程、单调时钟；generation2另保存少量触发栈位置，不操作张量、不加CUDA同步或强制GC。计时外记录进程CPU、缺页、切换和容器CPU限流计数。事件包围范围比真正计时略宽，容器计数也并非某个worker独占；分别保留这些局限。观察器本身会产生Python对象与计数读取开销，U用于检查其影响。原始事件与所有样本完整保留；不按结果追加试次，不调整原约束，不把无GC的测量解释成正式优化。

GPU作业串行执行，直接复用原参考，不复制多GB控制文件。错误/质量失败完整写入receipt。样本级GC重叠与限流变化只是定位证据，需要结合干预、触发位置及观察器影响判断原因。

## 真实审阅角色单文件对照

从真实初始快照 `generation-d880018dc713` 分别创建control和treatment新会话。保持原prompt、模型、工具配置、生产核心和其它9个输入文件完全一致；只将 `inputs/tp-summary.json` 从94,970 bytes改为30,049 bytes，比较对象作为显式键、数值作为表格行。

全部6组projection、全部变体对三个reference的status/exact/elements/max/RMS逐项验证不变；保留权重/输入映射和原始观测/输出路径，移除重复output metadata等展开信息，可回查原始索引。**不加入人工纠正答案、GC新结果或新的因果提示。** 两臂实际生成前快照再次核对唯一文件差异。外部文件系统/provider状态并未完整冻结，一次对照不代表稳定普遍收益。无论角色回答如何，本批不自动修改生产prompt或采集器。

## 准备与执行

脚本 `v6_ddp_gc_probe_20261028.py`、`v6_gc_review_batch_20261028.py`。CPU隔离检查已验证回调捕获generation2、返回值不变、GC启用状态/loader/回调列表恢复；TP呈现转换逐项数值相等。该检查不代表GPU质量或长尾因果成立。

工作目录 `/root/autodl-tmp/dev/v6-work/gc-review-diagnostics-20261028`。源代码Git冻结，使用现有真实provider配置，API/费用只统计、不限额。启动前数据盘约48.91 GiB可用。本批不启动大型TP重测，不大体积转存本地；原始trace/张量/样本留服务器，GitHub仅存代码、协议和小型证据。

旧DDP独立拒绝、TP完整32-token失败、旧Qwen拒绝及80样本、ResNet独立拒绝及80样本、旧不同adapter的36.68%接受与missing images事实均保持。长批次挂唯一普通完成监听；不启用Goal、不增加投递测试或模型进度轮询。

## 已启动

2026-10-27 18:31:35 UTC（香港时间10月28日02:31）启动，冻结提交 `00a8d89e12b3a77ecc3984abba32824632b35e88`，控制器PID `309314`。生产核心SHA仍为 `cb4ea28c3d90ccad5bc723ed5b6dc124488d572fca8a69b3ed3b12ec0e994016`。冻结环境已核对真实重放的model/agent/tool settings与原快照一致。

启动核查时首个U端点 `valid=true`，后续双卡worker已运行，数据盘剩52,405,944,320 bytes（约48.81 GiB）。普通完成监听PID `52568`，状态 `.scratch/v6-gc-review-wakeup.json` 为waiting，已真实探查控制器存活。这里只有启动与首端点确认，不预判GC因果或角色对照结果；详见[启动证据](evidence/v6-gc-review-diagnostics-20261028/README.md)。

## 终态审计

本批于2026-10-27 18:47:05 UTC结束，监听实际唤起审计。全部6个端点、两次真实角色重放完成，没有执行错误，GPU worker退出。数据盘可用约48.31 GiB。`v6_audit_gc_20261028.py`已逐项核对实际receipts、120样本/240逐rank值、两步质量、覆盖、全部源文件和请求：除声明的GC观察/干预模式外，任务、参数与代码一致。完整公开审计约27KB，GC原始事件保留服务器，未转存大文件。

| 端点 | 整步median / MAD（ms） | 超过200 ms的样本（从0起） |
|---|---:|---|
| U0 | 125.437 / 2.644 | 3：442.754 ms |
| O1 | 109.818 / 1.492 | 无 |
| D2 | 113.307 / 1.610 | 无 |
| D3 | 118.334 / 2.688 | 无 |
| O4 | 121.607 / 2.855 | 3：483.073 ms |
| U5 | 110.005 / 1.741 | 无 |

全部端点质量、数学策略和目标覆盖通过；没有记录fallback。D阶段GC确实关闭，未发生measure阶段generation2；但不能因D没有长尾就宣布原候选问题由GC造成，O1也无计分样本长尾，端点时间还在变化。没有修改原120.89597625 ms上限或独立拒绝。

### 观察器的实际干扰已定位

O4 rank0的generation2 GC持续360.060 ms，发生在本观察器的`cpu_observation()`内（Path.exists/stat调用路径）。观察器把CPU计数读取放在原`before_run`同步屏障之后、计时之前。此时rank1已经进入相同样本的整步计时，因此这次GC与rank1索引3的区间重叠359.861 ms：rank0该步为122.336 ms，rank1为483.073 ms。其容器`nr_throttled`和`throttled_usec`增量均为0。代码位置、事件时序及双rank差异支持“观察器在屏障后产生额外延迟，污染另一rank计时”；不是原候选同一步自身的GC证据。

O1 rank0的另一次251.108 ms GC发生在第二次warmup内，触发于GPT2激活路径，两rank对应warmup约386/389 ms，不在20个计分样本中。O1 rank1另有193.948 ms、O4 rank1另有343.569 ms GC，均在自身计时后，栈位于`torch.cuda.memory_stats`的字典展开中。所有160条带观察的逐rank计分区间中，容器限流计数增量均为0；这只能说明这些记录区间没有观察到配额限流，不能排除其它调度原因。

U0没有观察器仍有442.754 ms长尾，但没有GC记录，故根因仍未证明。保留这次实验结果，后续观察器取消屏障与计时之间的文件读取/对象采样；不关闭生产GC，不从分数扣除暂停时间。

### 审阅单文件对照：局部改善，不足以采用新策略

生成前快照已重新核对：control没有输入变化，treatment只变`inputs/tp-summary.json`；prompt、模型、工具配置、核心SHA完全相同，均是新会话，所有TP比较值与预定转换一致。会话分别为`ses_e815b0f04ffezeME9edf3rSSgf` / `ses_e8156179bffe7FyhNLI7bgeGHn`，provider报告费用0.83508279278 / 0.78786202118，均仅统计。

control已区分Q分片与完整GEMM，但漏掉rank1误差，并错误声称K/V所有变体相等；treatment给出两rank误差，并正确限定K/V的BF16相等、FP32不等，但开头仍有“every variant pair”与后文矛盾的过宽表述。control混合了不同端点的长尾，treatment则错误把旧B端点也写成有444–519 ms长尾。两臂均仍有Gloo范围/性能下限等措辞问题。因此仅能说本次更紧凑的呈现改善了部分比较对象描述，不能声称稳定解决了事实误读。

control建议直接用CPU锚点映射GPU kernel时间，仍缺GPU时钟校准误差依据，不采用。treatment建议在通信目标的profile中加CPU锚点，可作为今后具体采集缺口的候选，但不能解释本批观察器污染，也未据此改生产collector。两次完整原回答均保留；不继续为这项审阅反复重试直到答对。

### 下一步最小实现验证

已检查实际Torch实现：`memory_allocated/max_memory_allocated/memory_reserved/max_memory_reserved`每次均调用`memory_stats()`再取一个键。当前框架每个invoke在相邻的计时前/后位置共做8次此类读取，虽然位于计时外，其字典分配可能影响后续GC。准备只在隔离实验副本中把相邻读取合并为两个快照，字段、生命周期高水位和计时边界不变；生产模块暂不改。先做真实GPU字段/生命周期控制，再固定候选对照，并使用不在屏障后读取计数的轻量GC观察。只有验证后才决定采用，不预判能消除旧长尾。
