# allocator快照终态审计

承接[实验协议](v6-memory-snapshot-diagnostics-20261028.md)。唯一替换监听已实际触发核查。首批于19:05:28 UTC以`completed_with_errors`结束：CUDA worker未启动，DDP跳过，真实review完成。原因是本批代码先写protocol创建了executor要求不存在的目录，并非review所说的“旧尝试残留目录”。修正版19:14:15 UTC正常结束。逐文件比较确认controller只改了worker的`run`子目录；两次尝试均完整保留，没有重启。

审计脚本为`v6_audit_memory_snapshot_20261028.py`，[小型证据](evidence/v6-memory-snapshot-diagnostics-20261028/terminal-audit.json)记录原始receipt、源码、GC文件及review快照的SHA。审计脚本首次假设controls返回`setup`而报KeyError；删除这个错误假设后审计通过，未修改实验。原始大文件仍留服务器，本轮只取约90KB审计/回答摘要。

## CUDA和全部DDP端点

四个真实CUDA控制的reset、identity、错误输出及缺覆盖负例均通过。每种生命周期的两次样本、六项framework内存字段逐项相等；prepare高水位8,912,896 bytes保留。legacy当前allocated为1,703,936 bytes，invocation为1,638,400 bytes，相差65,536 bytes，符合前次返回tensor的生命周期差异；metrics后持有的分配也计入。完整控制流程调用数均为74→20（9次invoke各8→2，加两次初始化读取），并非每次调用74→20。

八个DDP端点均完成两步全量质量检查，逐rank覆盖无fallback，工作量、参数及源码清单一致。实际加载的evaluator分别匹配原SHA `5cbf179a…`和副本SHA `1bbf3af4…`，副本在invoke以外AST相同。全部160个max-rank样本及320个逐rank样本保留；`valid`表示质量/覆盖通过，不是正式性能接受。

|端点|实现|中位数ms|MAD ms|大于200ms的样本（从0编号）|
|---|---|---:|---:|---|
|0 L|原实现|123.743|2.594|无|
|1 S|快照|114.456|1.215|9:438.785；18:427.147|
|2 P|原实现+被动GC|114.104|1.155|无|
|3 Q|快照+被动GC|123.790|3.039|5:459.366；18:499.109|
|4 Q|快照+被动GC|122.943|2.946|9:461.119；14:508.266|
|5 P|原实现+被动GC|110.244|1.141|无|
|6 S|快照|122.170|4.003|9:293.873；18:490.678|
|7 L|原实现|115.371|2.306|无|

两rank的host、boot及CLOCK_MONOTONIC一致。P的rank0在第二个warmup前向中触发399.668/322.328ms Gen2 GC，rank1的277.260/191.479ms Gen2发生在计时外memory_stats。Q的四次Gen2则发生在正式样本的模型调用中：端点3样本5/18为339.544/372.491ms，端点4样本9/14为339.296/377.715ms；每次都与两rank对应长尾的近似时间窗重叠，触发栈属于Torch模型调用，未出现旧CPU文件观察器。所有GC事件均配对闭合，未关闭或强制GC。

metrics入口比真实timer结束晚了未测量的调用延迟，不能据此精确扣除暂停；回调本身也可能改变GC触发。上述记录证明有观察端点内存在计时中GC，并支持减少计时外分配会移动GC触发位置这一机制；不能外推为S端点或历史未观测长尾的完整原因，也不能把原实现本批无长尾解释为永远没有长尾。

**决定：不将allocator快照改动合入生产。** 字段映射和小型CUDA控制通过，但端点观察没有建立测量稳定性收益，反而暴露明显的时序变化。保持原evaluator、GC策略、timer、样本及接受规则。停止重复此候选的性能测量；无正式候选接受。

## 真实审阅核对

两次真实回答均保留：[失败首批回答](evidence/v6-memory-snapshot-diagnostics-20261028/review-initial-failed.json)、[修正版回答](evidence/v6-memory-snapshot-diagnostics-20261028/review-corrected.json)。provider估算费用分别0.34929387612和0.28355649064，仅记录，不设限。修正版输入逐字匹配其生成前快照，只有16,725 bytes，未收到人工结论。

修正版正确核对了字段键/default、控制结果、8→2调用、八端点及快照端点长尾分布。但有以下局限：

- 称“两处读取都在synchronize之后”不准确：pre读取在before_run与本次synchronize之前；post读取虽在同步后，adapter.metrics仍可能发起新分配，其他线程也不受该同步约束。字段读取合并不保证任意并发分配下数值一致。
- 称计时外改动“不会影响elapsed”过强：直接调用耗时不在timer内，但分配/GC触发位置可间接改变后续计时。当前Q记录提供了相应机制证据。
- 称没有逐样本GC资料遗漏了receipt旁独立的`gc-passive-rank-*.json`；初始索引只给receipt，未直接列出旁文件。不能从receipt没有GC字段推断没有采集。这是一个已证实的证据发现缺口。
- 非零reserved字段不足以单独证明历史allocator backend；本轮不据此宣称cudaMallocAsync被实验排除。相同键的库API检查与真实控制，仅覆盖所测环境。

下一步仅对该角色的`ddp-index.json`增加原有GC旁文件路径/哈希及格式说明，从同一生成前快照做真实control/treatment；不预塞GC结论、不修改生产提示。同时按主计划E推进相同DDP workload的整步目标变体，使用已审计baseline adapter、新合同/controls、三轮真实搜索及独立完整版本检查。它明确是同adapter目标变体，不称新原入口cohort，也不导入旧通信候选。旧DDP独立拒绝、TP完整32-token失败、Qwen/ResNet旧拒绝及样本、另一memory adapter的36.68%接受/missing images事实均不变。

终态核查无GPU计算进程；容器df可用51,031,908,352 bytes（47.53 GiB）。没有恢复大体积本地转存。
