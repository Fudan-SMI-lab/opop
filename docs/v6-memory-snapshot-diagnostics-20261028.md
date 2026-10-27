# 相邻显存统计读取的最小实现验证

承接[GC批次审计](v6-gc-review-diagnostics-20261028.md)。原候选长尾仍未完成归因，但已定位到观察器屏障后读取计数导致对侧等待，以及多次计时外GC触发于显存统计展开。本批只验证一项通用实现改进，不改变生产框架、GC设置、候选或接受规则。

## 实验中的代码变化

从冻结的真实`evaluation.py`生成独立worker副本，只将嵌套`invoke()`内相邻显存读取合并：计时前两个peak读取用一个`torch.cuda.memory_stats()`；计时后的生命周期高水位和四个内存字段共用另一个快照。将每次invoke的8次完整统计展开变为2次。其它prepare、安装、adapter metrics、同步、reset、计时、质量检查及返回对象生命期不变。每个字段仍读取实际Torch便利函数使用的同一键，并保留缺失键默认0。

AST检查确认`invoke`之外无变化，计时调用不变；基准源码SHA `5cbf179a62de3e3ce313582a2e2ad6b4a117d79b73f41e4aa97c8fdb36128a9d`，实验副本SHA `1bbf3af429b0eb25c07cbb937578a2e2a2220446e54e84bdc3474fb0a3e7d487`。每rank另外记录实际加载的evaluator路径/SHA；生产核心SHA不代表实验副本未变化。相邻读取合并也会改变并发分配时观察的时刻，需在审计中明确，不泛称所有并发环境数值必然相同。

## 固定验证顺序

1. 真实CUDA小型控制：分别对legacy与invocation返回对象生命期运行原实现/实验副本。覆盖prepare临时峰值、持久状态、run临时张量、返回张量和adapter.metrics后持有的分配。两臂必须通过相同controls，全部内存字段逐样本相等，生命周期高水位保留；另外统计实际memory_stats调用次数。控制不通过则不进入DDP对照，错误保留。
2. 同一完整C候选按 **L/S/P/Q/Q/P/S/L** 执行，每端点20样本。L/S是无观察器的原实现/快照副本，P/Q是在对应实现上加相同轻量GC回调。参数、adapter、质量、工作量和原整步上限不变，全部160样本及逐rank值保留，不追加至通过。
3. 使用现有provider做一次范围收窄的真实测量语义审阅：仅给修改前后invoke、CUDA控制结果、DDP所有样本的小型索引。需要时读取对应receipt，核对字段、生命周期、同步/计时边界与并发限制；不预载旧大trace或约800KB审阅输入，不自动采用模型结论。

轻量GC观察器不替换`before_run`，不读取CPU计数文件，不在屏障与计时之间加入取样，不关闭/强制GC；仅记录GC start/stop及计时结束后的metrics入口时间。后者有未测量的Python调用延迟，倒推时间窗只能近似；记录host/boot/时钟实现，保留观察器本身改变分配与GC触发位置的可能。L/S用于检查观察器影响，任何GC暂停都不能从成绩扣除。

## 留档与自动续行

工作目录 `/root/autodl-tmp/dev/v6-work/memory-snapshot-diagnostics-20261028`，脚本`v6_memory_snapshot_probe_20261028.py`、`v6_memory_snapshot_batch_20261028.py`。GPU串行租用，参考只读复用，原始数据留服务器。数据盘启动准备时约48.31 GiB可用；不恢复大体积本地转存，不启动大型TP实验。

旧DDP独立拒绝、TP完整32-token质量失败、Qwen/ResNet旧拒绝及全部样本、不同memory adapter的接受与missing images事实不变。本批没有新的正式候选接受；生产是否采用显存读取改进，要在真实控制、固定端点和代码审阅后决定。使用一个普通完成监听，不启用Goal、额外投递测试或模型进度轮询；API费用仅统计、不限额。

## 启动错误与修正

首次冻结`0d3dec2`于19:00:57 UTC启动，PID315735。CUDA控制在创建worker前失败：协议写入已创建`controls/legacy-original`，而LocalExecutor要求运行输出目录不存在，抛出FileExistsError；DDP按协议跳过。没有任何GPU质量/性能结论，后续真实语义审阅仍保留运行，不取消、不重启。

已将worker输出改为其独立`run`子目录，协议留父目录。首个根目录、错误日志及审阅均保留；修正后的验证使用新目录`/root/autodl-tmp/dev/v6-work/memory-snapshot-validation-20261028`。首次普通监听被标记停止，由新验证批次的唯一监听接管；新续行指令同时核对首次审阅终态。该修正只涉及实验目录布局，不涉及生产逻辑或实验判据。

## 修正后的实际执行

修正版冻结`f8d717fe61e3f87b60c6af3b29d227c1bd18ddec`于19:04:40 UTC启动，PID317239。首次失败批次后来以`completed_with_errors`结束，其审阅正常完成，旧错误没有覆盖。新普通监听PID28004，状态`.scratch/v6-memory-snapshot-validation-wakeup.json`为waiting；首次监听已停止。

截至本轮启动核查，legacy与invocation共4个真实CUDA控制均通过，全部内存字段逐样本相等，prepare生命周期峰值保留；每种生命期的整个控制流程中`memory_stats`调用次数从74减为20。8个固定DDP端点也均执行完成、质量检查通过；所有160样本及逐rank值留在服务器。尚未据此给出性能收益或长尾归因，仍需终态审计实际副本SHA、全部样本、GC时间窗和观察器影响。

当前真实`measurement_review`会话`ses_e813b810dffecH7QzlXGDM1rDd`已启动，初始快照`generation-1697496b45a9`。此次输入6文件总16,725 bytes，仅含invoke前后代码、CUDA控制、全部DDP样本小型索引和环境，不预载旧大trace。GPU阶段已结束，当前等待该审阅；生产是否采纳仍未决定。核查时数据盘剩51,032,162,304 bytes（约47.53 GiB）。
