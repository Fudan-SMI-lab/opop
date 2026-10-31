# 分片排程完整版本的独立验证

香港2026-10-31 **19:43:12–19:44:44**，controller302405，原 `promote_version(publish=False)` 完成并接受完整版本 **c283f6a55dc1f758fa0fb9575f757b27384b5bb5b381592251c6fa4808c5f6a6**。trial version-df2217162ef2，source正式accepted7401459f/CURRENT eb8、独立CURRENT原始EMPTY7a49均保持。这是exploration接受，未发布指针。

服务器终审 `6806d751cb3bed582e5f78b75d752537c27f4ab7f16151ea457d9f1afd3d82d9`，[公开摘要](evidence/v6-ddp-parallel-schedule-independent-20261031/public-terminal-audit.json) `56855e5cbff464bdfd2e89c599f02cd0e61dcbb705e41f9d04f2a01c988ae400`。4jobs/8ranks、40clean、16全量检查、24自身trace、8当前关键文件核对。无Agent/新native/source模型运行、control副本或历史hash sweep。

| pair | 原始EMPTY B ms | 完整7e6fb7 C ms | 收益ms | 噪声界限ms |
|---|---:|---:|---:|---:|
| AB | 116.880095 | 44.255325 | 72.624770 | 3.517290 |
| BA | 103.554279 | 44.594827 | 58.959452 | 2.257932 |

汇总 **110.217187→44.425076ms，改善59.6931502162%**。原anchor133.379621/limit140.04860205通过；B131.730335/C48.691093尾部和全部40样本保留。完整版本共同包含计算、分片通信与更新排程，不称独立通信贡献或数据量下降，不与正式相对9bdd的2.1027%、旧58.4625%/57.8546%/53.9102%相加。

每次全量比较248879620元素，12检查误差0，4次C第二检查r0最大9.5367431640625e-7/r1最大1.1920928955078125e-7，原容差/exact replicas通过。C首parent、次graph_setup_capture_parallel_pg2_tail4_deferred_small/reference1，不能称两次optimized-only检查。没有全量tensor留存或新source比较；原cbf30d10非零但容差内source结果及精确replica负例只hash复用，既有人工source/metric/CPU/user上下文辅助边界保留。

24原始trace重新计算：B3999kernels/0Graph/13AR，C1141kernels/1130Graph/17launch/11AR，AR在Graph外。B12×28351488+157541376，C6144+6×56702976+4×39383808，总497759232B相同。B通信并发0（浮点roundoff），C28.576652–29.704933ms；compute/comm重叠B7.994248–12.984939/C14.810226–15.832540ms，各自period内描述，不是wire、关键路径或收益分解。B peak1.981GB/lifecycle2.480GB/reserved2.590GB，C4.613GB/5.656GB/6.354GB，内存增加，不加新约束。所有8日志的deprecation/barrier/loss/profiler/AccumulateGrad警告保留，无未捕获traceback。

精确7e6fb7 native只复用[前一正式验证](v6-ddp-parallel-schedule-validation-20261031.md)的nsys-0b0b77dfd6与两SQLite/normalized哈希，未再采集。条件性限制仍包括copied inputs、有限guards、shared pool、PG初始化/缓存生命周期、setup hooks、post-update fallback只恢复grad而不恢复weights、rank非对称异常/globaloverride；固定原质量通过不变成通用保证。没有新增fault injection、CPU fixture或修复链。

读取器首次把已排序实际载荷与未排序期望列表比较而assert，原reader/error保留在服务器 `terminal-audit-reader-attempt-1.py` / `terminal-audit-reader-attempt-1-error.json`；仅排序期望列表后第二次审计通过，未重跑实验/角色/GPU/native/source。批次在listener预检前已结束，**没有独立监听或排队完成事件**，不启动旧批次监听。

[现有通用编排](../scripts/experiments/v6_promote_accepted_version.py) SHA b27c96c77ae87644e0bb76acce964e7d95ac1e23e075b5eeb0b39a1054541de3，working core e893与fixed/formal一致，原任务/质量/性能规则不变。singlefixed-parallel-schedule、params{}调用代码自身fine8/comm_groups2/tail_splits4；code7e6fb7a723590a93b289f43fcfef84663fc618b5da6d559362144d178827eed3，proposal442ac3f39dc9d13f8591f534be804b42b77b625c933a184038de64200a241403。无参数重试或人工修改。启动320MiB预算依据此前完整独立root及实际quota增长，8GiB警戒保持，没有新增清理。

**下一步已实际启动**：[精确新接受版本的正常discovery](v6-ddp-schedule-communication-evidence-20261031.md)。版本仍继承9bdd registry，需要实际新代码记录供后续优化决策；普通planner自主选择现有采集器，不预设下一个算法或直接串候选。
