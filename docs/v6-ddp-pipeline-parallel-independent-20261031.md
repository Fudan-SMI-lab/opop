# 并发通信完整版本的独立验证

香港2026-10-31 **18:29:46–18:31:15**，controller286820，原 `promote_version(publish=False)` 完成并接受完整版本 **b769fc35d58a472eafa11e7206746b3dff9ed46901aa721469002aebac63da7d**。源正式accepted187c/current0bc与独立CURRENT EMPTY7a49均保持；这是exploration接受，不是自动发布。

[新独立审计摘要](evidence/v6-ddp-pipeline-parallel-independent-20261031/public-summary.json)：服务器audit `6448a8c583329931568e511452ffa63fe91039dd1daafb97eb9efdafb1e97f43`，完整public `4a28d92ad9bb7823a985b5cae387129ac9c6adfbacba3e5e60540de6b8994547`。4jobs/8ranks、40计时样本、16全量检查、24自身trace及8当前关键身份核对，首次审计通过，没有角色、模型、native或source重跑。

| pair | 原始EMPTY B ms | 完整9bdd C ms | 收益ms | 噪声界限ms |
|---|---:|---:|---:|---:|
| AB | 112.956407 | 45.537191 | 67.419216 | 1.735502 |
| BA | 106.259760 | 45.519790 | 60.739970 | .210106 |

汇总 **109.6080835→45.5284905ms，改善58.4624700604%**。原约束通过，两对均超过噪声，尾部B114.689114/C45.812710保留。与正式对比02b933的5.6824%、旧57.8546%或53.9102%不相加，也不跨周期计算增量。完整版本包含计算、分段通信并发与更新安排；不是纯通信收益或数据量下降。

每次全量比较248879620元素，12次检查误差0，4次C第二检查r0最大9.5367431640625e-7/r1最大1.1920928955078125e-7，原容差及exact replicas通过。C仍首次parent、第二次graph_setup_capture_parallel_pg2/reference1，不能称两次优化检查。所有样本/原warning保留；源证据仅引用既有实际source/controls比较和replica负例记录，不是新source比较，没有全量tensor留存。

24份自身trace重新核对：原始EMPTY B为13AR，C为8AR，总497759232B不变；不能移用正式02b933父版本的8AR/零通信并发结论。新C通信依赖及kernel区间来自本批原trace；native只复用前一正式验证的精确9bdd Nsys/hash，未再次采集。区间重叠和NCCL sum不等于wire、关键路径或可回收收益。继承的有限guard、复制输入、共享pool、setup hook、更新后fallback只恢复grad而不恢复weights、rank非对称异常和globaloverride限制仍存在，原固定任务质量通过不等于通用故障安全保证。

[通用编排](../scripts/experiments/v6_promote_accepted_version.py)从真实正式结果推导accepted revision、prepared baseline和bundle，controller SHA `b27c96c77ae87644e0bb76acce964e7d95ac1e23e075b5eeb0b39a1054541de3`。复用工作core e893，单一fixed-pipeline-parallel/空参数调用自身fine8/两组默认值，无case路径/hash/模型名/预期答案分支。没有候选生成、参数搜索、采集器、source-control、reference副本、框架archive或部署门槛；语法与实际preflight通过，未添加冗余测试。

因剩余容量不足覆盖近期独立批次，在GPU空闲的18:28:03–08按已有授权对24份**已关闭FSDP诊断**metric JSON作服务器内无损冷存：263018192→23889822B，回读hash均通过，820个其他文件metadata不变。当前DDP/source/controls/accepted/models/provider/deps未触及。恢复说明位于`/root/autodl-tmp/dev/storage-audit-20261031/ddp-parallel-independent/RESTORE.md`，journal SHA `502e06a9c354f4d91ba67e2fdaa421cd78ea0b5de9d5a9a3c8747d961be93ed0`；旧trace是冷存，不是丢失。[存储记录](evidence/v6-ddp-pipeline-parallel-independent-20261031/cold-storage.json)。启动余9132503040B，384MiB预测按近期完整独立root324767744/321847296B安排，不是API cap。

监听39344已handled，晚到事件去重。已实际启动[新接受版本的一次普通discovery](v6-ddp-parallel-communication-evidence-20261031.md)，更新仍继承02b933的旧registry，供下一步实际优化选择使用；不自动追加候选或调参。
