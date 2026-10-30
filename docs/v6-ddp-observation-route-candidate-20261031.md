# DDP 原观察反馈选择的一次候选生成

声明一次现有`ProjectAgents.generate(candidate)`，沿用selector完整execution_plan选择，普通containment/sort及selected-group反馈路由。四微批批处理/完整梯度融合规约是假设，非已验证收益；不手选group/bucket/算法、不写优化代码、不提供审计纠正。无selector/prompt/reviewer重试，无自动GPU/search/参数/collector/native/验收/pointer阶段。完整代码和工具记录审查后，才能决定是否另行声明执行。

来源为`ddp-observation-feedback-selection-20261031`真实generation999c08dfa37c，selection SHA`4249c7d0864a37e31000af3e080e4f2a872eae98e7990221376501f4ab6bba92`。1055 affected calls仅按普通API排序，rationale/hypothesis/groups不改。复用实际56份selector材料，加正常selection/targets/target-resources；保留原77ac CPU事实和15份原始B/C观察材料。原trial02ed5436b4反馈14,429,015B SHA`a0234920605970500c870b2ec98fc952b264a79f264bd19d6a53c99b796cf459`不变；candidate按新选择正常路由，supplement不变，不加入日志范围答案/终审表。既有source/metricmapping/CPU辅助边界保留，非无辅助或单变量AB。

原sourcecurrent1fc24545/代码ccd90/单bundle/emptyparams、accepted4541/independent7a49/历史Graph09b7保持，使用accepted-code discovery；无新discovery/source/control/reference副本。生产core`ed24af7acc8caa15578e0b859c85ef8a901be51354448ed39094606ed9e10a2f`、普通candidate prompt/model/tools、合法参数接口/正常修复保持，API observe-only，无新cap。

原两rank、B1/S128/accum4、FP32/eager/noTF32、两次连续全量梯度/参数/loss/exact replicas及140.04860205ms约束保持。当前ccd90独立36.1578233%仍仅为计算调度收益；融合实际执行、浮点顺序/连续状态/回退安全、质量和整步收益均未知。不得以线速下界/跨profile减法/理论等价代替检查。新通信接受仍需实际机制及独立完整版本收益。

增长预估384MiB，保留8GiB警戒；当前约9.94GiB，无清理。不是API/工具cap。新任务一个隐藏监听，旧52336 handled后不重启。observer/mechanics、77ac、Qwen关闭，最新FSDP拒绝不重跑。实际启动和输入核验另记。
