# DDP 原观察反馈选择的一次候选生成

19:37:07UTC真实生成完成，reason=`batch_finished`，不是磁盘警报。终审[小型公开记录](evidence/v6-ddp-observation-route-candidate-20261031/public-terminal-audit.json) SHA`541438bba4cb254a361435bd01a7f5f9fdf8e4634973c5c76450fc486c64e07d`；服务器完整审计SHA`74209119a32ada66f1d298a4f3c67bbe5e1c043a5c796529986ef77ad8d2c123`。1020冻结文件、59实际输入32,668,900B、真实registry反馈路由/prompt/model/tools/core/current全部保持。55 assistant/59工具/0 schema修复；provider估计2.50923005388，非账单。4921 read行/2显式截断前缀及15份纯JSON或框架源码输出逐字核对。candidate一次写入、三次编辑；proposal一次写入无编辑，重建最终字节一致。

新代码`508ad3823d3a7b980df7076023b3567242f12a2544cb9bbeae0ada66c6f52f0a`/14,898B，proposal`853ae1eb048e1d590e5b2ba03d75dfec378a4fc20bf4a7b0faa478ba03e1de89`/9,828B，emptyparams，requires空/replaces旧ccd90 bundle。Agent实际读了旧Tensor-bool异常及两rank fallback，最终正确区分“旧并发实现未执行”与“原eager回退变慢”；该纠正不是开发者提供。仍未打开任何15份新观测材料或current collector JSON；工具18只读到完整generic索引指针/限制，不能声称消费了新GPU观测。

实际CPU活动包含Linear梯度视图/绑权重和两层小GPT2的FW/BW；顺序/合批toy梯度max3.72529e-8，归一化relative3.55985e-7，不是原模型CUDA/连续更新/精确replica验证。后续模拟删去CUDA guard，FakeGraph.replay为空、all_reduce为空、捕获上下文直接执行一次SGD，所以“参数变化/Graph标签”不能证明真实重放。保留Parameter不可调用、None.numel、patch-before-import TypeError、最初拿零flat比较导致setup失败、错误fallback标签、lambda-yield SyntaxError及pipeline exit0。Agent第三次代码编辑将全梯度gate移到实际batched backward后；未人工改代码。Agent删除自己的cache，完整工具记录和最终CPU脚本服务器保留，不声称所有被覆盖临时输出存续。审计首轮误将tool11的ls前缀当纯JSON，attempt1保留，仅修正读取器；未重跑Agent/CPUfixture/GPU。

代码是新的batch-four FW/BW Graph→单个497,759,232B SUM→除world→SGD Graph假说。第一质量调用仍parent，第二才setup/可能fast；局部2e-7/2e-5 gate和sentinel不是原全量质量权威。cat输入复制一次、有限identity/ptr guards、finallyFalse全局标志恢复、update后异常可能再次parent更新及rank非对称错误等边界明确保留；不承诺通用回退/可变输入/并发owner安全。33ms wire下界、4–5ms可回收间隙、55–62ms预测及“构造即保质量”不采纳。没有新GPU质量、native或接受结果。

下一步已声明[一次固定B/C验证](v6-ddp-observation-route-fixed-20261031.md)：当前acceptedccd90为B，原样508a为C，原完整质量/约束/通信指标不变，不做search/参数/collector选择/独立接受/pointer发布。终审时实际150GiB盘剩10,571,677,696B，GPU空闲；512MiB预计增长加8GiB警戒线有余量，不清理。旧51908完成事件已handled/进程不存在，不重启。闭合77ac/observer/Qwen及最新FSDP拒绝不重开。

实际19:16:07UTC（香港03:16）启动，冻结`b4dafb428b8536630fe3a1213b96080edf425279`，controller195086，唯一隐藏监听51908。local OpenCode实际start/health/stop通过、server_url=null，没有model预检prompt。旧52336完成事件已审计/handled，不重启；当前有新代码生成任务，无GPU阶段或接受预言。

19:17:58UTC[实际启动核验](evidence/v6-ddp-observation-route-candidate-20261031/startup-verification.json)SHA`e1be0686219dc36a6a3f92f1024990ef6bc16baea193d6ba0c499f745d86fdb8`：generation8078c7933f7b/session ses_e71c221b7ffesynuBbcSziZFG6，59文件32,668,900B与真实snapshot逐字相同，原56个selector非feedback输入保持；1020冻结身份、core/current、普通candidate prompt/model/tools核验。canonical selection SHA`106569f869bee2115b015e430c890aa19de3d3507b38c0c81de523e3dd54f07d`，原选择语义不变；candidate route6,251,759B SHA`7c08054f78a35a0de5d7be54d77a6def10d90edad1d334e9ec1ce05a1c397062`，原CPU事实/观察索引supplement保持，无新审计答案。与旧普通candidate输入12变/20增/5移除，不作单变量AB。

11 assistant/13工具的[早期审查](evidence/v6-ddp-observation-route-candidate-20261031/startup-tool-review.json)SHA`087e18314d084f777ff4b375e2d6037e1e706db1922ce0d44f3253a422b3bc4a`：3912 read行/2截断前缀，工具8/9/10/11纯JSON、7头部、13源文件输出重建一致。已读完整selection/adapter/旧失败5e3a/原source及旧拒绝、原约束；前13未打开新观察/currentcollector/installedccd90，早期事实不能当终态无读取证明。无观察到模型/GPU/collective/私有配置内容读取，完整后续审查待终态。GPU空闲/无jobs，实际余10,576,162,816B约9.85GiB，生成阶段而非整个项目停工。

声明一次现有`ProjectAgents.generate(candidate)`，沿用selector完整execution_plan选择，普通containment/sort及selected-group反馈路由。四微批批处理/完整梯度融合规约是假设，非已验证收益；不手选group/bucket/算法、不写优化代码、不提供审计纠正。无selector/prompt/reviewer重试，无自动GPU/search/参数/collector/native/验收/pointer阶段。完整代码和工具记录审查后，才能决定是否另行声明执行。

来源为`ddp-observation-feedback-selection-20261031`真实generation999c08dfa37c，selection SHA`4249c7d0864a37e31000af3e080e4f2a872eae98e7990221376501f4ab6bba92`。1055 affected calls仅按普通API排序，rationale/hypothesis/groups不改。复用实际56份selector材料，加正常selection/targets/target-resources；保留原77ac CPU事实和15份原始B/C观察材料。原trial02ed5436b4反馈14,429,015B SHA`a0234920605970500c870b2ec98fc952b264a79f264bd19d6a53c99b796cf459`不变；candidate按新选择正常路由，supplement不变，不加入日志范围答案/终审表。既有source/metricmapping/CPU辅助边界保留，非无辅助或单变量AB。

原sourcecurrent1fc24545/代码ccd90/单bundle/emptyparams、accepted4541/independent7a49/历史Graph09b7保持，使用accepted-code discovery；无新discovery/source/control/reference副本。生产core`ed24af7acc8caa15578e0b859c85ef8a901be51354448ed39094606ed9e10a2f`、普通candidate prompt/model/tools、合法参数接口/正常修复保持，API observe-only，无新cap。

原两rank、B1/S128/accum4、FP32/eager/noTF32、两次连续全量梯度/参数/loss/exact replicas及140.04860205ms约束保持。当前ccd90独立36.1578233%仍仅为计算调度收益；融合实际执行、浮点顺序/连续状态/回退安全、质量和整步收益均未知。不得以线速下界/跨profile减法/理论等价代替检查。新通信接受仍需实际机制及独立完整版本收益。

增长预估384MiB，保留8GiB警戒；当前约9.94GiB，无清理。不是API/工具cap。新任务一个隐藏监听，旧52336 handled后不重启。observer/mechanics、77ac、Qwen关闭，最新FSDP拒绝不重跑。实际启动和输入核验另记。
