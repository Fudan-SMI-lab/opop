# DDP 通信观测机制的一次 CPU 事实辅助修复

通信优先与原始项目优化目标保持。前一作者已完成并审计，但精确探针在独立CPU检查中将歧义/缺失关联标为observed，且部分observer安装失败阻断原入口并遗留patch，因此不执行旧探针GPU。此步仅修复具体观测合同，不是优化候选、reviewer叠加、prompt对照或重开77ac实现链。

调用现有`ProjectAgents.generate(diagnostic_author)`一次。原11初始文件保持字节相同（10显式+自动worker环境）；增加精确旧probe/plan、一次CPU结果和小索引四文件。索引指向已执行fixture与记录，不提供终态审计表、手写修复、通信算法/group/bucket/collector选择。明确是开发者CPU事实辅助；生产prompt/model/tools/core不变，只追加实验修复请求。API仅统计，无新增caps。

作者必须自行决定可靠的最小观测实现或明确unknown；原entry一次、返回/异常/validity及原DDP语义仍权威。仅允许元数据/API检查及不执行真实计算的Python mock，禁止小型模型训练、任何process group/真实collective（含单进程Gloo）和CUDA初始化。完整工具实际行为仍需审查；上轮违规CPU活动不会被改称合规。新代码只做生成/schema验证，审查前不执行GPU，不自动继续修到通过。

原accepted ccd90/current1fc、accepted4541/independent7a49/历史Graph09b7、source/controls/adapter均保持。完整质量两连续更新/两warmup/五样本、原容差/精确rank replica、原step约束140.04860205ms不改；本控制器没有模型、GPU、collective、search、parameter、collector执行或验收阶段。既有计算Graph收益仍不是通信接受。

空间预估128MiB（冻结框架、约1.5MB输入、正常作者记录），保留8GiB警戒；不是API或输出限制。启动前核实实际150GiB df及GPU空闲；当前不足大型GPU批次预算。无本步清理/raw本地转移/reference副本。原始历史与私有配置留服务器，公开仅代码/小摘要/hash，任何凭据不得回显。新长任务只设置一个隐藏完成/磁盘监听；旧作者监听处理后不重启。实际启动与快照核验另记。
