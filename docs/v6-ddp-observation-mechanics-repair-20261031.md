# DDP 通信观测机制的一次 CPU 事实辅助修复

终态18:33:01UTC（香港02:33），reason=`batch_finished`，controller正常completed；不是low_disk_space，也没有新GPU质量/性能接受。[完整终审小摘要](evidence/v6-ddp-observation-mechanics-repair-20261031/public-terminal-audit.json)对应服务器audit SHA`82015a1648a79c69f473422ffec22fd80946d59e7541377db9a5297c6ffc7deb`。15输入/原11逐字节相同、967冻结身份/core/current/model/tools保持。29 assistant/41工具/0 schema修复，provider估计0.91127245004，非账单。

新probe `92279e28dc1e548efe86ef44467a3b0d09f7d94586da3dc16c0a3f55b773db96`（46,532B）、plan `0a82e46248377b94a7ec21954eb34d2bdf556d65796557556923ebf9eeefcf13`（40,074B），各写一次、无编辑；实际write内容重建与最终字节相同。完整41工具输入/输出及CPU源代码审查，2853 read行逐行核对、无截断。实际只有元数据/API、2元素CPU Parameter及不执行模型/collective的synthetic harness；没有观察到模型、process group、真实collective或CUDA初始化。tool11错库路径exit2、tool23错schema打印None保留；tail管道exit0不当通过证据。作者`all(identities())`实际遍历字典键、`or True`断言无效，终审没有据此认证恢复。无观察到private config/后续审计或reviewer答案读取，文件系统非hermetic，非绝对无外部访问证明。

独立CPU夹具只替换EXPECTED probe SHA，原执行序列不变，在新目录执行一次；旧fixture/结果保持。结果SHA`6b48e77486273497e79b06b80f5ba6f23e9d93234a2e79655081b568d5273c23`：唯一range正控保留；歧义range→unknown；缺index1不再错配2→ordinal10，而为2→11、10缺失；不可读window关联unknown；无新zero_grad的3回调ordinal均null；4096广播硬cap无None增长；原返回SAME对象/validfalse保持、原RuntimeError SAME对象、一次调用、正常六patch恢复。

注入non-class profiler的partial setup仍在原入口前抛TypeError，但之前五patch恢复，setup_failed与原执行异常明确分开。它不是当前已安装API故障，也不证明observer失败时总会执行原入口。批准范围仅当前固定API/request的观测验证；实际安装失败就保留失败，不修复重跑。代码仍以zero_grad为部分Python锚点，Graph事件可能unknown；phase/重复index与bucket顺序是假设，CPU回调不是GPU ready。teardown zeros-like bucket约497,759,232B，plan称无kernel/1–2GB及通用小于2MB硬上限不能采信；owners、hook与同步扰动不为零，不证明任意异常/并发安全。

下一有界动作已明确为[一次精确B/C通信观测](v6-ddp-communication-observation-20261031.md)。仅使用Agent原始92279字节，不再追加author修复。B原空baseline作为观测控制，C当前accepted ccd90；原完整两次质量/两warmup/五样本/三独立metric profiles每rank不改，时间均为观测诊断，不接受/不减observer成本。先完成安全空间预算再启动；旧48592已handled，不重启。

实际已于2026-10-30 **18:21:12UTC（香港10月31日02:21）**启动，冻结`c1a63c3fb5cca2eae49acfa516b5b682b6f907c8`，controller **187340**，唯一隐藏完成/磁盘监听 **48592**。原48992事件已处理且进程不存在；没有重启旧批次。实际local OpenCode启动/health/stop通过，无model预检请求，server_url=null。

18:23:29UTC[实际启动核验](evidence/v6-ddp-observation-mechanics-repair-20261031/startup-verification.json)SHA`c9ab53ad335df0f51dd1b9cd9f6306c8f9cd1278d315d3cc7b17eae4d3793f09`：generation`1d16ccdd6338`/session`ses_e71f4697affen3q5JgSGmWI8QJ`，15初始文件1,471,240B（14显式1,467,670B）、原11逐字节相同、967冻结身份及模型/tools/core/current保持。仅追加实验repair段落SHA`9bae159d91d0cf272052ce5095034349e352fcadc2b65278b518baf300828de9`；无生产prompt修改。

5 assistant/10工具，前10全部read；[早期核验](evidence/v6-ddp-observation-mechanics-repair-20261031/startup-tool-review.json)SHA`876f61449d2c8f5d962291f0eb700d01ef1eb80c76381b6dcaa4b6076b28ca47`核对1727行、无截断，实际读到完整旧probe/plan/CPU结果/index、named fixture及原adapter/entry。这只是早期访问证据，不是完整终态推理/代码正确性证明。无GPU/jobs，150GiB实际余8,769,523,712B约8.17GiB；controller与单监听均存活。私有历史SHA`8dfb0569e0358af36fd4453c31d1687238c9e350b22e5de0bd780da8ad8ebbf6`留服务器，未打印配置/凭据。

通信优先与原始项目优化目标保持。前一作者已完成并审计，但精确探针在独立CPU检查中将歧义/缺失关联标为observed，且部分observer安装失败阻断原入口并遗留patch，因此不执行旧探针GPU。此步仅修复具体观测合同，不是优化候选、reviewer叠加、prompt对照或重开77ac实现链。

调用现有`ProjectAgents.generate(diagnostic_author)`一次。原11初始文件保持字节相同（10显式+自动worker环境）；增加精确旧probe/plan、一次CPU结果和小索引四文件。索引指向已执行fixture与记录，不提供终态审计表、手写修复、通信算法/group/bucket/collector选择。明确是开发者CPU事实辅助；生产prompt/model/tools/core不变，只追加实验修复请求。API仅统计，无新增caps。

作者必须自行决定可靠的最小观测实现或明确unknown；原entry一次、返回/异常/validity及原DDP语义仍权威。仅允许元数据/API检查及不执行真实计算的Python mock，禁止小型模型训练、任何process group/真实collective（含单进程Gloo）和CUDA初始化。完整工具实际行为仍需审查；上轮违规CPU活动不会被改称合规。新代码只做生成/schema验证，审查前不执行GPU，不自动继续修到通过。

原accepted ccd90/current1fc、accepted4541/independent7a49/历史Graph09b7、source/controls/adapter均保持。完整质量两连续更新/两warmup/五样本、原容差/精确rank replica、原step约束140.04860205ms不改；本控制器没有模型、GPU、collective、search、parameter、collector执行或验收阶段。既有计算Graph收益仍不是通信接受。

空间预估128MiB（冻结框架、约1.5MB输入、正常作者记录），保留8GiB警戒；不是API或输出限制。启动前核实实际150GiB df及GPU空闲；当前不足大型GPU批次预算。无本步清理/raw本地转移/reference副本。原始历史与私有配置留服务器，公开仅代码/小摘要/hash，任何凭据不得回显。新长任务只设置一个隐藏完成/磁盘监听；旧作者监听处理后不重启。实际启动与快照核验另记。
