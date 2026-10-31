# DDP 已有产物协议归属取证编写（2026-10-31）

用户最新约束（2026-10-31）：所有实施以实际项目优化推进为优先。不要把辅助证据分析扩展成安全性建设、分析器完善、冗余测试或版本管理门槛；所有框架调整必须可泛化，禁止按具体case的路径、哈希、模型名或预期答案硬编码。当前已运行的单次协议分析仅用于判断下一步通信优化依据，不再追加分析器修复/作者/评审链，也不默认执行生成reader。角色返回后优先直接提取足以决策的原始依据；只有一个明确缺失且会改变优化决策的事实，才可安排一次必要的现有产物读取。已经完成的检查不重复展开；验证聚焦原始数值正确性、性能收益及会改变这些结论的错误。原质量/性能接受规则不放宽，但不另加与推进无关的门槛。实验实例用于复现，不成为生产框架中的case分支。


## 已结束：提取决策依据，关闭读取器支线

终态为 **2026-10-30 23:58:45.858004UTC / 香港10月31日07:58:45，batch_finished**，不是低磁盘告警。初次核查150GiB盘余9,375,412,224B，GPU无worker、旧controller退出。旧51296监听已handled，不重启。

[终态摘要](evidence/v6-ddp-protocol-attribution-author-20261031/public-terminal-audit.json) SHA **6c75a3c39c627154d45c63971c843e98b643487d6028cb55596e0b4922552aa9**，服务器完整审计SHA **87c38409f80123fc422c5163a806274cd95ddd7c7cbdd26b1c1777b99e5a4b2b**。17初始文件3,037,071B、当前代码及普通模型/工具保持；控制器终态核1947身份，本轮独立读取仅核实际输入、输出重建和当前关键状态，没有重跑全部历史检查。91 assistant/104工具/0 schema修复；provider估算3.57431362012，非账单。读取器1写5改、计划1写1改均重建吻合；原错误、测试修改及工具历史留服务器。

足以决策的结论是：**现有原始记录不能确定计时区间逐次AllReduce实际采用LL还是Simple，因而不能承诺切换协议的收益。** 一次必要的六份已有日志读取确认，当前discovery/native日志没有逐次协议决策记录；另一次tuning进程rank0的26条决策不能移接到计时事件。kernel入口名、二进制符号与跨capture时差也不足以完成这个归因。计划关于长kernel“只在等peer”、所有环境均未固定协议、重复打印证明每次重新选算法等强结论均未采用。

生成读取器 **e939a566… /43,273B** 写死job ID、主机/PID日志名、产物名及部分答案文字，inert fixture还被改成匹配这些名字，违反用户的泛化要求。因此**不采用、不由审计者执行、不继续修复**。作者曾在inert fixture运行完整reader，并对真实旧日志调用解析helper；这不等于真实完整分析成功。104工具包含原产物JSON/SQLite/日志、安装库二进制与API只读检查和inert测试；未观察到模型/tensor运算/CUDA初始化/进程组/真实collective/live采集。外部文件系统不封闭，不作绝对无访问声明；本轮没有打印私有凭据。

下一步已落实为[一次普通优化候选生成](v6-ddp-pipeline-next-candidate-20261031.md)：保留原selection和当前02b933，原计划仅作为未验证证据加入，另明确传入用户泛化要求；没有传读取器代码、手写优化或审计纠正表。可以提出可被原质量/性能验证否证的假设，不再以补齐协议归因为前置分析工程。没有新质量、性能或接受结论。

以下启动记录保留为历史。

## 实际启动与早期核验

2026-10-30 **23:25:36.752019UTC**（香港10月31日07:25:36）启动controller **246575**，冻结 **cb34d3eaa55661a1f0df34678cf3d0f9ebd691ac**；唯一隐藏plain watcher **51296** 已通过实际waiting预检。旧44808 handled且进程不存在，没有旧角色/批次重启。

[23:27:11启动核验](evidence/v6-ddp-protocol-attribution-author-20261031/startup-verification.json) SHA **7c7b060d07cbb3c884b51014c63c67b29ac13734ea74900ff9d918b46ca2c789**：真实generation-b09ae091708d/session ses_e70ddb446ffe5mgY7t7ZOS27kJ收到17文件3,037,071B，逐字节与预检一致；1947冻结、当前0bc/02b933/source/核心与model/tools保持。实际prompt3164B为声明实验body加原575B wrapper，SHA b82e61f55838b22bc9e439c6751d9997b417b71e8ff7ac41489ce92cba9fa18f，无feedback suffix；不是生产prompt变化或对照实验。真实localOpenCode start-health-stop/server_urlnull，无model precheck。

19 assistant/28早期工具，private history SHAa876d33b24540d18ce0d9d89f921e421250b39c542a620a8d102f401ab371c49留服务器。[早期工具核验](evidence/v6-ddp-protocol-attribution-author-20261031/startup-tool-review.json) SHAe74c38dade19bd6f1fa6be055518efd1259f40f11415154061c6820956d830ae：1962显示read行、3截断prefix一致；18–27十个纯JSON输出完整重建。实际读取question/interface/index/原selection/当前02b933/原训练源/adapter/task；四collectorJSON均解析但不同字段有输出截断，不能声称全文消费或正确推理。工具1–3先读错误根目录路径，随后改inputs路径，原错误保留。目录列表有配置文件名，不等于内容读取；此前缀无模型/tensor/CUDA/processgroup/collective/private-config内容命令。仅早期记录，不能代替最终全部访问/生成代码审计，外部FS非hermetic。

启动时GPU空闲、无jobs、controller活跃，150GiB盘余9,380,257,792B约8.74GiB，无清理。新角色尚未证明协议归属、代码正确性或任何质量/性能接受。完成后按监听执行完整审计，再明确下一项有界工作，不自动执行reader或再开修复链。


这是一次明确有界的证据获取，源于[普通selector完整终审](v6-ddp-pipeline-communication-selection-20261031.md)后的具体缺口，不是自动候选链、prompt对照或reviewer堆叠。问题是：原始当前代码、采集、日志和安装库源码能否识别steady AllReduce的实际协议，以及配置/communicator创建的生效范围；无法证明的部分明确unknown。没有指定所需protocol、group、算法或优化答案。

现有 `ProjectAgents.generate(diagnostic_author)` 只生成 `artifact-reader.py` 和 `evidence-plan.json`，syntax/schema验证；controller不执行生成reader。新实验question/interface/body独立声明，生产role prompts/model/tools/core不变，没有feedback repair段。代码只读JSON/text/SQLite/相关已安装源码、输出有界报告；禁止CUDA初始化、模型/tensor运算、进程组、真实collective、profiler、live workload、参数试验或修改env/config。该接口不调用原evaluator，不是已关闭runtime observer的再次修复。工具实际行为和所有输出仍需终审，外部FS非hermetic。

只输入当前02b933原代码/原request、adapter、原训练源/common/README、原search/ownership、四个当前collectorJSON、原selection（作为假说）、question/interface/index和正常自动environment。共17文件3,037,071B；原selection c1e508b9字节不改。索引引用当前五个原始capture的requests/results/logs/SQLite/JSON及SHA，不复制raw traces/controls。**不输入终审、单位纠正、因果表、手写实现、旧repair答案**。既有source/metric/CPU事实辅助沿版本历史保留，不声称未经辅助或单变量AB。

源current0bc64a5a与singlefixed-fused-pipeline、空参数/自身fine8、代码02b93360…25781B不变。预检1947冻结身份，核心ed24af7…163文件；实际盘余9,400,045,568B约8.75GiB，GPU空闲。新256MiB输出/框架空间预测加8GiB警戒通过，非API cap；无清理/新controls/referencecopies。APIobserve-only、正常schema/provider行为原样，错误保留。private配置/history只在服务器，禁止未过滤metadata输出凭证。

生成后审计真实snapshot/工具/代码/错误及可答性，再决定有界CPU读取。若仍不能建立归属，保留unknown和具体缺失证据，不自动再次修复/选择/候选直到满意。没有新质量/clean/formal/independent/生产接受；02b933独立57.8545851%为已完成联合收益，不能解释为纯通信份额，也不与formal10.8304661%或旧收益相加。Qwen/最新拒绝FSDP/77ac/observer及eef828 correction边界保持。

启动记录随后追加。只为本新批创建一个隐藏普通终态/磁盘监听；旧44808已经handled/退出，无Goal、模型轮询、投递测试或旧批重启。low_disk_space不是完成；仍活跃时按既有授权安全处理且保留全部数据，不杀worker/盲删。
