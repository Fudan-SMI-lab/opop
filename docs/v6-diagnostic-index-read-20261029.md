# 针对未读取诊断索引的有限提示实验

后续[精确生成代码固定执行](v6-diagnostic-index-read-fixed-20261029.md)现已完整审计：Qwen f065/FSDP beb5默认K2通过12次完整连续质量和原约束、保留30样本/48通信trace。这只确认两份实现可执行，不修正proposal过度因果主张，不证明433B段落稳定优越，也不把部分索引读取称完整自主诊断闭环。段落实验仍关闭、生产不采用；仅接续一次声明的[整版独立AB/BA](v6-diagnostic-index-independent-20261029.md)，无新角色或参数搜索。

## 终态与本机重启后接续

本批于 **05:37:35 UTC（香港13:37）** 正常结束；本机重启不影响远端controller。旧本地监听最后检查05:22:28后中断，未投递完成通知；本轮用户检查时直接处理终态，已标记旧监听handled，不补发/重启旧实验。服务器GPU空闲，df约10.74GiB，无本轮清理或raw本地转存。

[终态小型审计](evidence/v6-diagnostic-index-read-20261029/public-audit.json)对应完整审计SHA `e761df4c0dfcc27ec89a44923e96fc2ae79b15220dcf86ea8d94c65f0a625285`。两角色实际初始32/37文件与前批逐字相同，只有同一433B prompt追加，model/tools/core不变，122冻结哈希保持。Qwen39 assistant/52工具、FSDP62/75工具，均0 schema修复；provider估算1.83347941892/2.96689327026，不作账单或调用限制。

**两角色这次确实打开了索引和部分观察记录**，与前批只暴露指针而未打开不同。但两者仍未阅读Agent诊断脚本/plan，未重算raw输出，故段落只部分遵循；一次重放差异不证明稳定因果优势。关闭本轮有限提示实验，不继续改段落直到成功、不增加reviewer，也不自动采用生产prompt。全工具历史、命令、首次生成代码和每次edit留服务器audit；没有观察到读取后续接受实现或interpreter/verifier答案，非隔离文件系统不能证明全部外部访问不存在。FSDP目录列表暴露audit文件名，不等于读取答案。

Qwen tool16实际读取索引，17–20/26读取candidate诊断JSON、phase事件及B/C摘要。正确获得两次greedy对冻结controls一致、第一次teacher一致、第二次teacher发生12 token/max31.125差异；独立源摘要只覆盖firstgreedy。它没有载入保存的完整张量。tool19曾按infer完成事件的ordinal分组，错把下一次greedy的routes分到teacher，tool20随后打印完整序列；不能以缺少teacher标签/慢计时直接证明原生路径。stride差异有binding代码支持，但proposal将可变cache解释写成唯一原因、排除graph/pool等机制，以及用run-only诊断证明路径健康，均越出实际证据。观测计时不能保证干净TTFT约束。

Qwen代码SHA `f0656da99b5d90d11fb8ec8d1a2649e196de5761b9ecfda926ec1deda146a2fe`，43,200B，空参数。仅prefill返回新cache/layer对象；decode实际返回预建mirrors[step]，不是每次返回都fresh。仍共享master tensors，以首层data_ptr及各层shape识别/刷新；任意混合alias/dtype/并发owner不获保证。31decode Graph加独立prefill pool、capture限定全局patch及rolled/non-greedy/setup直接调用检查；这些没有真实经过框架dispatch，也不是原样两次run/observe的完整顺序。实际runtime只有异常fallback，不自动检测所有数值漂移。CPU工具保留缺失cache_messaging import（复合命令exit0）、非法import语法/类layers AttributeError；view更新探针比较的是更新后的同一属性，不能当旧view身份验证。最终两层prompt8/4step guard/refresh打印与语法/import通过，均不是完整CUDA质量。

FSDP tool38/39/42打开索引，43/45读取B/C rank0诊断摘要；另读原failure输入/rank0原结果，未读rank1或全部path统计。proposal唯一shard-GEMM原因、transport修复已证明、一般确定性、每collective时长完全位于关键路径和K2无条件显存界均不采纳；引用路径误写dev-v6-work，实际工具访问为dev/v6-work。新诊断只覆盖K6，不能外推0/8。初始list.get/安装路径/grep错误、错误mock clamp断言、rollback UnboundLocalError与部分回滚失败均保留；Agent自己修正了代码，最终11项CPU mock通过。

FSDP新代码SHA `beb542b6fcf112d20c829b9c6860f3bec881863917e64f2e39bc19578446566a`，3,559B；parent forward原样委托，仅trailing K的reshard policy。默认K2，choices0/1/2/3/4。在setter前保存每group的post_forward_mesh_info，失败时恢复；成功策略持续整个worker生命周期，rollback仍吞掉恢复异常，不能作无条件恢复保证。tool72确实执行Gloo world_size1、4block/hidden32小模型，复制安装逻辑而非导入精确candidate；每个update后reset、下一次新随机输入，不是原两步连续质量/双rank通信或独立参考检查。project_agent_config隐藏CUDA设备，实际CPU参数成功运行与CPU默认mesh一致，不能仅凭脚本“CPU”标签作范围结论。

下一步按[原样固定验证](v6-diagnostic-index-read-fixed-20261029.md)各B/C一次，Qwen空参数/FSDP声明默认K2，不测其他参数、不改代码。只检验真实生成实现，不能替其因果措辞背书；没有新搜索/正式接受/生产采用。所有历史接受、拒绝和样本保持。

已于 **2026-10-29 05:02:38 UTC（香港13:02）** 启动，冻结 `c7e2189a242f9cdb60fc5c4b23f69309b8a72e12`，controller968454；真实本地OpenCode health预检通过，server_url=null。见[启动](evidence/v6-diagnostic-index-read-20261029/launch.json)、[预检](evidence/v6-diagnostic-index-read-20261029/preflight.json)。生产core保持。

05:04:19启动审计：Qwen generation-31d64128862c/session ses_e79f5ef19ffeD4N1q1jsgQ7Ffu，32初始输入逐字相同，只改433B prompt，model/tool配置/core相同，122冻结文件复核。真实9assistant/14工具，当前已读原任务、selection、adapter、binding、历史candidate、feedback和原run-only诊断；此时尚未观察到打开新增索引，不能提前判断后续读取或最终成功。完整工具/消息快照留服务器startup-access，公开[部分启动审计](evidence/v6-diagnostic-index-read-20261029/startup-verification.json)。FSDP按序执行，不是额外并行任务。

唯一普通完成/8GiB监听PID7216已启动，[正确root预检及回执](evidence/v6-diagnostic-index-read-20261029/wakeup-setup.json)确认controller运行；CIM核对仅一个watcher进程。旧14176已处理并标记退出，不重启。无投递测试/模型等待轮询/Goal。当前df 11,618,119,680B（10.82GiB）、GPU空闲；相较清理后的变化包含新角色文件/运行环境，不臆测为某一产物。终态后审查完整工具/代码/错误和证据消费，未预设生产采用或GPU验证。

前两次candidate重放都有索引文字可见，却没有打开索引或其中的观察文件。这是已经实际确认的输入消费缺口。本次从那两份**生成前初始快照**分别重放一次既有candidate角色，唯一变化是在prompt末尾追加同一通用段落（含分隔换行433 bytes）。32/37个输入文件、原失败feedback/候选/selection、model/tool配置和core全部保持，不输入新固定GPU结果、审阅报告、人工事实/原因或修复实现。CPU快照预检通过，122冻结文件核验。

> When prior feedback supplies an evidence index, open that index and inspect the relevant observation artifacts and their producing code before choosing a repair or asserting a cause. In the rationale, cite the inspected artifact and location, distinguish directly observed facts from hypotheses and instrumentation limits, and state any remaining evidence gap; an index pointer or a diagnostic status alone is not a quality result.

两任务各一次真实请求，不追加第三个reviewer、不重复核验到通过。这是与历史原样prompt的有限对照，单个输出差异不证明稳定因果优势；任务之间不是单变量A/B。此前accepted-feedback措辞研究和verifier研究仍关闭。本批段落仅为隔离实验，**production prompt、sampler、allocator、接受规则不变**，core仍 `ed24af7a…`。

Qwen初始快照generation-9e85d07b98af，历史失败a866；FSDP初始generation-433d566258dc，历史失败86e的df3d代码与原多参数feedback。原selection可能含错误推断，不手动修正；Agent需要自己把原反馈与已有观察范围核对。索引仅列原始来源/参考、Agent编写的诊断代码与固定B/C产物，没有采用过的解释/核验答案。FSDP新诊断只涵盖K6，不外推其他参数。环境文件系统可访问，终态必须审查完整工具路径/输出/脚本，不能只看初始文件名或关键词断言无污染。

判断重点是实际是否打开索引、阅读了哪些证据和产生证据的代码，是否引用准确、区分观察/假说/工具效应、保持工作量/精度/质量/约束，以及新代码和参数究竟改变什么。输出schema通过不是质量或推理正确；CPU/mock/语法探针和异常各自保存，pipeline exit0不掩盖内部失败。保持API observe-only，不设置新调用上限。控制器不安排GPU/model/search/接受；原candidate工具可自行探针，需审查实际行为后才能报告活动范围。

若未读取或仍有事实错误，保留结果，不重复直到成功；若有可用代码，也须先审查再决定是否需要声明的原样固定执行。两份既有固定代码已通过质量但未正式AB/BA，本批不以它们充当新优化接受，不追加独立重测来替代工作流缺口。没有自动生产collector/角色或固定失败路由。

控制器 `scripts/experiments/v6_diagnostic_index_read_20261029.py`；服务器目录 `dev/v6-work/diagnostic-index-read-20261029`。启动前核查真实150GiB容器df/GPU与冷维护结果。raw和private配置留服务器，只上传代码/哈希/小型脱敏回执到GitHub v6；新批配一条普通完成/8GiB监听，禁止Goal、模型轮询和额外投递测试。

启动前授权冷维护已完成：仅旧DDP/ResNet明确worker RUN JSON，229 payload/1,601保留文件/10当前版本/6参考校验，实际净释放2.63GiB，余11.11GiB。恢复journal独立保存在 `storage-audit-20261029/diagnostic-index-read/run/`；本批将访问的索引、观测、参考和源码均未压缩。见[上批审计及维护](v6-diagnostic-index-fixed-20261029.md)。
