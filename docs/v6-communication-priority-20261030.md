# 通信方向优先级调整

最新终态：accepted-version DDP 普通一轮已完成完整审计，trial-02ed5436b4 **拒绝**。父版本两组71.284842/72.901868ms，候选108.537192/111.151905ms；收益-37.252350/-38.250037ms，噪声门槛4.525422/0.450708ms。50个clean rank样本、20次零误差完整检查、30条独立通信trace、647冻结身份均核验。候选初始化 `params.index(s)` 触发Tensor布尔值不明确，实际退回原始eager，未执行拟议并发Graph；每rank仍13 AllReduce/497,759,232B，不能称通信优化成功。完整审计SHA `0ce0048441d64d0dd4fd20dcf52463b83561c24a91f4cefa07d0518a7f542131`。原accepted/current与历史拒绝保持，无独立验收或重试。

下一步明确为[一次隔离的普通selector决策](v6-ddp-communication-selection-20261030.md)：只用本轮原始拒绝反馈、历史失败代码/原始sidecar和不变的当前accepted discovery，保持原prompt/model/tools，由现有selector自主选择范围和假设。本步不生成candidate、不启动GPU或collector、不注入审计因果答案；输出后审查证据依据再决定是否值得实施。当前两卡空闲、余约8.54GiB，不足完整普通轮次1GiB预算；仅为小型角色材料预留384MiB，保留8GiB警戒线，无清理。旧13432已handled；新启动与唯一隐藏监听另行记录。Qwen关闭/FSDP拒绝不重跑，研究方向仍为通信，不把本轮收尾当作整个研究结束。

以下运行记录为历史。


当前实际运行（香港20:05核验）：[一轮accepted-version DDP普通续行](v6-ddp-accepted-communication-trial-20261030.md)已20:03启动，冻结2a0bb13a/controller145514/唯一隐藏监听13432。647身份、真实selector41文件50,078,774B及原accepted feedback route核验；原prompt/model/tools、当前installed bundle/source与刷新registry交付。12 assistant/16工具中已实际读取旧质量/配对/约束及已安装代码，当前collector仍仅早期索引曝光，终态完整消费审计待做。两卡空闲/无GPU jobs、余约9.14GiB；无新接受或生产变更。首次launcher旧文件名预检中止于实验启动前，错误保留、只修正部署路径。旧50252 handled，Qwen关闭/FSDP拒绝不重跑。以下为前批终态与历史。

最新终态（香港19:47完成，审计已落盘）：[已接受DDP执行的发现/通信证据](v6-ddp-accepted-communication-evidence-20261030.md)四任务八rank完成、548身份保持，审计51dfcb11。新registry1055 calls/390 groups，原代码/单bundle/contract不变；每rank仍13 AllReduce/497,759,232B，当前采集2 Graph launches。真实planner选择tuning/校准/timeline，12 assistant/14 tools/0修复；其“无Graph/无候选”和跨采集归因过强的说法不采纳，原输出保留。没有新增完整质量/clean/优化接受。

下一步已明确为[一轮普通accepted-version DDP续行](v6-ddp-accepted-communication-trial-20261030.md)：共享刷新后的registry/当前诊断/原controls，原accepted feedback逐字不变，现有Agent自主选范围/实现，不提供审计答案或手选通信组。两卡空闲、余约9.28GiB；既往最大普通feedback批次实际640MB，本次估算1GiB增长并保留8GiB告警线、不清理。仅新接受才另做独立闭环；通信成功仍需实际机制及独立整步收益。Qwen关闭、FSDP拒绝保持，旧50252已处理。以下为历史，启动核验另行记录。

迟到通知去重（香港19:46）：独立批次11:34:33UTC的完成事件在其完整审计及后续证据批次启动后才送达。重新核对terminal hash、审计9d85463c及source/independent指针一致，按已处理事件去重；没有重跑模型、样本或旧监听。当前controller142528/唯一隐藏监听50252仍运行，GPU暂闲、实际盘余9,988,984,832B。本轮同时修复上述新增中文段落经PowerShell管道编码变成问号的问题；原实验/审计JSON/冻结输入不变，错误文档版本保留在git历史。

当前实际运行（香港19:44）：[DDP已接受版本正式发现与通信诊断规划](v6-ddp-accepted-communication-evidence-20261030.md)已启动，冻结21bcce86/controller142528/唯一隐藏监听50252。548冻结身份和新discovery双rank核验，隔离branch1fc24545仍原代码/单bundle/contract，真实collection-planner6文件262,792B、原prompt/model/tools。没有selector/candidate/trial；source4541/独立7a49不变。独立36.1578233%完整step接受已收尾，但13 eager AllReduce未减少，通信成功缺口保持。当前planner运行/两卡暂闲/余约9.30GiB，终态接回完整证据审计后再决定有界下一步。Qwen关闭/FSDP拒绝不重测。以下为历史。

最新独立终态（香港19:34）：[DDP精确整版B/C/C/B](v6-ddp-feedback-independent-20261030.md)完整审计接受，112.6878425→71.9423715ms（36.1578233%），40样本/16零误差全检查/24通信trace/483身份保持，publish=False/current7a49不变。搜索39.3258618%另列；仍13 eager AllReduce/相同逻辑payload，只能归为计算调度收益，通信正例缺口仍在。下一步[已接受版本正式发现与诊断规划](v6-ddp-accepted-communication-evidence-20261030.md)，因为现有registry还是旧空baseline；仅新隔离分支discovery+既有真实collection-planner/自主collectors，不生成candidate、不搜索、不手选通信算法。两卡空闲/余约9.37GiB，实际历史同代码profile+六诊断共约85MB，预留1GiB；无清理。Qwen关闭/FSDP拒绝保持，53536已handled。以下为历史。

当时实际运行（香港19:34）：[DDP原字节独立整版B/C/C/B](v6-ddp-feedback-independent-20261030.md)已启动，冻结49aa288f/controller140320/唯一隐藏监听53536。483身份与实际B/C/C请求核验，两rank GPU工作中、余约9.51GiB，无独立接受预言。原搜索39.3258618%保留；native仍13 eager AllReduce，计算Graph收益不自动归为通信成功。Qwen关闭/FSDP拒绝保持，完成后接续终态审计和下一有界通信证据步骤。此为启动时历史。

搜索终态：DDP原始失败反馈普通试验已完整审计并搜索接受新代码ccd90dc1，118.214785→71.725802ms（39.3258618%），50个rank样本/20次完整零误差检查、30通信trace、312冻结文件保持。实际native每rank2 Graph launches、2981 Graph kernels，但仍13个eager AllReduce/497,759,232B逻辑负载；目前归为计算调度收益，不称通信优化成功。详见[终态审计](v6-ddp-communication-feedback-20261030.md)。当时声明[一次独立整版B/C/C/B](v6-ddp-feedback-independent-20261030.md)：精确接受4541830b/代码ccd90/空参数，对原当前prepared7a49，publish=False、原质量/140.04860205ms约束、所有样本/通信profile保持；仅hash复用已审计源与当前native，无新Agent/搜索/Nsys/生产变更。483文件与1GiB预算预检通过。搜索结论不被独立结果覆盖；通过也不自动算通信接受。Qwen关闭/FSDP拒绝保持，旧监听12400已handled。

当前实际运行（2026-10-30香港18:25）：新[DDP原始反馈普通续行](v6-ddp-communication-feedback-20261030.md)已18:23启动，冻结6c680992/controller125768，唯一隐藏监听12400。312冻结身份/32初始文件/实际原反馈route已核验，真实selector已读反馈片段与旧代码；无GPU job或新接受预言。标准有反馈prompt后缀有别于前次无反馈selector，模型/工具/生产core不变。约10.18GiB余量；Qwen关闭、FSDP拒绝保持，结束通知接回完整审计和下一证据步骤判断。

2026-10-30香港18:12接续：用户明确期待结果分析后继续推进。前一轮审计已完成，但结束后没有新controller/监听，空闲约7小时，现修正“有界轮次收尾即整个研究停止”的处理。准备[一轮DDP原始反馈续行](v6-ddp-communication-feedback-20261030.md)：仅未修改framework failure与历史candidate/诊断，复用当前空baseline discovery/controls，由现有Agent自主决策，无开发者状态因果表/修复答案、手选bucket或生产变更。新步骤逐批声明并保留失败，不是旧candidate固定重测或无限到通过；Qwen链关闭、FSDP拒绝不重跑。实际启动、隐藏监听和输入核验随后记录。

最新终态（2026-10-30香港11:11完成，11:27审计）：[当前DDP一轮](v6-ddp-communication-trial-20261030.md)拒绝`no_valid_parameter_candidate`。18 rank worker退出正常，但screen第二连续loss两rank失败，0 clean/0配对/0新接受，不安排独立重试。捕获诊断有AccumulateGrad流引用异常；回退恢复初始参数，第二实际loss精确等于已有第一步control，支持连续状态被重置的代码/顺序解释，但无完整参数转移trace。候选timeline仍13 AllReduce/rank，拟议one-flat-buffer未成为通信成功证据。115身份、完整角色记录、4timeline/2baseline Nsys/校准样本已审计，SHA `70b28f9d3d09ed8078461a325a40e0ce021a5c00e3b13d12e4bb5fff467746bf`。本次FSDP/DDP有界轮次均收尾；Qwen链关闭，拒绝不重跑。后续通信工作仍需完整连续更新/回退状态、原质量、实际native机制和独立整步收益，不能靠Graph、NCCL sum或重试取得接受。当前GPU空闲、约10.24GiB余量，无新后台试验或生产改动。以下运行描述为历史。

当前实际运行（2026-10-30香港10:12）：[一轮普通DDP通信试验](v6-ddp-communication-trial-20261030.md)冻结65dfbef4/controller100672，空baseline新发现已通过，真实collection-planner开始，115冻结身份/6初始文件262,446B已核验。唯一隐藏监听44088，约10.41GiB余量。没有新候选质量或接受结论；FSDP拒绝封存、Qwen链关闭，不并行GPU、不手选组/实现或更改生产。

当前推进：FSDP一轮已审计拒绝，不做独立重测；[DDP当前设备准备](v6-ddp-communication-device-20261030.md)于01:58:19UTC完成并通过完整源/质量/跨rank负例审计。新空基线133.379621ms，原1.05上限140.04860205ms，10样本/4reset/2identity/6条通信trace，每rank13 AllReduce RING_LL。源梯度/参数有微小容差内差异，replica exact。准备在监听创建前结束，未建listener或排队。下一步仅[一轮普通DDP通信试验](v6-ddp-communication-trial-20261030.md)，先补足2GiB预算；Agent自主选组/实现，Qwen链保持关闭。

最新终态：本轮 FSDP 已完成并拒绝，完整审计见[一轮通信试验](v6-fsdp-communication-trial-20261030.md)。质量和原显存约束通过，实际每 rank AllGather 49→25、ReduceScatter 25不变；但配对第二组176.066436→223.383338ms，整体未改善且未超过噪声门槛。90个clean rank样本、36次零误差完整检查、216条独立通信trace、4份Nsys SQLite及84冻结身份已核验，413–441ms长尾全部保留。不追加FSDP参数或独立重测，Qwen链保持关闭；下一步DDP当前设备空基线、完整源验证与通信试验，先满足空间预算。旧启动描述保留为历史。

香港08:21状态确认：Qwen独立收尾与FSDP新基线的两条迟到通知均按已有审计去重，未新增批次；当前仅一轮FSDP普通通信试验在selector阶段，DDP仍为后续任务。完整质量、原约束、独立整步收益与实际通信机制证据的成功标准不变。

执行更新（2026-10-30香港08:14）：当前设备FSDP基线与源验证已通过，现已实际启动[一轮普通通信试验](v6-fsdp-communication-trial-20261030.md)，冻结f200b79f/controller75690，唯一隐藏监听18872。先新baseline发现，再由Agent自主选范围/参数/实现；无旧K3、手选组或生产更改。当前尚无新优化结果。Qwen必要独立收尾完成，DDP随后；下列较早“正在等待Qwen”的描述仅保留历史。

当前主线已转入通信（2026-10-29 23:53 UTC／香港10月30日07:53）。[Qwen 独立收尾](v6-qwen-cache-independent-20261030.md)完整 B/C/C/B 接受：721.4966495→374.1322145 ms，48.1449824%，20样本/8零误差完整检查、原TTFT约束通过，publish=False；搜索48.0946416%另列，非跨设备增量，Qwen候选链结束。

[当前设备 FSDP 基线](v6-fsdp-communication-device-20261030.md)已真实完成，两rank共10样本、4 reset/2 identity检查、4次完整独立源比较及负例通过；24条通信trace全部复算。先按样本取rank最大再中位数，step175.753368 ms；原生每rank49 AllGather/25 ReduceScatter RING_LL，不能据此推出物理transport或关键路径。原peak上限为新测量2239245312×1.05=2351207577.6 B。

接续[一轮普通 FSDP 通信试验](v6-fsdp-communication-trial-20261030.md)：从当前空7231 baseline共享新controls，重新发现，由现有Agent自行选组/参数/实现，原完整工作、质量和接受规则保持。先补足磁盘预算，串行启动，不安装旧K3；只对实际新接受版本另做独立整版收尾，DDP随后。以下为历史记录，不代表旧批次仍运行。

用户于2026-10-30香港时间要求：保留原始核心目标，将研究重心更多放在通信。此要求调整后续任务优先级，不修改正在运行的冻结实验、生产提示、采样器、分组算法或接受规则。

核心目标继续是通用项目优化闭环：Agent从项目与目标理解完整执行，按真实证据选择范围，生成候选，完整回装执行，读取失败或接受反馈，并取得可独立复验的项目收益。通信作为同一流程中的重点资源；任务名、硬件型号、固定collective或手写修复答案不进入通用控制逻辑。

## 已有证据与缺口

- [FSDP首次搜索](v6-fsdp-search-20261028.md)已有一个通信相关正例：Agent选择K3参数保留策略，每rank原生AllGather从49降到46，ReduceScatter仍25；完整版本独立176.517374→171.053986ms，收益3.095099%，完整质量和原显存约束通过。baseline先有开发者证据辅助的源owner修复，不能称完全无辅助原入口。
- [FSDP续行](v6-fsdp-continuation-20261028.md)的2.157792%是同有效K3策略相对原baseline的另一组独立测量，不与3.095099%相加，也不是新增机制收益。后续prefetch/K扩展及新K2独立噪声拒绝均保持，不重复测到通过。
- [DDP整步正例](v6-ddp-step-followup-20261028.md)的Graph40.61%和另一批batch55.73%分别保留；原生collective数量/逻辑负载没有随之减少，不能将整步收益直接归为通信优化。[旧DDP通信候选](v6-ddp-tp-diagnostics-20261028.md)独立拒绝保持。TP完整32-token质量失败保持。

因此重点缺口是：在当前设备上，Agent能否从通信证据作出可验证的选择，取得完整质量/约束合格、独立整步收益，并在后续任务中复用该流程。较低NCCL kernel sum、少几个API调用或代码宣称异步均不能单独算成功；通信机制证据也不等于唯一性能因果归因。

## 执行优先级

1. 已运行的`qwen-cache-constraint-feedback-20261030`完成既定一轮及终态审计，保留全部样本/错误；若产生新接受版本，只做已声明且必要的独立整版收尾。之后不自动追加Qwen候选/提示/审阅试验，将后续GPU与真实角色预算优先转给通信。当前controller60363和唯一隐藏监听41852继续，不停止或修改冻结输入。
2. 首选双卡FSDP原完整训练任务。先声明当前设备准备协议：只迁移实际device UUID，复用已审阅的原任务/adapter/只读独立源参考，从空bundle原baseline开始；重新执行当前设备controls和锚点检查，比较完整源语义，固定本测量期的约束后再搜索。不安装旧K3、不重测旧K2直到通过、不把旧设备分数当新基线。当前基线adapter的人工源owner修复边界继续明示。
3. 准备通过后执行一轮明确冻结协议的普通真实搜索，由现有Agent自主选择组、通信/计算联合范围和候选，保留默认参数探索与真实反馈。使用完整训练step作为主收益，现有通信指标作为机制诊断；不人工指定K值、预取算法、分桶方案或手写候选。完整loss/logits/梯度/更新参数、连续两次更新、各rank实际分片、原精度、同步、总工作和显存约束均保持。
4. 对确实新接受的完整版本，安排独立B/C/C/B及单独原生路径验证。逐样本先取rank最大值再计算原中位数/MAD/噪声规则，全部慢样本保留。核对实际collective、逻辑负载、stream/依赖/等待证据和fallback，分别记录来源与测量范围。不同profile时期的中位数不可相减；NCCL sum不是wire time或暴露关键路径。通信机制有变化且整步独立通过，才记为当前期通信相关成功。
5. DDP作为后续第二个任务，用于检验同一流程在梯度同步场景的可复用性。先使用正确的原语义基线，不能靠修正一个故意低效baseline或纯计算Graph收益冒充通信正例。TP等仍有完整质量缺口的任务降低优先级。后续轮数和证据需求逐批声明，失败保留，不形成无限重试或reviewer堆叠。

## 当前硬件与启动门槛

2026-10-29 23:08:53UTC仅做只读检查，未运行GPU/model/collective：两张4090 UUID为`GPU-20e8003c-f531-d50c-084c-a9a97fe94004`与`GPU-a24af2e6-99bb-9729-5d14-50fa10a2b321`，`nvidia-smi topo`显示卡间NODE，P2P读写均CNS。不能复用旧GPU UUID，也不预设本机具备NVLink/P2P/RDMA；实际NCCL transport需由未来真实worker日志/trace确认。此证据只限定当前设备，不写成通用框架特判。

原FSDP request仍引用已消失的旧两卡，历史current仍3007940f、prepared7231e858，均保持。新一轮尚未启动，当前约9.20GiB可用空间不能直接视为足够容纳双rank完整controls及profile。下一步在当前Qwen结束/GPU退出后，先估算完整输出和参考的峰值占用；必要时仅按既有授权压缩明确旧冷RUN JSON并逐文件解压SHA校验。保护当前/冻结/accepted/source/controls/model/provider；已有只读独立源参考不复制，缺失冷JSON按各自journal服务器端恢复。保留8GiB告警阈值，不在空间不足时盲启双卡任务。

本次只修改研究优先级、计划与当前监听的后续指令；未启动并行通信试验、未清理数据、未更改当前Qwen输入或生产实现。后续新长批次仅一个隐藏完成/磁盘监听，不使用Goal、模型轮询、投递测试或raw本地转存。所有历史接受/拒绝、人工映射/事实/owner辅助边界和已关闭研究保持。
