# 已接受 DDP 版本的正式发现与通信诊断规划

后续[普通accepted-version一轮](v6-ddp-accepted-communication-trial-20261030.md)已完整审计拒绝：初始化Tensor索引异常使新候选退回原始eager；旧accepted Graph/source/本页discovery不变，通信量仍13 AllReduce/rank。原证据不因新失败而覆盖。下一步只做[一次普通selector决策](v6-ddp-communication-selection-20261030.md)，无新GPU或自动候选链。


终态已审计：2026-10-30 11:47:13 UTC，事件为 `batch_finished`，不是低磁盘告警。审计 SHA `51dfcb11aad2539abc8b579422791b0ce6d1c52b97aac603a4bfdae58d583f48`，小型公开记录 SHA `d172542ef245535ef3488e682b8befd99cafe10262515d97a13b8775d53d8079`，见[审计记录](evidence/v6-ddp-accepted-communication-evidence-20261030/terminal-audit.json)。548 个冻结文件、源码/adapter/entry/core/设备/controls 均核验；原 source current4541、独立 current7a49 保持。新分支1fc24545仅附加实际发现的 plan/registry，原代码ccd90/单bundle/空参数/contract未变。四任务八rank均status passed/valid true；没有新候选、搜索接受、完整模型质量检查或clean样本。校准的20个rank样本仅属合成collective正确性与耗时。

两个rank各4106 kernels、1052 calls/387 groups，合并registry1055 calls/390 groups；实际退役1419调用/619组、新增948调用、失效2项旧证据。lineage是存活调用身份或已安装bundle包含关系，不能证明多bundle共存或独立组归因。原stale empty discovery不再用于本分支。

当前discovery与独立timeline的原始trace均逐条复算。每rank每次均13 AllReduce，12×28,351,488B+157,541,376B=497,759,232B逻辑输入，仍无通信量减少。discovery每rank2次cudaGraphLaunch；r0/r1 NCCL kernel sum192.993544/33.218863ms属于该扰动采集。另一次timeline墙钟83.081355/83.052759ms，sum35.766444/39.508218ms，各4106 kernels、2 Graph launches、148个梯度callback和13个bucket；这些callback不能代表四个microbatch的全部反向Python事件。CPU时钟界限下最后一个collective提交差约2.596–2.768ms，不能解释为GPU到达差、wire时间或原discovery异常的唯一原因。timeline序号52–64不同于discovery51–63，禁止跨采集按序号拼接。所有sync、clock、interval、gradient事件和cross-rank提交界限已从自己的raw trace重算。

真实planner自主选择nccl_tuning、collective_calibration、execution_timeline，没有新Nsys。tuning日志的大载荷RING/SIMPLE、2 channels包含setup/warmup/peer检查；另外trace的RING_LL仅为kernel入口名称，不能冒充该次tuning决策或逐事件join。实际worker channel记录SHM/direct/direct，不是从NODE/P2P能力推断。校准两种payload的全部20样本、median与逻辑速率核验；不是497MB单次模型通信、带宽下界或可减去的关键路径成本。

planner6输入262,792B全部等于初始snapshot，与初始planner的prompt/model/tools相同；12 assistant/14 tools、0 schema repairs，provider估计0.21640881636（不是账单）。全部工具参数/输出、目录读取、CPU命令及session parts留服务器。它只读初始输入和部分observations，bash仅计数/grep及JSON schema验证；未观察到外部答案读取或自行模型/GPU实验，FS仍非封闭。生成collection.json仅一次，未手改。其“没有CUDA Graph／没有候选”说法与实际安装accepted bundle及当前Graph API矛盾；“另一次timeline能确切归因先前rank不对称”也过强。初始六输入未显式携带当前bundle/code元数据是信息边界，不是已证唯一原因，不据此改生产prompt或再堆reviewer。

下一步明确为[基于当前registry的一轮普通DDP续行](v6-ddp-accepted-communication-trial-20261030.md)。采用现有普通selector/candidate接口的installed_bundles/source、当前collector材料和原框架accepted feedback，不注入上述人工审计结论，不手选group/bucket/实现。实际两卡空闲、150GiB额度余9,964,486,656B约9.28GiB；既往普通trial/feedback实际占用262,836,224/640,086,016B，故本次复用discovery/controls按1GiB增量预算并维持8GiB告警线，不清理、不设新API/collector上限。只有真实新接受才安排独立整版闭环；通信成功仍需机制证据和独立整步收益。此前独立36.1578233%计算调度收益保持，Qwen关闭、FSDP拒绝不重跑。下列为启动历史。

实际启动2026-10-30 11:43:03 UTC（香港19:43），冻结`21bcce8615d27c89afee7466dce33be2856a0cfb`，controller142528，唯一隐藏监听50252；旧53536已退出/handled。实际本地OpenCode start/health/stop通过，无模型测试prompt。11:44:10启动核验SHA `ba0fd682282213b90807f0472f93a7051a703119a7e6fea64f93c29d876e1217`：548冻结身份通过，discover-27942843ca双rank完成，隔离branch新registry版本`1fc24545217087adefa1248704619fcdc882a19057336b231d8e02918bf4d0d8`保留原源码/bundle/contract，source current4541不变。

真实collection-planner generation-fa06929739bd/session ses_e7360b60effeKmiOf83y9r5NRJ已启动，6初始文件262,792B逐字等于snapshot，9 assistant/11工具的私有history已留服务器。与旧普通初始planner generation-fb697d303320相比prompt/model/tools一致；六文件只含request/actual observations/coverage/available collectors/schema/worker environment，无反馈或候选输入。此为初始身份/活动核验，不是全后续读取、collector成功、GPU质量或新接受证明；终态还需完整工具/输出/所有collector审计。发现worker退出后两卡空闲，planner仍在运行，盘余9,988,993,024B约9.30GiB。

[独立整版验证](v6-ddp-feedback-independent-20261030.md)已接受36.1578233%完整step收益，原搜索39.3258618%另列。但实际通信仍13 eager AllReduce/497,759,232B逻辑负载，当前证据支持计算Graph调度，通信优化成功缺口未闭合。实际检查发现source `ddp-communication-feedback-20261030/ddp-search/discovery.json`仍引用旧空baseline e925，而current为新接受4541；此前post-evaluation Torch profile没有更新其registry。不能据旧可优化范围继续候选链。

因此声明一个有界证据步骤，根`/root/autodl-tmp/dev/v6-work/ddp-accepted-communication-evidence-20261030`：从精确接受版本4541830b分叉，仅调用一次现有`ProjectSearch.discover()`，真实profile后由既有collection-planner按原输入/模型/工具/提示选择既有collectors。此API正常将新plan/registry附在隔离分支；源码、bundle、contract逐字不变，source accepted current4541和独立current7a49保持。**不运行selector、candidate、trial、参数探索或新独立验证**，不手选组/bucket/算法/collector，不改生产prompt/sampler/acceptance。诊断结果只决定下一证据步骤，不自动追加candidate或无限链。

精确代码`ccd90dc158788a164d6fda242572530e9cdc9ca50563af1a1d9d597f0ac12e0b`、单bundle `trial-d06aa22e2d-param-a72285d2`、空参数；不导入旧历史Graph。原current-device任务/request/adapter/controls和完整工作/精度/owner保持，原anchor133.379621/limit140.04860205ms不重设。共享只读约4GB controls，没有源/model/control重跑或参考复制。既有manual源/metric映射辅助与当前代码对象/回退/全局flag限制保持。

collection-planner只收到框架原request、available collectors、此次真实communication observations、coverage、schema及自动worker environment；没有开发者终态事实表、独立接受结论、手写通信方案或reviewer答案注入。必须审计实际快照/读取/全部工具/正常schema repairs，外部FS不是隔离；禁止凭输入交付宣称正确消费。planner本来只规划不自行跑工具实验，其实际行为仍须核查。它选择的collector可以启动真实worker，不能称整个步骤无GPU活动。没有人为强制Nsys或“必须减少AllReduce”的实现答案。

profile/collector完成不是新的完整GPU质量或性能接受；run-only没有observe/compare。新registry lineage只说明调用身份/已装bundle包含关系，不证明多个active bundles、独立组归因或唯一机制。通信trace逐条自己的时间窗口复算，真实kernel/API/channel和逻辑payload与物理wire/关键路径分开。不能把不同profile减去clean wall，也不能从NODE/P2P能力推实际transport。当前完整质量由已审计版本独立验证支持，不能推广为任意owner或并发安全。

控制器[scripts/experiments/v6_ddp_accepted_communication_evidence_20261030.py](../scripts/experiments/v6_ddp_accepted_communication_evidence_20261030.py)只编排既有API，core保持`ed24af7acc8caa15578e0b859c85ef8a901be51354448ed39094606ed9e10a2f`。预检548冻结文件/两卡空闲/10,058,711,040B余量通过。实际先前同代码profile占45,445,120B、六诊断合计39,698,432B；给新discovery/planner/collectors预留1GiB并保留8GiB告警线。没有清理或并行GPU，不根据完成标签预设collector成功。

实际launch/startup另记，长步骤仅一个隐藏完成/磁盘监听。旧独立53536已handled，旧批次不重启；所有后台/SSH/queue显式CREATE_NO_WINDOW，不关闭VSCode终端。raw/private configs/snapshot/history/trace留服务器，GitHub仅代码/hash/小摘要。若低空间先查实际阶段/GPU，不终止实验或盲删；按既有授权只在GPU退出后处理明确冷范围并保存恢复journal。

通信优先、通用优化目标不变。Qwen链关闭，FSDP最后拒绝不重测，既有DDP失败保留。此步骤使研究继续获取当前已接受执行的有效证据，既不把一个批次收尾当作全项目停止，也不以重复搜索到通过替代证据。
