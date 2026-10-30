# DDP 已独立接受联合版本的一次正常发现

本批已于2026-10-30 20:25:14.637825UTC完成，reason为`batch_finished`，不是低磁盘告警。终审`7cd9fa02d28a60b4fec48a69cb1c4365842e079cc928dd03db409d3f91da3ab9`，公开摘要[de751966…](evidence/v6-ddp-fused-communication-evidence-20261031/public-terminal-audit.json)。1310冻结身份、core、source current7a49及精确508a文件/bundle/contract保持。5jobs/10ranks均passed/valid，属于发现和4个正常规划采集器；没有新的完整模型质量、clean样本、候选或接受。GPU已空闲，实际150GiB磁盘剩9,982,173,184B约9.30GiB，无清理。旧51188完成已handled，不重启。

新current10992e0d真正附着9calls/8groups，旧registry退休1049calls/385groups，新增3calls、失效2份旧证据；继承关系仅表示调用存活或bundle包含，不是因果归属。两个原始发现trace各1105kernels、2GraphLaunch、1个497,759,232B AllReduce；1103kernels仅归到project层，不能把100%关联解释为全部算子精细归因。发现NCCLsum34.601452/33.079367ms；独立timeline约55.006329/55.008339ms、1AR、4sync、0梯度回调、1bucket。逐trace及跨rank CPU提交界限重算通过：29.304688–173.617188µs仅是本次CPU提交差，不是GPU到达、wire或另一capture的等待原因。0回调不表示无梯度。

此次校准实际覆盖124,439,808个Float/497,759,232B完整载荷，2warmup/5samples每rank，共10合成样本，原合成数值检查通过；median32.517463/32.509024ms，尾样本完整保留。这不是模型完整质量、性能下界或wire吞吐。单独tuning记录完整载荷RING/SIMPLE/channel0–1共3条，含warmup/setup，不与steady调用按序拼接；原生kernel入口仍为RING_LL，不将两者混成一个协议证明。

新Nsys `nsys-28991fc3d0`的两份真实SQLite均重新核对，各1105kernels/1103Graph/2个`cudaGraphLaunch_v10000`，correlation4:955、67:148；1个eager AllReduce在Graph外。r0 SHA`edf46c30f8068ebfed207d70b48f70e5bbf22a88e0edfa992fdf468008e1d508`，r1`900f619ff16abc841333418a9c199f9b0d5148c6d4d1d2a2be267bfa35dabb48`，normalized`3d23a95a534deaba937827fdff95f0c99167b8ab9c91e53cd52e290159af38f8`。本次真实日志SHM/direct/direct；native NCCLsum31.532702/321.602219ms为两个受采集扰动的rank期间，不能减clean时间、当wire或与timeline拼成critical path。stale AccumulateGrad等warning原样保留，无uncaught traceback。

真实规划角色35assistant/39tools/0schema repair，provider估计0.74683934934而非账单。6初始文件74372B及普通prompt/model/tools准确，全部39工具输入/输出已审阅，2756read行与原文件核对，JSON/拼接输出另行重建。工具10空git历史exit128，29/30空路径与shell结果保留。工具11读了自己的私有provider配置；19/22实际读本批launch/protocol/frozen controller，32实际读两个20261021历史collection计划。因此不再声称仅六文件或无外部答案暴露；这不改变真实worker数据。记录中未打开当前508a代码或自动复制的latest-feedback全文，未执行模型/CPU tensor/CUDA/processgroup/collective探针，只有JSON/schema操作；FS仍非hermetic。原计划“四次eager微批次”、确定bucket归属、校准分解wait/wire、paired测量隔离binding成本等说法不采纳，原输出不手改、不重跑planner。

审计首次元数据提取误带完整output/display字段，存在私有配置进入输出流的风险；后续只核对敏感字段存在性和哈希，已告知用户，未修改credential。单独[输出处理记录](evidence/v6-ddp-fused-communication-evidence-20261031/audit-output-handling-note.json)不含配置值；raw/history/config继续只在服务器。

下一项明确为[一次普通项目选择](v6-ddp-fused-communication-selection-20261031.md)：以原样508a新注册表、原始独立完整版本反馈和正常当前collector材料作输入，不加入上述人工终审表，不指定group/算法，不自动候选或GPU。不把本批发现当新接受，也不停止整个项目。以下启动记录是历史状态。

已于2026-10-30 20:19:44UTC（香港10月31日04:19:44）启动，controller210849、冻结`5a140af46d3efa7925323b8b8a70d3f9b4b5d53b`，唯一隐藏监听51188。local OpenCode真实start/health/stop通过、server_url=null，无模型prompt预检。20:22:06[实际启动核验](evidence/v6-ddp-fused-communication-evidence-20261031/startup-verification.json) SHA`be5b5d3a3323f64b0f42d36b1bbdd2304854dcfcaf4baef693b4bbd11866866e`：1310冻结、sourcecurrent7a49、精确508a文件/bundle/contract、request/prepared/core/adapter/controls/UUID保持。discover-0ee3a52759两rank通过，新隔离current10992e0d329f45c941eb2edcd31f60fc1e7f17c7a6bba8241bad93edde56a979只附着发现记录；不是新质量/接受。

真实collection-planner generation-c195abefadc1/session ses_e7187b33effexqMmCIaCpX8ABr，6实际初始文件74372B与snapshot相同，prompt/model/tools等于原始initial planner；29 assistant/32 tools早期私有history留服务器，尚未完整审阅工具输出/访问/最终计划，不能宣称全面消费或无额外读取。GPU在discovery后空闲，controller/规划角色仍活动；df剩10,004,762,624B约9.32GiB。

两个只读启动reader错误保留：第一次经stdin wrapper漏--root，在argparse退出、尚未读取实验；第二次错误预期无latest-feedback。现有fork_run对非current的accepted exploration会复制其原始complete-version result；已核对本批latest-feedback与source version-aefa021e172f/result.json字节相同。它不是planner初始输入（实际6文件、无previous-feedback、无feedback suffix、原prompt一致），但FS非hermetic，后续工具访问仍需终审。仅改reader，冻结controller/Agent/worker未修改或重启。旧54144/45316事件已处理，延迟排队必须去重。

基于独立终审bd738743…，声明一次现有`ProjectSearch.discover`及其正常真实collection-planner/自主选择的既有collectors。目的为原样508a建立真实当前调用/registry/communication记录：完整独立版本仍继承旧ccd90 registry，之前formal post-profile并未更新它。不是新候选优化、接受、独立验证或生产发布，不自动接下一候选。

从ddp-observation-route-independent-20261031/ddp-independent中accepted exploration `3cd9fc79f28ac3398d4bdfc5af76b089d7d64ce00a43532f02a8a571671529ca`显式fork，源current仍7a49不变。新ddp-evidence只附着discovery/plan/registry，code/bundle/contract必须与3cd9相同。精确code508ad3823d3a7b980df7076023b3567242f12a2544cb9bbeae0ada66c6f52f0a/14898B、单fixed-observation-route/emptyparams；正式sourcecurrent1fc/旧accepted4541/旧Graph09b7保留。

原请求、当前设备controls/adapter/source均共享只读，无source模型/controls重跑或reference复制。原GPT2 2rank B1/S128/accum4/effective8seq FP32/eager/noTF32/dropout0/usecacheFalse，全FWBW/globalmeanSGD/twosync/owner及2连续质量、2warmup、5clean、133.379621→140.04860205约束不改；本步profile/诊断并不自动执行这些全量质量或提供clean接受样本。每个collector worker的真实状态/valid/errors必须终审，不把controller完成当正确。

collection-planner仅按现有接口接收原request、available collectors、新实际communication observations、coverage/schema及自动environment；不注入独立终审表、developer因果答案、手选group/bucket/算法/collector或代码。生产prompt/model/tools/采样与接受规则不改，API observe-only无新caps。实际完整snapshot/tools/outputs/probes/sessionparts仍需审计，FS非hermetic，意图不是无GPU探针证据。私有config/credentials不得输出，历史留服务器。

已有独立全版本117.446641→54.130922ms/53.91019995%，正式对已接受ccd90为71.582732→54.347069ms/24.07796199%，不同期间不相加。实际13→1规约、497759232B总量不变，属于计算合批与通信合并联合案例，非纯通信因果份额/字节减少。之前508a native1105kernels1103Graph/2launch/1 eager AR，旧ccd90 native4103/2981Graph/13AR保持各自代码归属。如果planner实际选择新native必须审计新捕获身份，不能沿用旧count当新证明。NCCLsum含等待，不能跨trace扣clean wall或当wire/criticalpath；拓扑仅能力观察。

原控制/源码援助边界、C质量首次parent第二次setup、有限guard/复制tokens/全局override/fallback/rank非对称错误等条件限制保持。Qwen、最新FSDP拒绝、77ac与observer/mechanics链关闭，不重启。结果到达后先完整审计，再明确下一有界通信问题，不无限搜到通过。

实际原formal10jobs目录du为160950272B（含配对结果/metric、profile、5正常collector）。本步预留1GiB新发现/Agent材料/其自主collector增长加8GiB警戒，是空间预算而非API/collector cap。启动检查150GiB实际df/GPU空闲；无清理/并行GPU/reference复制。低磁盘不作完成、不盲删/停实验；必要时GPU结束后在既有授权冷压范围处理，保护active/frozen/currentaccepted/sharedcontrols/source/models/provider/deps。缺少旧plain cold文件不等于丢失，按既有RESTORE/journal服务器恢复。仅代码/hash/小摘要入GitHub，禁止raw本地批量传输。一个隐藏plain监听，旧54144/45316均handled，后台SSH/queue显式CREATE_NO_WINDOW。
