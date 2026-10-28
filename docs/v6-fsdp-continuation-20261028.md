# FSDP 接受后重新发现与继续搜索

承接[首批三轮与独立审计](v6-fsdp-search-20261028.md)：已接受同合同版本`3007940f…`，独立完整版本176.517374→171.053986ms（3.095099%），实际每rank减少3次AllGather。旧两轮超时/质量失败和全部样本保留。baseline evaluator有开发者证据辅助修复，因此后续仍是已核实baseline上的自主搜索，不称新的无辅助原入口队列。

新目录`/root/autodl-tmp/dev/v6-work/fsdp-continuation-20261028`，脚本`scripts/experiments/v6_fsdp_continuation_20261028.py`。从上批accepted current创建隔离branch，复用同一原始baseline锚点、adapter、源代码、request、两步完整质量、FP32/noTF32/eager及peak<=1.05 baseline；controls只读共享。只把上一轮框架原样latest-feedback作为首次反馈，记录SHA，不传入本轮开发者审计答案或新候选。后续每轮反馈由原框架生成。

固定再执行三轮；第一轮前应对已接受版本重新discovery，以观察真实剩余collective和已装bundle。Agent自主决定组/联合/区域/执行计划范围及替换/组合，保持现有选择和验收逻辑；不预设必须换算法或必须接受。角色请求与费用仅统计，不设开发调用限额。

若三轮结束后active bundle集合相对上批有新接受，另起原始baseline分支执行一次完整版本独立AB/BA，并做单独B/C原生路径检查。若无新接受，明确not_run，不把相同候选重新复测到通过。所有样本、质量失败和约束失败保留。历史3.095099%接受不被后续结果覆盖。

重点审计：接受后重发现是否发生、真实反馈读取、旧bundle替换或共存、实际FSDP策略/通信/计算路径、完整源语义、显存生命周期以及独立完整版本判据。四项通信次级指标仍分开采样，各自max-rank后median；不从NCCL sum推断wire time，不把跨字段中位数拼成物理分解。

空间准备只使用服务器内无损压缩，范围为已完成的ddp-tp-diagnostics、gc-review-diagnostics和memory-snapshot-validation三批；当前FSDP、源参考、已接受Qwen/DDP/ResNet项目、模型和私有provider状态保持在线。依据明确清单、无活动进程、逐文件gzip解压SHA和保留文件SHA验收，不下载raw。见前批证据目录的cold-diagnostic-plan与维护结果。

维护已于10:16:57 UTC完成：179份冷payload共2,235,372,450 bytes压缩为191,322,914 bytes，全部gzip和解压内容SHA复验通过，另2,827个保留文件SHA不变。实际df净回收2,044,243,968 bytes（约1.90 GiB），可用15,109,062,656 bytes（14.07 GiB）。服务器恢复清单和指令位于`/root/autodl-tmp/dev/storage-audit-20261028/terminal-diagnostics/`；仅回传小型结果，未转存raw，也未删除任何样本内容。见[验收结果](evidence/v6-fsdp-search-20261028/cold-diagnostic-result.json)。

本批只开一个GPU任务。启动前记录实际150GiB配额df；一个普通完成/磁盘监听每30秒查终态及余量，低于8GiB提前通知，不停止实验或自动删除。没有Goal、模型轮询或额外投递测试。代码、协议和小摘要归档GitHub v6；raw留服务器。

## 启动配置错误及恢复范围

初次冻结`0e77a659`，10:23:49 UTC启动PID630857，10:25:14以completed_with_errors结束。原生GPU重新发现已经完成（新revision`799a80d2…`），但首次collector planner的Runtime启动前失败，没有任何Agent会话/生成快照，也没有新候选或独立性能测量。原因是本轮部署脚本误将原配置`server_url=null`改成`http://127.0.0.1:19060`；框架将非空URL解释为连接既有服务，因此不会自动启动OpenCode，而该端口没有服务。不是provider请求失败，也不是FSDP质量失败。

原目录、配置、发现结果和终态均冻结；监听已于10:25:39成功queue一次真实失败通知。当前用户轮次已直接处理该错误，后续若这条通知到达，应检查本节及恢复批次，不能再次启动旧批次或覆盖原失败。

恢复输出为`/root/autodl-tmp/dev/v6-work/fsdp-continuation-followup-20261028`，还原`server_url=null`使用原自动启动方式；生产框架/prompt保持不变。控制器可选`--discovery-from`复用上述已完成GPU发现：先验证相同bundle/source/contract及raw trace SHA，再fork到新目录，通过现有plan_collectors/collect_extra执行尚未开始的发现尾部，随后用同一CLI --resume开始原定三轮。原样前轮反馈不变，不重复GPU发现或参考生成。独立检查和原生路径阶段仍只在新bundle接受时执行。新批使用一个新的完成/磁盘监听，旧已投递监听不复用。

恢复批次已冻结`85a0707d8775e9468ffa7e0a3da94645238b51ca`，10:29:33 UTC启动PID631783、PPID1，生产core仍`ed24af7a…`。部署先真实启动并关闭OpenCode健康预检，随后由框架自行启动服务；真实collector-planner会话`ses_e7df11836ffeHbaVCwEUL88m5o`及生成前快照已记录。新分支adapter SHA保持`8b014f6f…`，controls_path与已核实baseline相同，新目录没有reference-rank张量副本。见[启动](evidence/v6-fsdp-continuation-20261028/followup-launch.json)、[协议](evidence/v6-fsdp-continuation-20261028/followup-protocol.json)、[复用依据](evidence/v6-fsdp-continuation-20261028/discovery-reuse.json)、[真实会话核验](evidence/v6-fsdp-continuation-20261028/startup-verification.json)。

唯一普通完成/磁盘监听PID23708，状态`.scratch/v6-fsdp-continuation-followup-wakeup.json`，配置`.scratch/v6-fsdp-continuation-followup-wakeup-launch.json`；每30秒检查，8GiB提前通知阈值。启动核验为running，实际余约13.98GiB，[监听记录](evidence/v6-fsdp-continuation-20261028/followup-wakeup-setup.json)保存。本机以后重启应先查此新批终态与监听状态，不能恢复旧fsdp-search或初次continuation的已处理监听。

## 12:34 UTC终态与磁盘告警处理

监听12:30:52 UTC提交的是low_disk_space，非完成事件，随后退出；当前会话实际核查时，本批已于12:34:47 UTC完成，控制器与GPU worker均退出。因此直接审计终态，没有重启实验或恢复已完成批次的监听。[告警投递](evidence/v6-fsdp-continuation-20261028/disk-attention-delivery.json)与[实际完成](evidence/v6-fsdp-continuation-20261028/followup-completion.json)分别保留。

全部7个参数点、12个搜索配对端点和4个独立端点通过完整两步质量：92份rank/step检查最大误差均为0；230个clean rank样本全保留。552份次级指标raw trace逐份重算通过，各字段仍是独立的三次profile采样。source/adapter/合同、共享controls及六份完整source/reference SHA核验不变。见[端点与原生审计](evidence/v6-fsdp-continuation-20261028/followup-terminal-audit.json)、[代码与继承审计](evidence/v6-fsdp-continuation-20261028/continuation-identity-audit.json)。

| 轮次 | 真实实施与结果 |
|---|---|
| c0a2a514fe | selection提出forward prefetch，candidate改为扩展原keep参数域0..5；实际仅筛选K=1/2/3，仍选3，搜索接受并替换旧bundle。 |
| 423948159b | 实际实现FSDP forward-prefetch API，深度2/1均质量通过；配对收益2.756/0.845ms，噪声界3.711/0.143ms，拒绝。 |
| c0e29fed87 | 实际尝试K=5/4，质量通过；选择K=5后配对收益2.773/2.226ms，噪声界1.499/3.557ms，拒绝。 |

第一轮必须与新增优化区分：新旧代码只移除模块docstring并将DEFAULT_KEEP从2改为3；两个已选bundle均显式传入K=3，默认值不参与该路径。调用边界与实际策略相同。新bundle ID及搜索接受不能证明在旧接受版本之上又得到收益，原判定与全部波动样本仍冻结保存。其参数域确实扩展，不能称代码逐字相同或候选完全没有提出变化，但新值4/5在该轮未执行。CPU使用相同seed、原参数空间和已记录分数重放TaskSearch，精确复现1→2→3及选择3；原candidate request中没有显式parameter_trials，使用框架默认3。没有补测未访问点或将其当作失败。

完整版本独立AB/BA相对**原始修复baseline**为176.6046395→172.7938785ms（2.157792%），配对收益4.839005/2.782517ms超过界限0.096898/2.649656ms，原peak约束通过。它验证的是K=3策略的另一组独立测量，不能与此前3.095099%叠加、不能覆盖旧结果，也不证明继续搜索获得增量收益。原生每rank仍为baseline49AllGather+25ReduceScatter、candidate46+25，与原K=3机制一致；不以profile时长推断wire bandwidth或关键路径。

复用了失败初次批次的完整GPU发现并核对raw SHA；本轮第一轮接受后又实际重发现（9951946f→a6ea4101），后两轮使用新registry和call映射。共11个真实角色会话，selection/candidate无schema repair；真实工具记录读取反馈与候选代码，第二轮selection也明确识别前轮未实现预取。所有版本只有一个active bundle；本批证明接受后重发现和替换链，没有多bundle共存正例。Agent关于协议饱和、仅剩一个优化方向及精确内存上界的推论未被审计采纳。

明确提示问题：存在反馈时统一追加`Read inputs/previous-feedback.txt and repair the concrete failure.`，即便反馈已经接受。第一轮candidate确实把成功后的参数域扩展表述为修复concrete failure，但不能据此认定该句是唯一原因。下一步按[成功反馈最小对照](v6-accepted-feedback-replay-20261028.md)，从真实生成前快照仅替换此句，做两个新的真实会话；生产提示、调参逻辑与接受规则暂不变，不追加开发者优化答案。

补充审计初次从snapshot读取selection.json漏了inputs/前缀，产生KeyError。端点审计已完成，之后仅修正读取路径并复用已核实端点结果完成继承检查。这是审计脚本错误，不是实验或质量失败。

空间整理12:44:18 UTC完成。明确范围仅旧next-stage与ddp-semantics的7个jobs/sandboxes目录：479份大JSON的6,159,922,685 bytes原内容保留在514,399,020 bytes gzip中，全部压缩及解压SHA验收，另2,650个未选文件SHA不变。实际df净回收**5,644,873,728 bytes（5.26GiB）**，余**13,114,265,600 bytes（12.21GiB）**。未下载raw或压缩当前FSDP、当前接受项目、参考张量、模型、环境、provider状态。15个保留run的源码/bundle与参考路径、六份完整FSDP参考SHA在维护后复查通过；不把维护后检查称为全盘维护前快照。[清单](evidence/v6-fsdp-continuation-20261028/disk-attention-plan.json)、[回执](evidence/v6-fsdp-continuation-20261028/disk-attention-cleanup.json)、[保留版本核查](evidence/v6-fsdp-continuation-20261028/retained-versions-check.json)。完整journal及RESTORE.md位于服务器`/root/autodl-tmp/dev/storage-audit-20261028/disk-attention/`。本批新增约7GiB主要是测量/profile JSON，并非模型重新下载。
