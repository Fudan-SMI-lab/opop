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
