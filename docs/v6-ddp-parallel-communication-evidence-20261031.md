# 新接受并发通信版本的普通 discovery

香港2026-10-31 **18:33:44**已实际启动，controller **288526**，唯一隐藏完成/磁盘监听 **32992**。当前对象是独立接受完整版本 `b769fc35d58a472eafa11e7206746b3dff9ed46901aa721469002aebac63da7d`、精确9bdd代码、单一fixed-pipeline-parallel/空参数自身fine8/两组默认值。源独立CURRENT仍EMPTY7a49，显式fork accepted exploration；原正式current0bc及全部历史接受/拒绝保持。

只调用一次现有 `ProjectSearch.discover`，随后真实普通collection-planner自主选择现有collector。新branch只更新registry/plan/discovery，代码、bundle、contract必须保持。接受版本仍继承02b933的13call/12group registry，正式Torch profile的10call/9group没有挂到current；因此这一步为下一次优化决策提供实际当前代码证据，不是另建分析器或接受门槛。

[启动记录](evidence/v6-ddp-parallel-communication-evidence-20261031/launch.json)，通用controller [v6_discover_accepted_version.py](../scripts/experiments/v6_discover_accepted_version.py) SHA `62d4e9bf967d2a90c828304d58f7a164fd5f2b47cf328e95e3d8fa65ea35afa1`。复用工作core e893，仅6个当前关键文件。输入为普通原request、available collectors、实际新observations、coverage、schema和environment；没有手工collector、组/算法、终审纠正表或因果答案。既有fork可能复制原version result到latest-feedback，但不是普通初始planner输入；实际访问需在新角色结束后核对，不能凭控制器意图断言没有探测或外部读取。

不运行selector/candidate/参数搜索/独立重测/source-control/reference复制或发布。profile/run-only诊断不等于新全量质量/clean/接受。所有新采集均按自身时间窗口/身份解释；不得把旧formal native或SHM当作本批证据，也不把NCCL sum或跨周期差值当wire/关键路径。保留正常错误，不修到通过。

启动150GiB磁盘余8909717504B（约8.30GiB），GPU空闲；256MiB预测+8GiB告警按近期完整evidence root140967936/139771904B安排，不是API/collector cap，没有新清理。低磁盘不是完成，原恢复journals保持；必要冷存只在GPU结束后进行并保护当前工作/源/controls/accepted/models/provider/deps。旧正式45968、独立39344及候选22944已handled，晚到事件去重；固定B/C从未创建listener。

新结果后只审本批实际registry/planner/collector对下一优化决策有影响的事实，再选择一个有依据的步骤；不重复已完成候选、B/C、正式或独立批次，不自动开启新的修复/selector循环。

香港 **18:38:11**当前身份核验 SHA `ec1da1f707f9b7f69a11501a4a65be7821036ba01135d145b1bbdc48ece500b4`：[记录](evidence/v6-ddp-parallel-communication-evidence-20261031/startup-verification.json)。6当前关键文件保持，discover-ea580a14c8两rank已返回；孤立branch current `eb8cd90563b6c4dea210b819d5fff65a82cd80818bbde076d300c71c0b68a214`与接受版本files/bundle/contract相同。真实planner generation **56f9f57e7f45** / session **ses_e6e79f1abffeTtMJVQ99ACpuhW**，6实际输入100308B逐字节核对，普通初始prompt/model/tools一致；没有previous-feedback输入或suffix，latest-feedback只是原version result的自动精确副本。此时controller/planner仍运行，GPU空闲，余8892116992B约8.28GiB；完整工具内容/探测/后续collector审查待本批结束，不能把身份交付当作消费或正确性。未重做旧批审计、modelhealth或历史hash扫描。

## 本批终态和下一步

香港 **18:39:42**实际 `batch_finished`，没有低磁盘中断。完整聚焦审计 SHA `e92a2201a053db1f501e51a1c3b7b0c5e60d043279f00fb489c3e7492ac66faf`，自身原trace区间与校准数值复算 `0ae97fce09abd2f7ad3dd68b36bf0a15fd59330de91ffcc9f2ea3db4fc59ba49`；[脱敏小结](evidence/v6-ddp-parallel-communication-evidence-20261031/public-terminal-audit.json) SHA `8f18561606b57c85c5ac4c7b0767508787663b9325c9580728348b78a26bf04d`。只核本批6当前关键文件，不重复历史1947扫描。源CURRENT仍EMPTY7a49，新的孤立current eb8cd905与接受b769的files/bundle/contract相同；registry10calls/9groups，retired4calls/4groups、新1call、失效2旧证据，lineage containment不证明因果分组。

6jobs/12ranks有效：discovery与角色自主选择的nsys、nccl_tuning、device_telemetry、collective_calibration、execution_timeline。没有新完整模型质量、clean或接受；已有正式5.6824%和独立58.4625%结论保持。本批所有新请求、结果、日志、原trace、native与collector输出已按自身身份/周期核对，错误和尾部不删除。

| 本批原记录 | rank0 | rank1 | 解释范围 |
| --- | ---: | ---: | --- |
| discovery NCCL sum / union ms | 64.657021 / 39.181387 | 54.822887 / 33.422348 | 同capture通信区间，不是wire/关键路径 |
| discovery compute-comm overlap ms | 12.868373 | 12.324608 | 描述性重叠，不是可回收收益 |
| timeline wall ms | 46.516995 | 46.514155 | 另一次profile，不与clean混合 |
| timeline NCCL sum / union ms | 54.151922 / 35.135853 | 52.129514 / 33.922681 | 两通信组真实区间并发 |
| timeline compute-comm overlap ms | 14.332764 | 14.328081 | 不据此分解收益来源 |
| 新native NCCL sum ms | 52.369164 | 125.940899 | 两个受观测扰动周期，rank1长尾保留 |

各discovery/timeline自身窗口1138kernels/1130Graph/17launch/8eagerAR，payload6144 + 6×56702976 +157535232 =497,759,232B。timeline67sync/0gradient callbacks/1inventory bucket：生产就绪和消费关系仍unknown，零CPU callback不是没有梯度，一个inventory bucket也不证明八chunk的唯一owner。

新native `nsys-cd5a7ab1b5`：rank0SQLite `bc1577591f8e682a2f5131efec0c6e88a2492790b90844c49ea7c67105af6683`、rank1 `2a574e0b339d0222094f0123bc5a7cecb6880ab765b79d4e819a6f669a6eff17`，normalized `37005e8280c84b469ac96bbab882434262cca11d9d7adf3edcf96021e6129b52`。各1128kernels/1120Graph、17个versioned Graph launch、8eager RING_LL entry均在Graph外；实际本批日志channels00/01为SHM/direct/direct。kernel入口不等于逐次实际协议；另一次tuning包含初始化、warmup、flags、完整载荷及chunks，日志次数不是每step调用次数。rank1最前35.170306/44.467713ms尾部保留，不拿旧capture覆盖。

三种payload各2warmup/5samples/rank，共30合成样本质量通过。对齐maxrank后median为6144B **0.074284ms**、56702976B **3.882601ms**、157535232B **10.609086ms**；这是default-group独立SUM，不是当前双通信器并行复现、模型质量或wire下界。设备遥测另有每rank3窗口，每窗口一组5字段，NVLink API not_supported，无sampling failure。NVML PCIe为device-wide原生约20ms采样、顺序读约21ms，可能跨越step结束，不能积分成每collective字节/判断因果链；51–53%利用率不等于单模型通信利用率，7,544,832,000B设备占用也不是Torch峰值内存。所有deprecation、barrier、AccumulateGrad stream、profiler日志保留，无未捕获traceback。

真实planner generation56f9f57e7f45 / session ses_e6e79f1abffeTtMJVQ99ACpuhW：6输入100,308B逐字节相等，普通初始prompt/model/tools一致；9assistant/12tools、0schema repair，provider估算0.15096544024（非账单）。2401展示行核对，tool9字节截断后10续读补齐全部2088 observations行，6初始文件全部展示；tool2错误路径保留、4纠正；tool11计划一次写入4318B/无改，12只有JSON/schema检查。全部实际工具输入/输出核对，只观察到初始文件、目录元数据和自身plan/schema操作，未观察到model/tensor/CUDA/processgroup/collective、private config或后续答案内容访问；外部FS非hermetic，不能绝对证明不存在。完整history `4204b44190f9893313d8b1b3cdcf5e8aad7945b16d30eef4b7003ce3c79ba05f` / access `4bb3b7e97a7f06ab5a47b19dbfadd2ee35168f1a3604063c90940d9d9f974ea3` 仅server。计划“across repeats”实为每rank单profile、“没有candidate”忽略已安装接受bundle；这些措辞不采纳，原plan保留、未手改输入或另加评审链。

主审和原trace复算首次通过。后续只读摘要曾取不存在的telemetry摘要end_ns触发KeyError，保留调用记录、只改摘要读取；本地读取脚本一度在父工作目录报不存在路径，也仅纠正cwd，没有实验/角色/模型/GPU重跑。没有新增框架实现或测试门槛。

已继续[一次普通项目选择](v6-ddp-parallel-communication-selection-20261031.md)，使用真实新current资源与原独立反馈，不喂终审解释或人工算法。旧32992监听已handled/退出，延迟队列事件去重；无新清理，无重复formal/independent审计。
