# 分片排程接受版本的正常通信采集

**新批终审完成**：服务器audit `cd52e751ceba7236ae91595add7a2f4eb0f6676af5a0be1a8087d2eaf366cdee`、补充review `3ffa77319a0956d3eee66a0df316e4c8f571bb8f5daec2f7738cd1990faa3171`、[公开摘要](evidence/v6-ddp-schedule-communication-evidence-20261031/public-terminal-audit.json) `e998ae68737b55010878714bd054eafbffc711bdf0342ca0f59df3668c918232`、独立原event sweep `53c5772d5f9537bba5bcb025e517a59656b7ae4c5a9a0aa08f7b43e2d482a538`。5jobs/10ranks、六当前身份、实际requests/materialization/entry/adapter/controls引用/UUID/core、新registry/lineage及controller结果核对。source CURRENT EMPTY7a49保持，新current88fa文件/bundle/contract精确7e6；10calls/9groups中retired2calls/2groups、新增2calls、invalidated2旧trace。lineage是containment，非因果分组。只做discovery和四类自主collector，没有新的模型全量质量、clean或接受。

| 本批原始记录 | rank0 | rank1 |
|---|---:|---:|
| discovery profiled wall ms | 69.669154 | 47.606918 |
| discovery NCCL sum / union ms | 106.269651 / 54.850176 | 61.733800 / 32.934369 |
| discovery compute/comm overlap ms | 10.930852 | 15.807166 |
| timeline profiled wall ms | 46.241034 | 46.069625 |
| timeline NCCL sum / union ms | 63.742915 / 33.953090 | 62.789788 / 33.454062 |
| timeline compute/comm overlap ms | 15.070157 | 15.654991 |

各自workload window原event重新扫union/overlap。每rank1141kernels/1130Graph root-only、17Graph launch、11个eager RING_LL entry在Graph外，stream21:6/25:5；6144+六56702976+四39383808=497759232B，总载荷不变。timeline有88sync、0gradient callbacks、1inventory bucket；GPU readiness未知，不是无梯度或11piece owner证明。不同capture/profiled period不能相减成clean、wire、关键路径或可回收收益。

新Nsys `nsys-b03ee9aa25` 的r0SQLite3ecd07822d443af3b8b7aba1015ec93a61b0ea10762b9ceb147821bb76af45b3、r17d2a9d4c4207410f4dee678355bea22a256150622bd2373555c5117f8ae93d1e，normalized5309c899498441929b0b53e8b7233f5c61c4a7aa678f81d3fe6ebfe1e106861f。每rank1131kernels/1120Graph、17个cudaGraphLaunch_v10000、11AR/stream21:6和25:5。实际native日志channel00/01为SHM/direct/direct。NCCL sum85.489144/61.912751ms、union44.505021/33.025960、通信区间并发40.984123/28.886791ms；r0前两个17.321292/19.278957ms尾部完整保留。eager graphNodeId在SQLite中为NULL，不是Torch trace的0表示。新capture单独核对，未借用正式native计数。

普通tuning的56702976/39383808B为RING/SIMPLE双channel、6144B为RING/LL单channel，另有setup/warmup的64B/1MiB/40B/full497MB记录；occurrences不等于每step计数，不和timed kernel entry硬做协议join。校准实际14175744/9845952/1536 Float元素，2warmup、5/rank，30个synthetic样本quality均true；先aligned rankmax后median为3.886971/2.720739/.077179ms。原全部样本保留；default-group独立SUM不复现双通信组模型排程，不是模型质量、wire速率/下界或协议与等待的因果分离。本批未选择device_telemetry。所有10rank日志的deprecation/barrier/loss/AccumulateGrad/profiler warning保留，无未捕获traceback。

真实planner **7e627ef4afe5/session ses_e6e32d78cffekUWW2Ogv5gU7B0**：6实际输入108641B、7assistant/11tools/0schema修复、provider估算0.16615950932非账单。2682展示行逐行核对；tool6 byte cap后tool9从1372续读，完整暴露2369行observations及所有六输入。tool1目录、2 glob只显示config名字，3–9输入读取，10一次plan写入7136B无编辑，11仅JSON/schema/collector名单校验。完整实际commands/outputs没有model/tensor/CUDA/PG/realcollective/privateconfig/later-answer内容访问记录；FS非封闭，不能据此绝对排除外部访问。私有完整session/history仅在服务器，history4cddd7046796bf49bcf066543a596760f0f52b0ab4497a3298c3e94e5d126ce1/accessfbcdffdac2a00d20e52ef21f821745b23f847c2145952a83d6c8ba7066811f04。

planner自主选择nsys/nccl_tuning/collective_calibration/execution_timeline。原plan保留，但其“no candidate exists”、普通last-backward描述不适用于当前已安装whole-step replacement；跨capture校准不能隔离tuner-versus-wait，较大的首个kernel不能单独证明peer readiness，NVML not_supported也不是所有物理fabric不存在的证明。没有把这些审计纠正写回plan或注入下个角色。

主终审第一次实际执行通过。局部builder最初匹配多一个空行，ValueError后reader缺文件，均发生在远端审计前，仅修正字符串定位；额外native汇总读取器第一次把eager graphNodeId限定0而实际为NULL，仅改nullable表示后完成，失败reader保留。摘要打印溢出改为紧凑字段读取；没有角色、CPU probe、模型或GPU重跑，没有扩建分析器。

**下一步已执行**：[一次普通项目selector](v6-ddp-schedule-communication-selection-20261031.md)，香港20:02:40 controller307469/唯一隐藏监听46500，使用精确新current88fa、新原始四collector/原独立接受反馈与既有用户约束。新的已测量版本与新证据支持一次优化决策，不预设算法，不在旧数据上重试直到获得想要答案。旧53652完成事件已handled，迟到事件去重。

以下为启动及终态事件记录。

**19:55:28终态已到达**：batch_finished/completed，discovery和preservation结果已写入；唯一监听53652已成功queued完成事件（returncode0）。完整新planner工具/原始trace/native数值审计由该事件续接，当前不预称终态数据或新接受已核验，不重复创建listener。终态观察df8769634304B约8.17GiB，GPU为空。

香港2026-10-31 **19:51:28** 已启动一次原 `ProjectSearch.discover`，controller **304369**，唯一隐藏监听 **53652**。使用[新独立接受版本](v6-ddp-parallel-schedule-independent-20261031.md) **c283f6a55dc1f758fa0fb9575f757b27384b5bb5b381592251c6fa4808c5f6a6**，源 `ddp-parallel-schedule-independent-20261031/ddp-independent` CURRENT仍原始EMPTY7a49，explicit fork accepted exploration。新root `ddp-schedule-communication-evidence-20261031` 仅附加当前registry/plan/discovery，文件/bundle/contract保持。

理由是新接受7e6fb7仍继承旧9bdd的10call/9group registry；正式profile虽也是10call/9group，并没有把新代码registry接上。需要实际新版本调用与通信记录供后续优化使用，不构建分析器、不自动预排candidate。原fixed-parallel-schedule/params{}、自身fine8/2groups/4tail片、code7e6fb7和proposal442ac3f保持，旧物理文件不是多个active bundles。source/formal7401/旧eb8/b769/187c及其他既有指针不变。

现有通用 [v6_discover_accepted_version.py](../scripts/experiments/v6_discover_accepted_version.py) SHA62d4e9bf967d2a90c828304d58f7a164fd5f2b47cf328e95e3d8fa65ea35afa1，直接复用working runtime core e893。六当前关键文件，不复查历史1947hash、不复制runtime、不做model health/额外测试。普通initial collection-planner使用原request/available collectors/new observations/coverage/schema/environment自主选择现有collectors；无开发者审计表、手选collector或算法。API observe-only无cap，普通prompt/model/tools保持，所有错误保留。profile/run-only不是新增全量质量/clean/接受。

**19:53:19实际启动核对** [1ae30303](evidence/v6-ddp-schedule-communication-evidence-20261031/startup-verification.json)：discover5f47565d3a两rank结束，新current **88fa1d104cd1777b723521cfd9bb37540abfcefbb9a5edb3a04ca3cc21f67057** 与原7e6文件/bundle/contract一致，源CURRENT EMPTY7a49、六当前身份保持。真实planner **7e627ef4afe5/session ses_e6e32d78cffekUWW2Ogv5gU7B0**，六实际输入108641B与snapshot逐字节相等，普通初始prompt/model/tools相同、无previous-feedback suffix。fork自动复制原version-df2217162ef2反馈存在，但不在初始六输入内；后续外部访问待终态完整审计，非封闭FS，不能预称无读取。完整工具/输出/探测、新collector语义尚待终态核对。此时GPU空闲、controller planner阶段仍运行，df8793882624B约8.19GiB。

容量所需无损cold仅在GPU结束后 **19:50:54–56** 实施：4份旧正式拒绝02ed C的profile/timeline原trace加12份拒绝e610 C metrics，105566660→8590190B，释放96976470B。roundtrip全部校验，1914其他文件metadata保持；接受/来源/基线/controls/materialization/native/current/model/provider/deps均未修改。服务器RESTORE `/root/autodl-tmp/dev/storage-audit-20261031/ddp-schedule-evidence/RESTORE.md`，journal7a62edaf77df19c3b498f2de608369797a27f0a9fb9164783df6eb870979a9d4、plane06e000b5dbd87a35b60ce27b5e84e2aef8e96e762d72c222853b3981677ea17；旧journal独立保留，cold不是丢失，无raw本地转移。

本批192MiB增长预测依据实际三个完整evidence目录133746688/140967936/139771904B；8GiB警戒不变，非API/collector cap。启动df8810954752B约8.21GiB。low_disk_space不是完成，先看workers；不盲杀/删，必要容量仅按授权在GPU结束后保护当前数据做lossless cold。局部launcher builder的PowerShell/python-c引号SyntaxError发生在脚本创建/SSH/远端启动前，改成文件builder后启动一次，无实验重启。

旧formal7304已处理、迟到事件去重；独立短批未建listener，其他旧watcher不重启。完成后按实际当前registry、普通planner工具访问及真实collector结果选择下一有用优化步骤，不把NCCL sum/重叠或跨capture差值当wire/关键路径，不重复closed角色/参数/修复链。原质量/性能门槛、既有辅助输入与代码条件性限制保持。
