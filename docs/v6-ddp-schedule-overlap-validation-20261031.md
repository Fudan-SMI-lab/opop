# 重叠候选的原正式配对验证

香港2026-10-31 **21:04:58**正式AB/BA正常完成，完整新结果终审 `3d34cf680ab7b0fbf30d2b023365bf194c2fee05550b820a90e24d1036012ab7`，只读数值汇总 `9f2109cf1abd76b1627a86fec32bc98c04c863d42958f5d3fe1432b0f6f91af7`，[公开终审](evidence/v6-ddp-schedule-overlap-validation-20261031/public-terminal-audit.json) SHA `76669d9e317a99495653c135fd94a7b492a3005c8e55eb8674607d04bdc1b731`。10jobs/20ranks/40clean/16fullchecks（每次248,879,620元素）/24自身metric traces/2新native SQLite，8当前关键身份与原core e893保持。原判定已核实为接受ea18e465，publication=exploration，CURRENT仍88fa，没有发布指针。

|原fresh配对|B accepted7e6 ms|C exact2166 ms|收益ms|MAD噪声界限ms|
|---|---:|---:|---:|---:|
|AB|44.308969|43.144499|1.164470|0.487990|
|BA|44.234766|43.296156|0.938610|0.091448|

两对均通过，聚合 **44.2718675→43.2203275ms，改善2.3751878097%**。全部40样本保留，最大B44.643096/C43.573637；原anchor133.379621/limit140.04860205、数值/精确副本和原约束均通过。8个first checks误差0，8个second checks rank0最大9.5367431640625e-7、rank1 1.1920928955078125e-7，在原容差内。first parent/second graph_setup_lead2_pg2_tail2_perpiece_foldscale/reference1不是两个optimized checks；没有全raw tensor保存或新source比较。此2.3752%相对7e6，不与旧59.6932%或固定2.5789%相加，也不称纯通信贡献。

全部自身metric trace：B1141 kernels/1130 Graph/17 launches/11 eager AR，C1139/1127 Graph/22 launches/12 eager AR；B group1/stream21:6、group2/stream25:5，C为5/7。AR均在Graph外、RING_LL入口，C载荷6144+2×28351488+5×56702976+50419200+3×35705344B，总497,759,232B与B相同。自身计算通信重叠C15.027464—16.341379/B14.679553—15.683146ms，通信之间重叠C26.081054—29.748434/B28.331830—29.959752ms；只是各自capture内区间，不能作wire、关键路径或可回收收益。新profile12calls/11groups不等于已更新registry。

新精确2166 Nsys来自nsys-b6971ce879：r0 SQLite605d822095a04e332212eba13241670f6a20944f740bf77edd8d274332e0f95a、r1 46e34355530198edf4301516b2162bec98ff0e7f5bd091e9fd374e25c5bb5cd3、normalized585aa6dbab3d8aae48201395c175387801f458ab00ee6f0253030df217da2156。每rank1128 kernels/1116 Graph/22个cudaGraphLaunch_v10000/12 eager AR，stream21:5/25:7；实际native日志SHM/direct/direct双channel，不从NODE/P2P推断transport。NCCL sum59.693460/439.302237ms，union33.208888/221.342357ms，rank1首两段190.94/191.12ms尾部保留；不同受扰周期，不是clean/wire/关键路径。后续独立阶段只hash复用这些新证据，不再采一次Nsys。

独立timeline 43.861086/43.846283ms、12AR/102sync/0callback/1inventory bucket，GPU readiness未知，不能解释为无梯度或完整12片ownership。30个合成校准样本使用旧parent discovery的14175744/9845952/1536元素，aligned rank-max中位数3.865863/2.709768/.060604ms；是default group独立SUM，不是新lead2大小/双组模型排程、模型全量质量或wire下界。tuning含setup/warmup及当前chunks，不做每step或kernel协议join，无env pin。run-only诊断两rank各6个正常False generator位置334/356/365/384/385/406，共460返回；无exception，也不执行数值observe比较。

C Torch peak4,619,939,328B/lifecycle5,672,500,224B，runtime reserved5,888,802,816B，但lifecycle reserved6,354,370,560B；B peak约4.613GB/lifecycle5.656GB/reserved6.354GB。不能只选runtime reservation称生命周期内存收益，外部NCCL并未全量计入，无新内存门槛。全部deprecation/barrier/AccumulateGrad/profiler警告保留，无uncaught traceback；8.705609—11.089830ms安装时间不是完整setup。

真实planner generation7a806babeaa1/session ses_e6df335cfffem66IQy15doYEBK，10assistant/14tools/0schema修复，provider估算.19024736154不是账单；8输入116,884B与snapshot/普通prompt-model-tools一致。2964展示行、0逐行prefix截断；工具8字节截断后11续读，全部2369 observations行及8输入实际暴露。工具7正确读替换组成、排除binding_cost。**工具12确实读了自己的provider配置**，1504B/hash8adbdd58ab2165c422393fe885fce15bc685c1e835b4ed5ebd730acd6ab3a890，63行仅在服务器精确核对，内容没有输出；不能称8-input-only。13一次plan写入无编辑，14纯JSON/schema。没有记录到model/tensor/CUDA/PG/真实collective/后来答案内容命令，外部FS非封闭。private history9a37766219bccd71f07af5cf244c3eafb75d6db34ec917941ef466c283ba36fc/accessa7ba6d958045b30fef9828daf9cd0a87921d32da49eb7d16525ac17e553d87ee留服务器。原plan首kernel“等待主导”未获证明、将单独diagnostic profile称clean不当、校准大小来自parent；原plan保留，未作为新角色纠正答案。主审首遍通过；初始metadata检查错找results.json的FileNotFound保留initial-inspection-error.json（实际followup-results.json），无角色/实验重跑。

基于真实新正式接受，香港 **21:18:24已实际执行[一次完整版本独立B/C/C/B](v6-ddp-schedule-overlap-independent-20261031.md)**，对原始EMPTY，不改代码、参数、质量或接受规则。旧34508已handled，原固定/候选不重审。以下为历史启动记录。

香港21:04:58，正式批已返回 batch_finished，唯一34508已成功queue完成事件；不是low_disk_space。完整正式数值/noise/采集与planner终审由这一已queued事件续接，当前不把完成等同接受，不重复监听或重启。终态磁盘available 8606490624B。

香港2026-10-31 **20:59:39**已启动一次`ProjectSearch.validate_bundle(publish=False)`原fresh AB/BA，controller **315531**，唯一隐藏完成/磁盘监听 **34508**。[实际launch](evidence/v6-ddp-schedule-overlap-validation-20261031/launch.json)复用现有working runtime/core e893，与固定阶段和本次B/C相同；现有通用v6_validate_fixed_candidate.py SHA96586fdc2a611253b32b5a5e63ceb49d63132a63db78bb7cfb3c014cd51229da未改。8当前关键身份，没有历史sweep、环境副本、model-health或新增测试。

依据[候选完整终审](v6-ddp-schedule-overlap-candidate-20261031.md)与[原固定B/C数值及性能](v6-ddp-schedule-overlap-fixed-20261031.md)，B为独立接受7e6的CURRENT88fa1d104cd1777b723521cfd9bb37540abfcefbb9a5edb3a04ca3cc21f67057，来源`ddp-schedule-overlap-fixed-20261031/ddp-fixed`，不是EMPTY。C staged `ea18e465a9b6570b779732346c63f94bd55be57c8c524a572fab29effbd17471`，singlefixed-schedule-overlap，空参数{}调用代码自身lead2/comm_groups2/tail_splits4；精确代码`2166f860330679a93f335985183675815b2e7e7053dc5be9453277aa68497a61` /41127B、proposal`b3fae015c2007ce75443c39fcceb8f5c51cb06403265bd99ced30c584b29e4c6` /11239B。没有手工选择参数或改代码；requires[]/replacesfixed-parallel-schedule不变，不尝试其他lead/组数/tail参数直到通过。

trial-schedule-overlap-validation复用原request、prepared EMPTY7a49、实际current discovery和只读controls。原GPT2 12×768/12heads、B1S128×4accum×2ranks、FP32 eager/noTF32/dropout0、SGD.001/momentum0/weightdecay0/foreachFalse保持。完整zero/FWBW/lossdiv4/SGD/two sync、两连续full grads/params/lossfinite/exact replicas、atol1e-6/rtol1e-4、2warmup/5clean/rank、aligned rank-max-before-median-MAD及anchor133.379621/limit140.04860205保持；不加内存约束。first parent/second setup仍不是两个optimized checks，旧source对照仅hash复用，不称新source比较。

正常resource_feedback profile及真实普通collection-planner自主选择既有collectors，API observe-only无cap；controller不指定collector，不新建selector/candidate/parameter search/source-control/reference副本/independent或发布指针。当前与旧接受c283等均保留，即使publish=False正式接受也不发布CURRENT。只有新正式接受和完整终审才支持另行声明的独立完整版本验证；拒绝、尾部与错误保留，不重跑直到通过，不池化固定样本。

香港 **21:04:34** [实际请求与planner输入核实](evidence/v6-ddp-schedule-overlap-validation-20261031/startup-verification.json)，SHA7b744ac326e965f40e9ec46b38914d7801cc992fb41596600f892ed6378538a5：四ABBA及profile/native/timeline/tuning/calibration请求的原request、entry aed604ee、adapter b276a7ed、controls、devices、精确7e6或2166代码/params{}均核对；此时前四评价rank results存在不代表正式接受。真实planner generation7a806babeaa1/session ses_e6df335cfffem66IQy15doYEBK，8实际输入116,884B逐字节相同，普通prompt/model/tools/core相同；实际parent_bundles与added/removed/retained为parallel-schedule→schedule-overlap替换。完整工具/输出/探测/终态数值、noise、新native仍待终审。此时两个collector workers在GPU，df8,636,620,800B约8.04GiB。

[128MiB增量预算](evidence/v6-ddp-schedule-overlap-validation-20261031/disk-budget.json)依据最近三次完整正式批实际quota增长119,672,832—121,585,664B，134,217,728B大于最大值加10%，8GiB警戒不变；目录du含共享物化，不能等同新quota增长。不是API/collector cap。启动free8,732,909,568B，固定后没有追加cold。预算reader初次误索引df百分比列ValueError，仅改header读取Avail，原错误记录，不是实验重跑。low_disk_space不表示完成，勿杀任务/盲删；仅在GPU完成后必要授权无损cold并保护当前/来源/接受/controls/模型依赖，若仍活跃只替换一个监听。

旧33580候选已handled，fixed短批未创建监听，无fixed queued事件；其他旧完成事件去重不重启。所有SSH/queue隐藏，无model polling/delivery tests。新结果必须按自身capture解释，NCCL sum/重叠不当wire或关键路径，也不借旧SHM当新代码证明。此阶段推进实际原验证，不新增辅助分析器、手工优化或修复链。
