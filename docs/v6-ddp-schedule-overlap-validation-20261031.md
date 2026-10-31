# 重叠候选的原正式配对验证

香港21:04:58，正式批已返回 batch_finished，唯一34508已成功queue完成事件；不是low_disk_space。完整正式数值/noise/采集与planner终审由这一已queued事件续接，当前不把完成等同接受，不重复监听或重启。终态磁盘available 8606490624B。

香港2026-10-31 **20:59:39**已启动一次`ProjectSearch.validate_bundle(publish=False)`原fresh AB/BA，controller **315531**，唯一隐藏完成/磁盘监听 **34508**。[实际launch](evidence/v6-ddp-schedule-overlap-validation-20261031/launch.json)复用现有working runtime/core e893，与固定阶段和本次B/C相同；现有通用v6_validate_fixed_candidate.py SHA96586fdc2a611253b32b5a5e63ceb49d63132a63db78bb7cfb3c014cd51229da未改。8当前关键身份，没有历史sweep、环境副本、model-health或新增测试。

依据[候选完整终审](v6-ddp-schedule-overlap-candidate-20261031.md)与[原固定B/C数值及性能](v6-ddp-schedule-overlap-fixed-20261031.md)，B为独立接受7e6的CURRENT88fa1d104cd1777b723521cfd9bb37540abfcefbb9a5edb3a04ca3cc21f67057，来源`ddp-schedule-overlap-fixed-20261031/ddp-fixed`，不是EMPTY。C staged `ea18e465a9b6570b779732346c63f94bd55be57c8c524a572fab29effbd17471`，singlefixed-schedule-overlap，空参数{}调用代码自身lead2/comm_groups2/tail_splits4；精确代码`2166f860330679a93f335985183675815b2e7e7053dc5be9453277aa68497a61` /41127B、proposal`b3fae015c2007ce75443c39fcceb8f5c51cb06403265bd99ced30c584b29e4c6` /11239B。没有手工选择参数或改代码；requires[]/replacesfixed-parallel-schedule不变，不尝试其他lead/组数/tail参数直到通过。

trial-schedule-overlap-validation复用原request、prepared EMPTY7a49、实际current discovery和只读controls。原GPT2 12×768/12heads、B1S128×4accum×2ranks、FP32 eager/noTF32/dropout0、SGD.001/momentum0/weightdecay0/foreachFalse保持。完整zero/FWBW/lossdiv4/SGD/two sync、两连续full grads/params/lossfinite/exact replicas、atol1e-6/rtol1e-4、2warmup/5clean/rank、aligned rank-max-before-median-MAD及anchor133.379621/limit140.04860205保持；不加内存约束。first parent/second setup仍不是两个optimized checks，旧source对照仅hash复用，不称新source比较。

正常resource_feedback profile及真实普通collection-planner自主选择既有collectors，API observe-only无cap；controller不指定collector，不新建selector/candidate/parameter search/source-control/reference副本/independent或发布指针。当前与旧接受c283等均保留，即使publish=False正式接受也不发布CURRENT。只有新正式接受和完整终审才支持另行声明的独立完整版本验证；拒绝、尾部与错误保留，不重跑直到通过，不池化固定样本。

香港 **21:04:34** [实际请求与planner输入核实](evidence/v6-ddp-schedule-overlap-validation-20261031/startup-verification.json)，SHA7b744ac326e965f40e9ec46b38914d7801cc992fb41596600f892ed6378538a5：四ABBA及profile/native/timeline/tuning/calibration请求的原request、entry aed604ee、adapter b276a7ed、controls、devices、精确7e6或2166代码/params{}均核对；此时前四评价rank results存在不代表正式接受。真实planner generation7a806babeaa1/session ses_e6df335cfffem66IQy15doYEBK，8实际输入116,884B逐字节相同，普通prompt/model/tools/core相同；实际parent_bundles与added/removed/retained为parallel-schedule→schedule-overlap替换。完整工具/输出/探测/终态数值、noise、新native仍待终审。此时两个collector workers在GPU，df8,636,620,800B约8.04GiB。

[128MiB增量预算](evidence/v6-ddp-schedule-overlap-validation-20261031/disk-budget.json)依据最近三次完整正式批实际quota增长119,672,832—121,585,664B，134,217,728B大于最大值加10%，8GiB警戒不变；目录du含共享物化，不能等同新quota增长。不是API/collector cap。启动free8,732,909,568B，固定后没有追加cold。预算reader初次误索引df百分比列ValueError，仅改header读取Avail，原错误记录，不是实验重跑。low_disk_space不表示完成，勿杀任务/盲删；仅在GPU完成后必要授权无损cold并保护当前/来源/接受/controls/模型依赖，若仍活跃只替换一个监听。

旧33580候选已handled，fixed短批未创建监听，无fixed queued事件；其他旧完成事件去重不重启。所有SSH/queue隐藏，无model polling/delivery tests。新结果必须按自身capture解释，NCCL sum/重叠不当wire或关键路径，也不借旧SHM当新代码证明。此阶段推进实际原验证，不新增辅助分析器、手工优化或修复链。
