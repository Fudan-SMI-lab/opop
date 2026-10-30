# DDP 已独立接受流水版本的一次正常通信发现

依据[独立终审](v6-ddp-fused-pipeline-independent-20261031.md)5225b4bd，精确02b933完整版本已按原规则接受：114.757729→48.365121ms/57.8545851%，完整质量、原约束、两pair噪声通过，实际13→8 AllReduce且总497,759,232B不变。此为分段计算+通信/更新调度联合优化，不能把formal10.83%或旧53.91%相加。下一项只声明一次现有 `ProjectSearch.discover()`，包含正常真实collection-planner及其自主选择的existing collectors，更新当前版本发现与证据；实际启动另记。

新root `/root/autodl-tmp/dev/v6-work/ddp-pipeline-communication-evidence-20261031`。显式从 `ddp-fused-pipeline-independent-20261031/ddp-independent` 的accepted exploration `73ac265fbc8baff61b7c02bcd593a41c0ceb9a38e9586957cfff561c6e73c2a9` fork；source current仍为EMPTY7a49，不把current!=accepted当失败，也不手动publish。新隔离ddp-evidence只attach discovery/registry/plan，精确files/bundle/contract不改。02b93360c4d53add1ef14fc91fb32f757b09db43597888f880c1cd25793d614a 25,781B、proposal dabb839bd7b29b8b073ff0a134b3114f602b4797c7c37331b14b0ab4424b4a11 13,832B、单fixed-fused-pipeline/空参数{}，代码自身fine8不改。原928db266/source10992/3cd9/bbe28c/1fc/4541/历史09b7指针均冻结。

accepted版本还继承508a registry4f0a0324…/plan ce4d430f…，独立目录无discovery，formal profile不自动attach当前registry。需要新精确版本的实际发现后，才能决定任何后续候选；本次不是新优化实现/共存/接受。无selector/candidate/search参数/trial/独立阶段/source-control rerun/reference副本/生产pointer发布，不手选group/algorithm/collector，不追加observer或旧修复链。

planner只接原request/availablecollectors/真实新observations/coverage/schema及自动环境；原prompt/model/tools/APIobserve-only无cap。无developer独立审计facttable/causalanswer/手写实现或reviewer。现有fork_run会复制原version结果为latest-feedback，但普通初始planner没有feedback suffix/previous-feedback输入；需核实真实snapshot和后续完整工具访问，不能只依控制器意图断言没有CPU/GPU探测或外部访问。外部FS非hermetic，private configs/credentials绝不输出，metadata只用exit/truncated/status/hash白名单。raw history/config/snapshot保留服务器，GitHub仅代码/哈希/小型清理元数据。

原GPT2 12层768hidden12heads/vocab50257/npos1024、2ranks B1S128accum4/effective8seq1024tokens、FP32/eager/noTF32/dropout0/usecacheFalse seed20261020/input31000+rank，完整zero/FWBW/loss÷4/SGD.001无momentum/weightdecay/foreachFalse/twosync/owners不改。evaluate才执行2连续full grads/updatedparams/lossfinite/exact replicas1e-6/1e-4、2warmup5clean；profile/run-only不是新fullquality/clean/接受。原anchor133.379621→140.04860205保持、不新增memory constraint。当前两4090 UUID20e8003c…/a24af2e6…、driver580.105.08、adapterb276a7ed…、coreed24af7a…保持，shared当前controls/source只读、无新sourcecompare，原source/metric/CPU事实辅助边界保持。

实际collector如果返回局部error，不能因controller完成宣称全部成功；保留raw/errors，不手工替换collector或重复至通过。所有新trace按自己的window/算法入口/Graph/stream/载荷/interval计算；同capture overlap不是wire/关键路径/可恢复收益，跨profile不能相减，NODE/P2P只说明能力。若planner选newNsys审新实际02b933采集，不能拿已复用1128/1112Graph/17launch/8AR旧native计数当本批证据。registry lineage只是containment，不能证明因果group或多active bundles。

空间预检1769冻结身份保持/GPU空闲，actual150GiB df余9,480,597,504B约8.83GiB。当前精确02b933 formal9jobs实际du82,251,776B、之前完整508a evidence139,771,904B，据此为新的发现/普通planner与collectors保留768MiB增长预测及8GiB警戒；不是API/collector cap，不改原实验标准，无清理/大controls副本。运行中低盘不等完成，不终止活动实验/盲删；只在GPU结束后按已授权server-only无损cold并保护active/frozen/source/controls/accepted/model/provider/deps。既有RESTORE/journals保留，冷存plain缺失不等丢失。

所有后台SSH/queue显式CREATE_NO_WINDOW，不关VSCode。旧40904独立结果已handled/进程退出、晚事件去重，唯一新plain隐藏watcher只监新batch；terminal-before-precheck就不建listener，低盘处理后仍运行才一份替换监听、不重复无变化告警。终态完整审真实planner snapshots/tools/inputs/outputs/probes/errors与每collector/result/rawtrace，再决定下一有界通信步骤，不自动候选链或静默停整个项目。Qwen/latest rejected FSDP/77ac/observer mechanics和一次eef828纠正保持关闭。
