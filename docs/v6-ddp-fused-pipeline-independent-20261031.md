# DDP 精确流水完整版本的一次独立闭合

终态2026-10-30 22:42:21.885918UTC `batch_finished`；实时150GiB df余9,480,863,744B/GPU空闲/controller237970退出，非低磁盘完成误判。完整audit `5225b4bd7199980017a4ba9cda7322c23371f7bf3c6c818e2ca03168d4da966b` / [public](evidence/v6-ddp-fused-pipeline-independent-20261031/public-terminal-audit.json) `c3c56415c0b440f23bdad940cccbec516e08a789e8afc33998e3a1103c88d096`。1691冻结、完整版本files/bundle/contract、四实际请求及materialized manifest/entry/adapter/controls/core/UUID、sourcecurrent10992/新EMPTYcurrent7a49均保持。trial `version-19574e13ce62`，精确02b933独立接受 `73ac265fbc8baff61b7c02bcd593a41c0ceb9a38e9586957cfff561c6e73c2a9`，publication exploration，不改source/accepted/current指针。

| 配对 | 原EMPTY B中位 ms | C完整02b933中位 ms | 收益 ms | 噪声界限 ms |
|---|---:|---:|---:|---:|
| 0 | 120.591369 | 48.152025 | 72.439344 | 4.674488 |
| 1 | 108.924089 | 48.578217 | 60.345872 | 1.495542 |

aligned rankmax先于median/MAD，aggregate **114.757729→48.365121ms，57.8545851147%独立改善**。原anchor133.379621/limit140.04860205通过；没有旧周期或固定/formal样本池化。B两组完整max-rank样本115.843446/122.928613/120.591369/120.979157/112.898082及108.924089/119.108422/108.356951/112.212040/108.176318；C48.576944/48.230835/47.986754/48.108149/48.152025及48.593020/48.578217/50.093932/48.544833/48.294032。全部尾部保留，不减暂停/删慢样本/重定锚点。与formal相对508a的10.8304661%及旧508a独立53.91019995%各自分开，不相加。

4 jobs8 ranks全部valid/math unchanged，40rank samples、16 full checks每次248,879,620元素；12检查max0、4个C第二检查rank0=9.5367431640625e-7/rank1=1.1920928955078125e-7原容差通过，exact replicas observe通过。C第一quality parent_eager_baseline、第二graph_setup_capture/reference1，不称两次optimized检查；原全部grad/updatedparams/lossfinite/精确跨rank检查不改，未持久化全部原张量或新source比较。八份worker日志完整审阅，deprecation/barrier/defaultloss/AccumulateGrad stream/profiler start-stop均保留，无未捕获traceback。

24份自己的原始metric traces逐文件原window/名字/载荷/union/事件扫线/Graph/stream/rank聚合重算。B每份3999kernel、0Graph、13 AllReduce；C1138kernel/1122Graph/17launch、8 eager AllReduce outsideGraph，各总497,759,232B。C6144+6×56,702,976+157,535,232；B12×28,351,488+157,541,376。这里原EMPTY B已有8.427643–12.764656ms同trace通信计算区间重叠，C11.857317–12.457415ms；不能搬formal旧508a B重叠0，也不能跨capture相减成收益、wire或关键路径。按实际记录复算，不从profile/旧Nsys复制计数。

内存不是收益：B runtime1,981,029,376/lifecycle2,479,585,792/reserved2,589,982,720B；C两endpoint rankmax runtime4,613,118,464/4,612,856,320，lifecycle5,656,254,464/5,655,992,320，reserved6,354,370,560B。原任务无新增内存约束，保留增加，不用旧508a1.529GB代替本次原EMPTY B。安装时间不等完整setup/capture成本。

source-evidence-reuse只核此前真实source/controls/negative哈希，不新CPUtensor/model/source/controls比较；原微小非零withinTol和tiny bitwise-negative、人工source/metric/CPU事实辅助边界保持。native只复用前正式精确02b933 nsys56f3：r0 b86752ff…/r1 e866c778…/normalized d71e1b6c…，各1128kernel1112Graph17版本后缀Graphlaunch、8NCCL在Graph外和原SHM/direct/direct日志；本批没有新native。原SQLite只读重算计数/入口，NCCL36.194282/215.347664ms仍是不同扰动周期，非本次clean、wire或关键路径。结论为**分段计算+通信/更新调度的独立接受联合优化**，非纯通信占比、字节减少或通用owner/故障安全证明。

第一个审计reader整份receipt-rank与raw-rank直接判等失败：receipt合法增加rank/PID/exit_code，共有字段全部一致。原reader `terminal-audit-reader-attempt-1.py` 与执行错误保留；只改成共有字段精确核对及附加字段/exit0检查，第二次审计完成，没有实验/角色/GPU/source/native重跑。原实验一次完成、空参数fine8、无额外参数或修复。所有guard/copiedinput/sharedpool/cutgrad/partialsetup hooks/postupdate fallback未恢复weights/rank不对称/globaloverrideFalse条件局限保留。

当前accepted73ac仍继承旧508a registry `4f0a0324…`/plan `ce4d430f…`，独立目录没有discovery，formal profile1138kernels13calls12groups未attach registry。因此下一有界步骤是[精确版本正常发现](v6-ddp-pipeline-communication-evidence-20261031.md)，而非直接追加candidate。[预检](evidence/v6-ddp-fused-pipeline-independent-20261031/next-evidence-preflight.json)1769身份通过，当前formal9jobs实际allocated82,251,776B/前完整508a evidence139,771,904B，采用768MiB新增长预测+8GiB警戒，余约8.83GiB，无清理/参考副本/手选collector。旧40904已handled且进程不存在，延迟事件去重；Qwen/latest FSDP/77ac/observer关闭。

已于2026-10-30 22:40:32.500217UTC（香港06:40:32）实际启动独立controller `237970`，冻结 `33637dd3dd0a18d1fad16565a3fbc1a1222ef6d0`，唯一隐藏plain watcher `40904`。22:41:37[启动核验](evidence/v6-ddp-fused-pipeline-independent-20261031/startup-verification.json) SHA9e32a59c6e90e7738fcfacc01edee76d645695ae545de91bb9e4341759d21c23：1691冻结/currentsource10992/新EMPTY current7a49/request/prepared/core、精确928db266单02b933空参数及三份B/C/C实际source/entry/adapter/controls/UUID保持。第二C两rank238777/238778正在GPU执行，余9,588,985,856B约8.93GiB；没有Agent generation/新native/清理/并行GPU。此记录只证明身份与活动，不是终态fullquality或独立接受，完整四请求/40样本16检查24traces待终审。旧49392 formal已完整终审175995c0/handled/进程不存在，晚到事件去重不重跑。

前置[正式验证完整终审](v6-ddp-fused-pipeline-validation-20261031.md)175995c0确有新接受：928db266相对当前508a正式改善10.8304661%，两对收益超过原噪声、完整质量与原约束通过，并有新02b933 native通信机制。现在只声明一次 `promote_version(publish=False)`，完整版本对原当前设备EMPTY baseline7a49的fresh B/C/C/B。实际启动另记；此声明不预判独立接受。

新root `/root/autodl-tmp/dev/v6-work/ddp-fused-pipeline-independent-20261031`，source为formal `ddp-fused-pipeline-validation-20261031/ddp-validation`。精确accepted exploration `928db26604d7af4040d6a4a5547fbacdce24486e521a2317bbd48f1f4cc7bf1d`，sourcecurrent10992保持；新current为原EMPTY `7a49c6370b52d881a52550766b279df22684adc16cf3c91be56275f8fe2e1df9`，无论decision保持不发布。单fixed-fused-pipeline/空参数{}，02b93360c4d53add1ef14fc91fb32f757b09db43597888f880c1cd25793d614a 25,781B、proposal dabb839bd7b29b8b073ff0a134b3114f602b4797c7c37331b14b0ab4424b4a11 13,832B逐字不改。空参数调用代码默认fine8，不试fine9/coarse5、不手选切分。无Agent/search/参数/collector/newnative/source-control rerun/reference copies/初始discovery/pointerpublication。

原GPT2 12层768hidden12heads/vocab50257/npos1024，2ranksB1S128/accum4/effective8seq1024tokens，FP32/eager/noTF32/dropout0/usecacheFalse seed20261020/input31000+rank。完整zero/FWBW/loss÷4/SGD.001无momentum/weightdecay/foreachFalse/twosync/owners不改。2连续full grads/updatedparams/lossfinite/exact replicas、atol1e-6/rtol1e-4，2warmup5clean/rank，aligned rankmax先于median/MAD/noise。原anchor133.379621→140.04860205，非54ms新锚点，无新memory constraint。DDP gradients为replicas、不是FSDP shards。期望四workers八ranks/40rank样本16full checks/24各自metric traces；任何失败/尾部/拒绝保留，不重试通过、不覆盖formal verdict或混入固定样本。

当前设备GPU20e8003c-f531-d50c-084c-a9a97fe94004/a24af2e6-99bb-9729-5d14-50fa10a2b321、两4090/driver580.105.08、adapterb276a7ed5524fc0c47d6080a6349a0a65b9e14bda8270cbaf87d6cd81bceb193、coreed24af7acc8caa15578e0b859c85ef8a901be51354448ed39094606ed9e10a2f保持。shared current controls只读，source-evidence-reuse只核此前真正执行的currentdevice完整source/replica negative哈希，不是新tensor/model/source比较；cbf30d10微小非零withinTol及bitwise replica negative边界不变，原source/metric mapping/CPU事实辅助仍非unaided AB。

native-candidate-reuse只复用这次正式02b933 `nsys-56f3b3b7b4`：r0 b86752ff…/r1 e866c778…/normalized d71e1b6c…，各1128kernel/1112Graph/17版本后缀launch/8eagerAR outsideGraph；实际SHM/direct/direct。其36.194282/215.347664ms扰动NCCL周期不是clean/wire/关键路径。当前整步机制为分段反传、通信和更新调度，总497,759,232B不减少，不声称纯通信贡献或通用owner安全；不另跑Nsys。原baseline13AR、新C8AR的每一份metrictrace需独立核实，不能搬formal counts。

1691冻结身份只读预检通过，actual150GiB盘余9,713,926,144B约9.05GiB/GPU空闲；1GiB增长预测+8GiB警戒通过，预算不是API cap，无清理。低盘不等完成、不终止活动实验/盲删；仅GPU完成后按已授权server-only无损cold、保护frozen/source/accepted/shared controls/models/provider/deps。RESTORE journals保持，冷存plain缺失不等数据丢失。所有SSH/后台/队列CREATE_NO_WINDOW，不关VSCode，旧49392已handled，若新batch仍运行只创建一个plain隐藏watcher，若terminal-before-precheck则不建listener。

终态审每份实际请求/results/logs/coverage/fullchecks/samples/24owntraces、B/C/C/B噪声约束/source-native复用及指针冻结。结果前不声称独立接受，任何拒绝不再试参数或修复。若独立闭合成功，后续先依据新的精确版本与当前registry是否过时决定一个有界证据步骤，不自动候选链。Qwen/latest FSDP/77ac/observer关闭，整体研究不因单batch结束而静默停止。
