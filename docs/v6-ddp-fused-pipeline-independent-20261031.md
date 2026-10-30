# DDP 精确流水完整版本的一次独立闭合

前置[正式验证完整终审](v6-ddp-fused-pipeline-validation-20261031.md)175995c0确有新接受：928db266相对当前508a正式改善10.8304661%，两对收益超过原噪声、完整质量与原约束通过，并有新02b933 native通信机制。现在只声明一次 `promote_version(publish=False)`，完整版本对原当前设备EMPTY baseline7a49的fresh B/C/C/B。实际启动另记；此声明不预判独立接受。

新root `/root/autodl-tmp/dev/v6-work/ddp-fused-pipeline-independent-20261031`，source为formal `ddp-fused-pipeline-validation-20261031/ddp-validation`。精确accepted exploration `928db26604d7af4040d6a4a5547fbacdce24486e521a2317bbd48f1f4cc7bf1d`，sourcecurrent10992保持；新current为原EMPTY `7a49c6370b52d881a52550766b279df22684adc16cf3c91be56275f8fe2e1df9`，无论decision保持不发布。单fixed-fused-pipeline/空参数{}，02b93360c4d53add1ef14fc91fb32f757b09db43597888f880c1cd25793d614a 25,781B、proposal dabb839bd7b29b8b073ff0a134b3114f602b4797c7c37331b14b0ab4424b4a11 13,832B逐字不改。空参数调用代码默认fine8，不试fine9/coarse5、不手选切分。无Agent/search/参数/collector/newnative/source-control rerun/reference copies/初始discovery/pointerpublication。

原GPT2 12层768hidden12heads/vocab50257/npos1024，2ranksB1S128/accum4/effective8seq1024tokens，FP32/eager/noTF32/dropout0/usecacheFalse seed20261020/input31000+rank。完整zero/FWBW/loss÷4/SGD.001无momentum/weightdecay/foreachFalse/twosync/owners不改。2连续full grads/updatedparams/lossfinite/exact replicas、atol1e-6/rtol1e-4，2warmup5clean/rank，aligned rankmax先于median/MAD/noise。原anchor133.379621→140.04860205，非54ms新锚点，无新memory constraint。DDP gradients为replicas、不是FSDP shards。期望四workers八ranks/40rank样本16full checks/24各自metric traces；任何失败/尾部/拒绝保留，不重试通过、不覆盖formal verdict或混入固定样本。

当前设备GPU20e8003c-f531-d50c-084c-a9a97fe94004/a24af2e6-99bb-9729-5d14-50fa10a2b321、两4090/driver580.105.08、adapterb276a7ed5524fc0c47d6080a6349a0a65b9e14bda8270cbaf87d6cd81bceb193、coreed24af7acc8caa15578e0b859c85ef8a901be51354448ed39094606ed9e10a2f保持。shared current controls只读，source-evidence-reuse只核此前真正执行的currentdevice完整source/replica negative哈希，不是新tensor/model/source比较；cbf30d10微小非零withinTol及bitwise replica negative边界不变，原source/metric mapping/CPU事实辅助仍非unaided AB。

native-candidate-reuse只复用这次正式02b933 `nsys-56f3b3b7b4`：r0 b86752ff…/r1 e866c778…/normalized d71e1b6c…，各1128kernel/1112Graph/17版本后缀launch/8eagerAR outsideGraph；实际SHM/direct/direct。其36.194282/215.347664ms扰动NCCL周期不是clean/wire/关键路径。当前整步机制为分段反传、通信和更新调度，总497,759,232B不减少，不声称纯通信贡献或通用owner安全；不另跑Nsys。原baseline13AR、新C8AR的每一份metrictrace需独立核实，不能搬formal counts。

1691冻结身份只读预检通过，actual150GiB盘余9,713,926,144B约9.05GiB/GPU空闲；1GiB增长预测+8GiB警戒通过，预算不是API cap，无清理。低盘不等完成、不终止活动实验/盲删；仅GPU完成后按已授权server-only无损cold、保护frozen/source/accepted/shared controls/models/provider/deps。RESTORE journals保持，冷存plain缺失不等数据丢失。所有SSH/后台/队列CREATE_NO_WINDOW，不关VSCode，旧49392已handled，若新batch仍运行只创建一个plain隐藏watcher，若terminal-before-precheck则不建listener。

终态审每份实际请求/results/logs/coverage/fullchecks/samples/24owntraces、B/C/C/B噪声约束/source-native复用及指针冻结。结果前不声称独立接受，任何拒绝不再试参数或修复。若独立闭合成功，后续先依据新的精确版本与当前registry是否过时决定一个有界证据步骤，不自动候选链。Qwen/latest FSDP/77ac/observer关闭，整体研究不因单batch结束而静默停止。
