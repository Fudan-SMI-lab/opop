# DDP 原样流水候选的一次正式配对验证

前置[固定B/C完整审计](v6-ddp-fused-pipeline-fixed-20261031.md)155c368d已完成，当前任务原完整质量、约束通过，实际通信调度与整步表现具备进一步正式验证依据。下一有界步骤只调用一次现有 `ProjectSearch.validate_bundle(publish=False)`：fresh B/C/C/B、原噪声/质量/约束规则，不将固定样本池化，不手改候选或试其他参数。此处为启动前声明，实际启动另记。

新root `/root/autodl-tmp/dev/v6-work/ddp-fused-pipeline-validation-20261031`，source `ddp-fused-pipeline-fixed-20261031/ddp-fixed` current10992e0d（accepted508a），C精确staged `928db26604d7af4040d6a4a5547fbacdce24486e521a2317bbd48f1f4cc7bf1d`，单fixed-fused-pipeline/空参数。code02b93360c4d53add1ef14fc91fb32f757b09db43597888f880c1cd25793d614a 25,781B，proposal dabb839bd7b29b8b073ff0a134b3114f602b4797c7c37331b14b0ab4424b4a11 13,832B，requires[]/replacesfixed-observation-route。空参数调用代码自身fine8，不人工选桶、无fine9/coarse5探索。source/branch current保持，即使接受也仅exploration；新ID不是新机制。原prepared7a49/currentdevice controls、request/discovery/source共享只读，不初始rediscovery/source/control或reference副本。

正常API会执行resource_feedback profile，并由原真实collection-planner自主选择existing collectors；控制器不手选collector，不增加prompt/事实表/reviewer。普通prompt/model/tools/APIobserve-only无新cap，实际snapshot/tools/CPU-GPU探测/错误/collector结果均须终审，不能从控制器意图推断无模型探测。无selector/candidate/参数探索/新实现/独立阶段/生产采纳/指针发布；不打印私config或任何metadata.output/display，rawhistory/config/snapshots仅服务器。

原GPT2 12层768hidden/12heads/50257vocab/1024npos、两rankB1S128accum4/effective8seq1024tokens、FP32/eager/noTF32/dropout0/usecacheFalse、seed20261020/input31000+rank、完整zero/FWBW/loss÷4/SGD.001无momentum/weightdecay/foreachFalse/twosync不改。2连续full grads/updatedparams/lossfinite/exactreplicas1e-6/1e-4、2warmup5clean、aligned rankmax先于median/MAD/noise。anchor133.379621→140.04860205仍是原约束，不用当前B54ms重定标，不增memory constraint；C运行峰值4.61GB增加保留。每3独立postclean profile/rank/endpoint的原始trace分别重算，NCCL sums不是wire或关键路径，不跨profile减clean。同trace重叠不等于全部可恢复收益或独立因果归因。

保留代码局限及真实质量范围：quality第一parent、第二setup，不是两optimized；sharedpool/cutgrad/输入owner与有限guards、setup失败hooks、postupdate fallback不恢复权重、rank非对称失败/global override非通用安全保证。固定数据与formal分开，若拒绝不重试/覆盖。只有真正新正式接受，才另行声明原始当前设备baseline对完整版本的独立B/C/C/B与必要native证据；不自动独立或采纳。

[只读预检](evidence/v6-ddp-fused-pipeline-fixed-20261031/next-validation-preflight.json)1542冻结身份保持，GPU空闲，150GiB盘余9,839,620,096B（约9.16GiB），1GiB增长预测+8GiB警戒通过，无清理，不是API/collector cap。低盘不是完成，不终止活动任务/盲删；只在GPU完成后按已授权冷存范围处理，保护frozen/current/source/controls/模型/provider，旧RESTORE/journals保持。后台SSH/队列显式隐藏、唯一plain watcher；旧46572已handled/退出，晚事件去重不重跑。Qwen/FSDP最新拒绝/77ac/observer保持关闭。结果回来完整审查并有界决策，不静默停研或无限候选。
