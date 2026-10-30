# DDP 原样流水候选的一次固定 B/C 验证

依据[完整候选终审](v6-ddp-fused-pipeline-cpu-feedback-20261031.md)6cd3bb84，执行一次现有LocalExecutor/device leases下的B后C。root `/root/autodl-tmp/dev/v6-work/ddp-fused-pipeline-fixed-20261031`，此处为执行前声明，实际启动另记。不是正式ABBA/search/independent接受，不把单B/C收益当接受。没有新Agent、参数探索、collector选择、Nsys、observer、source/control运行或reference副本、指针发布。

B为 `ddp-fused-communication-evidence-20261031/ddp-evidence` current `10992e0d329f45c941eb2edcd31f60fc1e7f17c7a6bba8241bad93edde56a979`、单fixed-observation-route/508a。C为原样 `02b93360c4d53add1ef14fc91fb32f757b09db43597888f880c1cd25793d614a` 25,781B，proposal `dabb839bd7b29b8b073ff0a134b3114f602b4797c7c37331b14b0ab4424b4a11` 13,832B，requires[]/replaces原bundle、参数 `{}`。空参数采用生成代码自己DEFAULT_PLAN=fine8，不手选chunk计划，也不接着试fine9/coarse5。源和隔离branch current均不变，sourceaccepted/independent/formal/historical指针保留。

使用原GPT2 12层hidden768/12heads/vocab50257/npos1024、两rank B1/S128/accum4/effective8seq1024tokens，FP32/eager/noTF32/dropout0/usecacheFalse，seed20261020/input31000+rank，完整zero/FWBW/loss÷4/SGD.001 momentum0 weight_decay0 foreachFalse/twosync/owners。原2连续full grads/updated parameters/loss/finite/exact replicas检查atol1e-6/rtol1e-4、2warmup/5clean/rank不改；原3独立postclean通信profile/rank/endpoint最多12raw traces，分别重算，aligned rankmax先于median/MAD。原anchor133.379621及140.04860205ms约束不重定标，不新增memory constraint。adapter b276a7ed…/只读新设备controls/两GPU20e8003c与a24af2e6不改；原source比较仅既有证据，不新跑。

B两rank原valid/math、两完整检查、5样本通过才运行C；B失败则C不跑。时间不是B门。C的任何质量/运行/约束失败、fallback、尾样本全保留，不手改或反复执行。两端完成亦不预排正式接受/独立闭合；先完整审实际路线、状态、full checks、请求、trace、identity和错误，再作有界下一决定。

待验证机制为候选自主链式切分反向、分块通信及update等待；CPU测试没有验证精确CUDA graph共享pool/切分梯度地址/两rank异步通信。第一质量parent/第二setup的边界保持，setup失败hooks/cut_log、copied inputs/有限guards、postupdate fallback不恢复权重、rank非对称故障/global override等条件限制保留。NCCL sums含等待，不作wire/关键路径或跨profile减clean；实际通信证据与完整质量/独立整步收益才支持后续通信相关接受。旧508a独立53.91019995%属于联合计算合批+通信融合，与formal24.07796199%及旧ccd90收益不相加。

[只读预检](evidence/v6-ddp-fused-pipeline-cpu-feedback-20261031/next-fixed-preflight.json)1497冻结身份、current/request/prepared/discovery/core ed24af7…保持。GPU空闲/150GiB盘实际剩9,880,834,048B，512MiB新输出预测+8GiB警戒通过，非API上限，无清理。low_disk_space不是完成，不终止活动实验/盲删；旧cold RESTORE/journals保留。私有history/config只在服务器，GitHub代码/hash/小摘要；禁止打印私配置，后台SSH/监听显式隐藏。不关闭VSCode、不重启旧31676，长批最多一个plain completion/disk watcher；短批在预检已终态则不创建监听。

Qwen、最新拒绝FSDP、77ac和observer修复链继续关闭；本项不是新纠正、生产采纳或整个研究完成。
