# DDP 原样流水候选的一次固定 B/C 验证

本批实际22:19:56.955559UTC启动controller231902/冻结219c447c594f7102eb8408fbe5ed196fde46a5ac，22:21:02.069816UTC已完成；完整终审SHA `155c368da037515d41d33e15919746500734546acebde430eb7edfba54073b0f`，[公开结果](evidence/v6-ddp-fused-pipeline-fixed-20261031/public-terminal-audit.json) SHA `4550e7562d3c1eab69d5264347f5df1a9d5242a653d0fcc13e8339223a7a2f8a`。1497冻结/core/两实际请求/materialized sources/entry/adapter/controls/UUID/source及branch current保持。staged C `928db26604d7af4040d6a4a5547fbacdce24486e521a2317bbd48f1f4cc7bf1d`，原样02b933/单fixed-fused-pipeline/空参数，替换508a而非并存。4rank均valid/math通过，20rank样本/8全检查各248,879,620元素/12own raw traces核实；无重跑或新Agent。

两端第一检查均0，第二检查rank0 max9.5367431640625e-7、rank1 max1.1920928955078125e-7，原容差通过且observe exact replicas通过；不是全zero，也无完整tensor持久化/新独立source比较。quality路线各为第一parent、第二graph_setup_capture/reference1，非两个optimized检查。所有worker logs已读，原AccumulateGrad stream/deprecation/barrier警告保留，无uncaught traceback；安装8–12ms不当全部第二调用捕获成本。

aligned rankmax B样本54.738412/54.389143/54.668891/53.980413/53.452269ms，中位54.389143/MAD0.349269；C48.586987/48.359030/47.962373/48.328994/49.251586，中位48.359030/MAD0.227957。尾样本全留，两端原140.04860205ms约束通过。该单B/C不是正式噪声/搜索/独立接受，不能与旧数据池化。运行峰值B1,528,818,688B/C4,612,856,320B，生命周期2,427,159,040/5,655,992,320B，C reserved也更高；实际显存增加，不新增约束或称memory gain，不能只靠peak唯一归因为某个引用。

12份trace逐份重算各自workload窗口：B每rank1105kernel/1103Graph/2launch/1 eager AR；C1138/1122Graph/17launch/8 eager AR，均AllReduce Sum f32 RING_LL入口。C payload6144+6×56,702,976+157,535,232，总497,759,232B不变。无新Nsys/物理transport日志，不把旧SHM当本捕获证明。另一次只读[同trace事件扫描](evidence/v6-ddp-fused-pipeline-fixed-20261031/same-trace-interval-check.json) SHA78ea59e7…独立复算：B overlap0，C六周期11.824900–12.283282ms，comm stream17、全部NCCL在Graph外。这是同钟区间重叠，非wire/关键路径/可恢复收益；NCCL sum中位指标反而增加，不能做clean成本拆分。原setup-hook/有限guards/update后fallback/非对称失败/共享pool条件限制未变，实际质量仅当前任务范围。

监听本地builder第一次引号SyntaxError/随后文件未生成发生在SSH/listener前，留档后仅修文件式builder。唯一隐藏监听46572实际创建后正常完成，现已handled/进程退出，延迟事件去重。首次startup reader完成时实验已终态，5be7bca9…只作为请求/身份核验，不替代本终审。另一次只读本地读取误拼startup模板文件名失败，随后rg找到正确文件；无实验重启。完整审计首轮通过，无模型/CPUfixture/GPU重试。

下一步已声明[一次原样正式配对验证](v6-ddp-fused-pipeline-validation-20261031.md)，沿用staged02b933、B当前508a、空参数和原validate_bundle publishFalse/ABBA规则；正常真实planner自主collectors，不手选或预排独立。1542保护身份预检/盘余9,839,620,096B/1GiB预算+8GiB通过，GPU空闲，无清理。失败保留、不换fine9/coarse5重试，只有新正式接受才另行声明独立全版本闭合。以下为执行前声明。

依据[完整候选终审](v6-ddp-fused-pipeline-cpu-feedback-20261031.md)6cd3bb84，执行一次现有LocalExecutor/device leases下的B后C。root `/root/autodl-tmp/dev/v6-work/ddp-fused-pipeline-fixed-20261031`，此处为执行前声明，实际启动另记。不是正式ABBA/search/independent接受，不把单B/C收益当接受。没有新Agent、参数探索、collector选择、Nsys、observer、source/control运行或reference副本、指针发布。

B为 `ddp-fused-communication-evidence-20261031/ddp-evidence` current `10992e0d329f45c941eb2edcd31f60fc1e7f17c7a6bba8241bad93edde56a979`、单fixed-observation-route/508a。C为原样 `02b93360c4d53add1ef14fc91fb32f757b09db43597888f880c1cd25793d614a` 25,781B，proposal `dabb839bd7b29b8b073ff0a134b3114f602b4797c7c37331b14b0ab4424b4a11` 13,832B，requires[]/replaces原bundle、参数 `{}`。空参数采用生成代码自己DEFAULT_PLAN=fine8，不手选chunk计划，也不接着试fine9/coarse5。源和隔离branch current均不变，sourceaccepted/independent/formal/historical指针保留。

使用原GPT2 12层hidden768/12heads/vocab50257/npos1024、两rank B1/S128/accum4/effective8seq1024tokens，FP32/eager/noTF32/dropout0/usecacheFalse，seed20261020/input31000+rank，完整zero/FWBW/loss÷4/SGD.001 momentum0 weight_decay0 foreachFalse/twosync/owners。原2连续full grads/updated parameters/loss/finite/exact replicas检查atol1e-6/rtol1e-4、2warmup/5clean/rank不改；原3独立postclean通信profile/rank/endpoint最多12raw traces，分别重算，aligned rankmax先于median/MAD。原anchor133.379621及140.04860205ms约束不重定标，不新增memory constraint。adapter b276a7ed…/只读新设备controls/两GPU20e8003c与a24af2e6不改；原source比较仅既有证据，不新跑。

B两rank原valid/math、两完整检查、5样本通过才运行C；B失败则C不跑。时间不是B门。C的任何质量/运行/约束失败、fallback、尾样本全保留，不手改或反复执行。两端完成亦不预排正式接受/独立闭合；先完整审实际路线、状态、full checks、请求、trace、identity和错误，再作有界下一决定。

待验证机制为候选自主链式切分反向、分块通信及update等待；CPU测试没有验证精确CUDA graph共享pool/切分梯度地址/两rank异步通信。第一质量parent/第二setup的边界保持，setup失败hooks/cut_log、copied inputs/有限guards、postupdate fallback不恢复权重、rank非对称故障/global override等条件限制保留。NCCL sums含等待，不作wire/关键路径或跨profile减clean；实际通信证据与完整质量/独立整步收益才支持后续通信相关接受。旧508a独立53.91019995%属于联合计算合批+通信融合，与formal24.07796199%及旧ccd90收益不相加。

[只读预检](evidence/v6-ddp-fused-pipeline-cpu-feedback-20261031/next-fixed-preflight.json)1497冻结身份、current/request/prepared/discovery/core ed24af7…保持。GPU空闲/150GiB盘实际剩9,880,834,048B，512MiB新输出预测+8GiB警戒通过，非API上限，无清理。low_disk_space不是完成，不终止活动实验/盲删；旧cold RESTORE/journals保留。私有history/config只在服务器，GitHub代码/hash/小摘要；禁止打印私配置，后台SSH/监听显式隐藏。不关闭VSCode、不重启旧31676，长批最多一个plain completion/disk watcher；短批在预检已终态则不创建监听。

Qwen、最新拒绝FSDP、77ac和observer修复链继续关闭；本项不是新纠正、生产采纳或整个研究完成。
