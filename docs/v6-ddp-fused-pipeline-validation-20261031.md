# DDP 原样流水候选的一次正式配对验证

2026-10-30 22:31:24.174553UTC终态 `batch_finished`，不是低磁盘事件。完整审计一次成功，audit `175995c0f88e54a1b6e466c9da30375598070c6bd46a82609fd06c0663eb444a` / [public](evidence/v6-ddp-fused-pipeline-validation-20261031/public-terminal-audit.json) `daf91fed035550d25e9583e4c4c0a9ae42db5bb7d88081a25b3b7f9c837c8d0a`。1542冻结、core、source/current、请求、manifest/source/entry/adapter/controls/UUID全部保持；9 jobs18 ranks valid，40 rank clean/16 full checks（每次248,879,620元素）/24 own metric traces/2 native SQLite。精确空参数fixed-fused-pipeline默认fine8、代码02b933不改；接受928db266、publication=exploration，source/newbranch current10992不发布。控制器完成、full quality、正式接受分开核实。

| 配对 | B当前508a ms | C02b933 ms | 收益 ms | 噪声界限 ms |
|---|---:|---:|---:|---:|
| 0 | 54.067719 | 48.020123 | 6.047596 | 0.869176 |
| 1 | 53.795709 | 48.161193 | 5.634516 | 0.951302 |

原rankmax先于median/MAD/noise；aggregate53.931714→48.090658ms，**10.8304660964%正式改善**。原anchor133.379621/limit140.04860205通过，不用B54ms重定标、不池化固定B/C。全部样本含B54.614143/C48.247216保留。8个首检查max0，8个第二检查rank0=9.5367431640625e-7/rank1=1.1920928955078125e-7，在原容差内，exact replicas observe通过；非全部零。每rank quality第一parent_eager_baseline、第二graph_setup_capture/reference1，不冒称两次optimized检查；未持久化全部原张量、未新source比较。stream/deprecation/barrier警告保留，无未捕获traceback，6–12ms安装不是完整setup/capture成本。

每份24 metric trace独立按自己的workload window重算NCCL名字/载荷/总和/union/Graph/stream/rank聚合：B1 AR、C8 AR，C=6144+6×56,702,976+157,535,232，总497,759,232B相同。[另一次只读事件扫线](evidence/v6-ddp-fused-pipeline-validation-20261031/same-trace-interval-check.json) SHA4b6b55d6…验证B重叠0、C11.842126–12.383792ms；局部同钟区间描述不是wire、关键路径、可恢复收益或独立因果分解。运行峰值B约1.529GB/C约4.613GB、lifecycle2.427GB/5.657GB、C reserved6.354GB增加，不是内存收益、也不新增内存约束。

正常profile每rank1138kernels/13calls12groups，未更新registry。独立timeline每rank8AR/67syncAPI/0gradient callbacks/1bucket；0回调是未观测生产事件，不是没有梯度；一个bucket inventory不证明八个chunk唯一owner。rank0/1 wall49.335465/49.268949ms、NCCLsum37.722467/36.393849、overlap12.507179/11.953669均属该次采集。candidate diagnostics六个正常False generator位置125/147/157/176/177/198合计458返回/rank、无异常，run-only不执行observe/compare。本轮没有calibration。

新精确02b933 native `nsys-56f3b3b7b4`：r0 SQLite `b86752ffcfe7fec16a87b6704b4a1c2a77f1d5206badbb90f300c123a104377d`，r1 `e866c778c01ecdb52d4aad346c6b963a009c434094f6cb5d84a9413487363b0e`，normalized `d71e1b6cd5367e0b9fc749bfd80f6cf5393ed5d2e37140d4e143f56d712565d3`。实际各1128kernel/1112Graph/17版本后缀cudaGraphLaunch_v10000/8 eager AllReduce在Graph外，RING_LL入口；不是把Torch1138/1122或旧508a native计数搬来。原SQLite/API/correlation分别重算，NCCLsum36.194282/215.347664ms属不同扰动rank周期，非clean/wire/关键路径。当前native与tuning日志实际channel00/01 SHM/direct/direct。独立tuning6144B为RING/LL单channel，大chunk为RING/SIMPLE双channel，含setup/warmup/peer flags及一次497MB全规约；不按日志条数当每步计数，不跨capture合并成timed kernel协议证明。

真实planner `generation-fd0d55290f92`/session ses_e7113742fffeSb6n3aAQnhX6e1：8初始文件82,263B字节相等、普通prompt/model/tools/core不改，10assistant/13tools/0schema repairs/provider0.16317758512估算非账单。完整1873 read行逐行核对、0行内截断；tool8字节截断由tool9 offset1128续读，1353行observations的并集全部暴露。八输入实际均读；tools1/2只列目录（opencode.json名称不是内容），tool12一次写plan无编辑，13仅schema/known collectors/revision检查。无记录model/tensor/CUDA/processgroup/collective/privateconfig/外部审计答案内容读取；非hermetic，不能断言绝对无外部访问。原完整history/tools/outputs只在服务器，未输出敏感metadata。

自主选择timeline、candidate_diagnostics、nsys、nccl_tuning四项，非控制器手选；plan正确排除calibration硬下界，但“fully exposed”和用另一profile36.5–37.4ms除clean48ms的约78%不是关键路径成本。`binding direct/no functional replacements`不等于没有whole-step replacement。它读到的observations仍是父508a单AR/旧registry，不是新candidate代码或新native registry。原plan保留，不将审计纠正加入角色输入。

在上述真实新接受之后，声明[一次原版本独立闭合](v6-ddp-fused-pipeline-independent-20261031.md)，不是追加候选/修复或自动生产采用。[预检](evidence/v6-ddp-fused-pipeline-validation-20261031/next-independent-preflight.json)1691冻结身份通过，GPU空闲余约9.05GiB，1GiB预算+8GiB警戒；无清理。旧49392已handled/进程不存在，queued重复事件去重。02b933受限输入/owner/sharedpool/cutgrad生命周期、setup hooks、postupdate fallback未恢复weights/rank不对称/global override条件局限全部保留；本轮质量不证明任意输入/并发/故障安全。Qwen/FSDP最新拒绝/77ac/observer闭合链不重开。

香港06:27:48实核：迟到的固定B/C完成事件已按原22:21:02终态和终审155c368d去重，没有重跑或新增监听。正式controller233246/唯一监听49392仍存活，未终态；实际collection-planner generation-fd0d55290f92已启动，snapshot8初始文件82,263B（本次只核角色/初始规模，完整输入/工具待终审）。当时GPU无worker，处正常模型规划阶段，150GiB盘剩9,759,846,400B约9.09GiB；无清理、并行GPU或参数重试。

已于2026-10-30 22:25:22.159723UTC（香港06:25:22）实际启动controller `233246`、冻结 `5e19841d7009dca8c1795321607776f69baf05cf`、唯一隐藏plain watcher `49392`。真实LocalOpenCode start/health/stop通过，server_url=null，无模型预检。22:26:32[启动身份核验](evidence/v6-ddp-fused-pipeline-validation-20261031/startup-verification.json) SHA `de140538db9bbaa3445c1b0329a34c398da558b4a034d9bab43c9b05ba610a8d`：1542冻结/source与branchcurrent10992/原request/prepared/discovery/精确928db266及单02b933 bundle/空参数/core均保持，实际三份B/C/C请求的source/entry/adapter/controls/UUID核实；第二C两rank233958/233959在GPU运行，各6672MiB。实际150GiB盘余9,795,452,928B约9.12GiB，无清理或并行GPU。此时尚无planner generation；终态需审完整四请求/质量/噪声/正常profile/planner及collectors，不把worker状态当接受。旧46572固定结果已直接终审/handled/退出，晚到事件去重不重跑。

前置[固定B/C完整审计](v6-ddp-fused-pipeline-fixed-20261031.md)155c368d已完成，当前任务原完整质量、约束通过，实际通信调度与整步表现具备进一步正式验证依据。下一有界步骤只调用一次现有 `ProjectSearch.validate_bundle(publish=False)`：fresh B/C/C/B、原噪声/质量/约束规则，不将固定样本池化，不手改候选或试其他参数。此处为启动前声明，实际启动另记。

新root `/root/autodl-tmp/dev/v6-work/ddp-fused-pipeline-validation-20261031`，source `ddp-fused-pipeline-fixed-20261031/ddp-fixed` current10992e0d（accepted508a），C精确staged `928db26604d7af4040d6a4a5547fbacdce24486e521a2317bbd48f1f4cc7bf1d`，单fixed-fused-pipeline/空参数。code02b93360c4d53add1ef14fc91fb32f757b09db43597888f880c1cd25793d614a 25,781B，proposal dabb839bd7b29b8b073ff0a134b3114f602b4797c7c37331b14b0ab4424b4a11 13,832B，requires[]/replacesfixed-observation-route。空参数调用代码自身fine8，不人工选桶、无fine9/coarse5探索。source/branch current保持，即使接受也仅exploration；新ID不是新机制。原prepared7a49/currentdevice controls、request/discovery/source共享只读，不初始rediscovery/source/control或reference副本。

正常API会执行resource_feedback profile，并由原真实collection-planner自主选择existing collectors；控制器不手选collector，不增加prompt/事实表/reviewer。普通prompt/model/tools/APIobserve-only无新cap，实际snapshot/tools/CPU-GPU探测/错误/collector结果均须终审，不能从控制器意图推断无模型探测。无selector/candidate/参数探索/新实现/独立阶段/生产采纳/指针发布；不打印私config或任何metadata.output/display，rawhistory/config/snapshots仅服务器。

原GPT2 12层768hidden/12heads/50257vocab/1024npos、两rankB1S128accum4/effective8seq1024tokens、FP32/eager/noTF32/dropout0/usecacheFalse、seed20261020/input31000+rank、完整zero/FWBW/loss÷4/SGD.001无momentum/weightdecay/foreachFalse/twosync不改。2连续full grads/updatedparams/lossfinite/exactreplicas1e-6/1e-4、2warmup5clean、aligned rankmax先于median/MAD/noise。anchor133.379621→140.04860205仍是原约束，不用当前B54ms重定标，不增memory constraint；C运行峰值4.61GB增加保留。每3独立postclean profile/rank/endpoint的原始trace分别重算，NCCL sums不是wire或关键路径，不跨profile减clean。同trace重叠不等于全部可恢复收益或独立因果归因。

保留代码局限及真实质量范围：quality第一parent、第二setup，不是两optimized；sharedpool/cutgrad/输入owner与有限guards、setup失败hooks、postupdate fallback不恢复权重、rank非对称失败/global override非通用安全保证。固定数据与formal分开，若拒绝不重试/覆盖。只有真正新正式接受，才另行声明原始当前设备baseline对完整版本的独立B/C/C/B与必要native证据；不自动独立或采纳。

[只读预检](evidence/v6-ddp-fused-pipeline-fixed-20261031/next-validation-preflight.json)1542冻结身份保持，GPU空闲，150GiB盘余9,839,620,096B（约9.16GiB），1GiB增长预测+8GiB警戒通过，无清理，不是API/collector cap。低盘不是完成，不终止活动任务/盲删；只在GPU完成后按已授权冷存范围处理，保护frozen/current/source/controls/模型/provider，旧RESTORE/journals保持。后台SSH/队列显式隐藏、唯一plain watcher；旧46572已handled/退出，晚事件去重不重跑。Qwen/FSDP最新拒绝/77ac/observer保持关闭。结果回来完整审查并有界决策，不静默停研或无限候选。
