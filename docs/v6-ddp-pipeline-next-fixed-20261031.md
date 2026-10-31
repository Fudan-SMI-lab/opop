# DDP 新候选原样B/C结果（2026-10-31）

香港**08:26:13–08:26:54**执行一次B→C，正常完成，未重试。[终审摘要](evidence/v6-ddp-pipeline-next-fixed-20261031/public-terminal-audit.json) SHA **51c74ccfe5f2dd4056f79394d6690a885043546fbb6e67a52a4358036bae372a**，完整审计 **3a91f7ea039d00d4b893a27fdb62c92c02dbf3328baf35d41f47611ebc4f5b51**。9当前关键文件、实际请求/物化代码/adapter/controls/core/UUID保持，4rank、20计时样本、8全检查及12自有metric traces核对；没有重复历史研究。

| 原样结果 | B：已接受02b933 | C：e610e944 |
|---|---:|---:|
| 对齐rank-max样本中位数/ms | 48.194381 | 47.633574 |
| MAD/ms | 0.087564 | 0.083504 |
| 原质量/数学策略/约束 | 通过 | 通过 |

单次观察改善**1.16363565%**，不是正式噪声或独立接受。全部样本保留：B为48.194381/48.370858/48.106817/48.201495/48.057283；C为47.560757/47.777593/47.550070/47.800358/47.633574。原anchor133.379621、limit140.04860205未变；没有新显存约束。

每检查比较248,879,620元素，四个首次检查误差0，四个第二次检查rank0最大9.5367431640625e-7、rank1最大1.1920928955078125e-7，原容差和exact-replica观察通过。两臂均为首次parent/第二次graph setup，不能称为两次优化路径检查；未保存全部原始tensor或重新比较source。C实际声明`graph_setup_capture_fast_pg`，原始trace的process-group 1与B的0相互印证新组参与。警告保留，无未捕获traceback；7–12ms安装耗时不是完整setup/capture成本。

每条自有workload-window trace均8次eager AllReduce、17次Graph launch，总6144+六次56702976+157535232=497,759,232B不变。B为1138kernels/1122Graph，C为1138/1130Graph；B八个RING_LL入口，C小消息一个TREE_LL入口及七个RING_LL入口。这些入口名不能证明逐事件Simple配置。C同capture重叠约12.78–12.87ms，属于描述性区间，不是wire、关键路径或可回收收益。没有新增Nsys、tuning或传输日志，不能移用旧SHM证据为新代码证明。Torch峰值约4.613GB/lifecycle5.656GB/reserved6.354GB接近，未覆盖全部NCCL外部分配。

现有LocalExecutor/设备租约执行原evaluator，空参数调用候选自己的fine8；原连续检查、2warmup/5clean/三份postclean profile不变。新bundle fixed-pipeline-next，requires空、替换fixed-fused-pipeline；staged版本 **805deb71707914fe2eb9ae52259fa66ae536d8470a6d9e7808bb3911f2feada1**。源/branch current0bc保持，没有发布。通用[实验编排](../scripts/experiments/v6_fixed_candidate_validation.py)由原request读取规则，没有case常量；实际controller SHA331073e41abe3205d46f44484727daa53445aab656f623429459bc2d9ee46d73，复用已有cb34环境/core不变。

启动余9,348,747,264B，512MiB预测+8GiB告警线通过，无清理。任务在监听预检前已经结束，**未创建本批监听**，不重启。鉴于幅度较小，下一项已落实为[一次原正式AB/BA](v6-ddp-pipeline-next-validation-20261031.md)，由原噪声规则决定是否接受，不合并本次样本、不换参数、不修到通过。
