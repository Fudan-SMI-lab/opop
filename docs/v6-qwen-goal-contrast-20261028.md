# Qwen固定版本的同核心目标交叉测量

承接[反馈执行审计](v6-feedback-execution-validation-20261028.md)与方案E。已有TTFT与decode候选分别由真实自主搜索产生并通过独立检查，但旧交叉分数来自不同时间、不同核心版本和各自基线。本批只补足测量可比性，不重复提示对照，不生成新候选，不改变历史接受/拒绝。

目录`/root/autodl-tmp/dev/v6-work/qwen-goal-contrast-20261028`，控制器`scripts/experiments/v6_qwen_goal_contrast_20261028.py`。生产核心保持`ed24af7a…`。T为旧原入口TTFT接受版本59688cbb，D为同adapter decode接受版本6ca1faf2；BT/BD为各自原baseline，分别保留原合同与request，只在执行内存配置中统一device 0。禁止跨合同导入bundle，分支current及原始指针不发布/改写。

启动前已逐项比较：两baseline的五个文件SHA完全相同，adapter为24ebefd7…；request只有goal/objective/notes/devices不同，workload、quality、通信/测量配置一致。两合同的completion、state_protocol、workload、quality、world_size等逐字相等；invariants的目标相关表述经代码审阅保持完整32-token工作、两同步边界、BF16/SDPA/noTF32、全logits容差与exact tokens。两套只读参考的4个tensor、9,723,968个元素CPU逐值完全相同。不同目标合同仍各自保留，不因语义对齐就视为可互换合同。

固定串行顺序为 **BT/T/D/BD/BD/D/T/BT**，每个端点保留原warmup=2、repeats=5及完整质量/覆盖检查，共40个clean样本。每个样本同时记录TTFT、31个post-first-token的decode吞吐、完整生成时间及生命周期显存。没有按结果补跑、剔除异常值、修改参数或GC。全样本成功后仅做T/D、BT/T、BD/D两个目标上的配对诊断计算；任何端点失败均保留，不拼接成接受。不同目标原始生成过程并非同快照随机对照，即使候选排名相反，也不能把全部差异唯一归因于目标文本。

本批不做新的正式候选接受，也没有新Agent生成需要重复调用；GPU执行使用真实框架worker。机制沿用已审阅的冻结代码，记录实际quality coverage/routes；旧Nsight证明不能自动充当当前执行路径证明。若当前发生fallback，先根据本批证据报告，不能只引用旧Graph数量。没有新大型trace或controls生成，现有参考共享只读；估计新增以小型结果/日志为主，实际df由单普通完成/8GiB告警监听检查。启动前约10.65GiB，不额外转存或清理原始证据。

终态需检查八端点所有样本、actual core/adapter/bundle SHA、完整quality/coverage、原生内存生命周期、当前route、两baseline可比性、对称顺序与聚合，再判断目标偏好是否在本时段得到支持。保留旧固定Qwen拒绝及80样本，TTFT14.06%、decode88.8847%也不被本诊断覆盖；所有FSDP/DDP/ResNet/TP历史结论保持。
