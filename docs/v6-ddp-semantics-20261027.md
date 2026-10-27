# 采用 intake 语义说明后的原入口验证

依据[上轮真实审计](v6-semantics-lifetime-20261027.md)，生产仅采纳经过 A/B 的 intake v2 一段通用提示。生命周期提示不变。目录 `/root/autodl-tmp/dev/v6-work/ddp-semantics-20261027`，冻结提交与 SHA 在启动记录保存。

## 固定协议

1. 新 DDP 原入口完整三轮：原项目、原命令、原通信目标及整步约束，不给旧 adapter、分组、候选或人工诊断。真实 Agent 自主 intake、evaluator、发现/采集、选择、生成、回装、评价和反馈；API 用量只统计、不限额。三轮是该批协议，不是会话预算。
2. 源 serial/global-batch 全量梯度/参数、精确 rank 关系负例独立检查；若生成 observation 名称/容器不同，记录需人工审查，不将名称不同自动解释为数值失败。有接受版本则仅一次完整独立 AB/BA；保留搜索接受与独立拒绝的区别。审计说明是否真正允许等价通信策略，且没有省略工作量、质量或聚合语义。
3. TP 诊断串行在其后：原 BF16/SDPA 单次 prefill，单卡与官方双卡 hook 第一 block 内的归一化、Q/K/V、attention 输出投影、MLP 等输入/输出；hook 不改模型返回值。保留实际 shape/DTensor 元信息，区分直接比较与连续 rank 切片假设，不把未匹配形状当错误。
4. 对保存的**相同单卡输入与 checkpoint 权重**，另做两个 row projection 的 full BF16、分半 BF16 后求和、分半 FP32 后求和/转回等局部算术对照。只隔离舍入/分块影响，不声称复现 NCCL 执行，不把 FP32 用到原模型或改变质量门槛。此前 TP 完整32-token logits 失败继续有效；本批不是完整 TP 验收，更不是自主通信优化。

当前 df 可用约10.13 GiB，先前同类 DDP 占约6.2 GiB、独立目录约376 MiB。只开这一个项目，TP 保留第一 block 小型张量、不做大 trace。df 与 statvfs 在此容器不同，脚本分别记录，按 df 判断额度；不删除旧证据。运行代码从 Git archive 单独冻结，使用既有私有配置/依赖 overlay；两个 GPU 作业共享租约串行。

ResNet 同 adapter 显存候选的独立吞吐拒绝、其固定80样本、旧 Qwen 固定候选拒绝及全部80样本、旧 memory adapter 36.68%接受与缺失 images 事实，均保留原结果。本批不继续重测 ResNet 直到通过。

## 验证与自动续行

启动前做脚本编译、实际环境导入、配置与旧 TP 请求结构核查。已有 intake 6项检查通过，采用段落与实际 treatment 逐字一致。启动后记录真实会话/快照和单个普通完成守候。守候仅检查文件/进程，结束或控制器异常退出后 queue 一次；不启用 Goal、额外投递测试或模型轮询。

脚本：`v6_ddp_semantics_20261027.py`、`v6_tp_block_audit_20261027.py`。初始阶段/生成失败也应记录，不把 controller completed 当自主优化成功。终态后审计实际生成代码、inputs、controls、负例、完整版本、TP 输出，再决定下一步最小改动。
