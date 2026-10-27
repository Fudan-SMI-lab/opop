# 清理后恢复 DDP / TP 最小诊断

本批继续[阶段计划](plan-v6-next-stage-20261027.md)，承接[DDP / TP 终态审计](v6-ddp-semantics-20261027.md)。服务器清理已完成，启动准备时数据盘可用 53,383,962,624 bytes，约 49.72 GiB；双卡没有计算进程。用户已要求恢复推进，但继续禁止大体积转存本地。

## 问题与固定协议

DDP 当前候选 `trial-25f69b0d87-param-a8bd4641` 在搜索时接受、完整独立复测拒绝。通信指标下降，同时整步端点仍越过原上限，且一对收益未超过噪声。不能根据 NCCL kernel sum 推断 wire time，也尚未用直接的 CPU 等待证据确定 Gloo 作用。

- B：同一合同下无候选的父版本。
- C：固定完整候选 `75e28d3d2ccd5fab06175ef1b827a325ed2159d011758c7b2deccda6173b9e2d`，原参数 `pre_backward_align=none, per_collective_align=barrier`。
- N：完全相同候选代码，仅将已有参数 `per_collective_align` 设为 `none`。明确属于开发参数消融，不是原接受版本。

干净端点顺序固定为 **B/C/N/N/C/B**，每端点 20 个整步计时样本，保留全部样本和逐 rank 值。仅为本次诊断扩大原 5 次采样；原工作量、源数值/精确副本检查、两步连续质量检查、两次 warmup、目标及约束保持不变。通信目标仍在单独 profile 中采集，不与干净整步时间拼成同一次执行。每个端点保留执行前后 GPU、CPU 容器状态；不按得分追加端点或改变正式接受记录。

干净端点之后，对质量和执行通过的 B/C/N 分别使用现有 `execution_timeline` 做两次独立采集，关闭非必要 gradient callback。保留原 trace、时钟锚点、跨 rank 匹配、CPU c10d/Gloo ranges 和 CUDA 同步信息。CPU ranges 可能嵌套或重叠，汇总不是可相加的总成本；空事件不能证明没有等待；profiler 时间不能替代干净性能或直接解释因果关键路径。若当前采集能力不足，应据实记录缺口，再决定最小增补，而非现在修改生产 collector。

TP 使用先前保存的第一层输入/输出，重新加载官方 TP=2 BF16/SDPA 模块，核对 Q/K/V 实际本地权重与 checkpoint 连续行分片、实际输入与单卡输入均精确一致后，才进行相同操作数比较。比较官方模块、实际 local weight BF16、连续布局、checkpoint slice、完整矩阵计算后切片及单独 FP32 算术对照；K/V 是伴随对照，保存输出和哈希。布局/输入映射未成立则报告未解决，不默认错权重。本次不运行完整模型计分，不改变源精度或容差；之前完整 32-token logits max error=3.0 的质量失败继续有效。

GPU 阶段之后，使用现有真实 provider 和 `ProjectAgents` 发起一个**开发诊断审阅角色**：输入有界摘要、实际 candidate/adapter/collector 代码和原始证据路径；只在有具体不确定性时读取对应原始文件。保存生成前快照、会话、回答及费用统计；不设 API/费用限制。它提出证据、限制和下一项可证伪最小实验，不直接修改生产提示或发布候选。这不等同于新的原入口自主优化成功。

## 执行与保留要求

脚本为 `scripts/experiments/v6_ddp_tp_diagnostics_20261028.py` 与 `v6_tp_colwise_diagnostic_20261028.py`。只增加实验脚本与文档，生产框架不变。工作目录 `/root/autodl-tmp/dev/v6-work/ddp-tp-diagnostics-20261028`，Git 源码单独冻结，GPU 工作串行租用双卡；旧 artifact/reference 只读复用，不复制多 GB 参考。

旧 DDP 独立拒绝、TP 完整质量失败、旧 Qwen 固定候选拒绝及全部 80 样本、ResNet 显存候选独立拒绝及全部 80 样本、不同 adapter 的 36.68% 接受与 missing images 事实全部保留。本批不启用 Goal，不触发额外投递测试，不靠模型轮询；启动后挂一个普通完成监听，终态包含执行错误也会自动唤起审计。

启动前核对依赖、生成请求、唯一参数差异、来源哈希和脚本编译；启动/监听记录另附。大体积证据全部留在服务器，GitHub 仅存代码、协议、摘要和索引。
