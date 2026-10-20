# 实际资源响应、通信诊断与生成修复

环境为 box4-2 双 RTX 4090。本目录保存主开发路径的新增验证，不与冻结 `7936d90` 的正式策略对照混合。原始文件位置、大小和 SHA-256 见 [清单](raw-evidence-manifest.json)；候选来自真实 `zhipuai/glm-5.3` 会话。

## 可确认的结果

| 验证 | 实测结果与解释 |
|---|---|
| [PCIe／设备采样](device-telemetry-validation.json) | 两个 rank 各完成三个独立诊断窗口。NVML PCIe TX/RX、GPU／memory activity、device-used memory 均有原生读数。4090 的 NVLink 查询返回不支持。速率是设备级约 20 ms 窗口，记录每次读取起止时间；不能转写成某个算子组的 wire bytes。 |
| [真实 Collector Agent](telemetry-agent-plan.json) | 实际选择 `torch/nsys/device_telemetry`，请求 PCIe TX/RX 和设备活动比例，并保留缺测原因。Agent 的文字解释不是任务契约；其提及的通信目标／约束不能覆盖当前请求。GPU activity 也不能解释为 SM occupancy。 |
| [独立 allreduce 参考校准](collective-calibration-validation.json) | 从实际 trace 提取 28,351,488 B 和 157,541,376 B 两种 FP32 SUM payload。双 rank 全元素检查均通过；各 warmup 2 次、测量 5 次。rank 0 中位完成时间约 3.60／19.55 ms，rank 1 约 3.68／19.73 ms。使用独立 Gloo 控制组，NCCL Work wait 与设备同步在计时内。它是已加载项目环境中的连续缓冲区参考实验，不是 wire bandwidth、模型关键路径或性能下界。 |
| [参数资源响应](p6-new-signal-a/results.json) | 实际 Triton 融合核运行并通过完整输出检查。`num_warps` 8→16 时诊断 kernel sum 减少 0.576 μs、寄存器／线程减少 37、共享内存增加 32 B；16→4 时分别增加 3.648 μs、增加 115、减少 48 B。数据来自替换范围与原生 launch correlation，目标 J 和 workload 常量不再自动标成资源。 |
| 资源改善后的独立接受检查 | `trial-resource-audit-c9157758` 的父版本／候选中位延迟为 0.076228／0.0654985 ms，但第一对收益 0.005170 ms 未超过 0.006278 ms 噪声界，仍拒绝。参数筛选差分不是接受证据，也不是独立因果敏感度。 |
| [下一轮反馈消费](resource-feedback-consumption.json) | 真实 Selection Agent 读取上述资源响应和配对失败，转向测试融合核的 Graph replay。反馈文件哈希和实际选择均保存；将 Python 启动开销视为原因仍属于 Agent 的待验证假设。 |
| [Qwen 路径核查](qwen-fallback-path.json) | 原 decode 候选 62 次替换调用均调用了父实现，完整输出通过。该候选仅证明安装和回退兼容，不能作为融合计算生效或优化泛化证据。 |
| [约束表达式真实修复](p6-comparison-ddp-leaf-a/results.json) | 原 Agent 输出了旧表达式语言不支持的 `implies`；同一会话修复后，三个参数点均完成双 rank 全梯度／参数检查，但配对步时间 146.88→240.51 ms，拒绝。此项是开发修复验证，不补写为冻结正式对照成功。 |

首次资源诊断 `trial-resource-audit-51b80d5a` 暴露 Kineto 对带原始 JSON 引号的范围名导出无效 trace 的问题。该失败保留；框架改用十六进制元数据范围名，新的独立诊断成功，不修改旧 trace。新归属独立记录在 `bundle_resources`，不会把重分组前已退役的 call ID 重新塞入当前执行计划。

另保存 Qwen TTFT／吞吐、DDP 整步、跨目标与保留 workload 的新增负结果。它们包含数值失败、明确变慢和收益未超过噪声；无收益不改变质量标准。[测试回执](tests.txt)为本阶段受影响行为的 **36 passed**，不与前面阶段重复数量相加。

## 使用边界

`context['references']` 计数只能观察通过该接口调用的父实现；其他别名调用不能据此排除。候选 `record_route` 是自报信息，需实际 trace 佐证。嵌套替换范围包括其子工作和回退，延迟 backward 不会自动归属到已结束的 forward 范围。

当前设备采样每个 worker 记录其绑定 UUID；本次双进程 DDP 覆盖两卡。单进程控制多卡时，未采样设备在 coverage 中明确列出，不能宣称覆盖全部设备。NVLink／RDMA 实机效果仍未验收。

[会话用量快照](usage-snapshot.json)是观测到的 provider 报告统计，包含失败调用；不是账单或开发用量上限，且不包含快照之后的调用。
