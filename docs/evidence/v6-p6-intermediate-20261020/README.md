# P6 中间证据：目标差异、交叉复测和真实失败

本目录不是 P6 完成声明。原始 run、trial、候选、会话身份、参数筛选和 AB/BA 结果均随文件保存；`manifest.json` 校验导出的原始字节。GPU 为 box4-2 的两张 RTX 4090；模型调用为实际配置的 `zhipuai/glm-5.3`。

| 运行 | 实际结果 |
|---|---|
| ResNet B32 吞吐量 | 自动区域候选通过全部 32,032 个输出元素，两步最大绝对误差 6.79493e-6；AB/BA 中位吞吐量 7607.166→8576.284 images/s，约 +12.74%，接受 |
| ResNet B32 首轮显存 | 峰值 allocated 282,103,808→257,500,672 bytes，但吞吐量 6407.175 images/s 低于原始基线的 95% 约束；参数筛选拒绝，继续真实反馈迭代 |
| ResNet B1 已接受 Graph 后的下一轮 | 新候选 0.970322 ms，父版本 0.771454 ms；两对均变慢，拒绝并保留父版本 |
| DDP 首轮通信 | TaskSearch/TPE 实际测量 `single` 和 `split_small_large` 两个参数端点；均因返回的 c10d Work 缺少 capsule 而失败；rank 退出并回收；资源响应明确为 invalid_endpoint/unknown，未伪造数值斜率 |
| 新纯函数项目 | 无预设组，自动生成 evaluator 并发现 16 个调用/组；候选 graph 参数通过一次筛选，独立复测遭其他 CUDA 进程干扰，整次结果拒绝；保留两次受影响运行，待空闲设备重测 |
| ResNet 吞吐量候选跨显存目标 | 冻结代码/参数，在新契约下复测：allocated 282,103,808→93,936,640 bytes，吞吐量约 8563 images/s，满足原指标与约束；仅作 exploration，没有发布到主版本 |

吞吐量候选的真实 trace 含 60 个 CUDA kernel，其中 9 个 `bn_relu_kernel` 和 8 个 `bn_add_relu_kernel`；表明实际执行了生成的融合实现。构建、融合和 Graph capture 均在候选内，原始代码不作人工算法修补。

## 不能省略的测量边界

跨目标结果不代表物理显存减少。该 Graph 候选的 allocator reserved 从 396,361,728 增至 708,837,376 bytes；安装阶段 allocated 峰值为 299,457,536 bytes。warm-run allocated 计数器没有反映保留 Graph pool 的全部代价。后续实现额外提供 `framework.lifecycle_peak_allocated_bytes` 和 `framework.lifecycle_peak_reserved_bytes`，包含模型准备、候选安装及后续调用，并保留原有指标供明确指定口径的实验使用。它们仍是 PyTorch allocator 指标，不能替代驱动侧物理显存测量。

早期 workload 变体正确执行了 B32，但复制的自然语言 invariants 仍提及 B1。这是契约元数据缺陷，不应作为严格冻结契约的完整证据。现在 workload 改变时由真实 `variant_contract` Agent 只更新契约文字，评测代码保持原始哈希，再运行全部控制。`corrected-variant-contract.json` 和 prepared receipt 保存修正后的真实交叉复测，未回写或掩盖旧运行。

新项目失败记录中的 CUDA 进程占用监测发现 `/tmp/opencode` 的非框架进程。已向仍在运行的真实 Agent 发送调度说明，并加强候选提示；无 API 使用限额。worker 等待先前外部 CUDA 客户端结束，在执行过程中每 0.5 秒采样并记录后来出现的外部客户端；发现干扰则丢弃结果。同一 worker 的编译/helper 子进程按进程祖先归属识别。短于采样间隔的干扰仍可能漏检，不能把 sampled_clear 宣称为严格隔离。

DDP Agent 首轮把重插桩 trace 的约 263 ms 当成目标并推断“带宽已到下限”“暴露关键路径”。这些是未经验证且部分不成立的解释：当前独立通信目标样本约 64–70 ms，kernel duration 包含等待，未测 wire time 或因果关键路径。原始 proposal 保留；新一轮明确提供原始目标、plain profiler 开销和各测量域，不能用 Agent 的文字替代采集证据。

## 当前实现和剩余工作

新增冻结候选 replay：相同 evaluator/固定语义、相同 call 身份、当前 signatures，原代码和参数字节保持不变；无法映射的站点明确失败。CLI 使用 `--replay-from RUN --replay-bundle ID`，默认只评测；`--publish-replay` 仍须通过独立 AB/BA 才发布。

多 bundle 组合、隔离探索分支、三项目全部目标、保留 workload、等协议单组/联合和反馈消融仍在进行，不能以本目录的几个成功样本替代完整验收。旧单算子入口保持原路径，不运行无关完整历史套件。
