# 反馈驱动的接受版本与大图重分组复核

这组结果来自真实 Agent 产物和 box4-2 GPU 运行。它不替代仍在进行的冻结策略对照 B；[原始证据清单](raw-evidence-manifest.json)保留文件路径、大小和 SHA-256。

| 实验 | 结果 |
|---|---|
| [纯函数项目反馈后候选](p6-new-signal-a/results.json) | 上一轮融合核降低了 GPU 诊断时长但收益不稳定。实际 Agent 读取资源响应后，生成融合核的 Graph replay 实现。完整输出两步均通过，两个配对收益均超过噪声界；父版本 0.0792845 ms，候选 0.0498955 ms，已提交。原生关联中确实出现 `_fused_gating_kernel`；自报路线与此一致。 |
| [纯函数保留 shape／seed](p6-new-signal-heldout-b/results.json) | 相同候选字节和参数在 96×1536 上输出完全一致，但执行路线为 `verbatim_cuda_graph`，不是融合核。0.0657135→0.0738005 ms，收益不稳定且中位数变差，拒绝。不能把回退通过描述为融合核泛化。 |
| [Qwen 独立 TTFT 复测](p4-qwen-ttft-a/results.json) | 完整 logits／tokens／连续请求通过；两对父版本 29.239424／25.529808 ms，候选 22.362461／22.376298 ms，均超过各自噪声界。随后依据已完成的评价执行 CAS 提交并重新发现，见[独立发布记录](publication-replay-ba7ab0299386.json)；原 exploration 结果保持不变。 |
| [Qwen 候选来源](p6-comparison-qwen-score_only-a/results.json) | 真实 Agent 在 A 组产生按 decoder layer 保留原数学计算、精简 Python 调用并对 prefill 使用 Graph 的候选，覆盖 prefill/decode 两阶段的 72 个站点。A 组整轮因框架变化中断，不计入正式比较；保留原候选字节，在上行固定实现下独立复测才构成新的证据。 |
| [Qwen 组合 A 失败](p6-qwen-composition-a/results.json) | 后续 Agent 选择替换原 decode bundle，并出现完整 logits 数值失败。它既没有通过组合质量，也不证明两个有效优化共存；新的组合实验继续验证。 |

主要独立配对使用的框架源码 SHA 为 `e1ee82a70ff617f6932a77a0f2211de3569fa5d973c685e860aa3b1fc0492ed0`。时间是固定 GPU 输入、既定完成边界内的模型／项目时间，不包含线上请求排队与网络。

## 重分组索引

Qwen 提交后的真实发现产生 21,116 个调用和 15,059 个组，前一注册表有 17,852 个组。原算法为每个新组扫描旧组、重复读取 bundle 并构建集合。索引实现一次建立调用→旧组、安装边界→旧组映射，保留相同的直接身份与包含关系语义，不据此推断独立因果贡献。

[复核回执](p6-qwen-lineage-index-validation.json)显示：全部 15,059 个组的归属与原实现完成的结果完全一致。索引构建及全部查询约 0.313 秒；原算法对分散抽取的 121 个组耗时约 2.113 秒。该测量反映协调器 CPU 效率，不能加入模型加速比。

另有两个局部检查通过：归属中的共享成员／退休边界／未知父调用，以及失败候选代码的独立回传。新反馈附件保留原候选代码、目标 metadata、bundle 身份和缺失原因；失败实现不再只能通过 blob ID 引用。附件接入后的真实 Agent 消费仍需后续调用回执，不以格式检查替代。
