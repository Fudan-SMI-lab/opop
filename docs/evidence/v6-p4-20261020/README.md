# P4 中间检查点：真实整流程候选与反馈迭代

本检查点验证 ResNet-18 固定 B=1 FP32 推理负载的执行计划候选，不代表 P4–P6 全部完成。父版本来自 P3；候选由 box4-2 上实际配置的 `zhipuai/glm-5.3` 生成，选择和实现两次请求都读取了 P3 联合候选的拒绝原因和逐组资源反馈。未设置开发 API 用量额度。

## 实际路径与结果

- 选择会话 `ses_ea9e83652ffeaPoGijeAj14H67`，实现会话 `ses_ea9df1260ffemN5Gpgohbpkjlv`；会话报告成本分别为 $0.74564288 和 $0.60865264，仅作统计。
- `trial-c11a1129f8` 生成完整模型 CUDA Graph 策略，实际替换项目入口和模型入口。安装时预热、捕获及额外输入自检；每次调用仍执行输入复制、graph replay 和输出 clone。计时包含这些逐次成本，安装成本属于独立 setup。
- 新的项目适配层调用既有 `TaskSearch` / `TPETuner`。本候选的参数空间为空，因此只有一次筛选实验；这不能证明参数选择或相邻参数资源响应有效。
- 两个连续质量步骤均比较完整 1000 logits 和 top-1，共 1001 元素，最大绝对误差为 0。两个声明站点均实际调用，完整 bundle hash 一致。
- AB 顺序：父版本 1.382076 ms，候选 0.755926 ms；BA 顺序：父版本 1.593735 ms，候选 0.755423 ms。增益 0.626150 / 0.838312 ms，超过相应噪声界 0.013246 / 0.034046 ms，因此原子提交。
- 接受的候选 revision 为 `1501faec68c2bd33bd183b1b7fd89be1c4b132a3ee8c728169a370d3cd0d0b77`。重新发现后的 metadata revision 为 `7c7dae44166789072045916c08e99a106add9815555ccb854a97cc50930efc5f`，保留相同 bundle。
- 接受版本重新采集得到 8 个调用站点、7 个组；81/81 个 CUDA kernel 可关联，81 个均有 graph node 信息。旧的 268 个调用、199 个组退役，旧 trace 失效。图内部的独立组效果仍未知，不能把整体收益分摊给旧组。

## 证据

- `trial-result.json`：完整选择、实现、TaskSearch、质量、AB/BA、逐组反馈及提交决策。
- `selection.json` / `proposal.json` / `generated_candidate.py`：原始生成产物。
- `discovery-summary.json` / `accepted-discovery.json`：候选及接受后重新发现摘要；原始 trace 留在 `/root/autodl-tmp/dev/v6-work/p2-resnet-a/jobs/`。
- `tests.log`：实际执行选择，涵盖上下文不同但形状相同的 functional 站点、事务、执行器、完整输出质量、配对决策及旧 TaskSearch 评价接口。

## 本检查点的实现边界

同时落地了准备失败的真实请求修复、未验证 reference 不发布、按依赖邻居与 lineage 路由反馈、安装成本记录，以及可选通信/Nsys 的接口基础。新增通信路径尚待真实 DDP 闭环验证，不能用接口存在代替 P5 完成。

后续还需验证多 bundle 共存、更多反馈轮次、参数资源响应、Qwen/DDP 的完整自动优化与目标切换。单个固定 ResNet 负载上的结果不作为跨模型、跨形状泛化结论。完整显存占用也不能由 PyTorch allocated 指标代替。
