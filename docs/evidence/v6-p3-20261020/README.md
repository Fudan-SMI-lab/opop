# P3：自动发现、联合候选与实测拒绝

本轮从 P2 已验证的 ResNet 项目版本继续，未提供人工目标组或替换站点。[发现摘要](discovery-summary.json)记录 270 个调用点、204 个组及 81/81 CUDA kernel 关联；同时采集 module 与 functional 调用。数据依赖、包含关系与缺失的 alias/state/stream 信息分别记录，不把观测到的部分图当作任意重排的许可。

真实 GLM-5.3 自主选择卷积／BatchNorm 联合范围，生成 20 处完整复合替换。[候选代码](generated_candidate.py)、[选择](selection.json)、[方案](proposal.json)和[完整运行回执](trial-result.json)均保留。候选中的模型细节属于 Agent 生成产物，通用协调器不按模型名或固定算子名选组。

首次输出将 `implementation` 枚举误写成长段说明，框架在执行前拒绝；[原始错误](first-contract-failure.json)保留。接入 JSON schema、输出校验和原会话真实修复后，Agent 修正了元数据，候选 Python 文件哈希保持不变。修复请求没有调用额度或费用截止。会话成本按消息身份去重；修复后同一会话的累计成本不能再与修复前快照相加。

GPU 实测结果：

| 检查 | 结果 |
|---|---|
| 完整输出与连续两次调用 | 每次检查 1,000 logits 和 top-1；最大绝对误差 `6.914e-6`，通过 |
| 替换覆盖 | 20 个调用点各执行两次，框架 guard 无 fallback |
| AB 对，父版本／候选 | `1.329141 / 1.187234 ms` |
| BA 对，父版本／候选 | `1.342907 / 1.414494 ms` |
| 接受决定 | 拒绝；第二对退化，未满足每对收益高于噪声的条件；父版本保持不变 |
| 新观测图 | 151 个调用点、125 个组；81/81 CUDA kernel 关联 |
| CUDA 诊断总时长 | `731.385 → 734.659 µs`；不构成设备时间改善 |
| PyTorch allocated 峰值 | `86,361,088 → 132,884,480 bytes` |

上述短测证明完整自动通道、多组共同执行、反馈采集及拒绝行为，不作为稳定性能提升结论。框架已输出逐组／调用的前后资源信息，联合效果共享同一个 bundle 身份；未测的单组独立贡献保持未知。P4 继续验证真实下一轮消费这些反馈、旧内循环适配、多个 bundle 共存和执行计划改写；P5/P6 尚未完成。

复现入口为 `optimize-project --resume --request <原请求> --output <原运行> --scope joint --rounds 1`；重放已生成候选可用 `ProjectSearch.bundle_from_outputs` 与 `validate_bundle`，无需重复请求 provider。原始目录 `/root/autodl-tmp/dev/v6-work/p2-resnet-a` 保存逐 rank 日志、参考张量、所有原始 trace 和会话产物；公开摘要提供原始 trace/discovery 哈希。
