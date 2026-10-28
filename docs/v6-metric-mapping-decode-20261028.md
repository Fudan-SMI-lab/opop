# 指标接口说明对照与 Qwen decode 目标变体

承接[次级指标终态审计](v6-secondary-metrics-20261028.md)。显式 request 采集接口已按已验证代码采用，默认空映射；生产 intake/schema/prompt 未采用自动指标选择。原型与采用版仅有两项兼容性调整：空映射不初始化 collector、portable request 不输出空的新字段。`prepare_variant` 允许同质量/源代码下更换诊断采集配置。26 项相关检查通过；新批首阶段另验证采用版真实双卡执行。

本批按固定顺序执行：

1. **生产运行时确认**：使用此前已独立接受的 DDP Graph 完整版本、原 adapter/参考/参数/step_ms 主目标，仅显式增加通信次级映射，完整两步质量、5 次 clean、3 次独立通信 profile。不得回填旧结果或把此端点作为新接受。
2. **真实 intake 单段说明 A/B**：以 `secondary-metrics-20261028` 的 treatment 初始快照 `generation-25cac779e47c` 为起点。两臂同用基于本次生产版本复制的 intake overlay；明确记录相对旧快照的共同核心变化。control 完全保留输入，treatment 只在 `inputs/intake.schema.json` 的 diagnostic_metrics 描述追加一段通用说明，区分 discovery 字段名、collector 标识、输出 alias，以及 provider 参数与框架采样次数配置。其他输入、prompt、模型、工具设置相同。不给前次答案或诊断，不自动修改 Agent 输出，不因成功/失败追加重复会话。
3. **实际采集设置集成**：分别抽取两臂真实生成的 communication、required_observations、profiling、diagnostic_metrics，在原已核实 DDP adapter/candidate 上执行 discovery 和 evaluate。此前的 mapping-only 集成漏过了 discovery 名称检查，这次显式经过同一 worker 检查；保留失败。只验证采集设置接口，不能称为新的 evaluator/原入口 cohort；完整生成语义仍需逐项审计。
4. **Qwen 同 workload decode 目标**：复用 `autonomous-cohort-20261027/qwen-ttft` 的 baseline adapter SHA `24ebefd7…`，不导入任何旧优化 bundle。batch=1、prompt=256、32 个完整新 token、BF16/SDPA/noTF32、质量/生命周期/同步边界不变。目标从 TTFT 改为 `decode_tokens_per_second`（31 个 first-token 之后的 decode token），保留 prefill 和完整生成；原任务无性能约束，本次不凭空增加。仅 goal/objective/device placement 及相应 notes 改变。通过现有 `--reuse-evaluator` 执行真实 contract-only 适配、新 controls、三轮自主搜索与反馈。
5. **独立检查与交叉诊断**：验证冻结 adapter 字节不变、完整源 logits/精确 tokens；若有优化版本，执行完整版本独立 AB/BA。用各自独立端点交叉计算 TTFT/decode，仅作为不同测量时期下的诊断，保留原接受/拒绝，不跨合同导入 bundle。

未新增 Agent 上下文大包或失败类型路由；schema 段落只补足已观察到的接口误解。若两臂仍混淆名称或未保留语义，保留结论后分析，不能将未执行字段解释为生效。production 的自动 intake 字段与说明是否采用仍待终态审计。

原 Qwen TTFT cohort 的 14.06% 独立接受，与更早固定 RMS 候选的独立拒绝/全部 80 个诊断样本分别保留。DDP Graph 40.61%、同 adapter 合批 55.73%、旧通信目标拒绝；TP 完整 32-token 失败、ResNet 独立拒绝及全部 80 样本、另一个 memory adapter 的 36.68% 接受/缺少 images allocation 事实均不改写。allocator 合并与之前 reviewer 建议不采用。

执行脚本 `scripts/experiments/v6_metric_mapping_decode_20261028.py`，目录 `/root/autodl-tmp/dev/v6-work/metric-mapping-decode-20261028`。GPU 阶段串行，仅一个 Qwen 任务；旧同类 Qwen 主目录约 6.8 GiB，当前实际容器剩约 37.5 GiB，够开展此批且不需要本地转存。模型资产共享、raw 留服务器，GitHub 只存代码/索引/小摘要。新长批启用一个普通完成监听；没有 Goal、额外送达测试或模型轮询。

启动：2026-10-28 00:48:48 UTC，冻结 `04d966cf8d245f97cc74033a2bbb9b4c904a771a`，controller PID418751，[launch](evidence/v6-metric-mapping-decode-20261028/launch.json)、[protocol](evidence/v6-metric-mapping-decode-20261028/protocol.json)留档。启动实际 df 可用 40,313,094,144 bytes（37.55 GiB），GPU 空闲。生产核心 SHA `e2b2860f…`；[intake overlay](evidence/v6-metric-mapping-decode-20261028/intake-overlay-manifest.json)只修改 intake 文件，Qwen 阶段使用生产核心。

生产运行时双卡 smoke 已完整通过，step 中位数 70.1717 ms、次级通信 sum 33.3534 ms；这仅是接口执行检查，不替代旧正式接受。两步质量/覆盖、全部样本及逐 trace 重算见 [smoke 审计](evidence/v6-metric-mapping-decode-20261028/production-smoke-audit.json)。新 control intake 会话 `ses_e80046f18ffeeOu89iXmAtWG3s` 已建立；Qwen 搜索尚在后续队列。唯一普通监听 PID39216、状态 waiting，controller 存活，[设置证据](evidence/v6-metric-mapping-decode-20261028/wakeup-setup.json)保存；无恢复旧归档监听、无额外送达测试。
