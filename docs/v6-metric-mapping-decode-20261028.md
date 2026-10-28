# 指标接口说明对照与 Qwen decode 目标变体

承接[次级指标终态审计](v6-secondary-metrics-20261028.md)。显式 request 采集接口已按已验证代码采用，默认空映射；生产 intake/schema/prompt 未采用自动指标选择。原型与采用版仅有两项兼容性调整：空映射不初始化 collector、portable request 不输出空的新字段。`prepare_variant` 允许同质量/源代码下更换诊断采集配置。26 项相关检查通过；新批首阶段另验证采用版真实双卡执行。

本批按固定顺序执行：

1. **生产运行时确认**：使用此前已独立接受的 DDP Graph 完整版本、原 adapter/参考/参数/step_ms 主目标，仅显式增加通信次级映射，完整两步质量、5 次 clean、3 次独立通信 profile。不得回填旧结果或把此端点作为新接受。
2. **真实 intake 单段说明 A/B**：以 `secondary-metrics-20261028` 的 treatment 初始快照 `generation-25cac779e47c` 为起点。两臂同用基于本次生产版本复制的 intake overlay；明确记录相对旧快照的共同核心变化。control 完全保留输入，treatment 只在 `inputs/intake.schema.json` 的 diagnostic_metrics 描述追加一段通用说明，区分 discovery 字段名、collector 标识、输出 alias，以及 provider 参数与框架采样次数配置。其他输入、prompt、模型、工具设置相同。不给前次答案或诊断，不自动修改 Agent 输出，不因成功/失败追加重复会话。
3. **实际采集设置集成**：分别抽取两臂真实生成的 communication、required_observations、profiling、diagnostic_metrics，在原已核实 DDP adapter/candidate 上执行 discovery 和 evaluate。此前的 mapping-only 集成漏过了 discovery 名称检查，这次进入实际 profile worker；是否执行 required 模式检查须核对生成策略。只验证采集设置接口，不能称为新的 evaluator/原入口 cohort；完整生成语义仍需逐项审计。
4. **Qwen 同 workload decode 目标**：复用 `autonomous-cohort-20261027/qwen-ttft` 的 baseline adapter SHA `24ebefd7…`，不导入任何旧优化 bundle。batch=1、prompt=256、32 个完整新 token、BF16/SDPA/noTF32、质量/生命周期/同步边界不变。目标从 TTFT 改为 `decode_tokens_per_second`（31 个 first-token 之后的 decode token），保留 prefill 和完整生成；原任务无性能约束，本次不凭空增加。仅 goal/objective/device placement 及相应 notes 改变。通过现有 `--reuse-evaluator` 执行真实 contract-only 适配、新 controls、三轮自主搜索与反馈。
5. **独立检查与交叉诊断**：验证冻结 adapter 字节不变、完整源 logits/精确 tokens；若有优化版本，执行完整版本独立 AB/BA。用各自独立端点交叉计算 TTFT/decode，仅作为不同测量时期下的诊断，保留原接受/拒绝，不跨合同导入 bundle。

未新增 Agent 上下文大包或失败类型路由；schema 段落只补足已观察到的接口误解。若两臂仍混淆名称或未保留语义，保留结论后分析，不能将未执行字段解释为生效。production 的自动 intake 字段与说明是否采用仍待终态审计。

原 Qwen TTFT cohort 的 14.06% 独立接受，与更早固定 RMS 候选的独立拒绝/全部 80 个诊断样本分别保留。DDP Graph 40.61%、同 adapter 合批 55.73%、旧通信目标拒绝；TP 完整 32-token 失败、ResNet 独立拒绝及全部 80 样本、另一个 memory adapter 的 36.68% 接受/缺少 images allocation 事实均不改写。allocator 合并与之前 reviewer 建议不采用。

执行脚本 `scripts/experiments/v6_metric_mapping_decode_20261028.py`，目录 `/root/autodl-tmp/dev/v6-work/metric-mapping-decode-20261028`。GPU 阶段串行，仅一个 Qwen 任务；旧同类 Qwen 主目录约 6.8 GiB，当前实际容器剩约 37.5 GiB，够开展此批且不需要本地转存。模型资产共享、raw 留服务器，GitHub 只存代码/索引/小摘要。新长批启用一个普通完成监听；没有 Goal、额外送达测试或模型轮询。

启动：2026-10-28 00:48:48 UTC，冻结 `04d966cf8d245f97cc74033a2bbb9b4c904a771a`，controller PID418751，[launch](evidence/v6-metric-mapping-decode-20261028/launch.json)、[protocol](evidence/v6-metric-mapping-decode-20261028/protocol.json)留档。启动实际 df 可用 40,313,094,144 bytes（37.55 GiB），GPU 空闲。生产核心 SHA `e2b2860f…`；[intake overlay](evidence/v6-metric-mapping-decode-20261028/intake-overlay-manifest.json)只修改 intake 文件，Qwen 阶段使用生产核心。

生产运行时双卡 smoke 已完整通过，step 中位数 70.1717 ms、次级通信 sum 33.3534 ms；这仅是接口执行检查，不替代旧正式接受。两步质量/覆盖、全部样本及逐 trace 重算见 [smoke 审计](evidence/v6-metric-mapping-decode-20261028/production-smoke-audit.json)。新 control intake 会话 `ses_e80046f18ffeeOu89iXmAtWG3s` 已建立；Qwen 搜索尚在后续队列。唯一普通监听 PID39216、状态 waiting，controller 存活，[设置证据](evidence/v6-metric-mapping-decode-20261028/wakeup-setup.json)保存；无恢复旧归档监听、无额外送达测试。

## 终态审计（2026-10-28）

03:28:22 UTC 六个阶段均执行完成，普通监听实际投递本轮。依据原协议、初始快照、生成代码、private-runtime 真实工具记录、每端点完整样本、原生 SQLite 和独立结果进行复核；[终态审计索引](evidence/v6-metric-mapping-decode-20261028/terminal-audit.json)包含路径、SHA、全部 clean 数值和质量/覆盖结果，raw 留服务器。审计脚本为 `scripts/experiments/v6_audit_mapping_decode_20261028.py`。

### 指标接口

两臂初始输入只改 schema 的一段 description，10,621→11,345 bytes；其他文件、prompt、model/tool 设置与同臂共同 core 相同。control 的输入也与旧 `generation-25cac779e47c` 逐字相同，但执行核心相对该旧快照共同升级，已单独记录。两个新会话均 ready、零 contract repair，均选正确 discovery 字段 `communication_kernel_sum_ms`，没有生成前次被忽略的嵌套 `source.profiling`；3 次 profile 来自部署默认值。两臂均保留完整 FP32/noTF32 工作、原始 1.05 step 约束、max-rank 后 median、全量质量和精确 rank 关系；treatment 的关系要求部分在实际传递的 notes 中。

两臂都选 `communication=auto`，因此 **worker 的 required 模式强制检查没有执行**。这修正了启动协议中“经过 worker 即已检查”的过强说法。审计另外从实际 discovery 读取 observed 字段集合，两 rank 均包含所请求的正确名称；并核对 trace SHA。四个 profile/evaluate 集成全部有效，所有 5 个 clean 样本、两步质量、每 rank 3 个独立通信样本保留，逐 raw trace 重算和先 max-rank 后 median 一致。这里仅导入采集设置、复用旧已核实 adapter/candidate，不能推断新的 evaluator 保留了全部语义。

两臂都正确，不能将改善归功于新增段落或声称成功率提高。采用可选 intake 字段及经过对照的原段落，是为了让原入口能够表达已经验证的运行时接口并准确说明其边界；不是将该 A/B 解释成 superiority 证据。不修改生产 prompt、采样默认值、接受规则或 Agent 答案，不做名称自动修补。旧 unsupported discovery 名称和无效嵌套选项仍保留。

### Qwen decode：同 adapter 目标变体

真实 variant_contract 会话 `ses_e7ffb8b02ffe4TLZtvO5G87zP5`、快照 `generation-77a05769ef45`，零修复。新 baseline 无优化 bundle，adapter SHA 仍 `24ebefd7…`。工作量始终为 batch 1、prompt 256、完整 32 新 token；decode 指标只计算其中后 31 步，prefill、两次完成同步、BF16/SDPA/noTF32、完整 logits 容差和精确 token 均保留。不是新原入口 evaluator cohort。

| 自主轮次 | 实际结果 | 路径核查 |
|---|---|---|
| `20e10965a0` | 首个质量筛查失败，无有效参数候选 | tokens 精确但 logits 最大误差 1.0，拒绝；Graph 自报事件不能抵消质量失败，具体数值根因未独立定位 |
| `29f54aad41` | 全量质量通过，配对收益未超过噪声 | 两次完整调用中仅 2 个 decode 根调用进入新路径，60 个 fallback；返回自定义 cache 与原 DynamicCache 绑定签名不同，未成为完整 decode 替换 |
| `426d21c6f7` | 搜索接受，随后独立完整版本接受 | 两次调用共 62 个 decode 根调用、无 fallback；实际返回 DynamicCache，保留输出 logits 所有权；独立 nsys 有 31 次 `cudaGraphLaunch`、66,154 个 graph-node kernel，共 68,404 kernel |

第二轮真实工具记录读取首轮 rejection、screening error；第三轮读取上一候选代码及 quality/coverage、binding、measurement 等字段，实际看到 2 次命中/60 次 fallback。反馈文件约 5.54/35.89 MB，但 agent 使用程序提取字段，不能按文件大小声称整份进入上下文。第三轮候选直接使用当前模型权重/输入，逐层执行投影、RMSNorm、RoPE、SDPA、MLP，并按 decode 长度捕获/复用 Graph；prefill 保持原路径。候选范围声明覆盖 244 组/23,226 次调用，但绑定的是一个 decode 根边界，不是分别改写 244 个 kernel。候选关于“半数 clean 时间是 host gap”的定量解释未获无扰动因果分解支持，不作为结论。

完整版本独立 AB/BA 的 B 端点中位数 42.9785、47.8671 tokens/s，C 为 85.8020、85.7916；配对判据、完整质量与覆盖均通过：**45.4228→85.7968 tokens/s，+88.8847%**。全部四端点各 5 样本保留，不剔除慢样本。原 source 参考 SHA `af929f42…`，baseline 对独立源参考两步各 4,861,984 元素零误差；candidate 对 baseline 完整观察量每步 4,861,988 元素零误差，token 精确。新最终 bundle `trial-426d21c6f7-param-fdea28d5`，只有这一 bundle；本批只在 baseline 后发现一次，最后一轮接受后没有下一轮重发现/组合，不能将其他批次能力冒充本批证据。

交叉诊断：旧 TTFT 候选原 TTFT 接受仍为 14.06%，但其 decode 配对判据不通过；新 decode 候选原目标通过，TTFT 的两个配对方向不一致、判据不通过。两批时间、baseline anchor、核心版本不同，因此这支持各自目标下的效果差异，不能当作同时间同核心的候选横向排名。更早固定 Qwen 候选的独立拒绝及 80 样本不变。

审计时实际容器 df 可用 35,861,544,960 bytes（约 33.40 GiB），GPU 无 worker。下一步先将可选 intake 接口合入并做相关检查，再准备官方 FSDP2 分片训练的独立源就绪性验证。FSDP 的 all-gather/reduce-scatter 路径与已有 DDP 不同，但需实测质量、通信占比和完整工作，不预设一定能获得通信优化正例；不放宽旧 TP 的完整质量标准。
