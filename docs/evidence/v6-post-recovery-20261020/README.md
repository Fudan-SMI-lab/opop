# 独立复核、组合交互与真实环境修复

本目录继续记录 provider 恢复后的真实运行。完整结果、生成代码及原始文件哈希在相邻 JSON／Python 文件；失败、无改善和未完成验收保持区分。

## DDP 成功候选的独立复核

固定 `trial-9cf6ebd687-param-cb3e0316` 的代码和参数，使用框架 replay 接口重新绑定相同调用身份，在冻结实现 `a1b4ae1` 下执行：

| 条件 | 父版本→候选 | 判定 |
|---|---|---|
| 原 workload，整步目标 | 146.083→125.422 ms | 两对收益均超过噪声，影子评价通过 |
| 保留 workload，sequence=256、K=2 | 95.412→93.863 ms | 正确性通过；第一对收益未超过噪声，拒绝 |
| 原 workload，通信诊断目标 | 79.184→73.531 ms | 正确性与整步约束通过；通信收益不稳定，拒绝 |

三次均完成两个连续训练步，每步比较约 2.49 亿元素；24 个替换站点实际执行，未观察到 `context.references` 回退调用。保留形状不是全部回退后通过，但仍不能宣称获得可靠性能收益。通信目标的第二对收益为负，两对均未超过各自噪声界限；干净 step 约 121.55／119.79 ms，满足固定原始基线的 161.422 ms 上限。不能将计算优化的整步收益直接改称通信优化收益。

三个 trial 分别为 `replay-93432c81704a`、`replay-f0dc21acf59b`、`replay-0fbd6be241ce`；以各自文件中的完整配对数据和质量记录为准。

## A、B、A+B 的实测交互

`run_interaction.py` 使用同一冻结原始父版本和相同质量／采样协议，分别测量已有计算候选 A、通信策略 B，以及原始两候选组合 A+B。代码和参数保持不变，三个版本全部通过质量和覆盖检查，均未提交。

| 版本 | 本次父版本→候选 step |
|---|---|
| A | 149.738→152.682 ms |
| B | 144.448→247.901 ms |
| A+B | 141.627→228.397 ms |

每行各自独立 AB/BA，父版本身份相同但测量样本不同。基于配对收益计算的交互点值为 **+19.628 ms**；组合各对极值的范围为 **[-2.520, 41.776] ms**，再按端点 MAD 界限扩展为 **[-34.763, 74.019] ms**。范围覆盖零，不能确定交互方向，也不能把点值分配给某一组。这些范围不是统计置信区间。B 与 A+B 的整体退化由各自完整项目测量直接支持。

[交互结果](p6-ddp-interaction-c/interaction.json)、[固定协议](p6-ddp-interaction-c/protocol.json)和三个端点报告保留实际来源。该次 driver 为 `649f16b`，core 为 `8a528212ba65a71f41f7ac938cacdc939c6595fff373d87e43c5c97528eabb9b`；目标单位为 ms。后续 driver 在协议内显式写入 objective、workload 和 request 哈希，不回改该已完成实验。

## Agent 与 worker 的解释器不一致

实际工具记录显示，部分生成角色通过 `/root/miniconda3/bin/python` 检查库源码；实测该环境为 Torch 2.8.0+cu128、Transformers 5.20.1。框架 worker 使用 `/root/autodl-tmp/dev-venv/bin/python`，实际为 Torch 2.13.0+cu129、Transformers 5.16.1。库 API 检查可能因此出错；这不代表已经在正确 worker 上测得的质量／性能结果失效，也不将历史失败全部归因为此问题。

项目配置现在为角色设置 worker Python 优先的 PATH，并提供 `inputs/worker-environment.json`。生成、同会话修复和恢复请求都明确要求使用该解释器。旧单算子配置保持不变；没有修改系统库或私有 provider 配置。

[真实验证请求](p6-worker-environment-validation-a-result.json)使用配置中的 GLM-5.3 实际执行两次 CPU 元数据查询，默认 `python` 与显式路径均得到相同的开发环境版本，4 条 assistant 消息，报告费用约 $0.01596。box4-2 上解释器配置与 provider 恢复的 4 项相关检查通过。正在运行的四个开发会话收到[环境更正](p6-worker-environment-correction.json)，通过服务端 `noReply` 消息补充事实，没有中止或重建原会话；已完成的正式对照未修改。

## 自动探索分支的整版推广

四轮 ResNet 协议已经完成。第 2、3 轮未通过主分支的收益规则，但保留了质量与约束成立的探索分支；第 4 轮在该分支生成 `trial-93b540a61d`，局部评价通过后，框架自动触发 `version-b9e9208e6e53` 整版推广。主分支 0.730554 ms→完整候选 0.668637 ms，两对收益 0.061520／0.062314 ms 均超过对应噪声 0.002794／0.007582 ms，接受并提交。

这是真实继续探索和自动整版回传，不是只手工复测一份孤立代码。前三轮使用原冻结实现；第四轮因新分支缺少 discovery 的实际错误，使用 `a1b4ae1` 修复后恢复，完整记录保留这一源码变化，不能把它混入未改变实现的正式 B 组对照。

## 通信候选与 Qwen 回退诊断

`trial-35714beb71` 修复了通信策略的连续状态错误：两个 rank 的两个连续训练步完整梯度／参数均精确一致。独立通信诊断约 78.095 ms，但干净整步约 261.574 ms，超过锚定原始参考的 161.422 ms 上限，因此参数筛查拒绝，没有进入收益接受流程。该负例完成了质量、通信观测与项目约束的闭环，不是通信加速证明。

Qwen `trial-023e79293c` 在旧冻结实现下数值及配对计时通过，69.518→66.723 ms，已按原规则提交；新增 prefill 候选却在质量执行中 2/2 次调用原实现。其父候选仍有真实 functional 替换，因此这里只缺少新增候选有效执行，不能把两份安装文件算作有效共存。

新增通用 `candidate_diagnostics` 采集器对原候选字节做独立诊断。第一次追踪发现 `_try_build_graph` 返回 False，但 Python 报告的行号处于 `with` 清理语句；补充最近执行行后，第二次明确定位到 226–229 行：捕获、replay 和同步成功之后，`torch.equal(static_out.logits, warm.logits)` 不成立而回退。这不证明差异超过原质量阈值，也不证明可以忽略差异；后续 Agent 必须读取实际源码和原契约修复。诊断还保留父候选自测中的异常，未将其混为新候选失败原因。4 项相关检查在 box4-2 通过，真实诊断分别保留源码哈希和 worker 回执。

## 批量 Qwen 吞吐的有效新候选

前四次 Graph 候选都未通过完整质量检查；第五次 `trial-0b17ae710c` 改为 eager 解码、复用 KV 工作区及 RoPE 数据。框架使用原冻结实现 `d750162`、原 batch=8／prompt=256／生成32 token 契约重新测量：两个连续步骤各比较 38,895,872 个元素，最大误差为0；62 次 decode 根调用全部执行 `eager_flat_decode_ws`，`context.references` 调用为0。

独立 AB/BA 的生成吞吐为父版本 307.644→候选 349.959 token/s，提升约13.8%。两对收益分别为 50.139／34.491 token/s，超过对应噪声 8.664／5.349，接受并提交 revision `ccfeeb296e5581c30d9e23a96750f209f9212e0cb166eeb5735dda3bdf13fe69`。这是同一固定 workload 的结果；不能与 batch=1 TTFT 的数值相减，推导纯粹的目标选择效应。

失败反馈实际促成了不同实现，并经真实 GPU 检验成立；Agent 对历史失败根因的推断仍是候选分析，不因新方案通过就追认全部因果解释。原四次失败和相应成本继续保留。

## Qwen 组合验证的实际调整

`trial-fa955139a3` 将 Graph 初始化移至真实 prefill 调用，使用当时已启用的父版本 dispatch，代码也加入完整 greedy 自检，但本次在更早的 KV 守卫已经退出。筛查和 AB/BA 的外部质量均通过，但新路径仍为 `prefill_init_verify_parent`。自动诊断定位到 `_pairs_close` 的 key tensor `torch.allclose` 失败，随后 `_lazy_init` 禁用 Graph；这不能推导所有 logits 检查也失败，更不能证明内部 KV 差异的具体数值来源。

其配对 TTFT 65.866→65.338 ms，收益 −1.621／2.676 ms，均未超过各自噪声规则，因此拒绝，未计作有效共存。原候选、诊断、全部调用与实际失败成本保留。

随后在独立目录 `p6-qwen-throughput-composition-c` 从已经接受的 batch=8 eager 解码版本验证互补候选。此控制实验让 Selection Agent 自行在真实 groups 中选 group/joint 范围，并要求保留已有 decoder bundle；没有人工指定算子组，也没有改变该任务的 workload、目标、质量或采样协议。默认工作流仍支持四种 scope。这个调整验证的是同一 Qwen 项目的另一个明确负载，不能将结果补写成此前 interactive Graph 已通过。

该后续实验现已完成，正式提交结果见下文。草稿诊断与最终 Agent 提交分别留档，前者不替代自动提交和独立组合评价。

自动触发另经真实 TaskSearch 验证：`p6-qwen-automatic-diagnostics-c` 使用原候选字节和参数，筛查观察到全部经过原实现后自行运行诊断，再次定位上述 logits 条件。筛查和诊断的源码哈希均为 `cc2c1b9634d05a488ceda558a96641b79c3583c93245dad79aa9c9a98098334f`；结果保存进参数反馈，接受指针未改变，不计作新优化收益。

后续源码检查还发现组合提案中的一个未验证假设：安装阶段直接调用 parent forward，并不会自动进入旧 functional 候选的 dispatch mode，也没有自动建立 adapter 的 prefill/decode phase。已将真实执行语义补充给正在运行的候选会话，并作为通用 `inputs/binding-contract.md` 自动提供给以后选组、生成和恢复角色。Graph 的原生执行与 Python 调用计数分别解释，不允许通过无用调用填充计数。此轮 7 项诊断、角色环境和恢复检查通过；新的候选仍由冻结版本独立运行。

## 联合候选中的局部回退

16:37 左右 provider 再次返回账户窗口 HTTP 429，明确恢复时间为 17:12:49（UTC+8）。[实际重试回执](p6-qwen-provider-window.json)保留原会话和服务端时间；框架未增加费用、调用或时长额度。等待期间没有修改原 Agent 草稿，而是冻结副本 `c4707569b110cc756ba4f4d55df987af454e8f33396b8dac99ae1af2186d2127` 做独立诊断。

[局部 GPU 回执](p6-qwen-draft-gpu-diagnostics.json)使用真实 worker、原候选字节、原 1200／2300 次验证窗口和三个重复随机输入。归一化与 SiLU 链最终进入融合路径并逐位通过这些局部输出检查；两类 RoPE 始终回退。单独追踪发现 `_k_rope1` 编译时 `tl.arange(0, h)` 的 `h` 没有声明为 `tl.constexpr`。这是被异常处理隐藏的真实编译错误，不能把正确的父实现输出算成 RoPE 融合成功。探针源码和冻结草稿位于 `p6-qwen-draft-diagnostic-d/`。

旧自动条件要求整个 bundle 的所有站点都调用父实现，因而会漏掉这种部分融合、部分回退。当前实现改为任一活动站点全部通过 `context.references` 就运行独立诊断，回执记录触发的 rank／bundle／call 和次数；合法委托也可能触发，因此它不改变质量、约束或接受规则。[4 项检查](p6-partial-fallback-tests.json)在 box4-2 通过。

[完整 Qwen 草稿筛查](p6-qwen-draft-project-d-result.json)随后通过真实 TaskSearch 自动触发新诊断，定位到同一个编译错误。此实验使用开发者为诊断构造的临时 proposal、原自动 selection 和未改动的 Agent 代码，**不是 Agent 正式提交，也没有 AB/BA 接受或发布**。两个连续步骤各比较 38,895,872 个元素，原质量阈值通过，但最大绝对误差为 0.125，不能称为全模型逐位一致。父 decoder 执行 62 次、无父实现引用；新增 `norm_fused_mp` 4,290 次、`silu_fused` 1,032 次，RoPE 两类仍回退。单次筛查吞吐 102.398 token/s 仅是未配对的观测，不能据此计算可靠收益。原已接受指针保持不变。

该自动诊断的核心源码哈希为 `f7fe40118f9ef8ec5cc4e94653c7e716496dd1ca542d3cab66b8265bd647a85f`。仍在等待恢复的正式候选使用原冻结 `790286a`，没有在途中混换实现。[局部探针输入回执](p6-qwen-draft-gpu-diagnostic-context.json)和[完整筛查输入回执](p6-qwen-draft-project-context.json)证明真实资料已送入原会话，是否被读取由后续工具记录另行确认。

历史 packing 失败、草稿静态检查及上述诊断是开发期间显式提供的补充上下文，不宣称框架已经自动检索所有跨 run 历史。正常上一轮候选／反馈、worker 环境和 binding contract 则由框架自动注入。[工具访问记录](p6-qwen-composition-context-consumption.json)区分这些来源；文件名匹配只能证明对应访问或搜索，不能证明候选的因果推断成立。

## 最终 Qwen 有效共存与退化拒绝

原 `zhipuai/glm-5.3` 会话在 17:12:49 按服务端窗口恢复，随后正式提交 `trial-09def220ed`，由冻结 `790286a` 完成原 TaskSearch、完整质量、AB/BA 和原生 profile。期间发现 Agent 将修订文件写入拼错目录，已将[两个路径和实际哈希](p6-qwen-candidate-path-correction.json)回传；最终提交字节哈希为 `e34056e186fb42197a6d5d72224042bf66056e996f4b70227c7a10aa24be11e6`，不使用错目录中的未提交副本。

[正式结果](p6-qwen-throughput-composition-c/trial-09def220ed.json)与[候选代码](p6-qwen-throughput-composition-c/trial-09def220ed-candidate.py)保留全部端点和实际提交字节：

| 检查 | 实际结果 |
|---|---|
| 全量连续质量 | 每个候选端点的两个步骤各比较 38,895,872 个元素，均通过原阈值；最大绝对误差 0.125 |
| 继承 decoder | 每个质量端点 62 次 `eager_flat_decode_ws`，0 次 `context.references`；原 bundle 保持安装和有效执行 |
| 新增归一化融合 | 每个质量端点 `norm_fused` 4,290 次，另有 `norm_verify` 4,700 次；有效计算与校验计算分开 |
| 原生贡献证据 | 独立预热后的单次 profile 有 4,495 次 `_k_norm`：hidden 2,263、q 1,116、k 1,116；带旧 decoder 与新候选的嵌套归属 |
| 仍未生效的部分 | RoPE 两类经内部一致性守卫回退；rsqrt、SiLU、entry cast 和 weight scale 明确委托父实现，不称全部 26 个组都得到优化 |
| 配对吞吐 | 父版本 364.162／357.128，组合 105.033／107.559 token/s；均值 360.645→106.296 |
| 接受与回退 | 两对收益 −259.129／−249.569 token/s；拒绝整版，已接受指针保持父版本 |

[有效共存审查](p6-qwen-composition-assessment.json)结合[调用与原生摘要](p6-qwen-effective-composition-evidence.json)和实际代码：新 `_k_norm` 使用当前输入写入乘法结果与 BF16 tensor，后续 cast 返回该 tensor，父 weight mul 消费它；不是仅执行后丢弃的校验 kernel。旧 decoder 实现不包含此 kernel，其根路径继续贡献 GEMM／attention 等计算。原生范围嵌套，不能将两个 bundle 的 kernel 数或时间相加当成独立贡献。

来源是已接受的 batch=8 revision `ccfeeb296e5581c30d9e23a96750f209f9212e0cb166eeb5735dda3bdf13fe69`；分支重新发现后的实际比较父版本为 `0643ca5cc67f8d8ec2f7b2e8e4719a523a6ee99e581015ab0f3f9f87bfe98ceb`，待测组合为 `bddbf757f52fd5b47a58515262c859336bf0899f003e61d3c423b8d1b5961fd3`。拒绝后指针仍是实际父版本。该结果完成 Qwen 有效双 bundle 的功能验收，**不构成组合加速**。较早 batch=1 Graph 的失败仍为失败。

最终代码移除了草稿中的 rsqrt／SiLU 融合，但全模型最大误差仍为 0.125，因此不能将先前的误差独断地归因于这两个变换，也不能接受候选文档中的逐位一致推断。RoPE 已越过编译问题，但仍未通过内部守卫。端到端退化由配对实测成立；没有额外隔离实验时，不能把全部损失因果分配给某个 kernel 或 Python dispatch。

## Provider 恢复、备用请求与最终用量

等待账户窗口期间，OpenCode 已列出的 `opencode/glm-5.3-flash-free` 通过[真实响应与文件输出探针](p6-provider-alternate-probe-e-result.json)，未向探针发送项目内容。随后在[独立分支](p6-qwen-composition-fallback-e-composition-protocol.json)复用原自动 selection，向真实候选角色提供草稿和实际诊断；模型配置只在该控制器内覆盖，没有修改私有全局配置。这是显式开发调整，不宣称框架自动选择了备用 provider。

原会话完成必要质量、配对和原生共存验证后，尚未提交产物的备用角色被[显式取消](p6-qwen-composition-fallback-e-development-cancellation.json)。服务端确认中止，控制器以 `MessageAbortedError` 退出；没有把它计作有效试验或优化结果，也没有设置调用／token／费用／时长额度。[实际工具访问](p6-qwen-fallback-context-consumption.json)及全部已发生用量仍保留。

[最终用量快照](p6-resumed-final-usage.json)截至 17:34（UTC+8）包含 162 个会话、4,534 条 assistant 消息，provider 报告费用 $150.0315571213。包含嵌套探索分支、失败和取消的请求，按 session/message 去重；此次刷新读取 160 个已记录项目会话，均可获取，其余早期会话快照保留在账本。统计不含外层 Codex 对话，不是账单；当前没有活动项目角色，费用始终只统计。
