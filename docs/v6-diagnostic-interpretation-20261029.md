# 两任务诊断解读与既有反馈接口实验

2026-10-29 02:46:31 UTC 批次正常结束，终态已独立核查。**两份报告均不原样采纳或传给 candidate**：有用观测与错误、范围过度推断并存。生产 core/prompt/allocator/接受规则不变，无新增 GPU 实验或优化接受。见[小型审计回执](evidence/v6-diagnostic-interpretation-20261029/public-audit.json)，服务器 `terminal-audit.json` SHA `b3d242a3cc6a33c7380825026866e7a680288522ef4474acc4df705b03d76525`。

两次实际初始快照各 5 文件，显式输入分别 23,371 / 38,309 bytes，通用 prompt 和 model/tool settings 相同；不同任务，不是单变量 A/B。Qwen `generation-4f922b3e9bd7` / FSDP `generation-bfbc2edfc7da` 均 0 schema repairs，分别 37/35 条 assistant 消息、44/39 次工具调用；provider 费用估计 1.02682051884 / 1.03636267628，仅统计。原样 packet 为 33,780 / 37,609 bytes，SHA 分别 `0fb0d6ba65b63a4f61815b933260bcb5fb3e8f796a1f5944880f975ecb3ba4d1` / `6d781d5d00ac55e7c0ee736a525ec6ad79c24343bfcd18d1c9541deb53853cfa`。完整 session/tool 记录及 CPU 脚本保存在服务器 audit/sandbox；实际访问审阅未见后续成功答案或开发者事实表，但文件系统不是 hermetic。Qwen 的格式化异常、错误 stride toy、FSDP 错误安装路径与不匹配的自拟 core 哈希算法均保留，工具退出码不能单独证明检查通过。

所有被引用文件的 SHA 均匹配（21 / 20 个不同文件），116 个冻结文件再次一致。CPU 重放 `feedback.route` 确认原样 packet 保留且其他路由字段相同；这不是实际 selector/candidate 输入或使用证明。原 worker request 的实际差异均为 entry、output、parameters.diagnostic_context；function/wait 原本已相同。未改原文件、pointer、观测值或质量结果。

Qwen 解读确实读取四个 candidate 原始 CPU 输出并计算两次 greedy、两次 teacher 的关系；本次独立审计另对 B/C 全八个原始输出逐项对完整独立 source 比较：除第二 teacher 外全部 tokens/logits 精确相同；第二 teacher 12 token 不同、首 decode/maxabs 31.125、4,596,832 个 CPU float32 容差违例，全部 finite。其有用定位成立，但报告存在以下问题：

- summary/unresolved 把第二步 compare 未达到扩张成 evaluator compare 从未执行；实际第一 observe 返回后已调用第一步 compare，才进入第二 run。异常返回没有 checks 数组不等于没有历史 compare 调用。
- fact[11] 声称 negative 同时验证 token/logit；实际两个负例都只改 token，framework 记录路径为 `output.generated_tokens`，没有 logit 负例。
- fact[4] 的 margin 18.75 高于 logit 差异尺度不成立，实际首 decode maxabs 31.125；较大 margin 可描述，但不能使用错误大小关系。
- 相同 prompt/teacher 与错误仅出现在第二 teacher，不足以唯一归因跨调用状态或排除其他执行因素。静态 stride/signature 代码支持 fallback 推断，但缺 route 事件不能独立证明每个实际 native 路径；观测时间差也不是 Graph 路径证明。
- baseline 正确是一次正控制，不提供量化 observer 影响上界；request 不止 entry/function 改动。独立摘要局限不改变原 full-logits 质量合同。

FSDP 解读确实遍历两 rank 摘要，识别第一步 loss/logits 精确、梯度已存在容差内差异、第二步 logits 才超限，保留原 `status=passed/valid=false` 区别与缺少 actual raw 的限制。但报告仍有：

- 从最终 loss/logits 精确直接定位差异“进入 backward/reduction、而非 forward”过强；没有观测全部 forward 中间值或表示。条件性 GEMM/通信假设不能升级为唯一机制。
- fact[6] 声称参数恰在梯度不同的位置不同；实际非精确梯度 292/291 路径，参数 244/243，每 rank 另有 48 个梯度不同但参数精确的路径。
- top_mismatches 是全体元素按绝对误差排序；本次再次计算，第二 logits 的每 rank top8 **全部在相对容差内**，不能据这些位置推断真正 1898/1592 个违例的空间分布。
- `observer_wall_s` 从原 compare 之前开始，包含 compare+遍历；不能称额外增加的 observer 耗时或相减隔离。一次零误差 baseline 不能证明普遍“noise floor=0”或排除环境交互。
- fact[13] 引用的 `original_invocation.mirror.math_policy_unchanged` 不存在；真实 worker result 有该字段且为 true。文件 hash 正确没有修正错误 locator。

未把上述开发者核查内容人工写回 Agent packet。下一步为[一次独立证据核验角色实验](v6-diagnostic-verification-20261029.md)：原始输入加原样 packet，不提供本段错误清单，让新的真实角色核对支持范围并生成更精简的事实包。只检验这个已展示的缺口，不自动添加生产审核层，不重复旧角色直到通过，也不把两个案例当作泛化证明。

核查时 GPU worker 为空，实际 150 GiB 数据盘余 10,232,963,072 bytes（约 9.53 GiB）。本轮没有清理、raw 本地下载或 reference 复制；历史接受/拒绝及全部样本保持不变。

已于 2026-10-29 02:21:11 UTC（香港10:21）启动，冻结 `727de6bbccf41a335897c17af57106899b458560`，controller943330，真实 OpenCode 本地服务健康检查通过。02:22:48 核验 Qwen 初始 `generation-4f922b3e9bd7`／session `ses_e7a89bf86ffeHjr599gRaIxQ3T`：5 个初始文件（4 显式文件 23,371 bytes 加环境）逐字相同，11 条真实 assistant 消息、16 次工具调用；初步未见开发者审计或后续成功答案访问命中，完整访问和事实有效性留待终态审计。FSDP 此时尚未开始，其显式输入预检为 38,309 bytes。共 116 个冻结文件哈希核验，core 不变。见[启动回执](evidence/v6-diagnostic-interpretation-20261029/launch.json)和[实际快照核查](evidence/v6-diagnostic-interpretation-20261029/startup-verification.json)。

唯一普通完成/磁盘监听 PID47044 已用实际进程命令核实，旧验证监听已标记处理。实际 df 余 10,236,854,272 bytes（约9.53GiB），无 GPU worker、清理或 raw 本地转存；阈值8GiB。见[监听回执](evidence/v6-diagnostic-interpretation-20261029/wakeup-setup.json)。启动核查脚本首次因本地 UTF-8 BOM 被 GBK stdin 拒绝，尚未执行远端代码；移除 BOM 后核查完成，没有重启实验或创建重复监听。

Qwen 和 FSDP 的原样 Agent 诊断脚本已分别执行，但采集后的事实解读、证据边界判断和反馈整理仍由开发者承担。下一阶段用同一套通用 `diagnostic_interpretation` 指令，按 Qwen、FSDP 顺序各发一次新的真实角色请求，验证这一缺口。不同任务与产物，不是单变量 A/B，不自动采纳新角色或扩大生产上下文。

每个角色只获得原 task、原失败记录、证据文件索引、通用反馈 schema，框架自动增加 worker 环境。索引指向原 adapter/candidate/source/reference、Agent 自己生成的 probe/plan 和已完成的固定 B/C worker/诊断产物；不内嵌所有梯度摘要或大张量，不提供开发者 terminal-audit、上轮事实表、数值原因、其他任务输出或后续成功实现。Agent 可选择读取原始 JSON/CPU 张量和相关源码，实际工具访问与计算必须留档。外部文件系统并非隔离，不能仅凭输入文件名就宣称没有额外访问。

输出 `supplemental-evidence.json`：简短摘要、事实及其 path/SHA/具体位置/phase/rank/调用范围、局限、未解决问题和明确分开的条件性假设。目标是把真实观察交给后续角色，不是要求写修复实现。可以明确报告不足或错误；schema 通过不等于事实正确，不能把 worker 完成、run-only 诊断或摘要 hash 当成质量/因果/性能接受。

此批只做真实文件解读和小型 CPU 核查，无新模型/GPU/collective、候选搜索、源参考复制、性能接受或生产改动。Agent 不能重新运行实验；不得修改原任务/容差/冻结数据。API 费用仅统计，无调用限额。原测试错误与所有记录保留，不重复角色直到通过。

生成后控制器仅在 CPU 将**原样 packet**放入现有 `supplemental_evidence`，对原历史反馈运行既有 `feedback.route`，检查该字段保持且其他路由字段不变。该检查的 current_revision 来自原物化请求，仅验证接口传输；不是新的正式 decision、真实 selector/candidate 输入或下游使用。不会手工补正角色答案。最终须审计真实初始快照、所有读取、生成文件/引用/计算与局限，才能决定后续用途。

Qwen 产物来自 `diagnostic-author-validation-20261029`，FSDP 来自 `fsdp-diagnostic-validation-20261029`；后一脚本经过一次明确的 CPU 持久化缺陷修复。两者的观测扰动、原样数据保留程度、source mapping 与开发者协助边界仍存在，不能合并成无人干预原入口成功证明。

批次目录 `/root/autodl-tmp/dev/v6-work/diagnostic-interpretation-20261029`，控制器 `scripts/experiments/v6_diagnostic_interpretation_20261029.py`。保持生产 core `ed24af7a…`。启动前核实实际 df 与无 GPU worker，预计仅增加角色/session/小型 CPU 产物。大文件与 private provider 信息留服务器；只发布 code、hash、索引及小型脱敏摘要。设置一个普通完成/磁盘监听，无 Goal、模型等待轮询或投递测试。
