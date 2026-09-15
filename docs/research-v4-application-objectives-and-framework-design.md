# v4：应用目标、贡献边界与框架改造设计

> 日期：2026-09-15。性质：研究综合与设计建议，不是代码实施或实验完成报告。
> **先纠正旧判断：v4 已实现条件扫描与 fresh measurement，不能继续用 v3 的缺口描述当前实现。**
> 本文保留现有 A/B/C/D 框架，以 C1/C2 为主线；应用适配不是另建推理平台的理由。
> 本文写作与修订仅涉及此文档，没有修改源码、配置、测试或旧文档，没有运行测试、GPU 作业或下载权重；同期其他提交不属于本文工作。

## 执行摘要：四个问题的直接回答

**问题一，走出 KernelBench，什么应用能说明实际价值？**
首选 Qwen3-8B 的完整自回归生成，比较同一混合请求集合下的首 token 等待与输出流畅度。
第二个主案例是 SDXL-base 完整出图，比较共同约束下的端到端时间与峰值内存。
两者分别展示阶段偏好与资源取舍，论文说服力来自可解释的行动变化，不只是模型更大。
BGE reranker 可作低成本启动或替代，不强制成为第三个主应用；实时检测和服务系统后置。

**问题二，v4 实际改了什么？**
已有独立条件测量模块、固定搭档的 C4 双端复测、一次性方向 token、fresh 投递、生产 trial 回填与简报。
真实缺口是若干准入、隔离、软墙和日志细节，以及目标仍主要绑定 `robust_ms`，而不是“完全没有条件响应”。
本次静态证据不等于在当前 HEAD 重新运行了历史测试或真实 GPU 冒烟。

**问题三，三个贡献是否成立、是否新颖、怎样改写？**
C1 应写资源约束与结构方法对齐，不能保证到达物理峰值。
C2 应写固定搭档的程序旋钮响应指导有益结构改写，不能写最大化利用率或保证全局最优。
C3 若成立，应是应用目标驱动的 site/phase/action 选择，而不只是增加 objective adapter。
默认论文定位为“两项主贡献加应用验证”；只有独立消融证明 C3 的增量，才升级为三项机制贡献。

**问题四，如何具体改造、跑完整流程、做实验？**
先补核心协议缺口，同时设计应用身份；随后接入薄适配器与统一目标，再研究有界 C3。
正式评分始终针对完整已接受补丁集上的新产物，而非局部 kernel 的微基准替代品。
保留 M1/N1/P2′，新增应用绑定自检、候选与目标矩阵、C1/C2 机制消融及 C3 三臂实验。
不要等待可选 E4、完整 RR 或服务调度平台全部完成才开始应用身份与最小适配。

### 阅读约定与证据分级

- **现有源码事实**：限定版本、路径和调用链的静态判断，不表示某次 run 已走过该路径。
- **历史报告**：现有文档记载的测试和机器结果，本次未重新执行或复核原始运行数据。
- **来源证据**：一手模型卡、固定代码或论文的相关内容，不表示复现了作者实验。
- **拟议**：下文所有新模块、应用参数、实验和验收门；没有在本次交付中实现。
- **未知**：当前硬件容量、真实触发率、性能收益、统计功效和精确重启等价性。

## 1. 版本锚点：这次回答的是哪一个 v4？

本节引用已取得的只读版本快照，不重新查询移动中的 HEAD，也不把快照称作所有机器的最新状态。
**历史审计快照**：v4 位于独立 worktree，分支为 `v4`，当时工作区干净。
历史 HEAD 为 `7fc8a18bcb1bf15cb73b142bd1661a29ad545095`。
对应 v3 tip 与 merge base 均为 `391b4727f5132ccb64b8853b0d58284a44529321`。
相对该基线的历史统计为 29 个文件，新增 5004 行、删除 22 行。
**最终核验快照**：分支仍为 `v4`，HEAD 为 `ff402aa5bb13eb3193ca05830f81fadb3d92763a`，上述 base 不变。
最终相对基线为 31 个文件，新增 5129 行、删除 37 行；核验时仅本文档未跟踪，无已跟踪文件的 staged/unstaged 修改。
两次快照之间的提交由同期其他工作产生，不是本文写作或修订所作的代码更改。

| 版本记录 | 已知职责 | 本文如何使用 |
|---|---|---|
| `46aa50e` | 条件测量主实现 | 确认 v4 不再只是规格稿 |
| `16f23aa` | seed pairing 与实验配置 | 配对实验的工程基础，不替代运行核验 |
| `1fb9c7d` | 拒绝重复种子 | 减少配对集构造歧义 |
| `63d097d` | 端点与搭档日志 | 支持 P2′ 按当时截止重建边际对照 |
| `bc8b4ed` | 脚本参数 | 最终快照新增记录，不表示已全面重审脚本 |
| `eb4d908` | 排程超时量说明 | 实验文档行号变化，不能视为本次新增运行结果 |
| `942512c` | admission 的 axis_value/config_key 日志 | 增加已准入点的可识别字段，不修复当前搭档准入缺口 |
| `ff402aa` | 结果精度 | 提高结果记录精度，不修复 E 简报 role latency 缺值 |

两快照间唯一生产源码差异位于 bridge，已取得的静态差异核对限于日志字段与结果精度。
该核对不等于重新审计全部新增脚本或结果；§2 的 A、D 发现未被这些日志变更修复。

历史 29 文件变更中的新增生产模块为 `conditional/{__init__,identity,probe,gates,tokens,scanner,bridge}.py`。
现有生产接线修改涉及 `config.py`、`control/orchestrator.py` 和 `tuning/tpe.py`。
测试覆盖文件包括 `test_v41_conditional_scan`、`test_v41_fresh_intent`、`test_v41_seed_pairing`。
历史变更还包括 7 个配置、8 个脚本和实验文档；上述清单解释历史类别，不将其冒充最终 31 文件的完整清单。

旧文档 `../../v3/docs/research-task-objective-conditioned-optimization.md:140-147` 是 v3 语境。
其中“v4.1 条件化与 fresh repetition 尚未实现”已被此 v4 快照的源码事实取代。
该旧文档关于目标贯通、资源实体归属和新颖性约束的讨论仍有用，不需要删除或改写旧文件。

`experiment-plan-post-v4.1.md:143-158` 历史报告 Windows 1225、Linux 1239 个测试通过。
这里采用最终核验快照的 159 行文档：相较最初 142 行版本，排程超时量说明新增 17 行，后文行号随之移动。
同处报告 box4 真 GPU probe、C4 四次 fresh 复测、token 和 E1 链路冒烟通过。
这些是历史报告，不是本次在 HEAD 上新验证的测试结果，也不证明本文列出的每项边界已覆盖。
已知 4090 campaign 机器信息同样来自历史材料，本次没有查询设备或确认当前空闲状态。

## 2. v4 的真实能力与尚未闭合的协议

### 2.1 已有生产路径，不应重新发明

全文本地生产源码路径相对 `v4/src/kernel_optimizer/`；除明确更新者外，行号固定于历史审计快照 `7fc8a18bcb1bf15cb73b142bd1661a29ad545095`。
所有 `conditional/bridge.py`（含下表简写 `bridge`）锚点更新至最终快照 `ff402aa5bb13eb3193ca05830f81fadb3d92763a`；未变区段行号保持不变。
本地 v4.1 规格行号仍指历史审计版本；实验文档引用统一采用最终核验的 159 行版本，外部代码使用各自固定 SHA。

| 能力 | 源码锚点 | 实际含义与边界 |
|---|---|---|
| 逐 trial 接线 | `conditional/bridge.py:98-133` | 回填扫描结果，完成后尝试 E 准入，按 cadence 发 probe |
| 编译探针与答案读取 | `conditional/bridge.py:173-190` | 经 prescreen 批量获得资源回答，不是正式延迟测量 |
| fresh 投递 | `conditional/bridge.py:216-254` | 扫描点进入正常 tuner，并登记 pending、预算与点身份 |
| 冻结 C4 | `conditional/scanner.py:183-227` | 先冻结 F/N 与搭档；发现对、验证对分别独立随机，共四个 trial |
| 回填与方向门 | `conditional/scanner.py:395-427` | 按角色而非到达顺序计算发现响应与留出响应 |
| 一次性来源 | `conditional/tokens.py:104-157` | 完整新来源压旧、最多一个方向 token、准入即消费 |
| fresh 识别 | `tuning/tpe.py:79-158,191-207` | 重复配置可有明确 fresh 意图，不能当普通去重命中处理 |
| 生产真复测路径 | `control/orchestrator.py:1271-1335` | fresh 绕过 measured cache，执行正常 production trial |
| 条件简报 | `conditional/bridge.py:326-393` | 将资源墙、方向响应、经验不确定性送入简报 |
| 改写接线 | `control/orchestrator.py:2036-2066,3195-3210` | analyst/rewriter 复用已有反馈入口 |
| 端点来源日志 | `conditional/bridge.py:290-318` | 保存冻结轴值、搭档与 comparable key，便于截止时重建 |

这不是 TPE 的替代系统：扫描购买少量受控证据，E 追加点，生产测量仍进入正常选择链。
C4 的“C”是条件扫描内部来源相位；外层 A/B/C/D 的“C”是结构分析与改写，二者不要混称。
`g_d` 属于发现对，`y` 属于验证对；不能按 token 是否通过筛选 P2′ 的留出样本。
旧 incumbent 的缓存延迟不能充当 C4 新端点；四个来源 trial 都应有真实 fresh 测量意图。

现有 full forward 是继承能力，不是这次 v4 才支持完整模型。
`gpu/worker_main.py:2242-2280,2328-2345` 的模型构造和完整调用边界支持这一点。
不过完整 forward 不自动等于预训练权重可信绑定、KV 生命周期或完整应用请求计时。
`models/core.py:45-52` 的任务描述主要是 task source；`111-144` 的 `robust_ms` 仍为延迟口径。
`conditional/bridge.py:272-281` 选 incumbent、`tuning/tpe.py:227-253` 的 tell 仍依赖延迟。
资源向量基础可复用 `dimensions.py:230-363`，原组合入口见 `wiring.py:108-162` 与 `ports.py`。

### 2.2 开关与剂量不是同一件事

`config.py:727-742` 的默认模式为 `off`，`f_frac=0.1`，cadence 为 10 个 told。
因此“代码已接入”不等于“所有默认运行都启用了条件扫描”。
active 下若 `B=40,f=0.1`，则 `Q=4`，冷启动只够 C4，不够再购买 E1。
这是已接受的剂量限制，不是 fresh 机制失效，也不能据此宣称 E 在所有场景都不可用。
campaign 的 `f=0.125,Q=5` 允许 C4+E1，但几何、噪声、来源、预算等门仍可能阻止 E。
验证配置若为 `B=20,f=0.125,Q=2`，冷启动无法覆盖 C4，更不能作为全链正对照。
同候选续空间与冷启动要分层报告；继承来源也仍须满足当前适用性与本空间付费要求。

### 2.3 静态发现：影响哪项主张，最小补救是什么？

下表综合两次独立源码核读，并回读了关键 bridge/scanner/token/gate 路径。
它是静态发现清单，不是运行影响估计；所有补救和测试都只是建议，未在此执行。

| ID / 优先级 | 已确认的实现事实 | 受影响的主张 | 最小补救与专门测试，均拟议 |
|---|---|---|---|
| A / 高 | `bridge:216-222` 传 walls/fits，不传当前搭档；`scanner:161-174,231-247,293-319` 使用已存搭档；`tokens:145-147` 只筛 execution identity | E 在 admission 时使用当前上下文；规格 `68-72,188-193` 要求精确当前适用 | consume 前核对 current partner projection；测试轴自身移动保留资格、其他搭档变化拒绝，准入后仍执行 committed 几何 |
| B / 高，隔离 | `bridge:9-15` 明示共享 compiler facts；dispatch 经 `evaluation/correctness.py:194-245` 写生产 screen cache；生产 guard `orchestrator:1244-1247` 与 TPE `116-119` 会读它 | observe 完全不影响生产轨迹；规格 `116-118` 明确禁止写生产 `_screen_cache` | 先裁定独立缓存还是共享事实政策，再测 probe 前后普通 sampler/PRUNED 与缓存差异；不能只查 scan trial 数 |
| C / 中，软墙 | `probe.py:56-64,380-397` 可接寄存器/spill 字段，但 bridge 未接真实 trial profile；`462-482` 过滤缺失点后取末两点，可能跨 domain hole | 可连续解释的真实 spill 转折 | 接入逐 kernel 真 profile，保留 field_present/read_source；测试未知、拒绝、缺 spill 中间点不构成相邻软墙 |
| D / 中，简报 | `scanner:400-403` 仅在 contrast 非空时存 role latency；E 无 contrast；`bridge:384-392` 却读 `E_new` | E 结果能完整进入自然语言简报 | 正确结果统一保存 role latency；测试 E1 正式完成后简报含新点数值，失败不伪造值 |
| E / 中，排序研究 | `scanner:249-262` 是 span-first 排序，无层间轮转；优先入口 `161-169` 跳过 decision log；`264-275` 只算 RR 反事实；`430-447` 漏斗仅稀疏 F/G | 已实现完整 RR 政策对照、层间公平与八门漏斗 | 排序实验前增加实际政策选择、轮转与全入口日志；测试 requested/served、竞争集、首失败三态及真实选择 |
| F / 低，可选 | 默认 mint E1，见 `tokens:81,117-125`；`scanner:295-319` 对 E4 预留四槽却仅产一新点，E2 需手动指定 | 可直接开展 E4 新曲线研究 | 未实现前明确不支持 E4 或补齐四点协议；E2/E4 各测登记、槽数与执行点；不阻塞最小应用适配 |
| G / 条件性 | `store/run_store.py:139-167` 的恢复未还原 tokens、pending、probes、fresh intents | 精确重启后条件决策等价 | 若承诺该语义则加版本化快照及中断点回放；同进程同候选 continuation 已有，不能泛称 resume 全坏 |
| H / 低 | `config.py:738` 暴露 max_probe_attempts，但 `probe.py:41,108-110` 使用固定上限 | 非默认配置真的控制尝试数 | 透传配置或限制为常数；测试非 4 值；当前 shipping=4 的行为正确 |

**A 的关键区别**：沿被扫描轴移动不改变其搭档投影，不能误杀该轴已有证据。
真正需要拒绝的是 admission 之前其他搭档已改变的历史来源，而不是执行中把已承诺点替换成 live 参数。
准入后即使 incumbent 再变化，也保留原 committed 配置，并标注其在启动时已历史化。

**B 是有意实现选择与规格的冲突**，不是仅凭注释就能消失的小问题。
observe 不产生扫描 latency trial，也不投递扫描改写简报，但共享 screen facts 仍可影响后续生产拒绝。
影响幅度和具体轨迹变化尚未测量；应修正隔离政策和处理定义，不宣布全部历史数据无效。

**C 不表示所有环境的字段必然为 None**。
实验文档实际在 `143-156` 记录该次 warmup compile 中 `n_regs/n_spills` 为 None，硬墙仍可用。
应把编译硬可行性、经验 spill 转折、校准性能包络分开，而不是把缺失值补成零。
完整 E trial 仍会测量、产生 `TRIAL_DONE`、进入 TPE 和日志；D 只是结果简报遗漏，不能说 E 未执行。
E 的缺口先限制排序/RR 论文主张，不自动否定所有 C2 off/active 对照。
G 的精确重启等价性未被证明，也不是现有规格明确全面承诺的功能。

### 2.4 当前经验门应怎样解释？

`conditional/gates.py:91-133` 使用两端 fresh 重复和 `tau=max(4%,rho_F,rho_N)`。
通过门说明在这个经验重测包络下方向有支持，不是获得了 95% 置信区间。
每端两个 trial、每 trial 约 20 次内部 timing 不是 40 个独立应用请求样本。
应用的 p99 更不能复用这组数字作为正式尾延迟证据。
保留来源、方向、消费和失效生命周期；应用门限、重复单位与成本模型必须另行校准。

## 3. 问题一：完整应用怎样体现目标条件化的价值？

### 3.1 选择原则：展示偏好与行动，不展示“大模型”标签

KernelBench 继续承担低成本机制试验；它已有组合算子和完整网络任务，不是单 kernel 集合。
应用层补的是预训练状态、完整用户工作与质量约束，让读者看到优化究竟改善了什么体验。
主图应展示同一工作量中目标、证据、行动位置、完整产物指标的关系，允许没有排名反转。
不能用更短回答、更少扩散步、更低分辨率或更少请求伪装同任务优化。

| 角色 | 建议应用 | 最有解释力的图 | 不作为必要前置的内容 |
|---|---|---|---|
| 旗舰 | Qwen/Qwen3-8B，BF16，non-thinking，Transformers | 相同混合请求下 TTFT 与 TPOT 的候选矩阵，以及 prefill/decode 行动迁移 | 连续批处理服务、跨卡分片、从头写生成器 |
| 第二主应用 | SDXL-base 完整 pipeline | 完整出图时间与峰值内存的散点/Pareto 图及质量核验 | 新 diffusion runtime、强制三个 cap 或预设取舍 |
| 启动/替代 | BGE-reranker-v2-m3 固定 top-100；或 BGE-M3 dense-only | 完整 query 组延迟、排名质量与 padding/调度成本 | 必须增加第三篇应用故事 |
| 可选期限案例 | RT-DETR-R18，固定 640、decoder 3 与后处理 | deadline 达标率和所有帧完成成本 | 降分辨率、减层、丢难帧 |
| 后置 | FLUX.1-schnell | 只有前两者不足时才评估额外解释力 | 不因模型名称或规模而增加必选应用 |

### 3.2 Qwen3：新用户等第一字，已有用户要连续输出

建议冻结 Qwen3-8B revision `b968826d9c46dd6066d109eabc6255188de91218` [A01]。
Apache 2.0 许可见固定模型卡；non-gated 状态来自 2026-09-15 的模型 API 检索 [A01]。
API 元数据是日期限定的仓库状态，不是该不可变 README 或 checkpoint 的内容；本文没有下载或加载权重。
关闭 thinking，固定 tokenizer、chat template、sampling、输出与 reasoning token 政策。
不能通过更改思考模式、截断答案或减少输出 token 来获取虚假的速度优势。

机制对照可预登记两类请求：短 prompt 长回答，以及长文档短回答。
建议候选长度族为输入 512 至 2K、输出 128 至 512；长输入为 8K/16K、输出 64/256。
这些是待容量 pilot 筛选的设计值，不是已证明在某张 GPU 上可行的配置。
原生上下文上限 32768；启用 YaRN 扩展改变模型配置，不是纯 kernel 优化。

纯目标实验必须对两种目标使用**同一个混合请求集合、同样权重和共同约束**。
不能让 TTFT 臂只跑长 prompt，让 TPOT 臂只跑长输出，然后称目标造成选择差异。
prefill 和 decode 的计算与访存形态为可能的差异提供机制动机，但不预告赢家。
单 stream 或多 stream 是执行机制；请求并发是工作负载条件；continuous batching 是调度政策。
三者不同，首轮完整生成无需同时引入多 stream、并发请求与连续批处理。

定义请求 i 的可信起始时刻为 `s_i`，第 k 个输出 token 真正可获取的时刻为 `t_{i,k}`。
`TTFT_i = t_{i,1} - s_i`，不能用异步提交或 kernel enqueue 时刻替代 token 可获取时刻。
`ITL_{i,k} = t_{i,k} - t_{i,k-1}`，其中 `k>=2`。
`TPOT_i = (t_{i,n_i} - t_{i,1}) / (n_i-1)`，只在 `n_i>=2` 时有定义。
`Complete_i = t_{i,n_i} - s_i`，覆盖从开始到最后一个 token。
首轮目标可分别定义 `J_TTFT=Σ_i w_i TTFT_i` 与 `J_TPOT=Σ_i w_i TPOT_i`，权重预先冻结。
后者是请求级平均 TPOT 的加权聚合，绝不是 p99 ITL，也不等于 token 数加权的全局平均。
对单 token 合法完成，预登记 TPOT 适用子集及固定聚合规则，另报数量、TTFT 和完成时间。
失败、超时或未完成请求保留在总体有效性与失败率分母，不能当作单 token 请求悄悄排除。

两条验证轨道应并存：
1. **受控机制轨道**：固定生成长度，但每一步仍自回归使用自己的前序输出与 KV，不是 teacher forcing。
2. **主应用轨道**：自然 EOS、自由生成，在冻结 sampling 与质量政策下验证完整答案和体验。
teacher-forced tokens 可用于 logits、PPL 或局部数值诊断，不能冒充完整生成应用结果。
相同 seed 不保证数值重排后 token 完全一致；不同 token 轨迹须按预登记质量政策判断。
报告实际输出长度分布，避免“优化”通过回答更短而改善指标。

### 3.3 容量与权重语义：不许把两份 8B 同时塞进验证器

按模型元数据中的 8,190,735,360 个参数计算，裸 BF16 权重为 `8,190,735,360 × 2 / 2^30 ≈ 15.26 GiB` [A01]。
这是参数存储量推导，不是运行显存实测，更不是“24GB 一定放得下”的承诺。
Qwen3-8B config 给出 36 层、8 个 KV heads、head dimension 128 [A01]。
不压缩、不分片、不共享时，BF16 KV 每 token 为 `2 × 36 × 8 × 128 × 2 = 147456 bytes`，即 144 KiB。
32768 个驻留 token 仅 KV 就约 4.5 GiB，加约 15.26 GiB 权重，尚未计 activation、workspace、allocator 和编译开销。
batch 与多请求同时驻留会增加总 token 占用；不能由上述加法推出 24GB 适配保证。
先用容量 pilot 确定可行 profile，固定后才跨臂比较，不在候选间动态换 dtype、offload 或输入长度。

Llama-3.1-8B-Instruct 可替代，revision `0e9e39f249a16976918f6564b8830bc894c89659` [A02]。
其 gated 状态来自 2026-09-15 模型 API 检索，community license 见固定模型卡 [A02]；本次未取得权重，不把它作为无门槛默认项。
32 层、8 个 KV heads、head dimension 128 的架构依据见 Llama 3 论文 Table 3 [A02]。
据此在同样 BF16 条件下推导 128 KiB/token、32768 token 约 4 GiB KV，仍不构成容量保证。
现有 correctness 路径可能同时保有 reference 与 candidate；应用适配应改为顺序运行或有限 CPU golden traces。
checkpoint 不可变，reference 与 candidate 显式绑定同一状态，不靠相同初始化 seed 猜测权重一致。

### 3.4 Qwen 的真实补丁边界

建议固定 Transformers v4.57.3 commit `47b0e478f324b54f177ea7998a0791870fdd0324` [A03]。
其 `modeling_qwen3.py:166-240` 在 attention interface 前完成 Q/K norm、RoPE 与 KV update。
只替换 attention interface 不能自动融合这些上游步骤；若要融合，须显式扩大被允许的补丁区域。
记录每个 site 覆盖的层、phase、shape 分派、原实现与新实现，确认请求实际调用了补丁。
attention mask 的注册必须配套核验，官方接口文档提示未正确注册时可能得到 `None` mask [A04]。
无 mask 的错误实现即使很快也不是候选；增加 causal、padding、长上下文与 KV cache 负例。
vLLM/SGLang 的原生模型路径不会因为改了 HF 文件而自动使用该补丁。

### 3.5 SDXL：同一次完整出图，时间与内存如何选择？

建议采用 `stabilityai/stable-diffusion-xl-base-1.0`，模型来源见 [A05]。
拟议起点是 1024×1024、30 steps、guidance 5，冻结 prompts、seeds、scheduler 与组件 dtype 政策。
实际模型权重 revision 与全部组件 hash 在实施时另行冻结，本文不把模型仓库 URL 当不可变版本。
VAE upcast、offload、watermark 配置与后处理规则必须包含在契约中。
完整边界为 raw prompt、两个文本编码器、含 CFG 的 denoiser、每步 scheduler update、VAE decode、配置的 watermark 和 CPU 图像后处理。
text-to-image 主流程不包含 VAE encode，也不应凭空假定 pipeline 一定有 safety checker。
固定 pipeline 实现与内存说明见 Diffusers commit `759164b7ad116e091e9d3e222211c9aa27d835f6` [A06-A07]。

纯目标版本让所有候选在共同约束下分别最小化 warm image time 与 peak memory。
若比较“不同内存 cap 下最快”，改变的是可行集合，应标为约束实验而非纯目标排名翻转。
不强制三个 cap，不假设一定存在 trade-off；同时改善时间和内存的 Pareto 改善也应如实报告。
内存报告区分 allocated、reserved、常驻增量与可观测的设备占用，说明外部库分配的盲区。
不可把常驻预分配从基线中扣掉而使候选缓存免费；时间与内存可分 pass，但产物和协议必须一致。

SDPA/xFormers 是强基线，不应只打未经优化的 naive attention。
attention slicing 与 SDPA/xFormers 叠加可能变慢；该警告来自固定版 `pipeline_utils.py:2046-2081`，不是内存指南 [A12]。
内存指南讨论的 VAE tiling 可能改变色调，不能静默归为同质量；offload 改变传输政策，也须固定或明确成为处理因素 [A07]。
质量检查同时覆盖局部输出、latents 与最终感知/任务质量，不只要求像素恒等，也不只看 CLIP/FID。
全流程改善必须由完整出图计时证明，不能把 denoiser 某个 kernel 的收益直接当作端到端收益。

### 3.6 备选应用与已有机制的限制

BGE-reranker-v2-m3 固定每个 query 的 top-100 query-document pairs，测完整组完成，而不是挑一个 pair [A08]。
冻结精确 token IDs、query 截断与排序质量政策；比较 score 容差和最终排名质量。
官方 encoder-only reranker 已有按长度排序的推理代码，不能把 length sorting 写成新贡献 [A09]。
BGE-M3 的 dense/sparse/multi-vector 能力见固定模型卡；若采用 dense-only 就明确限定任务，不混入其他输出的成本变化 [A10]。
RT-DETR 的 640 与 decoder 3 依据限定为固定 `rtdetr_r18vd_6x_coco.yml` 及其 include 配置，不泛指所有变体 [A11]。
其 deadline 实验必须对所有帧计数，固定分辨率、decoder 与后处理，不丢弃超时帧。
这些备选只用于降低启动成本或补充期限语义，不扩展本次最低实现承诺。

vLLM v0.10.2 官方说明较小 chunk budget 有利 ITL、较大 budget 可能有利 TTFT，并给 Llama3.1-8B 例子 [P01]。
这是已有系统机制，不是本框架实测结果，更不意味着应把 batch scheduler 纳入本项目重写范围。
POD 将 prefill/decode attention 结合，涉及资源与 CTA 分配；其 8B 为 Llama3，不是 Llama3.1，使用两张 A100 80GB [P02]。
FlashInfer 的相关 Llama3.1-8B H100 实验使用 f16，不是本文拟议 BF16 [P03]。
MuxWise 是更近的应用控制参照，涉及 SLO 与 prefill/decode 资源控制，但未认证其 artifact 与本协议完全匹配 [P04]。
本次已核验来源尚未提供“指定 BF16 模型 artifact、共同请求轨迹与共同约束下，仅改变目标便导致候选排名反转”的直接证据；本文将其作为待检验假设，不主张此类结果在文献中不存在。
这些来源约束 C3 新颖性；不能移用作者收益，也不是要求重建他们的系统。

## 4. 问题三：三项贡献的有效直觉、过度主张与改写

### 4.1 贡献对照表

| 项 | 原始表述方向 | 有效直觉 | 必须删除的过度主张 | 建议中文 / English | 最近参照 | 所需新增证据 |
|---|---|---|---|---|---|---|
| C1 | 算子和硬件资源对齐，逼近物理峰值 | 资源约束限制结构动作，实体级诊断可减少无效改写 | 保证达到物理峰值；利用率越高目标越好；多个 kernel 极值拼成单一瓶颈 | **将算子级资源约束与结构改写方法对齐，并用可校准性能包络解释剩余空间。** / Align operator-level resource constraints with structural transformations using calibrated performance envelopes. | KernelBand、KernelSkill/Pro、SAAKE、KTT [P05-P09] | 硬可行性正确率、归属与校准、相对普通诊断的独立增量 |
| C2 | 资源墙上的调参斜率引导优化 | 固定搭档的受控程序响应可帮助选择下一次结构改写 | 第一次测单轴；斜率最大化利用率；预测新重写必获益；全局最优 | **以固定搭档下的程序旋钮目标响应连接资源边界与有益结构改写，并检验有限预算收益。** / Use fixed-partner program-knob responses to connect resource boundaries with beneficial structural rewrites under a finite budget. | KTT、坐标下降、Ansor、KernelBand [P05,P09-P11] | P2′ 校准、独立改写消融、全部 probe 成本与失败分母 |
| C3 | 根据任务目标优化完整应用 | 同一完整目标可改变应优先改写的 site、phase 与动作 | objective adapter 本身新颖；换 prompt 就证明新机制；任何目标都普适提升 | **在冻结应用产物与任务契约下，以目标响应证据选择改写位置、阶段与行动。** / Select rewrite sites, phases, and actions from objective-response evidence under a frozen application contract. | Ansor、AgentCompile、MaxKernel、POD、MuxWise [P02,P04,P10,P12-P13] | 正确目标底座上的 R/P/T 增量，强固定 site 与最近机制基线，完整应用独立测量 |

### 4.2 三种资源边界不能混为一个“峰值”

硬可行性回答 shared memory、launch 或其他硬限制是否允许执行，不能给出性能最优点。
calibrated roofline 或 sustainable envelope 解释在当前环境中可期待的性能范围，需要校准与误差披露。
经验 spill 转折来自特定 kernel、shape、搭档与编译/运行证据，不是硬不可行边界。
高 occupancy、高带宽占用或更多并行度都是诊断量，不是用户目标本身。
若应用 J 变差，即使资源利用率更高也不能接受；若利用率降低但 J 改善且约束满足，可以接受。

### 4.3 新颖性定位的两条合法路线

**三机制路线**：C1、C2、C3 各有清楚处理差异，且在共同预算下证明独立增量。
**默认路线**：C1/C2 两项主贡献，Qwen 与 SDXL 证明应用意义；C3 只作为工程与探索性分析。
若 evaluator-only 重排就选出同样产物，应认可这个简单答案，不强行构造第三贡献。
若目标 prompt 有效而 evidence priority 无增益，结论是目标表达有帮助，不是 C3 机制获证。
传统坐标搜索已经测单参数邻域；Ansor 已有任务目标敏感性与预算分配，不能写首次性口号。
AgentCompile 的区域组合与 MaxKernel 的冻结 harness 也限制“第一次在完整任务上优化”的说法。
前序文档已有这些机制的细分讨论，本文只收窄本框架可检验的增量，不做穷尽新颖性证明。

## 5. 问题四：具体模块怎样改，哪些只是后续选项？

### 5.1 最小数据模型：完整产物而非孤立 kernel 文件

以下路径全部是**拟议职责**，不存在“本文已创建这些文件”的含义；允许实施时合并小模块。
定义已接受补丁集 `P_t`，行动 `a=(site,phase,family,params,source_delta)`，新产物 `P'=P_t⊕a`。
权威评分是 `J(P', frozen_contract)`，不是只测 `source_delta` 对应的局部函数。
每次行动只改一个 site/phase；site group 若跨层共享，须明确列出全层覆盖范围。
所有已接受补丁随新动作一起加载，避免测新 kernel 时丢失先前优化或隐含组合冲突。
身份至少覆盖 checkpoint、工作负载、状态政策、实现版本、目标、约束、测量协议、补丁集和参数。
原始指标向量与派生 J 分开存；禁止把内存字节数塞入 `latency_ms` 以求少改接口。

优先级约定：P0 为最小可信应用所需，P1 为核心证据或后续 C3，P2 为可选研究/服务扩展。
该表是依赖地图，不是要求一次性实现所有模块。

| 模块，拟议路径或现有修改点 | 修改前 | 修改后输入 → 输出 | 复用 / 新增收益 | 依赖与优先级 |
|---|---|---|---|---|
| `models/application.py`，新 | TaskSpec 主要承载 source | model/workload/state 契约 → 不可变应用身份 | 新增可信状态定义，不另造模型库 | 无；P0 |
| `models/application_artifact.py`，新 | 候选源与参数是主要产物 | accepted patch set + delta → 完整 manifest/hash | 保留候选血缘，明确组合测量 | application；P0 |
| `objectives/spec.py`，新 | `robust_ms` 隐含权威口径 | metric/unit/direction/aggregation/constraints → 冻结目标 | 防止目标在搜索中漂移 | application；P0 |
| `objectives/policy.py`，新 | 延迟 best/tell/accept 分散 | raw metrics + validity → J/可行性/比较决策 | 共用权威比较，保留原指标 | spec；P0 |
| `applications/qwen3.py`，新 | 无预训练应用可信绑定 | 契约 + 完整产物 → 原 Transformers 生成调用 | 薄适配，控制权重、KV 与采样 | contracts/artifact/patch utilities，不依赖 runner；P0，首应用 |
| `applications/sdxl.py`，新 | 无完整出图契约 | 契约 + 完整产物 → 原 Diffusers pipeline | 薄适配，固定组件和完整边界 | contracts/artifact/patch utilities，不依赖 runner；P0，第二应用阶段 |
| `applications/patches.py`，新 | 候选替换不声明应用 site | allowlist + delta → 校验安装/卸载与覆盖证据 | 防止补丁未命中和越界 | artifact；P0 |
| `applications/runner.py`，新 | worker 以 forward 为主 | 拥有 episode 生命周期，调用注入的 adapter → 原始时序、内存、质量、成本 | 复用 worker/job 隔离，不重写 pipeline | contracts 与注入的 adapter 接口；P0，无 adapter 反向依赖 |
| `applications/attribution.py`，新 | 资源极值难绑定应用阶段 | trace + site manifest → phase/kernel 归属 | 为 C1/C2 提供可解释实体 | 先做最小归属 P0，细分 P1 |
| `models/core.py`，现有 | latency-centric trial | 兼容旧 TrialRecord → 应用指标与身份引用 | 保留 KernelBench 路径，避免单位混用 | objective/artifact；P0 |
| `ports.py`、`wiring.py`，现有 | legacy evaluator 组合 | application mode → adapter/evaluator 注入 | 复用原 composition root | contracts；P0 |
| `cli.py`，现有 | 原任务与配置入口 | 冻结 manifest → 可复查 run 输入 | 不允许 agent 运行中修改契约 | wiring；P0 |
| job/worker/evaluator，现有链 | 模型构造及 forward 计时 | 完整 episode → 指标、质量、cost ledger | 权重顺序验证与 reset 证明 | runner；P0 |
| TPE/stats/families/convergence，现有链 | 延迟排序及终止 | 同一个 J → tell/history/best/停止 | 目标贯通，不只替换 TPE 一处 | policy；P0 |
| orchestrator/conditional bridge/identity/gates，现有链 | 延迟来源与条件简报 | full-artifact identity + J response → 合格行动证据 | 复用 C4/token 生命周期，校准应用重复 | 核心 A/B/C 修复；P0/P1 |
| store/report，现有链 | 事件、延迟与恢复状态 | manifest/raw metrics/cost/来源 → 可重放证据与报告 | 明确失效与最终独立复测 | 全链；P0，精确重启按承诺追加 |
| `control/application_actions.py`，新 | 无有界应用位置策略 | 合格 site/phase 响应 → 下一改写行动 | 仅研究 C3 优先级，不造 learned allocator | 正确目标 + C1/C2；P1 |

### 5.2 三组工作必须分开排期

**核心修复**：A 当前搭档准入、B 隔离政策、C 真实软墙来源与相邻性、D 简报缺值。
**新应用功能**：状态和补丁身份、完整 episode、统一目标、质量与调用覆盖，独立于可选 E4。
**可选研究支架**：E 的 RR/轮转/完整漏斗、F 的 E4、按承诺需要的 G 重启等价性，以及 serving。
排序研究需要 E 完整，但普通应用绑定设计不需要先完成 RR；配置 H 可独立修正。
所有实现都应保持旧 KernelBench 默认路径兼容，并以回归测试证明，而不是凭架构图保证。

### 5.3 C3 的有界选择规则，不能变成另一个大平台

以下是**首版拟议协议**，用于使实验可解释，不声称规则本身新颖或已经实证调优。
先用 trusted baseline profile 选少数 site groups，将有序 site/phase 条目冻结为跨臂相同的 manifest。
高成本 hotspot 本身只是常见启发式，必须作为强基线，而不是被包装为贡献。

**来源与资格。** 每个完整 contrast 记录 site、phase、axis、精确 partners、当前完整 base artifact、J 和 measurement protocol。
comparable key 包含上述身份及冻结端点几何；每个 key 只使用按完成序号确定的最新完整 contrast。
最新完整来源包括 unresolved 或 negative，绝不是“最新通过门”的来源；不得复活旧的合格结果或挑选历史最高分。
incomplete 来源保留但不覆盖已有完整来源；上下文不匹配的记录不具有当前资格。
仅当前上下文适用、通过当前应用噪声门且尚未尝试的方向行动进入 supported 集合；outward/inward 均可。
latest-complete unresolved 不受支持，但仍可探索；负墙向响应若反向过门，可支持 inward，不改写原始符号。

**分数。** 首轮只覆盖 J 严格为正的最小化目标，由统一 application policy 计算方向化改善。
沿候选方向定义 `d=(J_before-J_after)/J_before`；两个值均来自完整应用端点，不是局部 kernel 代理。
重复数据的经验包络冻结为 `I_d=[min_{r,s}(1-J_after,s/J_before,r), max_{r,s}(1-J_after,s/J_before,r)]`，只使用该最新完整来源的合法重复。
supported 行动的排序分数取 `score=lower(I_d)>0`，并且必须先满足该应用预登记的重复质量与噪声门。
这是保守的经验相对改善分数，不是统计置信下界，也不是按任意旋钮间距归一化的导数或物理容量导数。
不跨 J、base artifact 或不兼容协议混排；其他目标方向、负值或零值需要未来显式政策，不把负目标直接代入此公式。

**确定性选择。** 外层 C 的 rewrite 轮数 `k` 从 1 开始，每次改写尝试后递增，失败也计轮，不因结果好坏重置。
当 `k mod 3=0` 时，在 unknown/无合格未尝试行动的 site/phase 池中，按冻结 manifest 顺序轮转选择下一个条目。
若该探索池为空，则使用 supported 排序；非探索轮也先使用 supported 排序。
supported 按正的保守 score 降序，相同分数按稳定 manifest 序、axis ID，再按 comparable key 稳定序打破剩余平局。
若无 eligible supported 行动，则退回完整 manifest 的轮转探索，不停止为未知位置提供尝试。
探索池与完整 manifest 各有一个从首条目前起始的游标，沿冻结顺序扫描；仅实际选中时移到所选条目之后。
策略、完成序号规则、k 与游标更新规则在运行前冻结；每三轮一次是拟议默认值，不是经验最优值，也不允许看结果后调参。

**防重复占用。** 在当前 base 下记录 `(site,phase,axis,latest_contrast_id)` 已尝试标记，开始尝试即标记，失败不退款。
没有新的完整 contrast 时，不再用同一证据行动获得 supported 优先权；剩余 unknown/无合格行动条目继续轮转。
即使所有来源都已尝试，也退回既有 manifest 探索，而不是让最强旧分数垄断或令其他位置饥饿。
全部尝试计入原预算，不新增 allocator 或免费探针；来源证据只决定优先级，不保证新结构改写获益。

**R/P/T 边界。** R 与 P 都按相同冻结 manifest 做固定 site/phase round-robin。
P 仅向所选 site/phase 的 rewriter 增加目标描述，不能改变所选位置或阶段；T 保留 P 的提示并独自替换为上述位置优先政策。
三臂均有 target-aware C1/C2、同一完整任务 J、相同 probe 配额与协议；T 不独享额外测量。
若 T 没有相对 P/R 的独立策略增量，C3 不作为主贡献。

正式 B 调参与应用 C2 的每个端点都测完整应用 episode。
更便宜的局部 probe 可以解释资源或排查实现，也可成为另一个明确命名的研究处理。
但不能偷偷用局部 kernel latency 充当应用 J，尤其不能仅因局部变慢就拒绝全局可能更好的候选。
只有完整正确性、共同约束及噪声政策共同通过，才接受 `P'` 并更新 `P_t`。
accepted context 变化后，旧 token/score 失去当前资格；代码提案可留存为待重新验证的候选。
这与 C4 准入后保持 committed 参数不矛盾：已在途证据按冻结上下文完成，但不能自动服务新上下文。

## 6. 完整流程与可信测量边界

### 6.1 从用户输入到可导出产物

```text
冻结 model / workload / state / objective / constraints / cost 契约
  → trusted baseline：质量、容量、完整计时、identity、补丁 call coverage
  → initial generator 提案 → A 参数化与正确性 repair
  → B 按完整 J 调参，内部复用条件 C4 来源与 E 行动
  → 外层 C：C1 资源解释 + C2 响应；可选 C3 选择 site/phase/action
  → C rewriter 生成一个 delta，构造 P' = P_t ⊕ a
  → A 校验/修复 → B 在完整产物上验证与调参
  → 完整 J + correctness + constraints + noise 决定接受或拒绝
  → 接受则更新 P_t 并失效旧上下文证据；否则保留 P_t
  → 停滞时进入原有 D novelty/family fallback，再回 A/B/C
  → 预算或收敛停止 → 独立 reload 全产物 → 导出与报告
```

初始候选来自原 initial generator，不把 D 改名为初始生成器。
D 保持停滞后的结构族探索职责；新族、repair 和条件测量都没有免费预算。
外层 C 可以先采用固定 site 顺序，等应用测量可靠后才加入 C3 优先级。
首次原型的目标是可信闭环，而不是让图中的每个可选模块同时上线。

### 6.2 一个 trial 的测量契约

最低风险方案是每 trial 一个 fresh process、一个 GPU 上的一份模型，episode 内执行多条固定请求。
模型 load 与 compile 可排除在 warm 指标外，但必须计入 search cost；首次执行成本另列。
reference 与 candidate 顺序运行，避免双份 8B 驻留；有限 golden traces 不构成形式化正确性证明。
每个 episode 显式重置 KV、RNG、buffers、补丁安装和输入状态，checkpoint 从不被候选修改。
以后若改 resident worker，先证明 reset 与 fresh-process 结果等价，再谈节省开销。
完整 token 时间应由可信 harness 观测真实可获取事件，计时同步与 CPU 开销定义跨臂一致。

搜索主标量先选 weighted TTFT、weighted per-request TPOT 或 warm image time，并校准适合应用的重复政策。
保留 legacy C4 的独立发现/验证设计与来源消费，不硬编码其 4% 或约 20 timings 为应用统计标准。
当前 `Q=floor(fB)` 是 legacy 剂量，不是 8B episode 的自动成本模型。
成本账本同时记录 GPU 时间、总墙钟、load、compile、probe、失败、repair 和最终确认成本。
不发明 GPU 天数与可行默认 batch；容量和成本 pilot 后再冻结实际预算。
最终尾延迟结论需要足量且适当独立的数据，不能用四个来源 trial 宣称 p99 改善。

## 7. 实验：保留旧证据，分阶段增加新问题

### 7.1 既有 v4 实验不撤销

保留 `experiment-plan-post-v4.1.md` 的 M1 off/active、N1 配对 off/off 和 P2′ 发现到留出的设计。
N1 噪声底仅适用于同机、共享冻结种子集与该配对协议，不应称整个任务的通用噪声底。
所有端到端读数考虑最终赢家血缘；按总 GPU 成本比较，不能只按 trial 数判断预算公平。
P2′ 应按来源自己的截止时刻重建边际预测，保留全部合格留出分母，不能只选过门案例。
本次发现先转化为协议 fixture 与处理定义修订，不要求取消、重启或把所有历史实验判为无效。
若 B 影响 observe，明确其处理含共享编译事实；若 E 未闭合，则暂缓实际 RR 因果主张。

### 7.2 新实验总表，全部为拟议

E0 至 E4 是本节实验 ID，不等于扫描协议 E1/E2/E4 的行动长度。
每项都记录完整失败与预算消耗；“通过”可以是测量可靠但方法无增益，不预设正结果。

| 实验 | 问题与固定变量 | 比较对象 | 主指标与辅助图 | 通过 / 负结果解释 | 前置条件 |
|---|---|---|---|---|---|
| E0 协议 fixture | v4 的准入、fresh、隔离与日志是否符合声明；固定小参数域与事件序列 | 当前行为对照明确规格；修复前后 fixture | 准入前搭档变化则拒绝旧来源、轴移动保留、committed 不变、fresh 真调用；生命周期与漏斗表 | 精确满足规则才放行相关主张；失败定位到议题，不推断全部运行无效 | 无 GPU 自检可先做；本次未执行 |
| E1 应用绑定 | noop 是否等价，补丁是否命中，状态是否正确；固定权重、请求、版本、reset | trusted baseline、noop patch、故意错误 KV/mask/state 负例 | 质量、调用覆盖、完整原始时序；覆盖矩阵与差异图 | noop 在校准范围，负例被拒绝；否则先修 harness，不开始收益实验 | P0 contracts/runner，容量 pilot |
| E2 候选×目标矩阵 | 同一产物库中目标是否改变选择；固定 workload、权重、约束、grammar | 固定候选离线重评分，对照按各 J 的自适应搜索 | 完整 TTFT/TPOT 或时间/内存矩阵；散点与交叉选优图 | 排名反转提供动机；无反转同样有效；仅重评分足够则不主张 C3 | E1，候选集合与全部目标预登记 |
| E3 C1/C2 机制 | 普通诊断、条件响应各带来什么；固定任务、J、grammar 与总成本 | C1 对普通诊断；C2 off/active/no-response；有信号后加同 raw probe 无摘要面板 | P2′、改写胜率、完整 J 对累计成本；校准图与轨迹 | 条件证据增量须超共同底座；无差异可为成本不值或功效不足 | E0；优先低成本任务，不要求每个 8B 实验全展开 |
| E4 C3 策略 | 证据位置政策是否有独立增量；固定有序 manifest、完整 J、target-aware C1/C2、probe 配额/协议、grammar 与总成本 | R/P 均固定 site/phase round-robin；P 仅加所选位置内的目标提示；T 保留 P 提示并独自换为 §5.3 政策 | 独立最终 J、anytime cost、行动分配；site/phase 图 | T 胜 P/R 才支持 C3；P 胜而 T 不胜仅支持提示价值；不事后调三轮规则 | E1/E2，可信 C1/C2；冻结来源、分数、k/游标与防重复规则 |
| 应用展示 | 方法是否改善真实完整任务；固定 Qwen/SDXL 质量与状态契约 | 强库基线、固定 site/hotspot、完整框架 | 自然 EOS 质量与 TTFT/TPOT；完整出图质量/时间/内存；成本图 | 必须完整应用获益且质量合格；局部快但全局不快记负结果 | 前面测量门通过，独立 reload |
| 可选 serving 2×2 | kernel 与 runtime 设置的作用能否分离；固定 trace/window/cohort | stock/candidate kernel × default/tuned runtime | 服务 TTFT/ITL p99、goodput、失败；因子效应图 | 只有交互/主效应可归因，不能把调 batch 的收益全算给 kernel | 补丁在目标 runtime 确实命中；足量独立请求 |

### 7.3 消融公平性与最近基线

E3 的 no-response 保持目标正确和普通资源诊断，仅关闭待检验响应通道，不删除共有能力。
“同 raw probe 无摘要”面板专门区分多看数据与解释方式，不必让所有昂贵应用都复制全套臂。
E4 的 R/P/T 都已有 target-aware C1/C2 与正确 evaluator，probe 配额、协议和完整任务 J 相同。
R/P 都固定 round-robin 选择 site/phase；P 只在已选位置内增加目标描述，不能借 prompt 改选位置。
T 保留 P 的提示，只将位置优先级替换为 §5.3 的冻结政策，不独享额外测量、候选 grammar 或免费尝试。
各臂节省下来的开销可用于正常搜索，不通过闲置 baseline 预算帮助 T 获胜。
论文新颖性声明前，至少加入代表性的强固定 site sweep/hotspot 与最近机制基线。
不能只与弱随机 site 或未优化 eager 实现比较，就宣称超越已有资源/应用控制方法。

候选库矩阵是离线选择证据，自适应搜索轨迹是算法证据，二者需要分开画图和命名。
纯目标变化使用共同约束；memory-cap 实验单列，不通过改变可行集合制造“偏好反转”。
最终 holdout 独立于搜索选择，保留预登记任务与 seed，不只展示最好一次 trial。
若条件来源很少或竞争集始终只有一个，报告暴露不足或排序不可识别，不写策略失败或成功。
统计报告区分搜索 seed、模型 sampling seed、请求样本与 trial 内重复，不把嵌套样本当独立样本。

### 7.4 服务实验只在需要时做

2×2 是 kernel 与 runtime 两个明确因素，不是任选四组参数跑四次。
冻结到达 trace、计时窗口与请求 cohort；窗口结束后 drain，记录开始与结束归属政策。
失败、超时和未完成请求全部计入，goodput 需结合 SLO，不能只报低负载吞吐。
服务级 TTFT/ITL p99 是后续验证，不是首次证明完整应用价值的必要前置。
HF 补丁若未进入服务 runtime，就不能把两者放进这张因子表，须先通过调用覆盖门。

## 8. 推荐实施顺序与可观察交付

| 阶段 | 做什么 | 修改前 → 修改后可观察结果 | 放行边界 |
|---|---|---|---|
| 0 核心可信性 | 修 A，裁定 B，补 D；为 C 接真实 profile 与邻接测试 | 仅有注释/稀疏日志 → 可复现的准入、隔离与来源 fixture | 保留旧实验，按受影响主张逐项放行 |
| 1 身份与目标 | P0 application/artifact/objective 接线，保持 KernelBench 兼容 | source/latency 隐含契约 → manifest、raw metrics 与统一 J 可读 | 不等 E4/RR；不启动收益宣传 |
| 2 单应用闭环 | Qwen noop、权重顺序验证、KV/mask 负例、容量 pilot | 能运行模型 → 确认实际补丁与完整请求语义 | 冻结可行 profile 后才比较目标 |
| 3 目标矩阵与 C2 | E2/E3，小任务先测机制，Qwen 做完整生成验证 | 局部假设 → 完整 J 矩阵与付费响应证据 | 没有新颖性增量也可交付工程结论 |
| 4 第二应用 | SDXL 完整 pipeline 与强 attention 基线 | 单一生成故事 → 不同资源目标的完整质量/成本图 | 不强求排名翻转或多个 cap |
| 5 可选 C3 | application_actions 与 E4 R/P/T、强位置基线 | 固定位置 → 可归因的位置/阶段选择增量 | 无增量则维持两主贡献 |
| 6 可选扩展 | 实际 RR、扫描 E4、精确重启、serving | 声明缺口 → 专项承诺及对应测试 | 按研究需要追加，不绑架最小交付 |

这不是 GPU 排程承诺。历史 4090 信息不能替代容量、质量、噪声和 episode 成本 pilot。
若 24GB 容量限制迫使选择较短 profile，应在跨臂前冻结并如实声明，不边跑边缩任务。
优先决策是“修关键协议、做 Qwen 薄适配与统一目标、再接 SDXL”，不是全面重构优化器。

## 附录 A. 来源索引与核验深度

### A.1 本地源码与设计材料

本文直接读取前序研究文档、当前实验安排、v4.1 规格及关键 conditional 源码，并综合独立静态核读结果。
源码事实按 §2 的历史/最终版本锚点限定范围；最终 bridge 日志差异核对不等于全面重审新增脚本与结果。
没有扫描根目录数据库、结果目录或进行完整安全审计；同期其他提交不属于本文交付。
规格：[`implementation-conditional-scan-v4.1.md`](implementation-conditional-scan-v4.1.md)，尤其 `68-72,116-118,188-193`。
实验：[`experiment-plan-post-v4.1.md`](experiment-plan-post-v4.1.md)，最终快照 159 行版；历史测试和冒烟见 `143-158`，配对边界见 `119-133`。
前序贡献与文献语境：[`v3 研究文档`](../../v3/docs/research-task-objective-conditioned-optimization.md)，尤其 §2 和附录 A.3。
这些资料支持实现存在性与研究设计，不支持未经测量的加速、触发率或所有应用可行性结论。

### A.2 模型卡与固定实现，非性能结果

- [A01] Qwen3-8B 固定模型卡：https://huggingface.co/Qwen/Qwen3-8B/blob/b968826d9c46dd6066d109eabc6255188de91218/README.md
  配置：https://huggingface.co/Qwen/Qwen3-8B/blob/b968826d9c46dd6066d109eabc6255188de91218/config.json
  仓库 API 元数据（2026-09-15 检索，non-gated 与参数数量依据；不是不可变 checkpoint 内容）：https://huggingface.co/api/models/Qwen/Qwen3-8B
- [A02] Llama3.1-8B-Instruct 固定模型卡：https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct/blob/0e9e39f249a16976918f6564b8830bc894c89659/README.md
  gated 状态的 API 依据（2026-09-15 检索，日期限定）：https://huggingface.co/api/models/meta-llama/Llama-3.1-8B-Instruct
  架构依据：Llama 3 论文 Table 3，32 层、8 KV heads、head dimension 128：https://ar5iv.labs.arxiv.org/html/2407.21783
- [A03] Transformers v4.57.3 Qwen3 实现：https://github.com/huggingface/transformers/blob/47b0e478f324b54f177ea7998a0791870fdd0324/src/transformers/models/qwen3/modeling_qwen3.py
- [A04] 同版本 attention interface 与 mask 注册：https://github.com/huggingface/transformers/blob/47b0e478f324b54f177ea7998a0791870fdd0324/docs/source/en/attention_interface.md
- [A05] SDXL-base 模型卡：https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0
- [A06] 固定 Diffusers 完整 pipeline：https://github.com/huggingface/diffusers/blob/759164b7ad116e091e9d3e222211c9aa27d835f6/src/diffusers/pipelines/stable_diffusion_xl/pipeline_stable_diffusion_xl.py
- [A07] 同版本内存优化指南：https://github.com/huggingface/diffusers/blob/759164b7ad116e091e9d3e222211c9aa27d835f6/docs/source/en/optimization/memory.md
- [A08] BGE-reranker-v2-m3 固定模型卡：https://huggingface.co/BAAI/bge-reranker-v2-m3/blob/953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e/README.md
- [A09] FlagEmbedding encoder-only reranker 的长度排序，固定代码：https://github.com/FlagOpen/FlagEmbedding/blob/fd1a2bdf69488ffebe0327999d4400d8c8058a0b/FlagEmbedding/inference/reranker/encoder_only/base.py
- [A10] BGE-M3 固定模型卡，dense/sparse/multi-vector 任务边界：https://huggingface.co/BAAI/bge-m3/blob/5617a9f61b028005a4858fdac845db406aefb181/README.md
- [A11] RT-DETR 指定 R18 变体配置：https://github.com/lyuwenyu/RT-DETR/blob/29320b6fd828f8e0987a71426cf2d961b09dfed7/rtdetr_pytorch/configs/rtdetr/rtdetr_r18vd_6x_coco.yml
  所含基础配置：https://github.com/lyuwenyu/RT-DETR/blob/29320b6fd828f8e0987a71426cf2d961b09dfed7/rtdetr_pytorch/configs/rtdetr/include/rtdetr_r50vd.yml
- [A12] Diffusers attention slicing 与 SDPA/xFormers 组合警告，固定实现文档：https://github.com/huggingface/diffusers/blob/759164b7ad116e091e9d3e222211c9aa27d835f6/src/diffusers/pipelines/pipeline_utils.py#L2046-L2081

### A.3 官方机制、论文与选定代码

本节继承已提供的一手研究证据及前序文档的机制核对，不表示本次再次全文检索或复现论文。
- [P01] vLLM v0.10.2 优化说明，固定 commit：https://github.com/vllm-project/vllm/blob/01efc7ef781391e744ed08c3292817a773d654e6/docs/configuration/optimization.md
- [P02] POD，指定 v2，prefill/decode attention 与资源调度：https://arxiv.org/html/2410.18038v2
- [P03] FlashInfer，指定 v2，相关模型、H100 与 f16 实验条件：https://arxiv.org/html/2501.01005v2
- [P04] MuxWise，指定 v3，SLO 与应用资源控制：https://arxiv.org/html/2504.14489v3
- [P05] KernelBand，资源策略屏蔽、headroom 与历史奖励：https://arxiv.org/html/2511.18868v2
  固定代码示例：https://github.com/TongmingLAIC/KernelBand/blob/89dbcb25a3ead1c527d3726ec5efa86a26d4505c/kernelband/mab/masking.py
- [P06] KernelSkill，资源与结构规则：https://arxiv.org/html/2603.10085v1
- [P07] KernelPro，profiler、MCTS 与 energy 路线：https://arxiv.org/html/2606.26453v2
- [P08] SAAKE，硬件模型敏感性与搜索初始化：https://doi.org/10.1145/3192366.3192397
- [P09] KTT，counter 引导参数搜索：https://ar5iv.labs.arxiv.org/html/2102.05297
- [P10] Ansor 论文：https://www.usenix.org/system/files/osdi20-zheng.pdf
  后续 TVM v0.8.0 的 objective_func 敏感性实现，不全部倒归原论文：https://github.com/apache/tvm/blob/7b3a22e465dd6aca4729504a19beb4bc23312755/python/tvm/auto_scheduler/task_scheduler.py
- [P11] 固定 TorchInductor 坐标下降实现：https://github.com/pytorch/pytorch/blob/ba56102387ef21a3b04b357e5b183d48f0afefc7/torch/_inductor/runtime/coordinate_descent_tuner.py
  Droplet 实测坐标邻域论文：https://arxiv.org/html/2406.20037v1
- [P12] AgentCompile，区域模板及组合路线：https://arxiv.org/html/2606.07665v1
- [P13] MaxKernel，TPU harness 与冻结边界：https://arxiv.org/html/2609.04523v1
  选定 worker 代码：https://github.com/AI-Hypercomputer/accelerator-agents/blob/21a1d4174cabe08bc80f62afe3ed207b578e50f3/MaxKernel/auto_search/worker.py

## 最终建议与未决边界

对问题一，采用 Qwen3-8B 完整生成与 SDXL 完整出图，重点呈现目标如何改变行动及用户体验。
对问题二，承认 v4 条件/fresh 链已实现，再按 A 至 H 精确修补协议与表述，不沿用旧 v3 总评。
对问题三，默认保留 C1/C2 两项主贡献，只有 R/P/T 与强基线支持时才提升 C3。
对问题四，按核心修复、应用身份与目标、完整测量、机制验证、可选策略的顺序推进，不重建平台。
未决项是隔离政策裁定、应用重复标定、SDXL 权重版本冻结、容量 profile、真实质量与收益，以及 C3 是否有独立增量。
本文交付仅修改此设计文档；同期其他代码提交已按快照单列，不属于本文工作，没有在本次进行代码修复、测试运行、GPU 实验、权重下载或全源码安全批准。
静态问题已获独立源码核读支持，但运行影响仍未知；负结果与无增量结论都属于合法研究产出。
