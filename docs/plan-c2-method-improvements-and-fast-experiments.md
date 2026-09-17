# C2 方法改进与四卡压缩实验方案

## 1. 状态、目标与边界

本文是待实施方案，不是执行报告；本次只新增文档，没有改代码或启动 GPU、SSH、模型调用。
目标是先修正方法语义与交接，再以两台双 4090 的四个独占槽验证是否值得继续投入。
全文标记口径：**事实**来自既有材料或当前源码；**拟改**尚未实现；**实验**尚未执行；**估算**不是时长保证。
保留既有报告、旧计划及 T6 NOT RUN；无关的既有 framework 8B 任务仍未完成，不由本文接管。
不建平台，不增加恢复、身份、schema 迁移或强制账本体系，不做安全审计。
不禁 dtype，不降低 B40，不跨候选复用性能热启动，不追加第三轮或 12h 扩展。

## 2. 已有证据与不能推出的结论

| 类型 | 已有观察 | 本方案采用的解释边界 |
|---|---|---|
| 事实 | N256/SM64 为 5.199359894 ms，同卡 G0 为 5.019135952 ms | 漏域有实际后果，但补点尚未追回 G0 |
| 事实 | E1 不含 SM64；E2 含该配置但从未试到 N256+SM64 | 有限 B40 覆盖不足，不等于空间内没有好点 |
| 事实 | 纯 N128→N256 默认背景对照变慢约 10.8455% | 不把多参数好点的收益全部归于 N |
| 事实 | native retune 六个 study 保留默认 witnesses，结构 B 三对均更快 | 这是 candidate＋space 比较，不是 C2 单因素因果证明 |
| 事实 | T4 信息门槛通过，T5 两任务共同时间比较均失败 | 信息阶段收益不保证闭环净收益 |
| 事实 | task43 G0 的 projection/cast fusion 改善了全任务数据流 | 资源峰值不必来自延迟瓶颈，不以资源最小化为目标 |
| 事实 | 旧 adapter 的 `failed_hypotheses=[]`，新轮次未完整接收慢子代经验 | 修复已有反馈通道，而非发明第二轮专用策略 |
| 源码事实 | native 实验直接 `_tune`，绕过 expansion；S7/scanner 被关闭 | 不能将这些实验称为整套 native 框架检验 |
| 继承线索 | 40/96 个数值坐标落在声明域端点 | 只描述域边界，不能证明接近物理资源墙 |

40/96 是前序讨论的描述性计数，三份指定报告没有可独立复算的同名表，本文不冒充重新测量。
即使该计数无误，也必须区分声明 choices 边界、条件编译拒绝与真实硬件因果限制。
N256 的可运行性只在已测配套参数下成立；其他 N256 组合仍有 shared-memory 拒绝。
两次生成重复不保证统计识别；同卡 finals 只控制部分设备差异，不能消除生成随机性。
证据入口：[分阶段结果](result-c2-staged-program.md)、[原生重调参](result-c2-native-retune.md)、[解锁区域复核](result-c2-unlocked-region.md)。

## 3. 方法优先级

| 优先级 | 项目 | 机制、机会与范围 | 成本与验收方向 |
|---|---|---|---|
| P0 | 语义与全任务推理 | conditional response 不冒充 compile wall；允许融合、物化、布局、调度等全任务机会 | 小幅 prompt/input 改动；无响应也能普通结构推理 |
| P0 | rewrite intent 交接 | 把已有 `change_summary` 等意图交给 parameterizer，减少声称区域未表达 | 可选 advisory 输入；说明表达或遗漏，不硬编码轴 |
| P0 | 未晋级子代反馈 | 使用已有 failed-hypotheses 路径传达已试动作与真实结果 | adapter 接线；不把未试建议写成失败 |
| P1 | accepted-candidate 续接入口 | 一次接入 tuning、stats 和可选 expansion，生产与实验共用 | 小范围抽取，避免重复参数化和 witness 初始化 |
| P2 | 有界 conditional 后续 | 在现有 planner 能请求域内条件对照时测 native off/active，contrast 在实验内构造 | 最多一个分支；桥接修正超过 2 工程小时即跳过 C |
| P2 | 提前扩域提案 | 首次参数化表达预期区域，优先现有域内 conditional neighbors | 真正 mid-B40 开域延后，不冒充已实现 |

在现有 analyst `summary` 与 rewrite `change_summary` 中简短交代机制、机会、成本、适用范围。
不新设强制数值证明、引用完整性、region hardgate；“尚未知道”是有效回答，不应迫使模型编造。
评价标准仍是完整任务的正确性与 retune 后表现，不要求新结构在旧默认参数点先赢。
保持原质量规则 `dual_witness_relaxed`、FP64 rescue 口径、backend 选择和现有约束。

## 4. 拟改接口与有限实施文件图

以下 `src/` 和 `tests/` 路径均相对 `v5/`；接口名为建议，不声称已存在。

| 文件 | 现有接口与拟改内容 | 明确不做 |
|---|---|---|
| `src/kernel_optimizer/agents/modules.py` | `RewriterInputs` 分离响应与 wall；`ParameterizerInputs` 加可选 intent；调整 analyst/rewriter prompt | 不改输出为强制证明 schema，不扩大 backend/dtype 限制 |
| `src/kernel_optimizer/agents/task_rewriter.py` | 只作为原生符号、单位、未知值和全任务语义参考 | 默认只读，不迁移 direct-task 架构 |
| `src/kernel_optimizer/control/orchestrator.py` | `_candidate_pipeline` 与已接受候选的共用续接；真实 wall 保留；反馈/intent 接线 | 不重写控制器或恢复路径，不新建或宣称已有完整 action router |
| `scripts/experiments/c2_local_agents.py` | 拆开一次 rewrite 与一次 parameterize 的调用边界，支持 P/H 共用 rewrite | 不用 `wall_text=brief`，不生成候选池 |
| `scripts/experiments/c2_retune.py` | `tune_existing` 在 validation 后调用共用续接，显式选择实验开关 | 不再次调用 parameterizer，不另写 witness/cache 初始化 |
| `scripts/experiments/c2_closed_loop.py` | 两轮真实反馈、当前父传播、共同 ready-time 与截止规则 | 不追加轮次，不用 finals 决定父晋级 |
| `scripts/experiments/c2_method_program.py`（拟新增） | 最小入口调度 core/A/B/C 与四槽，复用既有实验 helpers 和新共用 native 续接 | 本次不创建代码，不建新 orchestrator 或预算平台 |
| `src/kernel_optimizer/conditional/{scanner,bridge,probe}.py` | 仅 C 被选中且所选模块确需小修时改对应文件 | 不批量修 E2/E4、递归 token 或新独立 axis selector |

### 4.1 响应与编译墙分开

拟给 `RewriterInputs` 增加可选 `conditional_response_text`，放入独立的响应输入文件。
保留 `wall_text` 给实际 compiler wall 输入，不用响应文本覆盖它；无真实墙就传空值。
`orchestrator._stats_and_analysis` 当前以 conditioned brief 覆盖 wall text，拟改为并列传递。
已有 soft-wall 信息另作明确标签，不能以一次 spill 观测冒充编译拒绝。
响应说明固定配套参数、原始 objective 方向和单位，名义轴只有 contrast，不伪装成数值 slope。
缺失、失败、未覆盖轴均为未知；12 endpoints 最多覆盖六轴，不代表全域资源调查。
L 保留旧的局部资源/受阻方向推理框架，但同样修正输入来源标签，不故意保留虚假编译墙陈述。
G0/P/H 使用全任务推理包：reference、运行语义、普通历史资源、局部观测的适用范围。

### 4.2 intent 是建议，不是准入硬门槛

拟增加 `rewrite_intent: str | None = None`，来源是该次 rewrite 已有 summary 与结构意图，不新增一轮模型分析。
内容包括结构动作、预期可达区域及配套条件；没有明确区域时可以只描述数据流目标。
parameterizer 在现有说明中简短说明意图如何进入 domains/defaults/constraints，或为何省略、未知。
不强迫输出特定 N、SM、dtype，不按文字与域逐项一致设硬拒绝，不要求每条都带 citation。
空 intent 保持旧调用兼容；G0 也获得自己 rewrite 的 intent，而不是取得 H 的 fresh 信息。
H 与 P 共用参数化之前的同一份 rewrite 字节及其 summary，各自只参数化一次。
参数化本身可能改写 kernel body，所以 H/P 最终源码不同不自动意味着交接无效。

### 4.3 已有反馈与 native 续接

恢复现有失败反馈通道，记录已试结构、未晋级原因、实际 tuning 结果及保留的父状态。
区分 validation 拒绝、有效但慢、正确性失败、未尝试建议；后者不得冒充已否证假设。
四臂历史输入一致，不从其他臂偷看新产物；同一臂下一轮才接收自己的真实反馈。
在 orchestrator 内建议一个 `_continue_accepted_candidate` 边界，接收已注册候选及接受结果/已有状态。
将原有 accepted-witness 到 anchors/cache 的转换归入唯一共用路径，再调用 `_tune`、stats 与可选 expansion。
生产 pipeline 和 retune adapter 都进入该路径，不调用 `_candidate_pipeline` 去二次参数化。
已恢复状态使用既有 anchors/cache 语义，不新造恢复层；stats 的数值计算与是否调用 analyst 显式分开。
核心实验不额外调用 post-tune analyst，以保持第 6 节的模型调用数；native 扩域仍用原 eligibility 判定。
不复制 witness 循环；扩域 validation 的 witnesses 也复用相同转换，避免两个版本逐渐分叉。
保留原始 default witness 和正常第二见证点；B40 是每个 space 的预算，不是删掉 witnesses 后再给 40。
保留合法且兼容的 prior-best anchor；参数合法不充分，缓存复用还要求旧配置下源码等价。
源码等价不能确认时，用现有 evaluator 重测该 anchor，不拿旧延迟代表新源码，更不跨候选 warmstart。
native tuning 完成选择与父保留后才做 finals；final 分数只用于报告，永不重新选优。

## 5. 并行实施、依赖与未来验收

本节是未来工作顺序，不是本次执行清单；不安排大型最终复审波次。

| 所有者 | 独占编辑范围 | 依赖与完成条件 |
|---|---|---|
| I：语义负责人 | `modules.py` | 先确定 response/intent 可选输入及 summary 约定；此文件只允许一个编辑者 |
| II：核心负责人 | `control/orchestrator.py` | 可先抽续接，接线依赖 I 接口；保留 defaults、anchors、stats 与 native 选择 |
| III：adapter 负责人 | 三个既有实验文件及拟新增 `c2_method_program.py` | 接口确认后接入；独占最小入口，不复制 helper、参数化或 orchestrator |
| IV：测试负责人 | 下表新测试文件 | 与 I/II 并行用 mock 写契约测试，文件互不重叠 |
| V：条件模块负责人 | 仅被选中所需的 conditional 文件 | 核心接口稳定后检查 C 就绪性，最多 2 工程小时，不影响主线 |

顺序为接口约定 → 语义与核心并行 → adapter 接线 → CPU 契约测试 → 固定实验输入 → 核心实验。
测试负责人可拆给多个 agent，但每个测试文件单一所有者；充足 agent 不等于多人同时修改 `modules.py`。

| 拟新增测试路径 | 最小验收内容 |
|---|---|
| `tests/test_c2_method_prompts.py` | 响应不是 compiler wall；真实 wall 不丢失；全任务 summary；intent 缺省兼容及遗漏说明 |
| `tests/test_c2_accepted_continuation.py` | validation 一次、参数化不重复、默认 witnesses 保留、B40、兼容 anchor 与重测分支 |
| `tests/test_c2_method_protocol.py` | 四 cell/四 arm；P/H rewrite 同源；12 链/16 参数化/48 acquisitions；无效不替换 |
| `tests/test_c2_method_program.py` | phase/slot/input 匹配、B/C 原始子集及 N/A、A ready/终测分离、四槽顺序与块计数 |
| `tests/test_c2_method_feedback.py` | 慢子代反馈、真实父传播、未尝试不记失败、ready 截点、final 不重新选优 |
| `tests/test_c2_method_conditional_budget.py` | 仅 C 所需：E1 可达、Q8、两 dispatch、120s、等待/重试/取消均计费 |

未来在 `D:/Pyhon_projects/opop/v5` 用已有 `.venv-v5` 执行以下 PowerShell 命令，不创建环境：

```powershell
$env:PYTHONPATH='D:/Pyhon_projects/opop/v5/src;D:/Pyhon_projects/opop/v5'
& ./.venv-v5/Scripts/python.exe -c "import kernel_optimizer; print(kernel_optimizer.__file__)"
& ./.venv-v5/Scripts/python.exe -m pytest -q tests/test_c2_method_prompts.py tests/test_c2_accepted_continuation.py tests/test_c2_method_protocol.py tests/test_c2_method_program.py tests/test_c2_method_feedback.py tests/test_c2_retune.py tests/test_c2_closed_loop.py tests/test_v5_hypothesis_failed_memory_attribution.py
```

若选择 C，再运行 `& ./.venv-v5/Scripts/python.exe -m pytest -q tests/test_c2_method_conditional_budget.py tests/test_v41_conditional_scan.py`。
这些新路径尚不存在；实施时先写失败用例，再完成对应改动，本次没有运行不存在的测试。
对未来改动文件做现有诊断与相关回归即可，不因写方案而加依赖、启动服务或扩建验证设施。

## 6. 核心实验：四 cell、四 arm、16 个 B40 机会

**实验定位：有界组件检验，不是完整框架测试。** expansion、S7、scanner 全部 OFF，以隔离提示效果。
cell 定义为一个任务的原始 P1 父状态与一次 fresh generation rep；task43/task21 各 rep0/rep1。
两个 rep 都从字节相同的各自原始 P1 出发，不接续另一个 rep 或历史 T4 winner。

| arm | analyst/rewriter 语义 | fresh responses | intent 给 parameterizer | rewrite 来源 |
|---|---|---|---|---|
| G0 | 修正后的全任务 prompt | 无 | 有，来自自身 rewrite | 独立 G0 链 |
| L | 已修复事实标签的 legacy 局部推理对照 | 有 | 无 | 独立 L 链 |
| P | 修正后的全任务 prompt | 有 | 无 | 独立 P 链 |
| H | 与 P 相同 | 同 P | 有 | **P 的同一份参数化前 rewrite** |

每臂接收完全相同的普通历史资源、reference、运行语义和恢复后的反馈接线。
P−L 检验全任务推理 prompt package，不能隔离错误标签修复，因为两者都修复了标签。
历史误标按语义正确性问题用 fixtures 验收，不需要额外 GPU 实验重新证明该 bug。
H−P 检验 intent handoff，但两次 parameterizer 的随机性仍在，不能称确定性单变量结果。
H−G0 比较匹配通用改进后的 fresh 信息包效果，仍包含新生成轨迹的随机性。
G0 无 fresh responses 不等于无历史资源；L 无 intent 不等于剥夺原本可见源码。

| 计数项 | 计算 | 总量 |
|---|---|---:|
| cell | 2 任务 × 2 fresh reps | 4 |
| endpoint acquisition | 每 cell 一批 12，L/P/H 共用 | 48 |
| fresh analyst→rewriter 链 | 每 cell G0/L/P 三链 | 12 |
| parameterization | 每 cell G0/L/P/H 各一次 | 16 |
| B40 机会 | 4 cell × 4 arm | 16 |
| 完整 tuner 记录上限 | 16 × 40，含 witness 复用 | 640 |

每次固定一个 proposal，不 best-of、不候选彩票、不用额外 quota 替换无效方案。
原调用的有界 transport/格式重试保留并计费，但不得将拒绝变成新结构 draw 或择优池。
sampler/evaluation/validator seed 均为 0；两个 rep 是独立调用，不承诺 LLM seed 可复现。
冻结父源码和普通历史可复用，四批响应必须 fresh；历史 GPU 测量不能作为本次性能结果。
P/H 共用 rewrite 只为控制结构输入，不表示部署 H 时生成成本为零。

## 7. 四卡排程与同卡终测

| 槽位 | 物理位置 | 固定 cell | arm 顺序 |
|---|---|---|---|
| A0 | HOST-A GPU0 | task43 rep0 | G0, L, P, H |
| A1 | HOST-A GPU1 | task21 rep0 | L, H, G0, P |
| B0 | HOST-B GPU0 | task21 rep1 | P, G0, H, L |
| B1 | HOST-B GPU1 | task43 rep1 | H, P, L, G0 |

每 GPU 同时只有一个计时 job；acquisition、witness、probe 与 finals 都不得与该卡其他计时交叠。
模型调用提前流水化；独立 cell 和 G0/L/P 链并行，但 acquisition→分析→rewrite→parameterize 依赖不打乱。
即使某槽 H 先于 P 调优，P 的 rewrite 也必须先完成，H 不能因此另起 rewrite。
P/H 两次参数化可在共享 rewrite 到达后并行；廉价模型并发用于缩短等待，不扩大 proposal 数。
两 GPU 的 CPU compilation 可能争抢同一主机资源，不假设四卡提供严格 4 倍加速。
所有比较留在该 cell 的同一物理 GPU，不拿跨卡 raw latency 直接决定赢家。
native 选择冻结后，对父与 G0/L/P/H 的选中 incumbents 各测 3 个 full blocks。
每块保持 100 个性能样本、5 次 correctness；不缩短 finals 来填时间预算。
轮转顺序：首块按父加该槽 arm 顺序，第二块反序，第三块循环移位，事前固定。
四 cell 完整时为 4 × 5 × 3 = 60 块，6,000 个性能样本、300 次 correctness。
即使某臂保留父，也以该臂 incumbent 名义保留新测块，不把历史父成绩当本次结果。
无效 proposal 仍占机会；未完整 asked=40 的 study 不称完成 B40；缺失 final 标为 censored/inconclusive。

### 7.1 未来最小入口（实现后才可用，本次不执行）

拟定 CLI：`--phase core|A|B|C --slot A0|A1|B0|B1 --config PATH --inputs PATH --output PATH`。
`--inputs` 用既有 JSON 表达原始 parent bundle 或前阶段输出及其来源，不建 manifest/freeze 平台。
core 的 task/rep/arm 顺序严格取上表；A 取第 9 节槽表，B/C 保留原 core cell；输入路径必须匹配所选父。
入口核对 phase/slot/parent 对应关系，不隐式自动挑选父或更好的候选；N/A 槽只记录原因，不发 GPU 工作。
HOST-A 示例仅为未来命令模板：尖括号替换成执行时从既有环境/旧 run 取得并确认存在的绝对路径，不声称这些占位文件已存在。

```bash
PYTHONPATH=/root/autodl-tmp/c2-local-experiment/code/v5/src:/root/autodl-tmp/c2-local-experiment/code/v5 "<ORCH_VENV_ABSOLUTE_PATH>/bin/python" -m scripts.experiments.c2_method_program --phase core --slot A0 --config "<EXISTING_HOST_A_CONFIG_PATH>" --inputs "<EXISTING_TASK43_P1_BUNDLE_PATH>" --output /root/autodl-tmp/c2-method-improvements/core/A0
```

示例 interpreter 必须是 HOST-A 的已有 `orch-venv`；其 worker 仍用 `kernel-opt-venv`，不另建环境。

## 8. 前瞻判据与唯一分支决策树

令每臂 `M` 为三块原始 median 的中位数，`I=[min(block medians), max(block medians)]`。
主要门槛：`1-M_H/M_G0 >= 0.02` 且 `max(I_H) < min(I_G0)`。
至少 3/4 cell 同时满足，并覆盖两个任务，才称 H 对 G0 通过投资门槛。
这是本次事先选择的 2% 门槛，不是历史结果的 gate，也不是统计显著性或置信区间。
P−L、H−P 用同一尺度逐 cell 报告，但不另开分支或用次要结果替代主要失败。
全部四 cell 都报告，含失败、接近、inconclusive；不能只报成功的三个或推广到所有任务。

1. 全四 cell 固定留在主要 gate 分母；已确定的无效 H 是失败机会而非删行理由，不替换，也不单独阻止后续。
2. 核心或测量 censored 时全局 gate 为 inconclusive，不触发 A；全部比较已解析（含确定失败）且主要 gate 通过才选 A，A 无法就绪则停止。
3. 未选 A 时，至少一个预指定且有效 H 满足声明的 native expansion eligibility 即选 B，仅运行合格原 cell 的配对。
4. 没有 B 合格 cell 时，至少一个有效 H 的 ordered domain 可由现有 U/W planner 请求域内条件对照，且最小 bridge fixtures 通过，才选 C。
5. 无触发、证据不足或工程超界则停止，不能手选轴制造条件，也不能换 cell 补齐。

eligibility 在分支前用已有 stats、space、incumbent 和编译信息判定，不按有利延迟筛选；不合格/无效 cell 明记 not applicable。
C 不要求 scanner-OFF 核心已有完成 contrast；只要求可请求，C4 在 C 的 Q/时间内构造验证，未形成可用 contrast 记负机会。
B/C 是合格子集诊断，不是总体 gate；censored 核心不提供正向全局结论，但已解析有效 cell 仍可作有界诊断。
最多 A/B/C 中一个；B 优先于 C；任一分支最多 8 个新增 study 机会，总计最多 24。

## 9. 可选 A：两任务、两臂、两轮真实闭环

触发只由第 8 节主要 gate 决定；从两任务原始 P1 重启，不挑核心最好 H 作父。
任务 × G0/H × 2 轮 = 8 个 B40；每轮各臂都做新 rewrite 和一次 parameterization。
H 对实际当前父每轮采 12 fresh endpoints，2 × 2 × 12 = 48 次；G0 不采。
首轮 sampler=1，第二轮 sampler=2，evaluation/validator=0；expansion/S7/scanner 仍 OFF。
传播真实 native selected source、params、space、history 及未晋级反馈；保留父也照实传播。
固定 A 槽表：A0=task43/G0，A1=task21/H，B0=task21/G0，B1=task43/H；每臂两轮顺序执行，臂间不共享新历史。
每臂在自己的 GPU 完成每轮当前父与 child 的 native selected 配置各三块；保留父时可复用该轮父块作为 ready 状态。
每任务共同截点记作 `H_time=min(total_elapsed_G0,total_elapsed_H)`，避免与 H arm 混淆。
取各臂 `elapsed_ready <= H_time` 的最新完整状态，没有则原父；全程计入采集、模型、等待、tuning、finals。
ready-trace 的 `R` 分母为该臂自己 GPU 上初始父 fresh 三块 median；原父 fallback 为 R=1、范围 [1,1]。
使用既有保守比值范围 `[min(child)/max(parent), max(child)/min(parent)]`，不是置信区间。
A 的前瞻门槛是两个任务都满足相对 R 改善至少 2% 且 H 范围上界低于 G0 下界。
同时完整报告两轮最终终点，不能用晚到终点替换共同截点结果；任一不完整即 inconclusive。
上述跨设备 ready 比较只用如旧 T5 的归一化 R；明确设备分配限制，不直接比较跨 GPU raw latency。
随后做配对 terminal finals：task43 的两臂终点都到 A0，task21 的两臂终点都到 B0，每产物三块，顺序正/反/正。
terminal 使用对应比较 GPU 的本次初始父 fresh 三块作共同归一化分母；不取另一卡或历史父测量。
跨主机产物转移与这批终测在 ready-trace 后单独计时，绝不回填或改变先前 elapsed_ready/H_time。
完整计数：4 臂 × 2 轮 ×（父3块＋child3块）=48块，已含首轮初始父12块；额外终测2任务×2臂×3=12块。
合计60块、6,000性能样本、300次 correctness；全部及转移等待均纳入 A 的2.5h调度预算，不为补终测延长。
预计 1.5 至 2.5h，调度上限 2.5h，不增加第三轮，不因一任务先赢就延长另一任务。

## 10. 可选 B：native 扩域对额外同域搜索

仅在 A 未触发且至少一个预指定有效 H 有 native eligibility 时运行；匹配子集只按该条件选择，不按有利延迟。
其余原 cell 明记 not applicable，原因包括无效或无扩域触发；不替换候选，不加事后好点 anchors。
每个合格 H 从相同核心后状态分两支：同域 fresh B40，或 native cap=1 expansion 后 fresh B40。
令合格数 k∈[1,4]，2k≤8 studies，sampler=1、evaluation/validator=0；S7/scanner OFF，无结构 rewrite。
扩域允许既有 parameterizer 扩域调用及原有限验证重试，计入模型成本；不增加独立 axis selector。
两支初始为同一源码、状态及 identical eligible incumbent anchors，保留原始 default witnesses；不能拿核心 B40 充当对照。
旧 best 仅在合法且旧配置源码等价时复用缓存，否则沿现有 evaluation 重测，不以旧分数保护新源码。
若扩域拒绝或 no-op，记录失去的机会，不手动添加 N256/SM64 或改轴重跑。
逐 cell 同卡测 native selected finals，报告域增加了什么、失败、耗时以及是否优于同域额外预算。
保留原 core cell GPU；rep0 对照先、rep1 扩域先，配对间不得同时计时；父＋两支 selected 各三块=9k块。
完整时至多36块，按正/反/正轮换终测；用2%与范围不重叠作子集描述，非总体 gate，不再授权 C。
预计 45 至 90min，调度上限 90min；不同时执行 C。

## 11. 可选 C：域内 conditional 的有限正确性与效用

仅 A 未触发、B 无合格 cell 且最小 bridge fixtures 通过时考虑 C；不是安全研究或全面 scanner 修复。
至少一个预指定有效 H 的 ordered domain 与既有 U/W planner 能请求域内条件对照即可，不要求已有完成 contrast。
仅这个 eligibility 子集配对；其他 cell 明记 not applicable，不按结果筛选，不替换；合格数 k∈[1,4]。
每对同 source/domain/状态与初始 anchors，OFF/ACTIVE 各一个 fresh B40；2k≤8 studies，sampler=1、evaluation/validator=0，无 LLM/扩域/S7。
`f_frac=0.2` 给 B40 内 `Q=8`，是相对旧实验 0.125 的**新预注册剂量**，不是未改动旧策略。
Q 是 B40 内份额，不是 B40 外赠送 8 个性能 trial；诊断编译工作单独计入时间和成本。
在所分配 Q 和时间内构造并验证 C4，再按现有路径到 E1；没有可用 contrast 就记录负机会，不在 cap 外重试。
`pcap=12`，每 study 最多 2 次 diagnostic dispatch，累计 probe 预算 120s，必须真实可执行。
等待、重试、取消与失败都消耗同一预算；重试也是 dispatch，不能另开免费额度。
仅事后累计 elapsed 或预计预算 admission 不足以保证 120s，必须能传递剩余 deadline 并取消在途工作。
若现有 worker 无法可靠执行该限制，不宣称 timeout 已受控，C 判 not ready 并跳过。
默认只用 E1，复用现有选择；不修 E2/E4、不造递归 token、不写新轴选择器。
partner admission 或 continuous neighbors 仅在所选 E1 路径确实不可达且小修足够时纳入。
最小修改加测试超过 2 工程小时，或依赖广泛 scanner 重写，立即跳过 C。
报告 admission、contrast 完整度、E1 结果、native selected finals 和成本；未产出 contrast 不是成功。
仍用各原 core cell GPU，rep0 OFF先、rep1 ACTIVE先；父＋两支 selected 各三块=9k≤36块，正/反/正轮换，禁止同卡同时计时。
结论仅为该 eligibility 子集诊断，不作总体门槛或通过者筛选；全部负机会与未完成项保留。
预计 45 至 90min，调度上限 90min；不追加替代候选或另一分支。

## 12. 提前扩域：当前能做什么，什么延后

当前 `_maybe_expand_space` 在初始 B40 和统计之后按边际改善/余量策略触发，不是持续开域。
优先在首次 parameterization 表达 rewrite 预期区域，同时用现有域内 conditional neighbors 利用局部信息。
不要直接在正在运行的 Optuna study 中修改 categorical choices，那不是有效的同一 study 域变更。
更频繁的真正 mid-B40 space opening 是延后的新实验策略，B/C 不代表已经验证它。
重访前提：B 显示额外域而非单纯额外 B40 有可用收益，或记录到可重复的开域时机损失。
还需预先定义独立新 space/study 边界、合法 anchors、来源兼容与每 space 完整 B40 的新增预算。
只有这些条件和独立预算获确认后才能另做设计；本方案不偷偷切分 B40 或声称覆盖所有搜索想法。

## 13. 环境、时间与成本口径

未来新产物统一放 `/root/autodl-tmp/c2-method-improvements/`，按 core/cell/arm 与所选 branch 分目录。
沿用 `/root/autodl-tmp/c2-local-experiment/code/v5` 部署源码，或执行前明确指定一个新源码副本。
Linux 显式设 `PYTHONPATH=/root/autodl-tmp/c2-local-experiment/code/v5/src:/root/autodl-tmp/c2-local-experiment/code/v5`。
若改源码副本，两段路径一起替换并输出 `kernel_optimizer.__file__`，防止脏 v2 导入。
HOST-A worker 用已有 `kernel-opt-venv`，HOST-B worker 用已有 `orch-venv`，两侧 orchestrator 都用 `orch-venv`。
不建新环境或设施；主机地址沿用既有 handoff，旧 GPU 空闲快照不代表执行时仍可用。

| 时间项 | 依据或估算 | 口径 |
|---|---|---|
| 历史 native 六 studies | 55m04s；单 run 约 20.7 至 29.3min | 并发执行窗口，不是串行和 |
| 历史 12 fresh 机会 | 2h51m34s | 含生成与 native 工作 |
| 历史四条两轮轨迹 | 2h17m36s | 共八 B40，不能按四卡机械除二 |
| 并行实施 | 4 至 8h | 有阻塞则预留 1 至 2 工作日，不作当天保证 |
| 核心四卡实验 | 2 至 3.5h，调度硬上限 4h | 16 机会，含 fresh acquisition 与 finals |
| 可选 B/C | 0.75 至 1.5h，上限 1.5h | 只选一个 |
| 可选 A | 1.5 至 2.5h，上限 2.5h | 与 B/C 互斥 |
| 分析 | 0.5 至 1h | 全 cell、失败和成本均报告 |

强制部分典型合计为实施 4 至 8 + 核心 2 至 3.5 + 分析 0.5 至 1 = **6.5 至 12.5h**。
仅强制实验加分析为 **2.5 至 4.5h**；此范围不含实施或可选分支。
含可选分支的 GPU program 调度最大值为 4+max(2.5,1.5)=**6.5h**，不是 12h。
再加分析为 7 至 7.5h 的最大调度预算口径，实施另算，在途 drain 另报。
这里的硬上限是停止调度上限，除非取消机制能保证，否则不是严格墙钟完成承诺。
到期不启新 arm、study、acquisition 或模型工作；不为凑数破坏正在进行的单次计时。
在途工作正常完成或经现有 worker timeout 终止，drain 单列；未完成 arm 计 censored/inconclusive。
用首批完成 job 校准预测与排队，不修改已声明 N、B、gate，不隐藏延期或补跑。
成本只读已有 events：实际共享排程 span 与每种方法独立部署的估计成本分开列。
共享排程只计一次 acquisition/P rewrite；独立 L/P/H 方法各应承担自身 acquisition、模型与参数化成本。
等待、重试、validation、screening、失败、finals 均计入；worker wall 不等于 GPU busy，不能重复相加。
provider 未提供的 token/货币成本写 unknown，不写零；P/H 实验复用不等于部署免费。

## 14. 交付判定

未来报告应列出 16 个核心机会和唯一分支的至多 8 个机会，实际完成数与机会数分开。
附所有 cell 的三块值、原生选择、父保留、拒绝/截断理由、实际排程和独立方法成本估计。
结论只回答提示包、intent 或被选组件是否值得继续；失败和不确定同样是完整结果。
本文件仅授权方案层面的明确设计，不声称改动已落地、可选分支已执行或完整框架收益已获验证。
